from __future__ import annotations

from collections.abc import Mapping
from typing import TypedDict

import torch
import torch.nn.functional as F
from anydataset.types import Modality
from torch import Tensor

from ..._oom import annotate, tensor_report
from ...task import Task
from ..protocol import GenerationRuntime, TextEvaluationModel
from ..service import generate_responses
from ..types import Request


class TextProbe(TypedDict):
    instruction: str
    reference: str


class TextProbeResult(TypedDict):
    generated: str
    nll: float


@torch.no_grad()
def evaluate_text(
    probes: Mapping[str, TextProbe],
    model: TextEvaluationModel,
    *,
    max_new_tokens: int,
) -> dict[str, TextProbeResult]:
    runtime = model.runtime
    prompts = {
        name: _prompt_ids(runtime, probe["instruction"])
        for name, probe in probes.items()
    }
    requests = [
        Request(
            prompt_ids=prompts[name],
            task=Task.T2TT,
            audio_input_positions=None,
            audio_context=None,
        )
        for name in probes
    ]
    try:
        generations = generate_responses(
            requests,
            model,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    except torch.OutOfMemoryError as error:
        annotate(
            error,
            phase="text_evaluation_generation",
            inputs={
                "type": "TextGenerationRequests",
                "prompt_ids": [tensor_report(value) for value in prompts.values()],
                "padded_prompt_shape": [
                    len(prompts),
                    max((value.numel() for value in prompts.values()), default=0),
                ],
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "use_cache": True,
            },
        )
        raise

    results: dict[str, TextProbeResult] = {}
    for (name, probe), generation in zip(probes.items(), generations):
        results[name] = TextProbeResult(
            generated=decode_text_ids(runtime, generation["response_ids"]),
            nll=_reference_nll(model, prompts[name], probe["reference"]),
        )
    return results


def _prompt_ids(runtime: GenerationRuntime, instruction: str) -> Tensor:
    ids = runtime.text_tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=False,
    )
    local_ids = torch.as_tensor(ids, dtype=torch.long)
    return runtime.layout.to_global(Modality.TEXT.value, local_ids)


def _reference_nll(
    model: TextEvaluationModel,
    prompt_ids: Tensor,
    reference: str,
) -> float:
    runtime = model.runtime
    text_start, _ = runtime.layout.blocks[Modality.TEXT.value]
    local_reference = torch.tensor(
        runtime.text_tokenizer.encode(reference, add_special_tokens=False),
        dtype=torch.long,
    )
    reference_ids = runtime.layout.to_global(Modality.TEXT.value, local_reference)
    response_ids = torch.cat(
        (reference_ids, torch.tensor([runtime.eos_token_id], dtype=torch.long))
    )
    input_shape = [1, prompt_ids.numel() + response_ids.numel()]
    try:
        device = model.backbone.get_input_embeddings().weight.device
        input_ids = torch.cat((prompt_ids, response_ids)).to(device=device)[None]
        hidden_states = model.token_hidden_states(
            input_ids,
            attention_mask=torch.ones_like(input_ids, dtype=torch.bool),
        )
        predictors = hidden_states[0, prompt_ids.numel() - 1 : -1]
        prediction = model.token_logits(predictors, Modality.TEXT).float()
        target = input_ids[0, prompt_ids.numel() :] - text_start
        return float(F.cross_entropy(prediction, target).detach().cpu())
    except torch.OutOfMemoryError as error:
        annotate(
            error,
            phase="text_evaluation_reference_nll",
            inputs={
                "type": "TextReferenceNLL",
                "input_ids_shape": input_shape,
                "prompt_tokens": prompt_ids.numel(),
                "reference_tokens_with_eos": response_ids.numel(),
            },
        )
        raise


def decode_text_ids(runtime: GenerationRuntime, token_ids: Tensor) -> str:
    if token_ids.numel():
        local_ids = runtime.layout.to_local(token_ids).detach().cpu().tolist()
    else:
        local_ids = []
    return runtime.text_tokenizer.decode(local_ids, skip_special_tokens=True)


__all__ = ["TextProbe", "TextProbeResult", "decode_text_ids", "evaluate_text"]
