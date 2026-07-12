from __future__ import annotations

from collections.abc import Mapping
from typing import TypedDict

import torch
import torch.nn.functional as F
from anydataset.types import Modality
from torch import Tensor

from ..datamodule.types import Task
from ..model.acoustic import SpeechToSpeechFlowModel
from .generation import Request, generate


class TextProbe(TypedDict):
    instruction: str
    reference: str


class TextProbeResult(TypedDict):
    generated: str
    nll: float


@torch.no_grad()
def evaluate_text(
    probes: Mapping[str, TextProbe],
    model: SpeechToSpeechFlowModel,
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
            acoustic_input_ids=None,
            acoustic_input_positions=None,
        )
        for name in probes
    ]
    generations = generate(
        requests,
        model,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    results: dict[str, TextProbeResult] = {}
    for (name, probe), generation in zip(probes.items(), generations):
        results[name] = TextProbeResult(
            generated=_decode(runtime, generation["token_ids"]),
            nll=_reference_nll(model, prompts[name], probe["reference"]),
        )
    return results


def _prompt_ids(runtime, instruction: str) -> Tensor:
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
    model: SpeechToSpeechFlowModel,
    prompt_ids: Tensor,
    reference: str,
) -> float:
    runtime = model.runtime
    text_start, text_end = runtime.layout.blocks[Modality.TEXT.value]
    local_reference = torch.tensor(
        runtime.text_tokenizer.encode(reference, add_special_tokens=False),
        dtype=torch.long,
    )
    reference_ids = runtime.layout.to_global(Modality.TEXT.value, local_reference)
    response_ids = torch.cat(
        (reference_ids, torch.tensor([runtime.eos_token_id], dtype=torch.long))
    )
    device = model.backbone.get_input_embeddings().weight.device
    input_ids = torch.cat((prompt_ids, response_ids)).to(device=device)[None]
    output = model(input_ids, attention_mask=torch.ones_like(input_ids, dtype=torch.bool))
    prediction = output.logits[
        0, prompt_ids.numel() - 1 : -1, text_start:text_end
    ].float()
    target = input_ids[0, prompt_ids.numel() :] - text_start
    return float(F.cross_entropy(prediction, target).detach().cpu())


def _decode(runtime, token_ids: Tensor) -> str:
    if token_ids.numel():
        local_ids = runtime.layout.to_local(token_ids).detach().cpu().tolist()
    else:
        local_ids = []
    return runtime.text_tokenizer.decode(local_ids, skip_special_tokens=True)


__all__ = ["TextProbe", "TextProbeResult", "evaluate_text"]
