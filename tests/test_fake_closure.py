from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from anydataset.types import (
    AudioItem,
    AudioView,
    Modality,
    Role,
    TextItem,
    TextMeta,
    TextView,
)
from anytrain.idspace import Layout
from torch import Tensor, nn

from speech_to_speech.datamodule.collator import Collator
from speech_to_speech.datamodule.types import Task
from speech_to_speech.loss import Loss
from speech_to_speech.model.acoustic import SpeechToSpeechFlowModel
from speech_to_speech.model.base import Config as ModelConfig
from speech_to_speech.pl_module.decode import (
    decode_generated_audio,
    decode_generated_codes,
)
from speech_to_speech.runtime.audio_tokenizer import NativeAudioTokenizer


class _TextTokenizer:
    _placeholder = "$$$PLACEHOLDER$$$"

    def __len__(self) -> int:
        return 32

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        if text == self._placeholder:
            return [7, 8]
        return [4 + sum(text.encode("utf-8")) % 4, 9]

    def apply_chat_template(self, conversation, **kwargs) -> str | list[int]:
        content = conversation[0]["content"]
        rendered = f"<user>{content}</user><assistant>"
        if not kwargs["tokenize"]:
            return rendered
        if self._placeholder in content:
            return [1, 2, 17, 8, 3]
        return [1, 2, 3]


class _Codec:
    acoustic_feature_dim = 4
    acoustic_codebook_sizes = (16,)

    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(0)
        self.semantic_codebook = torch.randn(8, 4, generator=generator)

    def acoustic_codes_to_features(self, acoustic_codes: Tensor) -> Tensor:
        if acoustic_codes.dim() != 3 or acoustic_codes.size(-1) != 1:
            raise ValueError("fake codec expects [batch, frame, 1] acoustic codes.")
        values = acoustic_codes[..., 0].to(dtype=torch.float64)
        return torch.stack((values, values.square(), values + 1, values * 0.5), dim=-1)

    def decode_features(
        self, semantic_codes: Tensor, acoustic_features: Tensor
    ) -> Tensor:
        if semantic_codes.dim() != 3:
            raise ValueError(
                "fake codec expects [batch, frame, codebook] semantic codes."
            )
        if semantic_codes.shape[:2] != acoustic_features.shape[:2]:
            raise ValueError("fake semantic codes and acoustic features must align.")
        semantic = semantic_codes.to(dtype=acoustic_features.dtype).sum(dim=-1)
        return semantic + acoustic_features[..., 0]


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.input_embeddings = nn.Embedding(32, 4)
        self.output_embeddings = nn.Linear(4, 32, bias=False)
        self.rnn = nn.GRU(4, 4, batch_first=True)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.input_embeddings

    def get_output_embeddings(self) -> nn.Module:
        return self.output_embeddings

    @property
    def base_model(self) -> _Backbone:
        return self

    def forward(self, *, inputs_embeds: Tensor, **kwargs):
        output_hidden_states = kwargs["output_hidden_states"]
        hidden, _ = self.rnn(inputs_embeds)
        return SimpleNamespace(
            last_hidden_state=hidden,
            hidden_states=(hidden,) if output_hidden_states else None,
            past_key_values=None,
            attentions=None,
        )


class _FlowRuntime:
    def __init__(self) -> None:
        self.sampled = False

    def training_sample(self, x_1: Tensor, *, x_0: Tensor | None = None):
        del x_0
        x_0 = torch.zeros_like(x_1)
        return SimpleNamespace(
            x_t=x_1 * 0.5,
            velocity=x_1 - x_0,
            t=torch.full((x_1.size(0),), 0.5, device=x_1.device),
        )

    def sample(self, model: nn.Module, x_0: Tensor, **model_extras: object):
        del model, model_extras
        self.sampled = True
        return SimpleNamespace(final=torch.zeros_like(x_0))


class _Runtime:
    def __init__(self) -> None:
        self.config = SimpleNamespace(audio_view=AudioView.LONGCAT)
        self.text_tokenizer = _TextTokenizer()
        self.audio_tokenizer = NativeAudioTokenizer(vocab_size=8)
        self.codec = _Codec()
        self.backbone = _Backbone()
        self.layout = Layout(text=(0, 32), audio=(32, 42))
        self.flow_matching = _FlowRuntime()
        self.pad_token_id = 0
        self.eos_token_id = 10
        self.boa_token_id = 40
        self.eoa_token_id = 41

    @property
    def codec_audio_range(self) -> tuple[int, int]:
        return 32, 40


class FakeClosureTest(unittest.TestCase):
    def test_flow_model_uses_runtime_sampler(self):
        rt = _Runtime()
        with _runtime(rt):
            model = SpeechToSpeechFlowModel(
                ModelConfig(
                    semantic_audio_adapter=None,
                    semantic_audio_output_adapter=None,
                    acoustic_prompt_adapter=None,
                ),
                runtime_snapshot=rt,
            )
            output = model.sample_acoustic(torch.zeros(2, 3, 4))

        self.assertTrue(rt.flow_matching.sampled)
        self.assertEqual(output.shape, (2, 3, 4))

    def test_all_tasks_build_expected_model_batches(self):
        rt = _Runtime()
        with _runtime(rt):
            for task in Task:
                with self.subTest(task=task.value):
                    batch = Collator({task: 1.0})([_raw_sample(0), _raw_sample(1)])

                    self.assertEqual(batch.tasks, [task, task])
                    self.assertEqual(batch.input_ids.shape, batch.labels.shape)
                    self.assertEqual(
                        batch.acoustic_input_ids is not None,
                        task.source_modality is Modality.AUDIO,
                    )
                    self.assertEqual(
                        batch.acoustic_labels is not None,
                        task.target_modality is Modality.AUDIO,
                    )
                    if task.target_modality is Modality.AUDIO:
                        supervised = batch.labels[0].ne(-100).nonzero().flatten()
                        first = int(supervised[0])
                        last = int(supervised[-1])
                        self.assertEqual(int(batch.input_ids[0, first - 1]), rt.boa_token_id)
                        self.assertEqual(int(batch.input_ids[0, last]), rt.eoa_token_id)

    def test_all_task_paths_forward_backward_and_update_parameters(self):
        for task in Task:
            with self.subTest(task=task.value):
                torch.manual_seed(0)
                rt = _Runtime()
                with _runtime(rt):
                    batch = Collator({task: 1.0})([_raw_sample(0), _raw_sample(1)])
                    model = SpeechToSpeechFlowModel(
                        ModelConfig(
                            semantic_audio_adapter=None,
                            semantic_audio_output_adapter=None,
                            acoustic_prompt_adapter=None,
                        ),
                        runtime_snapshot=rt,
                    )
                    loss = Loss(rt.layout, rt.flow_matching)
                    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                    before = {
                        name: parameter.detach().clone()
                        for name, parameter in model.named_parameters()
                    }

                    outputs = loss(batch, model)
                    optimizer.zero_grad()
                    outputs["loss"].backward()
                    optimizer.step()

                    self.assertTrue(torch.isfinite(outputs["loss"]))
                    self.assertEqual(
                        "flow_matching" in outputs,
                        task.target_modality is Modality.AUDIO,
                    )
                    self.assertTrue(
                        any(
                            not torch.equal(before[name], parameter.detach())
                            for name, parameter in model.named_parameters()
                        )
                    )

    def test_fake_semantic_and_acoustic_outputs_decode_to_waveform(self):
        rt = _Runtime()
        with _runtime(rt):
            batch = Collator({Task.TTS: 1.0})([_raw_sample(0)])
            model = SpeechToSpeechFlowModel(
                ModelConfig(
                    semantic_audio_adapter=None,
                    semantic_audio_output_adapter=None,
                    acoustic_prompt_adapter=None,
                ),
                runtime_snapshot=rt,
            )
            labels = batch.labels[0]
            start, end = rt.codec_audio_range
            semantic = labels[labels.ge(start) & labels.lt(end)][None]
            assert batch.acoustic_labels is not None
            features = model.acoustic_target_latent(batch.acoustic_labels)
            self.assertEqual(
                features.dtype,
                rt.backbone.get_input_embeddings().weight.dtype,
            )

            waveform = decode_generated_audio(
                semantic,
                features,
                codec=rt.codec,
                audio_tokenizer=rt.audio_tokenizer,
                audio_token_range=rt.codec_audio_range,
            )

            self.assertEqual(waveform.shape, (1, 3))
            self.assertTrue(torch.isfinite(waveform).all())
            decoded_codes = decode_generated_codes(
                semantic,
                batch.acoustic_labels,
                codec=rt.codec,
                audio_tokenizer=rt.audio_tokenizer,
                audio_token_range=rt.codec_audio_range,
            )
            self.assertTrue(torch.equal(decoded_codes, waveform))


def _runtime(rt: _Runtime):
    return patch("speech_to_speech.runtime.singleton._runtime", rt)


def _raw_sample(offset: int):
    def audio(base: int) -> AudioItem:
        return AudioItem(
            views={
                AudioView.LONGCAT: torch.tensor(
                    [
                        [base, base + 1],
                        [base + 1, base + 2],
                        [base + 2, base + 3],
                    ]
                )
            }
        )

    return {
        (Role.SOURCE, Modality.AUDIO): audio(offset),
        (Role.SOURCE, Modality.TEXT): TextItem(
            views={TextView.TEXT: f"source {offset}"},
            meta={TextMeta.LANG: "zh"},
        ),
        (Role.TARGET, Modality.AUDIO): audio(offset + 3),
        (Role.TARGET, Modality.TEXT): TextItem(
            views={TextView.TEXT: f"target {offset}"},
            meta={TextMeta.LANG: "en"},
        ),
    }


if __name__ == "__main__":
    unittest.main()
