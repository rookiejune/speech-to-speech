from __future__ import annotations

from collections.abc import Sequence
import unittest
from types import SimpleNamespace

import torch
from anydataset.types import Modality
from anytrain.module.idspace import Layout
from torch import Tensor, nn
from transformers.cache_utils import Cache

from speech_to_speech.loss.supervised import TokenObjective
from speech_to_speech.model.base import Model
from speech_to_speech.model.ctc import (
    CTCDecoderConfig,
    CTCDecoderRoutes,
    CTCDecoderRoutesConfig,
    CTCRoute,
)
from speech_to_speech.task import PredictionModality
from speech_to_speech.runtime.backbone import BackboneBodyAdapter, BackboneReadout


class _Output:
    last_hidden_state: Tensor | Sequence[Tensor]
    hidden_states: tuple[Tensor, ...] | None
    past_key_values: Cache | None
    attentions: tuple[Tensor, ...] | None

    def __init__(self, text: Tensor, audio: Tensor) -> None:
        self.last_hidden_state = (text, audio)
        self.hidden_states = (text + 10, audio + 10)
        self.past_key_values = None
        self.attentions = None


class _Body:
    def __init__(self) -> None:
        self.config = SimpleNamespace(return_dict=False)
        self.calls: list[dict[str, object]] = []
        self.text = torch.full((1, 2, 3), 1.0)
        self.audio = torch.full((1, 2, 3), 2.0)

    def __call__(self, *, return_dict: bool | None = None, **kwargs: object) -> _Output:
        if return_dict is not True:
            raise AssertionError("backbone body must return an attribute output")
        kwargs["return_dict"] = return_dict
        self.calls.append(kwargs)
        return _Output(self.text, self.audio)


def _encode(
    adapter: BackboneBodyAdapter,
    *,
    modality: Modality | None = None,
) -> Tensor:
    return adapter.encode(
        inputs_embeds=torch.zeros(1, 2, 3),
        attention_mask=torch.ones(1, 2, dtype=torch.bool),
        output_hidden_states=False,
        modality=modality,
    ).last_hidden_state


class _RoutingModel(Model):
    def __init__(self, encoder: BackboneBodyAdapter) -> None:
        nn.Module.__init__(self)
        self._encoder = encoder

    def _input_embedding(
        self,
        input_ids: Tensor,
        audio_input_positions: Tensor | None = None,
        **kwargs: object,
    ) -> Tensor:
        del audio_input_positions, kwargs
        return input_ids[..., None].to(dtype=torch.float32)

    def _audio_head_uses_sequence_context(self) -> bool:
        return False

    def modality_logits(
        self,
        hidden_state: Tensor,
        modality: Modality,
        **kwargs: object,
    ) -> tuple[Tensor, object | None]:
        del modality, kwargs
        return hidden_state, None

    def selected_logits(
        self,
        hidden_state: Tensor,
        token_ids: Tensor,
        **kwargs: object,
    ) -> tuple[Tensor, object | None]:
        del token_ids, kwargs
        return hidden_state, None


class _ObjectiveModel:
    def __init__(self, layout: Layout) -> None:
        self.layout = layout
        self.predictions: list[PredictionModality | None] = []

    def token_hidden_states(
        self,
        input_ids: Tensor,
        *,
        prediction: PredictionModality | None = None,
        **kwargs: object,
    ) -> Tensor:
        del kwargs
        self.predictions.append(prediction)
        return torch.zeros(*input_ids.shape, 3)

    def token_logits(
        self,
        hidden_state: Tensor,
        modality: Modality | None = None,
        **kwargs: object,
    ) -> Tensor:
        del kwargs
        if modality is None:
            raise ValueError("token loss must select a modality")
        start, end = self.layout.blocks[modality.value]
        return hidden_state.new_zeros(hidden_state.size(0), end - start)

    def project_audio_hidden(
        self,
        hidden_state: Tensor,
        **kwargs: object,
    ) -> tuple[Tensor, object | None]:
        del kwargs
        return hidden_state, None


class BackboneReadoutRoutingTest(unittest.TestCase):
    def test_body_forces_attribute_output_despite_config_default(self) -> None:
        body = _Body()
        adapter = BackboneBodyAdapter(
            body,
            readout=BackboneReadout("last_hidden_state[0]"),
        )

        selected = adapter.encode(
            inputs_embeds=torch.zeros(1, 2, 3),
            attention_mask=torch.ones(1, 2, dtype=torch.bool),
            output_hidden_states=False,
            extra={"return_dict": False},
        ).last_hidden_state

        self.assertFalse(body.config.return_dict)
        self.assertTrue(torch.equal(selected, body.text))
        self.assertIs(body.calls[-1]["return_dict"], True)

    def test_readout_parses_attribute_index_and_history_requirement(self) -> None:
        indexed = BackboneReadout("last_hidden_state[1]")
        self.assertEqual(indexed.attribute, "last_hidden_state")
        self.assertEqual(indexed.index, 1)
        self.assertFalse(indexed.requires_hidden_states)

        history = BackboneReadout("hidden_states[2]")
        self.assertEqual(history.attribute, "hidden_states")
        self.assertEqual(history.index, 2)
        self.assertTrue(history.requires_hidden_states)

    def test_indexed_last_hidden_state_does_not_request_layer_history(self) -> None:
        body = _Body()
        adapter = BackboneBodyAdapter(
            body,
            readout=BackboneReadout("last_hidden_state[1]"),
        )

        selected = _encode(adapter)

        self.assertTrue(torch.equal(selected, body.audio))
        self.assertIs(body.calls[-1]["output_hidden_states"], False)

    def test_hidden_states_readout_requests_layer_history(self) -> None:
        body = _Body()
        adapter = BackboneBodyAdapter(
            body,
            readout=BackboneReadout("hidden_states[1]"),
        )

        selected = _encode(adapter)

        self.assertTrue(torch.equal(selected, body.audio + 10))
        self.assertIs(body.calls[-1]["output_hidden_states"], True)

    def test_modality_readouts_route_homogeneous_training_predictions(self) -> None:
        body = _Body()
        adapter = BackboneBodyAdapter(
            body,
            readout=BackboneReadout("last_hidden_state[0]"),
            modality_readouts={
                Modality.TEXT: BackboneReadout("last_hidden_state[0]"),
                Modality.AUDIO: BackboneReadout("last_hidden_state[1]"),
            },
        )
        model = _RoutingModel(adapter)
        input_ids = torch.zeros(1, 2, dtype=torch.long)

        text = model.token_hidden_states(
            input_ids,
            prediction=PredictionModality.TEXT,
        )
        audio = model.token_hidden_states(
            input_ids,
            prediction=PredictionModality.AUDIO,
        )

        self.assertTrue(torch.equal(text, body.text))
        self.assertTrue(torch.equal(audio, body.audio))

    def test_mixed_prediction_rejects_modality_specific_readouts(self) -> None:
        body = _Body()
        model = _RoutingModel(
            BackboneBodyAdapter(
                body,
                modality_readouts={
                    Modality.TEXT: BackboneReadout("last_hidden_state[0]"),
                    Modality.AUDIO: BackboneReadout("last_hidden_state[1]"),
                },
            )
        )

        with self.assertRaisesRegex(ValueError, "mixed prediction modalities"):
            model.token_hidden_states(
                torch.zeros(1, 2, dtype=torch.long),
                prediction=PredictionModality.PARALLEL,
            )
        self.assertEqual(body.calls, [])

    def test_shared_default_readout_preserves_mixed_prediction_behavior(self) -> None:
        body = _Body()
        model = _RoutingModel(
            BackboneBodyAdapter(
                body,
                readout=BackboneReadout("last_hidden_state[0]"),
            )
        )

        hidden = model.token_hidden_states(
            torch.zeros(1, 2, dtype=torch.long),
            prediction=PredictionModality.INTERLEAVED,
        )

        self.assertTrue(torch.equal(hidden, body.text))

    def test_ctc_routes_select_two_readouts_from_one_backbone_forward(self) -> None:
        body = _Body()
        model = _RoutingModel(
            BackboneBodyAdapter(
                body,
                modality_readouts={
                    Modality.TEXT: BackboneReadout("last_hidden_state[0]"),
                    Modality.AUDIO: BackboneReadout("last_hidden_state[1]"),
                },
            )
        )
        model.ctc_decoders = CTCDecoderRoutes(
            CTCDecoderRoutesConfig(
                source=CTCDecoderConfig(
                    backbone_readout="hidden_states[0]"
                ),
                target=CTCDecoderConfig(
                    backbone_readout="last_hidden_state[1]"
                ),
            ),
            hidden_size=3,
        )

        output = model.objective_hidden_output(
            torch.zeros(1, 2, dtype=torch.long),
            ctc_routes=frozenset({CTCRoute.SOURCE, CTCRoute.TARGET}),
            prediction=PredictionModality.TEXT,
        )

        self.assertEqual(len(body.calls), 1)
        self.assertIs(body.calls[0]["output_hidden_states"], True)
        self.assertTrue(torch.equal(output.token, body.text))
        self.assertTrue(torch.equal(output.source_ctc, body.text + 10))
        self.assertTrue(torch.equal(output.target_ctc, body.audio))

    def test_generation_step_uses_the_existing_modality(self) -> None:
        body = _Body()
        model = _RoutingModel(
            BackboneBodyAdapter(
                body,
                readout=BackboneReadout("last_hidden_state[0]"),
                modality_readouts={
                    Modality.AUDIO: BackboneReadout("last_hidden_state[1]")
                },
            )
        )

        output = model.generation_step(
            torch.zeros(1, 2, dtype=torch.long),
            attention_mask=torch.ones(1, 2, dtype=torch.bool),
            output_hidden_states=True,
            token_ids=None,
            modality=Modality.AUDIO,
            past_key_values=None,
            use_cache=False,
        )

        self.assertIsNotNone(output.hidden_states)
        assert output.hidden_states is not None
        self.assertTrue(torch.equal(output.hidden_states[0], body.audio))

    def test_generation_token_kind_routes_homogeneous_constrained_tokens(self) -> None:
        body = _Body()
        model = _RoutingModel(
            BackboneBodyAdapter(
                body,
                readout=BackboneReadout("last_hidden_state[0]"),
                modality_readouts={
                    Modality.AUDIO: BackboneReadout("last_hidden_state[1]")
                },
            )
        )

        output = model.generation_step(
            torch.zeros(1, 2, dtype=torch.long),
            attention_mask=torch.ones(1, 2, dtype=torch.bool),
            output_hidden_states=True,
            token_ids=torch.tensor([4]),
            token_kind=Modality.AUDIO.value,
            modality=None,
            past_key_values=None,
            use_cache=False,
        )

        self.assertIsNotNone(output.hidden_states)
        assert output.hidden_states is not None
        self.assertTrue(torch.equal(output.hidden_states[0], body.audio))

    def test_mixed_constrained_generation_rejects_modality_readouts(self) -> None:
        body = _Body()
        model = _RoutingModel(
            BackboneBodyAdapter(
                body,
                modality_readouts={
                    Modality.TEXT: BackboneReadout("last_hidden_state[0]"),
                    Modality.AUDIO: BackboneReadout("last_hidden_state[1]"),
                },
            )
        )

        with self.assertRaisesRegex(ValueError, "mixed generation tokens"):
            model.generation_step(
                torch.zeros(1, 2, dtype=torch.long),
                attention_mask=torch.ones(1, 2, dtype=torch.bool),
                output_hidden_states=False,
                token_ids=torch.tensor([1, 4]),
                token_kind="mixed",
                modality=None,
                past_key_values=None,
                use_cache=False,
            )
        self.assertEqual(body.calls, [])

    def test_objective_passes_batch_prediction_to_model(self) -> None:
        layout = Layout(text=(0, 4), audio=(4, 8))
        model = _ObjectiveModel(layout)
        batch = SimpleNamespace(
            input_ids=torch.tensor([[0, 1]], dtype=torch.long),
            token_labels=torch.tensor([[-100, 1]], dtype=torch.long),
            attention_mask=torch.ones(1, 2, dtype=torch.bool),
            audio_input_positions=None,
            input_modalities=None,
            prediction_modality=PredictionModality.TEXT,
        )

        TokenObjective(layout)(batch, model)  # type: ignore[arg-type]

        self.assertEqual(model.predictions, [PredictionModality.TEXT])


if __name__ == "__main__":
    unittest.main()
