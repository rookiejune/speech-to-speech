from __future__ import annotations

import json
import re
import unittest
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from types import SimpleNamespace
from typing import Any, cast

import torch
from anydataset.types import AudioView, Modality
from anytrain.codec import (
    AcousticLayout,
    AudioBackendIdentity,
    AudioCodeSchema,
    AudioCodeSpec,
    SemanticAcousticCodes,
)
from anytrain.module.idspace import Layout
from peft import LoraConfig
from tokenizers import Tokenizer, normalizers, pre_tokenizers
from tokenizers.models import WordLevel
from torch import Tensor, nn
from transformers import PreTrainedTokenizerFast
from transformers.cache_utils import Cache

from speech_to_speech.model import (
    AdapterType,
    AudioInputAdapterConfig,
    AudioInputAdapterType,
    AudioOutputAdapterConfig,
    AudioOutputAdapterType,
    Model,
)
from speech_to_speech.model.checkpoint_contract import (
    MODEL_CONTRACT_GRAMMAR,
    ModelCheckpointContract,
    validate_checkpoint_contract,
)
from speech_to_speech.model.acoustic import DecoderConfig
from speech_to_speech.model.acoustic.flow import FlowModel
from speech_to_speech.model.acoustic.rvq import RVQModel
from speech_to_speech.model.base import Config as ModelConfig
from speech_to_speech.model.ctc import (
    CTCDecoderConfig,
    CTCDecoderRoutesConfig,
    CTCDecoderType,
)
from speech_to_speech.model.acoustic.contract import (
    FlowModelRuntime,
    FlowSample,
    FlowSamplingRuntime,
)
from speech_to_speech.model.toy import ToyConfig, create_toy_backbone
from speech_to_speech.pl_module import Config, SpeechToSpeechModule
from speech_to_speech.runtime import (
    AudioSequenceLayout,
    AudioTokenRegistry,
    AudioTokenSpec,
)
from speech_to_speech.runtime.audio_tokenizer import (
    BiCodecAudioTokenizer,
    FlattenedAudioTokenizer,
    NativeAudioTokenizer,
)
from speech_to_speech.runtime.backbone import (
    BackboneAdapter,
    BackboneBodyAdapter,
    BackboneExtra,
    BackboneOutputView,
)
from speech_to_speech.runtime.protocol import TokenModelRuntime
from speech_to_speech.runtime.codec_contract import (
    CodecBackend,
    SemanticCodec,
)
from speech_to_speech.runtime.audio_tokenizer.contract import AudioTokenizer
from speech_to_speech.runtime.backbone.contract import TextTokenizer
from speech_to_speech.runtime.backbone.contract import (
    Backbone,
    BackboneOutput,
    BackboneReadout,
)
from speech_to_speech.task import ControlToken


class ModelCheckpointContractTest(unittest.TestCase):
    def test_checkpoint_roundtrips_complete_model_contract(self) -> None:
        contract = _contract(1)
        module = _module(contract)
        checkpoint: dict[str, Any] = {}

        module.on_save_checkpoint(checkpoint)

        self.assertEqual(checkpoint["speech_to_speech_model_schema"], "v4")
        self.assertEqual(
            checkpoint["speech_to_speech_model_contract"],
            contract.checkpoint_payload(),
        )
        self.assertNotIn("speech_to_speech_audio_sequence_layout", checkpoint)
        module.on_load_checkpoint(checkpoint)

    def test_checkpoint_contract_uses_v7_grammar(self) -> None:
        self.assertEqual(MODEL_CONTRACT_GRAMMAR, "s2s-model-v4-contract-v7")

    def test_checkpoint_requires_model_v4_schema_before_other_contracts(self) -> None:
        module = _module(_contract(1), lora_config=LoraConfig())

        for schema in (None, "v2", "v3", 4):
            with self.subTest(schema=schema):
                checkpoint: dict[str, Any] = {
                    "speech_to_speech_model_contract": "invalid",
                }
                if schema is not None:
                    checkpoint["speech_to_speech_model_schema"] = schema
                with self.assertRaisesRegex(
                    ValueError,
                    "checkpoint model schema is incompatible",
                ):
                    module.on_load_checkpoint(checkpoint)

    def test_old_v3_checkpoint_is_rejected_before_contract_validation(self) -> None:
        checkpoint = {
            "speech_to_speech_model_schema": "v3",
            "speech_to_speech_model_contract": "legacy-v3-contract",
            "speech_to_speech_peft": None,
        }

        with self.assertRaisesRegex(
            ValueError,
            "expected 'v4', got 'v3'",
        ):
            _module(_contract(1)).on_load_checkpoint(checkpoint)

    def test_v4_checkpoint_without_model_contract_is_rejected(self) -> None:
        checkpoint = {
            "speech_to_speech_model_schema": "v4",
            "speech_to_speech_peft": None,
        }

        with self.assertRaisesRegex(ValueError, "missing the model contract"):
            _module(_contract(1)).on_load_checkpoint(checkpoint)

    def test_checkpoint_rejects_old_v5_model_contract(self) -> None:
        module = _module(_contract(1))
        checkpoint = _saved_checkpoint(module)
        payload = _model_payload(checkpoint)
        payload["grammar"] = "s2s-model-v4-contract-v5"

        with self.assertRaisesRegex(
            ValueError,
            "checkpoint model contract.*grammar",
        ):
            module.on_load_checkpoint(checkpoint)

    def test_checkpoint_rejects_invalid_model_contract_digest(self) -> None:
        module = _module(_contract(1))
        checkpoint = _saved_checkpoint(module)
        _model_payload(checkpoint)["sha256"] = "not-a-sha256-digest"

        with self.assertRaisesRegex(
            ValueError,
            "checkpoint model contract digest is invalid",
        ):
            module.on_load_checkpoint(checkpoint)

    def test_checkpoint_rejects_model_contract_payload_tampering(self) -> None:
        module = _module(_contract(1))
        checkpoint = _saved_checkpoint(module)
        components = cast(dict[str, Any], _model_payload(checkpoint)["components"])
        audio_sequence = cast(dict[str, Any], components["audio_sequence"])
        audio_sequence["variant"] = 2

        with self.assertRaisesRegex(
            ValueError,
            "checkpoint model contract digest is invalid",
        ):
            module.on_load_checkpoint(checkpoint)

    def test_model_contract_payloads_are_independent_snapshots(self) -> None:
        contract = _contract(1)
        first = contract.checkpoint_payload()
        components = cast(dict[str, Any], first["components"])
        audio_sequence = cast(dict[str, Any], components["audio_sequence"])
        audio_sequence["variant"] = 2

        second = contract.checkpoint_payload()

        self.assertEqual(
            cast(dict[str, Any], second["components"])["audio_sequence"],
            {"grammar": "semantic-v1", "variant": 1},
        )
        self.assertEqual(second["sha256"], contract.sha256)

    def test_checkpoint_rejects_valid_model_contract_payload_mismatch(self) -> None:
        checkpoint = _saved_checkpoint(_module(_contract(2)))

        with self.assertRaisesRegex(
            ValueError,
            "checkpoint model contract does not match model at components",
        ):
            _module(_contract(1)).on_load_checkpoint(checkpoint)

    def test_model_contract_is_validated_before_peft_contract(self) -> None:
        checkpoint = _saved_checkpoint(_module(_contract(2)))
        checkpoint.pop("speech_to_speech_peft")

        with self.assertRaisesRegex(
            ValueError,
            "checkpoint model contract does not match model",
        ):
            _module(_contract(1), lora_config=LoraConfig()).on_load_checkpoint(checkpoint)

    def test_real_model_contracts_build_and_json_roundtrip(self) -> None:
        models = (
            _token_model(),
            _flow_model(),
            _rvq_model(),
        )

        for model in models:
            with self.subTest(model=type(model).__name__):
                module = _real_module(model)
                checkpoint = json.loads(json.dumps(_saved_checkpoint(module)))

                module.on_load_checkpoint(checkpoint)
                payload = _model_payload(checkpoint)
                self.assertEqual(
                    payload["sha256"],
                    model.checkpoint_contract.sha256,
                )
                self.assertIsInstance(payload["components"], dict)

    def test_contract_binds_runtime_control_vocabulary_and_embedding(self) -> None:
        runtime = _ControlContractRuntime()
        model = _token_model(runtime=runtime)

        token_space = cast(
            Mapping[str, object],
            _component(model, "runtime", "token_space"),
        )
        tokenizer = cast(Mapping[str, object], token_space["text_tokenizer"])
        controls = cast(Mapping[str, object], token_space["text_controls"])
        embedding = cast(
            Mapping[str, object],
            _component(model, "interface", "control_embedding"),
        )

        self.assertEqual(tokenizer["vocab_size"], runtime.lexical_text_vocab_size)
        self.assertEqual(
            controls,
            {
                "grammar": "typed-text-control-v1",
                "tokens": {token.value: runtime.control_token_id(token) for token in ControlToken},
            },
        )
        self.assertEqual(
            embedding,
            {
                "family": "dense-v1",
                "rows": len(ControlToken),
                "dim": 8,
                "trainable": True,
            },
        )
        self.assertIn("tokens.control_embedding.weight", model.state_dict())

    def test_real_model_contract_rejects_audio_layout_mismatch(self) -> None:
        checkpoint_model = _token_model(
            runtime=_contract_runtime(
                audio_sequence_layout=AudioSequenceLayout.SEMANTIC,
            )
        )
        current_model = _token_model(
            runtime=_contract_runtime(
                audio_sequence_layout=AudioSequenceLayout.FLATTENED,
            )
        )
        checkpoint_layout = _component(
            checkpoint_model,
            "runtime",
            "token_space",
            "audio_sequence_layout",
        )
        current_layout = _component(
            current_model,
            "runtime",
            "token_space",
            "audio_sequence_layout",
        )

        self.assertEqual(checkpoint_layout, "semantic")
        self.assertEqual(current_layout, "flattened")
        _assert_contract_mismatch(
            self,
            checkpoint_model,
            current_model,
            "components.runtime.token_space.audio_sequence_layout",
        )

    def test_real_model_contract_rejects_tokenizer_state_mismatch(self) -> None:
        cases = (
            (
                _contract_runtime(text_tokenizer_state="text-v1"),
                _contract_runtime(text_tokenizer_state="text-v2"),
                "components.runtime.token_space.text_tokenizer.state_sha256",
            ),
            (
                _contract_runtime(audio_tokenizer_state="audio-v1"),
                _contract_runtime(audio_tokenizer_state="audio-v2"),
                "components.runtime.token_space.audio_schemas.input.tokenizer.state_sha256",
            ),
        )

        for checkpoint_runtime, current_runtime, path in cases:
            with self.subTest(path=path):
                _assert_contract_mismatch(
                    self,
                    _token_model(runtime=checkpoint_runtime),
                    _token_model(runtime=current_runtime),
                    path,
                )

    def test_fast_tokenizer_contract_tracks_backend_normalizer(self) -> None:
        lowercase = _fast_text_tokenizer(lowercase=True)
        normalized = _fast_text_tokenizer(lowercase=False)

        self.assertEqual(lowercase.get_vocab(), normalized.get_vocab())
        checkpoint_model = _token_model(runtime=_contract_runtime_with_text_tokenizer(lowercase))
        current_model = _token_model(runtime=_contract_runtime_with_text_tokenizer(normalized))

        self.assertEqual(
            _component(
                checkpoint_model,
                "runtime",
                "token_space",
                "text_tokenizer",
                "state_grammar",
            ),
            "serialized-tokenizer-backend-v1",
        )
        _assert_contract_mismatch(
            self,
            checkpoint_model,
            current_model,
            "components.runtime.token_space.text_tokenizer.state_sha256",
        )

    def test_real_model_contract_rejects_semantic_artifact_mismatch(self) -> None:
        checkpoint_model = _token_model(
            runtime=_contract_runtime(
                semantic_artifact_sha256="1" * 64,
            )
        )
        current_model = _token_model(
            runtime=_contract_runtime(
                semantic_artifact_sha256="2" * 64,
            )
        )

        _assert_contract_mismatch(
            self,
            checkpoint_model,
            current_model,
            "components.runtime.audio_codecs.output.semantic_artifact_sha256",
        )

    def test_real_model_contract_rejects_invalid_semantic_artifact_digest(self) -> None:
        model = _token_model(runtime=_contract_runtime(semantic_artifact_sha256="invalid"))

        with self.assertRaisesRegex(ValueError, "lowercase SHA-256"):
            _ = model.checkpoint_contract

    def test_bicodec_contract_records_global_units_separately_from_acoustic(self) -> None:
        codec = cast(
            Mapping[str, object],
            _component(
                _token_model(runtime=_GlobalContractRuntime()),
                "runtime",
                "audio_codecs",
                "output",
            ),
        )

        self.assertIsNone(codec["acoustic"])
        self.assertEqual(
            codec["global"],
            {
                "feature_dim": 6,
                "codebook_sizes": [5, 7],
                "unit_length": 2,
            },
        )

    def test_contract_records_schema_selector_payload_and_private_grammar(self) -> None:
        runtime = _GlobalContractRuntime()
        model = _token_model(runtime=runtime)
        schemas = cast(
            Mapping[str, object],
            _component(model, "runtime", "token_space", "audio_schemas"),
        )
        output = cast(Mapping[str, object], schemas["output"])
        tokenizer = cast(Mapping[str, object], output["tokenizer"])
        grammar = cast(Mapping[str, object], output["private_grammar"])
        audio_tokenizer = cast(BiCodecAudioTokenizer, runtime.audio_tokenizer)

        self.assertEqual(output["schema_id"], runtime.output_audio_schema_id)
        self.assertEqual(output["selector"], runtime.output_audio_token_spec.selector)
        self.assertEqual(output["selector_id"], runtime.output_audio_schema_token_id)
        self.assertEqual(output["payload_range"], list(runtime.output_codec_audio_range))
        self.assertEqual(output["spec"], runtime.output_audio_token_spec.contract_state())
        self.assertEqual(tokenizer["vocab_size"], runtime.audio_tokenizer.vocab_size)
        self.assertEqual(grammar["grammar"], "bicodec-streams-v1")
        self.assertEqual(
            grammar["private_marker_ids"],
            sorted(
                (
                    audio_tokenizer.semantic_token_id,
                    audio_tokenizer.global_token_id,
                )
            ),
        )
        self.assertEqual(grammar, runtime.output_audio_token_spec.grammar.contract_state())

    def test_coupled_audio_contract_records_shared_resources(self) -> None:
        model = _token_model()
        audio_codecs = cast(
            Mapping[str, object],
            _component(model, "runtime", "audio_codecs"),
        )
        schemas = cast(
            Mapping[str, object],
            _component(model, "runtime", "token_space", "audio_schemas"),
        )
        embeddings = cast(
            Mapping[str, object],
            _component(model, "interface", "audio_embeddings"),
        )

        self.assertEqual(audio_codecs["sharing"], "shared")
        detokenizer = cast(
            Mapping[str, object],
            audio_codecs["output_detokenizer"],
        )
        self.assertEqual(detokenizer["name"], model.runtime.codec_name)
        self.assertEqual(detokenizer["sharing"], "output")
        self.assertEqual(schemas["sharing"], "shared")
        self.assertEqual(schemas["input"], schemas["output"])
        self.assertEqual(embeddings["sharing"], "shared")
        self.assertEqual(embeddings["input"], embeddings["output"])
        self.assertNotIn("tokens.input_audio_embedding.weight", model.state_dict())

        restored = _token_model()
        incompatible = restored.load_state_dict(model.state_dict(), strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])

    def test_codes_only_audio_contract_records_no_output_detokenizer(self) -> None:
        audio_codecs = cast(
            Mapping[str, object],
            _component(
                _token_model(runtime=_CodesOnlyGlobalContractRuntime()),
                "runtime",
                "audio_codecs",
            ),
        )

        self.assertIsNone(audio_codecs["output_detokenizer"])

    def test_decoupled_audio_contract_records_glm4_input_and_bicodec_output(
        self,
    ) -> None:
        runtime = _DecoupledContractRuntime()
        model = _token_model(runtime=runtime)
        audio_codecs = cast(
            Mapping[str, object],
            _component(model, "runtime", "audio_codecs"),
        )
        token_space = cast(
            Mapping[str, object],
            _component(model, "runtime", "token_space"),
        )
        schemas = cast(
            Mapping[str, object],
            token_space["audio_schemas"],
        )
        input_schema = cast(Mapping[str, object], schemas["input"])
        output_schema = cast(Mapping[str, object], schemas["output"])
        embeddings = cast(
            Mapping[str, object],
            _component(model, "interface", "audio_embeddings"),
        )

        self.assertEqual(audio_codecs["sharing"], "independent")
        self.assertEqual(
            audio_codecs["input"],
            {
                "name": "glm4",
                "audio_view": "glm4",
                "frame_rate": 12.5,
            },
        )
        self.assertEqual(
            cast(Mapping[str, object], audio_codecs["output"])["name"],
            "bicodec",
        )
        self.assertEqual(
            token_space["blocks"],
            {name: list(bounds) for name, bounds in runtime.layout.blocks.items()},
        )
        self.assertEqual(schemas["sharing"], "independent")
        self.assertEqual(
            input_schema["selector_id"],
            runtime.input_audio_schema_token_id,
        )
        self.assertEqual(
            input_schema["payload_range"],
            list(runtime.input_codec_audio_range),
        )
        self.assertEqual(
            output_schema["selector_id"],
            runtime.output_audio_schema_token_id,
        )
        self.assertEqual(
            output_schema["payload_range"],
            list(runtime.output_codec_audio_range),
        )
        self.assertEqual(
            cast(
                Mapping[str, object],
                input_schema["tokenizer"],
            )["vocab_size"],
            runtime.input_audio_tokenizer.vocab_size,
        )
        self.assertEqual(
            cast(
                Mapping[str, object],
                output_schema["tokenizer"],
            )["vocab_size"],
            runtime.audio_tokenizer.vocab_size,
        )
        self.assertEqual(embeddings["sharing"], "independent")
        self.assertEqual(
            cast(Mapping[str, object], embeddings["input"])["rows"],
            runtime.input_audio_tokenizer.vocab_size + 3,
        )
        self.assertEqual(
            cast(Mapping[str, object], embeddings["output"])["rows"],
            runtime.audio_tokenizer.vocab_size + 4,
        )
        self.assertEqual(
            _component(model, "interface", "input_audio_projection"),
            {
                "sharing": "independent",
                "adapter": {"family": "identity-v1"},
            },
        )
        self.assertIn("tokens.input_audio_embedding.weight", model.state_dict())

    def test_decoupled_audio_state_dict_requires_input_embedding(self) -> None:
        model = _token_model(runtime=_DecoupledContractRuntime())
        state = dict(model.state_dict())
        state.pop("tokens.input_audio_embedding.weight")

        with self.assertRaisesRegex(
            RuntimeError,
            "tokens.input_audio_embedding.weight",
        ):
            model.load_state_dict(state, strict=True)

    def test_decoupled_contract_rejects_input_tokenizer_state_mismatch(self) -> None:
        checkpoint_model = _token_model(
            runtime=_DecoupledContractRuntime(input_tokenizer_state="glm4-v1")
        )
        current_model = _token_model(
            runtime=_DecoupledContractRuntime(input_tokenizer_state="glm4-v2")
        )

        _assert_contract_mismatch(
            self,
            checkpoint_model,
            current_model,
            "components.runtime.token_space.audio_schemas.input.tokenizer.state_sha256",
        )

    def test_decoupled_contract_rejects_input_codec_identity_mismatch(self) -> None:
        checkpoint_model = _token_model(runtime=_DecoupledContractRuntime(input_codec_name="glm4"))
        current_model = _token_model(
            runtime=_DecoupledContractRuntime(input_codec_name="glm4-audio")
        )

        _assert_contract_mismatch(
            self,
            checkpoint_model,
            current_model,
            "components.runtime.token_space.audio_schemas.input.schema_id",
        )

    def test_real_model_contract_rejects_backbone_readout_mismatch(self) -> None:
        checkpoint_model = _token_model(
            runtime=_contract_runtime(backbone_readout="last_hidden_state")
        )
        current_model = _token_model(runtime=_contract_runtime(backbone_readout="hidden_states[0]"))

        _assert_contract_mismatch(
            self,
            checkpoint_model,
            current_model,
            "components.runtime.backbone.encoder.readout",
        )

    def test_real_model_contract_rejects_cache_position_protocol_mismatch(self) -> None:
        checkpoint_model = _token_model(
            runtime=_contract_runtime(backbone_supports_cache_position=True)
        )
        current_model = _token_model(
            runtime=_contract_runtime(backbone_supports_cache_position=False)
        )

        _assert_contract_mismatch(
            self,
            checkpoint_model,
            current_model,
            "components.runtime.backbone.encoder.supports_cache_position",
        )

    def test_real_model_contract_rejects_audio_head_mismatch(self) -> None:
        checkpoint_model = _token_model(
            config=_model_config(audio_output=AudioOutputAdapterType.NONE)
        )
        current_model = _token_model(config=_model_config(audio_output=AudioOutputAdapterType.MLP))

        _assert_contract_mismatch(
            self,
            checkpoint_model,
            current_model,
            "components.interface.audio_head",
        )

    def test_real_model_contract_rejects_source_audio_tower_mismatch(self) -> None:
        checkpoint_model = _token_model(config=_model_config(audio_input=AudioInputAdapterType.MLP))
        current_model = _token_model(config=_model_config(audio_input=AudioInputAdapterType.NONE))

        _assert_contract_mismatch(
            self,
            checkpoint_model,
            current_model,
            "components.interface.source_audio_encoder",
        )

    def test_real_model_contract_rejects_source_audio_causality_mismatch(self) -> None:
        checkpoint_model = _token_model(
            config=_model_config(
                audio_input=AudioInputAdapterType.TRANSFORMER,
                audio_input_causal=True,
            )
        )
        current_model = _token_model(
            config=_model_config(
                audio_input=AudioInputAdapterType.TRANSFORMER,
                audio_input_causal=False,
            )
        )

        self.assertTrue(
            _component(
                checkpoint_model,
                "interface",
                "source_audio_encoder",
                "causal",
            )
        )
        self.assertFalse(
            _component(
                current_model,
                "interface",
                "source_audio_encoder",
                "causal",
            )
        )
        _assert_contract_mismatch(
            self,
            checkpoint_model,
            current_model,
            "components.interface.source_audio_encoder.causal",
        )

    def test_real_model_contract_records_route_local_ctc_decoders(self) -> None:
        model = _token_model(
            config=_model_config(
                ctc=CTCDecoderRoutesConfig(
                    source=CTCDecoderConfig(
                        type=CTCDecoderType.TRANSFORMER,
                        backbone_readout="hidden_states[0]",
                        pool_factor=3,
                        layers=3,
                        heads=2,
                        ffn_ratio=6.0,
                        dropout=0.125,
                    ),
                    target=CTCDecoderConfig(
                        type=CTCDecoderType.LINEAR,
                        backbone_readout="last_hidden_state",
                        pool_factor=2,
                        layers=5,
                        heads=4,
                        ffn_ratio=3.0,
                        dropout=0.25,
                    ),
                )
            )
        )

        decoders = _component(model, "interface", "ctc_decoders")

        self.assertEqual(
            decoders,
            {
                "source": {
                    "grammar": "ctc-decoder-v1",
                    "type": "transformer",
                    "causal": False,
                    "backbone_readout": "hidden_states[0]",
                    "pool_factor": 3,
                    "hidden_size": 8,
                    "layers": 3,
                    "heads": 2,
                    "ffn_ratio": 6.0,
                    "dropout": 0.125,
                },
                "target": {
                    "grammar": "ctc-decoder-v1",
                    "type": "linear",
                    "causal": True,
                    "backbone_readout": "last_hidden_state",
                    "pool_factor": 2,
                    "hidden_size": 8,
                    "layers": 5,
                    "heads": 4,
                    "ffn_ratio": 3.0,
                    "dropout": 0.25,
                },
            },
        )

    def test_real_model_contract_rejects_route_local_ctc_decoder_mismatch(
        self,
    ) -> None:
        cases = (
            (
                CTCDecoderRoutesConfig(),
                CTCDecoderRoutesConfig(
                    source=CTCDecoderConfig(
                        type=CTCDecoderType.LINEAR,
                    )
                ),
                "components.interface.ctc_decoders.source.type",
            ),
            (
                CTCDecoderRoutesConfig(),
                CTCDecoderRoutesConfig(target=CTCDecoderConfig(pool_factor=2)),
                "components.interface.ctc_decoders.target.pool_factor",
            ),
        )

        for checkpoint_ctc, current_ctc, path in cases:
            with self.subTest(path=path):
                _assert_contract_mismatch(
                    self,
                    _token_model(config=_model_config(ctc=checkpoint_ctc)),
                    _token_model(config=_model_config(ctc=current_ctc)),
                    path,
                )

    def test_realized_contract_ignores_inactive_adapter_config_fields(self) -> None:
        first = _token_model(config=_inactive_adapter_config(variant=1))
        second = _token_model(config=_inactive_adapter_config(variant=2))

        self.assertEqual(first.checkpoint_contract, second.checkpoint_contract)
        self.assertEqual(
            first.checkpoint_contract.checkpoint_payload(),
            second.checkpoint_contract.checkpoint_payload(),
        )

    def test_realized_contract_ignores_backbone_execution_metadata(self) -> None:
        first = _token_model()
        second = _token_model()
        first_config = cast(Any, first.backbone.config)
        second_config = cast(Any, second.backbone.config)
        first_config._name_or_path = "fixture/first"
        second_config._name_or_path = "fixture/second"
        first_config.architectures = ["FirstWrapper"]
        second_config.architectures = ["SecondWrapper"]
        first_config.use_cache = True
        second_config.use_cache = False
        first_config.return_dict = False
        second_config.return_dict = True

        self.assertEqual(first.checkpoint_contract, second.checkpoint_contract)

    def test_realized_contract_ignores_generation_and_initializer_config(self) -> None:
        first = _token_model()
        second = _token_model()
        first_config = cast(Any, first.backbone.config)
        second_config = cast(Any, second.backbone.config)
        first_config.temperature = 0.2
        second_config.temperature = 1.4
        first_config.max_length = 16
        second_config.max_length = 512
        first_config.initializer_range = 0.01
        second_config.initializer_range = 0.2

        self.assertEqual(first.checkpoint_contract, second.checkpoint_contract)

    def test_realized_contract_canonicalizes_integer_config_keys(self) -> None:
        checkpoint_model = _token_model()
        current_model = _token_model()
        cast(Any, checkpoint_model.backbone.config).pruned_heads = {0: [1]}
        cast(Any, current_model.backbone.config).pruned_heads = {1: [0]}

        _assert_contract_mismatch(
            self,
            checkpoint_model,
            current_model,
            "components.runtime.backbone.architecture_sha256",
        )

    def test_state_dict_schema_rejects_new_persistent_ownership(self) -> None:
        checkpoint_model = _token_model()
        current_model = _token_model()
        cast(Any, current_model).contract_probe = nn.Buffer(torch.zeros(2))

        _assert_contract_mismatch(
            self,
            checkpoint_model,
            current_model,
            "components.state_dict.schema_sha256",
        )

    def test_state_dict_schema_ignores_tensor_dtype(self) -> None:
        first = _token_model()
        second = _token_model().to(dtype=torch.float64)

        self.assertEqual(first.checkpoint_contract, second.checkpoint_contract)

    def test_real_flow_contract_rejects_decoder_topology_mismatch(self) -> None:
        checkpoint_model = _flow_model(layers=1)
        current_model = _flow_model(layers=2)

        _assert_contract_mismatch(
            self,
            checkpoint_model,
            current_model,
            "components.acoustic.decoder",
        )

    def test_real_rvq_contract_rejects_decoder_topology_mismatch(self) -> None:
        checkpoint_model = _rvq_model(layers=1)
        current_model = _rvq_model(layers=2)

        _assert_contract_mismatch(
            self,
            checkpoint_model,
            current_model,
            "components.acoustic.decoder",
        )

    def test_checkpoint_lora_contract_roundtrips_complete_config(self) -> None:
        config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["v_proj", "q_proj"],
            exclude_modules=["down_proj", "k_proj"],
            use_rslora=True,
        )
        module = _module(_contract(1), lora_config=config)
        checkpoint = _saved_checkpoint(module)

        module.on_load_checkpoint(checkpoint)

        payload = _peft_payload(checkpoint)
        self.assertEqual(payload["grammar"], "peft-lora-v2")
        config_payload = cast(dict[str, Any], payload["config"])
        self.assertEqual(config_payload["r"], 8)
        self.assertEqual(config_payload["lora_alpha"], 16)
        self.assertEqual(config_payload["lora_dropout"], 0.1)
        self.assertEqual(config_payload["target_modules"], ["q_proj", "v_proj"])
        self.assertEqual(
            config_payload["exclude_modules"],
            ["down_proj", "k_proj"],
        )
        self.assertEqual(config_payload["peft_type"], "LORA")
        self.assertTrue(config_payload["use_rslora"])
        self.assertNotIn("peft_version", config_payload)
        self.assertNotIn(
            "peft_version",
            cast(dict[str, Any], payload["defaults"]),
        )

    def test_checkpoint_accepts_only_default_peft_schema_additions(self) -> None:
        config = LoraConfig(r=16, target_modules=["q_proj"])
        module = _module(_contract(1), lora_config=config)
        checkpoint = _saved_checkpoint(module)
        payload = _peft_payload(checkpoint)
        config_payload = cast(dict[str, Any], payload["config"])
        defaults = cast(dict[str, Any], payload["defaults"])

        config_payload["future_option"] = False
        defaults["future_option"] = False
        module.on_load_checkpoint(checkpoint)

        config_payload["future_option"] = True
        with self.assertRaisesRegex(ValueError, "LoRA contract does not match"):
            module.on_load_checkpoint(checkpoint)

    def test_checkpoint_accepts_missing_default_peft_schema_fields(self) -> None:
        config = LoraConfig(r=16, target_modules=["q_proj"])
        module = _module(_contract(1), lora_config=config)
        checkpoint = _saved_checkpoint(module)
        payload = _peft_payload(checkpoint)
        config_payload = cast(dict[str, Any], payload["config"])
        defaults = cast(dict[str, Any], payload["defaults"])

        config_payload.pop("qalora_group_size")
        defaults.pop("qalora_group_size")
        module.on_load_checkpoint(checkpoint)

        config_payload.pop("r")
        defaults.pop("r")
        with self.assertRaisesRegex(ValueError, "LoRA contract does not match"):
            module.on_load_checkpoint(checkpoint)

    def test_checkpoint_requires_lora_contract_only_when_enabled(self) -> None:
        checkpoint = _saved_checkpoint(_module(_contract(1)))
        checkpoint.pop("speech_to_speech_peft")

        _module(_contract(1)).on_load_checkpoint(checkpoint)
        with self.assertRaisesRegex(ValueError, "missing the PEFT LoRA contract"):
            _module(_contract(1), lora_config=LoraConfig()).on_load_checkpoint(checkpoint)

    def test_checkpoint_rejects_lora_config_mismatch(self) -> None:
        checkpoint = _saved_checkpoint(_module(_contract(1), lora_config=LoraConfig(lora_alpha=16)))

        with self.assertRaisesRegex(ValueError, "LoRA contract does not match"):
            _module(
                _contract(1),
                lora_config=LoraConfig(lora_alpha=32),
            ).on_load_checkpoint(checkpoint)


@dataclass(frozen=True)
class _RuntimeConfig:
    backbone: str = "fixture/toy"


class _TextTokenizer:
    special_tokens_map: Mapping[str, str | Sequence[str]] = {
        "pad_token": "<pad>",
        "bos_token": "<bos>",
        "eos_token": "<eos>",
    }
    pad_token_id: int | None = 0
    bos_token_id: int | None = 1
    eos_token_id: int | None = 2
    chat_template = "fixture-chat-v1"

    def __init__(self, state: str) -> None:
        self.state = state

    def __len__(self) -> int:
        return 8

    def contract_state(self) -> dict[str, object]:
        return {
            "grammar": "fixture-text-v1",
            "state": self.state,
            "vocab": [f"token-{index}" for index in range(len(self))],
        }

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
    ) -> list[int]:
        del add_special_tokens
        return [ord(character) % len(self) for character in text]

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        del skip_special_tokens
        return " ".join(str(token_id) for token_id in token_ids)

    def apply_chat_template(
        self,
        conversation: Sequence[Mapping[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        enable_thinking: bool = False,
        return_dict: bool = False,
    ) -> str | list[int]:
        del enable_thinking, return_dict
        rendered = "\n".join(f"{message['role']}: {message['content']}" for message in conversation)
        if add_generation_prompt:
            rendered += "\nassistant:"
        return self.encode(rendered) if tokenize else rendered


class _StatefulNativeAudioTokenizer(NativeAudioTokenizer):
    def __init__(self, *, vocab_size: int, state: str) -> None:
        super().__init__(vocab_size=vocab_size)
        self.state = state

    def contract_state(self) -> dict[str, object]:
        return {
            **super().contract_state(),
            "state": self.state,
        }


class _Codec:
    sample_rate = 16_000
    frame_rate = 50.0
    codebook_sizes = (4, 3)
    acoustic_feature_dim = 8
    acoustic_codebook_sizes = (4,)
    acoustic_layout = AcousticLayout.FRAME_ALIGNED
    acoustic_unit_length = None
    semantic_codebook = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    semantic_codebook_sizes = (3,)

    def encode(self, audio: Tensor, sample_rate: int) -> Tensor:
        del sample_rate
        return audio.new_zeros((1, 2), dtype=torch.long)

    def decode(self, codes: Tensor) -> Tensor:
        return codes.new_zeros(1, dtype=torch.float32)

    def tokenize(self, audio: Tensor, sample_rate: int) -> SemanticAcousticCodes:
        del sample_rate
        shape = (audio.size(0), 1, 1)
        semantic = torch.zeros(shape, dtype=torch.long, device=audio.device)
        acoustic = torch.zeros(shape, dtype=torch.long, device=audio.device)
        return SemanticAcousticCodes(semantic=semantic, acoustic=acoustic)

    def detokenize(self, codes: SemanticAcousticCodes) -> Tensor:
        return self.decode_features(
            codes.semantic,
            self.acoustic_codes_to_features(codes.acoustic),
        )

    def acoustic_codes_to_features(self, acoustic_codes: Tensor) -> Tensor:
        shape = (*acoustic_codes.shape[:-1], self.acoustic_feature_dim)
        return acoustic_codes.new_zeros(shape, dtype=torch.float32)

    def decode_features(
        self,
        semantic_codes: Tensor,
        acoustic_features: Tensor,
    ) -> Tensor:
        del semantic_codes
        return acoustic_features.new_zeros(1)


class _SemanticCodec:
    sample_rate = _Codec.sample_rate
    frame_rate = _Codec.frame_rate

    def decode(
        self,
        semantic_codes: Tensor,
        *,
        mask: Tensor | None = None,
        reference_features: Tensor | None = None,
        reference_mask: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        del mask, reference_features, reference_mask, generator
        return semantic_codes.new_zeros(1, dtype=torch.float32)


class _FlowSample:
    def __init__(self, final: Tensor) -> None:
        self.final = final


class _FlowRuntime:
    def sample(
        self,
        model: nn.Module,
        x_0: Tensor,
        *,
        time_grid: Tensor | None = None,
        **model_extras: object,
    ) -> FlowSample:
        del model, time_grid, model_extras
        return _FlowSample(final=x_0)


class _BackboneAdapter:
    def __init__(
        self,
        model: Backbone,
        text_tokenizer: TextTokenizer,
        *,
        readout: str,
        supports_cache_position: bool,
    ) -> None:
        self._model = model
        self._text_tokenizer = text_tokenizer
        self._readout = readout
        self._supports_cache_position = supports_cache_position

    @cached_property
    def model(self) -> Backbone:
        return self._model

    @cached_property
    def text_tokenizer(self) -> TextTokenizer:
        return self._text_tokenizer

    @cached_property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    @cached_property
    def body(self) -> BackboneBodyAdapter:
        target = cast(Callable[..., BackboneOutput], cast(object, self.model))
        return BackboneBodyAdapter(
            target,
            readout=BackboneReadout(self._readout),
            supports_cache_position=self._supports_cache_position,
        )

    @property
    def has_modality_readouts(self) -> bool:
        return self.body.has_modality_readouts

    def input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def contract_state(self) -> Mapping[str, object]:
        return {
            "grammar": "fixture-backbone-v1",
            "execution": self.body.contract_state(),
        }

    def encode(
        self,
        *,
        inputs_embeds: Tensor,
        attention_mask: Tensor | None,
        output_hidden_states: bool,
        past_key_values: Cache | None = None,
        use_cache: bool = False,
        position_ids: Tensor | None = None,
        cache_position: Tensor | None = None,
        modality: Modality | None = None,
        extra: BackboneExtra | None = None,
    ) -> BackboneOutputView:
        return self.body.encode(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_ids=position_ids,
            cache_position=cache_position,
            modality=modality,
            extra=extra,
        )


class _ContractRuntime:
    config = _RuntimeConfig()

    def __init__(
        self,
        *,
        audio_sequence_layout: AudioSequenceLayout,
        text_tokenizer_state: str,
        audio_tokenizer_state: str,
        backbone_readout: str,
        backbone_supports_cache_position: bool,
        semantic_artifact_sha256: str | None,
    ) -> None:
        self._audio_sequence_layout = audio_sequence_layout
        self._backbone_readout = backbone_readout
        self._backbone_supports_cache_position = backbone_supports_cache_position
        self._semantic_artifact_sha256 = semantic_artifact_sha256
        self._codec = _Codec()
        self._semantic_codec = _SemanticCodec()
        self._text_tokenizer: TextTokenizer = _TextTokenizer(text_tokenizer_state)
        self._audio_tokenizer: AudioTokenizer = (
            _StatefulNativeAudioTokenizer(
                vocab_size=3,
                state=audio_tokenizer_state,
            )
            if self.audio_sequence_layout is AudioSequenceLayout.SEMANTIC
            else FlattenedAudioTokenizer(
                codebook_sizes=self._codec.codebook_sizes,
                codec_name=self.codec_name,
            )
        )
        text_end = len(self._text_tokenizer)
        audio_start = text_end
        audio_end = audio_start + self._audio_tokenizer.vocab_size + 4
        self._layout = Layout(
            text=(0, text_end),
            audio=(audio_start, audio_end),
        )
        self._boa_token_id = audio_end - 4
        self._eoa_token_id = audio_end - 3
        self._mask_token_id = audio_end - 2
        self._audio_schema_token_id = audio_end - 1

    @property
    def codec_name(self) -> str:
        return "fixture-codec"

    @property
    def output_codec_name(self) -> str:
        return self.codec_name

    @property
    def output_audio_detokenizer_name(self) -> str:
        return self.codec_name

    @property
    def input_audio_backend_identity(self) -> AudioBackendIdentity:
        return AudioBackendIdentity(preset=self.input_codec_name)

    @property
    def output_audio_backend_identity(self) -> AudioBackendIdentity:
        return AudioBackendIdentity(preset=self.codec_name)

    @property
    def output_audio_detokenizer_identity(self) -> AudioBackendIdentity:
        return self.output_audio_backend_identity

    @property
    def input_audio_decoupled(self) -> bool:
        return False

    @property
    def input_codec_name(self) -> str:
        return self.codec_name

    @property
    def audio_view(self) -> AudioView:
        return AudioView.LONGCAT

    @property
    def output_audio_view(self) -> AudioView:
        return self.audio_view

    @property
    def input_audio_view(self) -> AudioView:
        return self.audio_view

    @property
    def output_audio_code_spec(self) -> AudioCodeSpec:
        if isinstance(self._codec, _GlobalCodec):
            return AudioCodeSpec(
                view=self.audio_view.value,
                schema=AudioCodeSchema.SEMANTIC_GLOBAL,
                sample_rate=self._codec.sample_rate,
                frame_rate=self._codec.frame_rate,
                semantic_codebook_sizes=self._codec.semantic_codebook_sizes,
                global_codebook_sizes=self._codec.global_codebook_sizes,
                global_unit_length=self._codec.global_unit_length,
            )
        return AudioCodeSpec(
            view=self.audio_view.value,
            schema=AudioCodeSchema.SEMANTIC_ACOUSTIC,
            sample_rate=self._codec.sample_rate,
            frame_rate=self._codec.frame_rate,
            frame_codebook_sizes=self._codec.codebook_sizes,
            semantic_codebook_sizes=self._codec.semantic_codebook_sizes,
            acoustic_codebook_sizes=self._codec.acoustic_codebook_sizes,
            acoustic_layout=self._codec.acoustic_layout,
            acoustic_unit_length=self._codec.acoustic_unit_length,
        )

    @property
    def input_audio_code_spec(self) -> AudioCodeSpec:
        return self.output_audio_code_spec

    @property
    def codec_frame_rate(self) -> float:
        return self._codec.frame_rate

    @property
    def output_codec_frame_rate(self) -> float:
        return self.codec_frame_rate

    @property
    def input_codec_frame_rate(self) -> float:
        return self.codec_frame_rate

    @property
    def audio_sequence_layout(self) -> AudioSequenceLayout:
        return self._audio_sequence_layout

    @property
    def acoustic_generator_artifact(self) -> str | None:
        if self._semantic_artifact_sha256 is None:
            return None
        return "/fixture/semantic-codec"

    @cached_property
    def acoustic_generator_artifact_sha256(self) -> str | None:
        return self._semantic_artifact_sha256

    @property
    def semantic_codebook_sizes(self) -> tuple[int, ...]:
        return (3,)

    @cached_property
    def text_tokenizer(self) -> TextTokenizer:
        return self._text_tokenizer

    @cached_property
    def audio_tokenizer(self) -> AudioTokenizer:
        return self._audio_tokenizer

    @cached_property
    def output_audio_tokenizer(self) -> AudioTokenizer:
        return self.audio_tokenizer

    @cached_property
    def output_audio_token_spec(self) -> AudioTokenSpec:
        return AudioTokenSpec.create(
            codec_name=self.output_codec_name,
            sequence_layout=self.audio_sequence_layout.value,
            tokenizer=self.output_audio_tokenizer,
        )

    @cached_property
    def output_audio_token_registry(self) -> AudioTokenRegistry:
        spec = self.output_audio_token_spec
        return AudioTokenRegistry((spec,), spec.schema_id)

    @cached_property
    def input_audio_tokenizer(self) -> AudioTokenizer:
        return self.audio_tokenizer

    @cached_property
    def input_audio_token_spec(self) -> AudioTokenSpec:
        if not self.input_audio_decoupled:
            return self.output_audio_token_spec
        return AudioTokenSpec.create(
            codec_name=self.input_codec_name,
            sequence_layout=self.audio_sequence_layout.value,
            tokenizer=self.input_audio_tokenizer,
        )

    @cached_property
    def input_audio_token_registry(self) -> AudioTokenRegistry:
        if not self.input_audio_decoupled:
            return self.output_audio_token_registry
        spec = self.input_audio_token_spec
        return AudioTokenRegistry((spec,), spec.schema_id)

    @cached_property
    def layout(self) -> Layout:
        return self._layout

    @cached_property
    def lexical_text_vocab_size(self) -> int:
        return len(self.text_tokenizer)

    @cached_property
    def control_token_ids(self) -> tuple[int, ...]:
        return ()

    def control_token_id(self, token: ControlToken) -> int:
        if not isinstance(token, ControlToken):
            raise TypeError("control token lookup requires a ControlToken.")
        try:
            return self.control_token_ids[list(ControlToken).index(token)]
        except IndexError as error:
            raise ValueError("runtime does not reserve text control tokens.") from error

    @cached_property
    def pad_token_id(self) -> int:
        return 0

    @cached_property
    def eos_token_id(self) -> int:
        return 2

    @property
    def boa_token_id(self) -> int:
        return self._boa_token_id

    @property
    def eoa_token_id(self) -> int:
        return self._eoa_token_id

    @property
    def mask_token_id(self) -> int:
        return self._mask_token_id

    @property
    def audio_schema_token_id(self) -> int:
        return self._audio_schema_token_id

    @property
    def output_audio_schema_token_id(self) -> int:
        return self.audio_schema_token_id

    @property
    def output_audio_schema_id(self) -> str:
        return self.output_audio_token_spec.schema_id

    @property
    def output_audio_block_name(self) -> str:
        return Modality.AUDIO.value

    @property
    def output_boa_token_id(self) -> int:
        return self.boa_token_id

    @property
    def output_eoa_token_id(self) -> int:
        return self.eoa_token_id

    @property
    def output_mask_token_id(self) -> int:
        return self.mask_token_id

    @property
    def input_audio_block_name(self) -> str:
        return Modality.AUDIO.value

    @property
    def input_boa_token_id(self) -> int:
        return self.boa_token_id

    @property
    def input_eoa_token_id(self) -> int:
        return self.eoa_token_id

    @property
    def input_audio_schema_token_id(self) -> int:
        return self.audio_schema_token_id

    @property
    def input_audio_schema_id(self) -> str:
        return self.input_audio_token_spec.schema_id

    @cached_property
    def codec(self) -> CodecBackend:
        return self._codec

    @cached_property
    def output_codec(self) -> CodecBackend:
        return self.codec

    @cached_property
    def input_codec(self) -> CodecBackend:
        return self.codec

    @cached_property
    def semantic_codec(self) -> SemanticCodec:
        return self._semantic_codec

    @property
    def acoustic_side_channel(self) -> bool:
        return self.audio_sequence_layout is AudioSequenceLayout.SEMANTIC

    @property
    def structured_full_sequence(self) -> bool:
        return False

    @cached_property
    def bos_token_id(self) -> int:
        return 1

    @property
    def backbone_trust_remote_code(self) -> bool:
        return False

    @property
    def backbone_chat_template(self) -> str | None:
        return None

    @property
    def backbone_readout(self) -> str:
        return self._backbone_readout

    @property
    def backbone_readouts(self) -> Mapping[str, str]:
        return {}

    @property
    def backbone_supports_cache_position(self) -> bool:
        return self._backbone_supports_cache_position

    @property
    def gradient_checkpointing(self) -> bool:
        return False

    @property
    def backbone_module(self) -> str:
        return ""

    @property
    def backbone_body(self) -> str:
        return "base_model"

    @cached_property
    def backbone(self) -> Backbone:
        return create_toy_backbone(_toy_config(), len(self._text_tokenizer))

    @cached_property
    def backbone_adapter(self) -> BackboneAdapter:
        return _BackboneAdapter(
            self.backbone,
            self.text_tokenizer,
            readout=self.backbone_readout,
            supports_cache_position=self.backbone_supports_cache_position,
        )

    @cached_property
    def flow_matching(self) -> FlowSamplingRuntime:
        return _FlowRuntime()

    @property
    def audio_head_range(self) -> tuple[int, int]:
        return self.layout.blocks[Modality.AUDIO.value]

    @property
    def codec_audio_range(self) -> tuple[int, int]:
        start, _ = self.audio_head_range
        return start, self.boa_token_id

    @property
    def output_codec_audio_range(self) -> tuple[int, int]:
        return self.codec_audio_range

    @property
    def input_codec_audio_range(self) -> tuple[int, int]:
        return self.codec_audio_range

    @cached_property
    def audio_generation_allowed_ids(self) -> tuple[int, ...]:
        start, end = self.codec_audio_range
        return (
            self.boa_token_id,
            self.audio_schema_token_id,
            *range(start, end),
            self.eoa_token_id,
        )

    @cached_property
    def output_audio_generation_allowed_ids(self) -> tuple[int, ...]:
        return self.audio_generation_allowed_ids

    def generation_allowed_ids(self, modality: Modality) -> tuple[int, ...]:
        if modality is Modality.AUDIO:
            return self.audio_generation_allowed_ids
        if modality is Modality.TEXT:
            start, end = self.layout.blocks[Modality.TEXT.value]
            blocked = {self.pad_token_id, self.bos_token_id}
            return tuple(token_id for token_id in range(start, end) if token_id not in blocked)
        raise ValueError(f"unsupported generation modality: {modality.value}")

    def is_codec_audio_id(self, token_id: int) -> bool:
        start, end = self.codec_audio_range
        return start <= token_id < end


class _ControlContractRuntime(_ContractRuntime):
    def __init__(self) -> None:
        super().__init__(
            audio_sequence_layout=AudioSequenceLayout.SEMANTIC,
            text_tokenizer_state="text-v1",
            audio_tokenizer_state="audio-v1",
            backbone_readout="last_hidden_state",
            backbone_supports_cache_position=True,
            semantic_artifact_sha256=None,
        )
        self.lexical_text_vocab_size = len(self._text_tokenizer)
        text_end = self.lexical_text_vocab_size + len(ControlToken)
        audio_end = text_end + self._audio_tokenizer.vocab_size + 4
        self._layout = Layout(text=(0, text_end), audio=(text_end, audio_end))
        self._boa_token_id = audio_end - 4
        self._eoa_token_id = audio_end - 3
        self._mask_token_id = audio_end - 2
        self._audio_schema_token_id = audio_end - 1

    @cached_property
    def control_token_ids(self) -> tuple[int, ...]:
        return tuple(
            range(
                self.lexical_text_vocab_size,
                self.lexical_text_vocab_size + len(ControlToken),
            )
        )

    def control_token_id(self, token: ControlToken) -> int:
        return self.control_token_ids[list(ControlToken).index(token)]


class _GlobalCodec:
    sample_rate = 16_000
    frame_rate = 50.0
    semantic_codebook = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    semantic_codebook_sizes = (3,)
    global_codebook_sizes = (5, 7)
    global_feature_dim = 6
    global_unit_length = 2

    def tokenize(self, audio: Tensor, sample_rate: int) -> object:
        del audio, sample_rate
        raise NotImplementedError

    def detokenize(self, codes: object) -> Tensor:
        del codes
        raise NotImplementedError

    def global_codes_to_features(self, global_codes: Tensor) -> Tensor:
        shape = (*global_codes.shape[:-1], self.global_feature_dim)
        return global_codes.new_zeros(shape, dtype=torch.float32)

    def decode_features(
        self,
        semantic_codes: Tensor,
        global_features: Tensor,
    ) -> Tensor:
        del semantic_codes
        return global_features.new_zeros(1)


class _GlobalContractRuntime(_ContractRuntime):
    def __init__(self) -> None:
        super().__init__(
            audio_sequence_layout=AudioSequenceLayout.FLATTENED,
            text_tokenizer_state="text-v1",
            audio_tokenizer_state="audio-v1",
            backbone_readout="last_hidden_state",
            backbone_supports_cache_position=True,
            semantic_artifact_sha256=None,
        )
        self._codec = _GlobalCodec()
        self._audio_tokenizer = BiCodecAudioTokenizer(
            semantic_codebook_size=3,
            global_codebook_sizes=self._codec.global_codebook_sizes,
            global_unit_length=self._codec.global_unit_length,
        )
        text_end = len(self._text_tokenizer)
        audio_end = text_end + self._audio_tokenizer.vocab_size + 4
        self._layout = Layout(text=(0, text_end), audio=(text_end, audio_end))
        self._boa_token_id = audio_end - 4
        self._eoa_token_id = audio_end - 3
        self._mask_token_id = audio_end - 2
        self._audio_schema_token_id = audio_end - 1

    @property
    def codec_name(self) -> str:
        return "bicodec"

    @property
    def audio_view(self) -> AudioView:
        return AudioView.BICODEC

    @property
    def semantic_codebook_sizes(self) -> tuple[int, ...]:
        return self._codec.semantic_codebook_sizes

    @property
    def acoustic_side_channel(self) -> bool:
        return False

    @property
    def structured_full_sequence(self) -> bool:
        return True


class _CodesOnlyGlobalContractRuntime(_GlobalContractRuntime):
    @property
    def output_audio_detokenizer_name(self) -> None:
        return None

    @property
    def output_audio_detokenizer_identity(self) -> None:
        return None


class _DecoupledContractRuntime(_GlobalContractRuntime):
    def __init__(
        self,
        *,
        input_codec_name: str = "glm4",
        input_tokenizer_state: str = "glm4-v1",
        input_vocab_size: int = 11,
        input_frame_rate: float = 12.5,
    ) -> None:
        super().__init__()
        self._input_codec_name = input_codec_name
        self._input_codec_frame_rate = input_frame_rate
        self._input_audio_tokenizer = _StatefulNativeAudioTokenizer(
            vocab_size=input_vocab_size,
            state=input_tokenizer_state,
        )
        text_end = len(self._text_tokenizer)
        input_end = text_end + self._input_audio_tokenizer.vocab_size + 3
        output_end = input_end + self._audio_tokenizer.vocab_size + 4
        self._layout = Layout(
            text=(0, text_end),
            audio_input=(text_end, input_end),
            audio=(input_end, output_end),
        )
        self._input_boa_token_id = input_end - 3
        self._input_eoa_token_id = input_end - 2
        self._input_audio_schema_token_id = input_end - 1
        self._boa_token_id = output_end - 4
        self._eoa_token_id = output_end - 3
        self._mask_token_id = output_end - 2
        self._audio_schema_token_id = output_end - 1

    @property
    def input_audio_decoupled(self) -> bool:
        return True

    @property
    def input_codec_name(self) -> str:
        return self._input_codec_name

    @property
    def input_audio_view(self) -> AudioView:
        return AudioView.GLM4

    @property
    def input_audio_code_spec(self) -> AudioCodeSpec:
        return AudioCodeSpec(
            view=self.input_audio_view.value,
            schema=AudioCodeSchema.FRAME,
            sample_rate=16_000,
            frame_rate=self.input_codec_frame_rate,
            frame_codebook_sizes=(self._input_audio_tokenizer.vocab_size,),
        )

    @property
    def input_codec_frame_rate(self) -> float:
        return self._input_codec_frame_rate

    @cached_property
    def input_audio_tokenizer(self) -> AudioTokenizer:
        return self._input_audio_tokenizer

    @property
    def input_audio_block_name(self) -> str:
        return "audio_input"

    @property
    def input_boa_token_id(self) -> int:
        return self._input_boa_token_id

    @property
    def input_eoa_token_id(self) -> int:
        return self._input_eoa_token_id

    @property
    def input_audio_schema_token_id(self) -> int:
        return self._input_audio_schema_token_id

    @property
    def input_codec_audio_range(self) -> tuple[int, int]:
        start, _ = self.layout.blocks[self.input_audio_block_name]
        return start, self.input_boa_token_id


def _contract_runtime(
    *,
    audio_sequence_layout: AudioSequenceLayout = AudioSequenceLayout.SEMANTIC,
    text_tokenizer_state: str = "text-v1",
    audio_tokenizer_state: str = "audio-v1",
    backbone_readout: str = "last_hidden_state",
    backbone_supports_cache_position: bool = True,
    semantic_artifact_sha256: str | None = None,
) -> _ContractRuntime:
    runtime = _ContractRuntime(
        audio_sequence_layout=audio_sequence_layout,
        text_tokenizer_state=text_tokenizer_state,
        audio_tokenizer_state=audio_tokenizer_state,
        backbone_readout=backbone_readout,
        backbone_supports_cache_position=backbone_supports_cache_position,
        semantic_artifact_sha256=semantic_artifact_sha256,
    )
    _accept_token_runtime(runtime)
    _accept_flow_runtime(runtime)
    return runtime


def _contract_runtime_with_text_tokenizer(
    tokenizer: PreTrainedTokenizerFast,
) -> _ContractRuntime:
    runtime = _contract_runtime()
    if len(tokenizer) != len(runtime._text_tokenizer):
        raise AssertionError("fast tokenizer fixture must preserve the text token range.")
    runtime._text_tokenizer = cast(TextTokenizer, cast(object, tokenizer))
    return runtime


def _fast_text_tokenizer(*, lowercase: bool) -> PreTrainedTokenizerFast:
    backend = Tokenizer(
        WordLevel(
            {
                "<pad>": 0,
                "<bos>": 1,
                "<eos>": 2,
                "<unk>": 3,
                "hello": 4,
                "HELLO": 5,
                "world": 6,
                "!": 7,
            },
            unk_token="<unk>",
        )
    )
    backend.normalizer = normalizers.Lowercase() if lowercase else normalizers.NFC()
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    tokenizer.chat_template = "fixture-chat-v1"
    return tokenizer


def _accept_token_runtime(runtime: TokenModelRuntime) -> None:
    del runtime


def _accept_flow_runtime(runtime: FlowModelRuntime) -> None:
    del runtime


def _toy_config() -> ToyConfig:
    return ToyConfig(
        hidden_size=8,
        intermediate_size=16,
        layers=1,
        heads=2,
        max_position_embeddings=32,
    )


def _model_config(
    *,
    audio_input: AudioInputAdapterType = AudioInputAdapterType.MLP,
    audio_input_causal: bool = False,
    audio_output: AudioOutputAdapterType = AudioOutputAdapterType.NONE,
    ctc: CTCDecoderRoutesConfig | None = None,
) -> ModelConfig:
    return ModelConfig(
        semantic_audio_adapter=AdapterType.LINEAR,
        audio_input_adapter=AudioInputAdapterConfig(
            type=audio_input,
            layers=1,
            heads=2,
            ffn_ratio=2,
            causal=audio_input_causal,
        ),
        audio_output_adapter=AudioOutputAdapterConfig(
            type=audio_output,
            layers=1,
            heads=2,
            ffn_ratio=2,
        ),
        ctc=CTCDecoderRoutesConfig() if ctc is None else ctc,
        toy=_toy_config(),
    )


def _inactive_adapter_config(*, variant: int) -> ModelConfig:
    if variant == 1:
        layers, heads, ffn_ratio, dropout, causal = 1, 1, 2, 0.0, False
    else:
        layers, heads, ffn_ratio, dropout, causal = 3, 4, 7, 0.5, True
    return ModelConfig(
        semantic_audio_adapter=AdapterType.LINEAR,
        audio_input_adapter=AudioInputAdapterConfig(
            type=AudioInputAdapterType.MLP,
            layers=layers,
            heads=heads,
            ffn_ratio=ffn_ratio,
            dropout=dropout,
            causal=causal,
        ),
        audio_output_adapter=AudioOutputAdapterConfig(
            type=AudioOutputAdapterType.NONE,
            layers=layers,
            heads=heads,
            ffn_ratio=ffn_ratio,
            dropout=dropout,
        ),
        toy=_toy_config(),
    )


def _token_model(
    *,
    runtime: _ContractRuntime | None = None,
    config: ModelConfig | None = None,
) -> Model:
    return Model(
        _model_config() if config is None else config,
        runtime=_contract_runtime() if runtime is None else runtime,
    )


def _flow_model(*, layers: int = 1) -> FlowModel:
    return FlowModel(
        _model_config(),
        runtime=_contract_runtime(),
        decoder=DecoderConfig(
            hidden_dim=8,
            layers=layers,
            heads=2,
            ffn_ratio=2,
        ),
    )


def _rvq_model(*, layers: int = 1) -> RVQModel:
    return RVQModel(
        _model_config(),
        runtime=_contract_runtime(),
        decoder=DecoderConfig(
            hidden_dim=8,
            layers=layers,
            heads=2,
            ffn_ratio=2,
        ),
    )


def _real_module(model: Model) -> SpeechToSpeechModule[Any]:
    return SpeechToSpeechModule(
        Config(),
        model=cast(Any, model),
        objective=cast(Any, SimpleNamespace()),
    )


def _assert_contract_mismatch(
    test: unittest.TestCase,
    checkpoint_model: Model,
    current_model: Model,
    path: str,
) -> None:
    payload = json.loads(json.dumps(checkpoint_model.checkpoint_contract.checkpoint_payload()))
    with test.assertRaisesRegex(ValueError, re.escape(path)):
        validate_checkpoint_contract(payload, current_model.checkpoint_contract)


def _component(model: Model, *path: str) -> object:
    value: object = model.checkpoint_contract.components
    for key in path:
        if not isinstance(value, Mapping):
            raise AssertionError(f"model contract path is not a mapping at {key!r}")
        value = value[key]
    return value


def _contract(variant: int) -> ModelCheckpointContract:
    return ModelCheckpointContract.from_components(
        {
            "token_space": {
                "text_range": [0, 4],
                "audio_range": [4, 7],
            },
            "audio_sequence": {
                "grammar": "semantic-v1",
                "variant": variant,
            },
        }
    )


def _module(
    contract: ModelCheckpointContract,
    *,
    lora_config: LoraConfig | None = None,
) -> SpeechToSpeechModule[Any]:
    model = SimpleNamespace(
        checkpoint_contract=contract,
        lora_config=lora_config,
    )
    return SpeechToSpeechModule(
        Config(),
        model=cast(Any, model),
        objective=cast(Any, SimpleNamespace()),
    )


def _saved_checkpoint(module: SpeechToSpeechModule[Any]) -> dict[str, Any]:
    checkpoint: dict[str, Any] = {}
    module.on_save_checkpoint(checkpoint)
    return checkpoint


def _model_payload(checkpoint: dict[str, Any]) -> dict[str, Any]:
    return cast(dict[str, Any], checkpoint["speech_to_speech_model_contract"])


def _peft_payload(checkpoint: dict[str, Any]) -> dict[str, Any]:
    return cast(dict[str, Any], checkpoint["speech_to_speech_peft"])


if __name__ == "__main__":
    unittest.main()
