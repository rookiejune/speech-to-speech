from __future__ import annotations

from typing import Optional, Protocol, Union, cast

from semantic_acoustic_codec.config import Route
from semantic_acoustic_codec.runtime.artifact import AcousticGeneratorArtifact

from semantic_acoustic_codec.loss.repa import WavLMTeacher

from speech_to_speech.loss.module import FlowObjective, RVQObjective, TokenObjective
from speech_to_speech.model import Config as ModelConfig
from speech_to_speech.model import Model
from speech_to_speech.model.acoustic import (
    AcousticType,
    DecoderConfig,
    FlowRepaConfig,
)
from speech_to_speech.model.acoustic.flow import FlowModel
from speech_to_speech.model.acoustic.initialization import load_acoustic_initialization
from speech_to_speech.model.acoustic.rvq import RVQModel
from speech_to_speech.runtime import Runtime
from speech_to_speech.runtime.types import frame_codec

from .module import Config, SpeechToSpeechModule
from .protocol import FlowCompositionModel, RVQCompositionModel


class RepaConfig(Protocol):
    @property
    def weight(self) -> Optional[float]: ...

    @property
    def teacher_checkpoint(self) -> str: ...

    @property
    def teacher_layer(self) -> int: ...

    @property
    def student_layer(self) -> Optional[int]: ...


class AcousticConfig(Protocol):
    @property
    def type(self) -> str: ...


class FlowConfig(AcousticConfig, Protocol):
    @property
    def init_artifact(self) -> Optional[str]: ...

    @property
    def decoder(self) -> DecoderConfig: ...

    @property
    def repa(self) -> RepaConfig: ...


class RVQConfig(AcousticConfig, Protocol):
    @property
    def init_artifact(self) -> Optional[str]: ...

    @property
    def decoder(self) -> DecoderConfig: ...


CompositionModule = Union[
    SpeechToSpeechModule[Model],
    SpeechToSpeechModule[FlowCompositionModel],
    SpeechToSpeechModule[RVQCompositionModel],
]
CompositionModel = Model


def build(
    runtime: Runtime,
    config: Config,
    model_config: ModelConfig,
    acoustic: AcousticConfig,
) -> tuple[AcousticType, CompositionModule, CompositionModel]:
    acoustic_type = AcousticType(acoustic.type)
    if acoustic_type is AcousticType.NONE:
        module, model = token(runtime, config, model_config)
        return acoustic_type, module, model
    if not runtime.acoustic_side_channel:
        raise ValueError(
            "runtime representation has no independent acoustic side channel; "
            "configure model/acoustic=none."
        )
    if acoustic_type is AcousticType.FLOW:
        module, model = flow(
            runtime,
            config,
            model_config,
            cast(FlowConfig, acoustic),
        )
    else:
        module, model = rvq(
            runtime,
            config,
            model_config,
            cast(RVQConfig, acoustic),
        )
    return acoustic_type, module, model


def token(
    runtime: Runtime,
    config: Config,
    model_config: ModelConfig,
) -> tuple[SpeechToSpeechModule[Model], Model]:
    model = Model(model_config, runtime=runtime)
    module = SpeechToSpeechModule(
        config,
        model=model,
        objective=TokenObjective(runtime.layout),
    )
    return module, model


def flow(
    runtime: Runtime,
    config: Config,
    model_config: ModelConfig,
    acoustic: FlowConfig,
) -> tuple[SpeechToSpeechModule[FlowCompositionModel], FlowModel]:
    teacher = None
    weight = acoustic.repa.weight
    if weight is not None:
        teacher = WavLMTeacher(
            frame_codec(runtime.codec),
            checkpoint=acoustic.repa.teacher_checkpoint,
            layer=acoustic.repa.teacher_layer,
            device=runtime.backbone.get_input_embeddings().weight.device,
        )
    initialization = _initialization(runtime, acoustic.init_artifact, Route.FM)
    if initialization is not None:
        expected_repa_weight = 0.0 if weight is None else weight
        if initialization.spec.decoder.repa_loss_weight != expected_repa_weight:
            raise ValueError(
                "Flow REPA weight does not match initialization artifact: "
                f"{expected_repa_weight!r} != "
                f"{initialization.spec.decoder.repa_loss_weight!r}."
            )
    model = FlowModel(
        model_config,
        runtime=runtime,
        decoder=acoustic.decoder,
        repa=(
            None
            if teacher is None
            else FlowRepaConfig(
                feature_dim=teacher.feature_dim,
                student_layer=acoustic.repa.student_layer,
            )
        ),
        initialization=initialization,
    )
    objective = FlowObjective(
        runtime.layout,
        runtime.flow_matching,
        repa=(
            None
            if weight is None or teacher is None
            else {"weight": weight, "teacher": teacher}
        ),
    )
    return SpeechToSpeechModule(config, model=model, objective=objective), model


def rvq(
    runtime: Runtime,
    config: Config,
    model_config: ModelConfig,
    acoustic: RVQConfig,
) -> tuple[SpeechToSpeechModule[RVQCompositionModel], RVQModel]:
    initialization = _initialization(runtime, acoustic.init_artifact, Route.RVQ)
    model = RVQModel(
        model_config,
        runtime=runtime,
        decoder=acoustic.decoder,
        initialization=initialization,
    )
    module = SpeechToSpeechModule[RVQCompositionModel](
        config,
        model=model,
        objective=RVQObjective(runtime.layout),
    )
    return module, model


def _initialization(
    runtime: Runtime,
    path: Optional[str],
    route: Route,
) -> AcousticGeneratorArtifact | None:
    if path is None:
        return None
    return load_acoustic_initialization(
        path,
        codec=runtime.codec,
        route=route,
        device=runtime.backbone.get_input_embeddings().weight.device,
    )
