from __future__ import annotations

from omegaconf import DictConfig

from speech_to_speech.loss import FlowObjective, RVQObjective, TokenObjective, WavLMTeacher
from speech_to_speech.model import (
    DecoderConfig,
    FlowRepaConfig,
    SpeechToSpeechFlowModel,
    SpeechToSpeechRVQModel,
    TokenModel,
)
from speech_to_speech.pl_module import Config, SpeechToSpeechModule
from speech_to_speech.pl_module.protocol import FlowCompositionModel, RVQCompositionModel
from speech_to_speech.runtime import Runtime


def token(
    runtime: Runtime,
    config: Config,
) -> tuple[SpeechToSpeechModule[TokenModel], TokenModel]:
    model = TokenModel(runtime=runtime)
    module = SpeechToSpeechModule(
        config,
        model=model,
        objective=TokenObjective(runtime.layout),
    )
    return module, model


def flow(
    runtime: Runtime,
    config: Config,
    acoustic: DictConfig,
) -> tuple[
    SpeechToSpeechModule[FlowCompositionModel], SpeechToSpeechFlowModel, float | None
]:
    teacher = None
    weight = acoustic.repa.weight
    if weight is not None:
        teacher = WavLMTeacher(
            runtime.codec,
            checkpoint=str(acoustic.repa.teacher_checkpoint),
            layer=int(acoustic.repa.teacher_layer),
            device=runtime.backbone.get_input_embeddings().weight.device,
        )
    model = SpeechToSpeechFlowModel(
        runtime=runtime,
        decoder=decoder(acoustic.decoder),
        repa=(
            None
            if teacher is None
            else FlowRepaConfig(
                feature_dim=teacher.feature_dim,
                student_layer=(
                    None
                    if acoustic.repa.student_layer is None
                    else int(acoustic.repa.student_layer)
                ),
            )
        ),
    )
    objective = FlowObjective(
        runtime.layout,
        runtime.flow_matching,
        repa=(
            None
            if weight is None or teacher is None
            else {"weight": float(weight), "teacher": teacher}
        ),
    )
    return SpeechToSpeechModule(config, model=model, objective=objective), model, weight


def rvq(
    runtime: Runtime,
    config: Config,
    acoustic: DictConfig,
) -> tuple[SpeechToSpeechModule[RVQCompositionModel], SpeechToSpeechRVQModel]:
    model = SpeechToSpeechRVQModel(runtime=runtime, decoder=decoder(acoustic.decoder))
    module = SpeechToSpeechModule[RVQCompositionModel](
        config,
        model=model,
        objective=RVQObjective(runtime.layout),
    )
    return module, model


def decoder(config: DictConfig) -> DecoderConfig:
    hidden_dim = config.hidden_dim
    return DecoderConfig(
        hidden_dim=None if hidden_dim is None else int(hidden_dim),
        layers=int(config.layers),
        heads=int(config.heads),
        ffn_ratio=int(config.ffn_ratio),
    )
