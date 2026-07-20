from __future__ import annotations

from speech_to_speech.loss import FlowObjective, RVQObjective, TokenObjective, WavLMTeacher
from speech_to_speech.model import (
    Config as ModelConfig,
    FlowRepaConfig,
    SpeechToSpeechFlowModel,
    SpeechToSpeechRVQModel,
    TokenModel,
)
from speech_to_speech.pl_module import Config, SpeechToSpeechModule
from speech_to_speech.pl_module.protocol import FlowCompositionModel, RVQCompositionModel
from speech_to_speech.runtime import Runtime

if __package__:
    from ._config import FlowConfig, RVQConfig
else:
    from _config import FlowConfig, RVQConfig


def token(
    runtime: Runtime,
    config: Config,
    model_config: ModelConfig,
) -> tuple[SpeechToSpeechModule[TokenModel], TokenModel]:
    model = TokenModel(model_config, runtime=runtime)
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
) -> tuple[
    SpeechToSpeechModule[FlowCompositionModel], SpeechToSpeechFlowModel, float | None
]:
    teacher = None
    weight = acoustic.repa.weight
    if weight is not None:
        teacher = WavLMTeacher(
            runtime.codec,
            checkpoint=acoustic.repa.teacher_checkpoint,
            layer=acoustic.repa.teacher_layer,
            device=runtime.backbone.get_input_embeddings().weight.device,
        )
    model = SpeechToSpeechFlowModel(
        model_config,
        runtime=runtime,
        decoder=acoustic.decoder,
        repa=(
            None
            if teacher is None
            else FlowRepaConfig(
                feature_dim=teacher.feature_dim,
                student_layer=(
                    None
                    if acoustic.repa.student_layer is None
                    else acoustic.repa.student_layer
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
            else {"weight": weight, "teacher": teacher}
        ),
    )
    return SpeechToSpeechModule(config, model=model, objective=objective), model, weight


def rvq(
    runtime: Runtime,
    config: Config,
    model_config: ModelConfig,
    acoustic: RVQConfig,
) -> tuple[SpeechToSpeechModule[RVQCompositionModel], SpeechToSpeechRVQModel]:
    model = SpeechToSpeechRVQModel(
        model_config,
        runtime=runtime,
        decoder=acoustic.decoder,
    )
    module = SpeechToSpeechModule[RVQCompositionModel](
        config,
        model=model,
        objective=RVQObjective(runtime.layout),
    )
    return module, model
