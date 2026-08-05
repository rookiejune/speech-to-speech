from __future__ import annotations

from anydataset.types import Modality

from .contract import (
    FieldRole,
    PredictionModality,
    ResponseControl,
    ResponseLayout,
    ResponseSpec,
    ResponseStep,
    Task,
    TaskField,
    TaskObjective,
    TaskProgram,
)


DIRECT = "direct"
TARGET_COT = "target_cot"
FULL_COT = "full_cot"

_SOURCE_TEXT = TaskField(FieldRole.SOURCE, Modality.TEXT)
_SOURCE_AUDIO = TaskField(FieldRole.SOURCE, Modality.AUDIO)
_TARGET_TEXT = TaskField(FieldRole.TARGET, Modality.TEXT)
_TARGET_AUDIO = TaskField(FieldRole.TARGET, Modality.AUDIO)

_SOURCE_ASR = ResponseStep(_SOURCE_TEXT, ResponseControl.ASR)
_TARGET_ASR = ResponseStep(_TARGET_TEXT, ResponseControl.ASR)
_TARGET_MT = ResponseStep(_TARGET_TEXT, ResponseControl.MT)
_TARGET_TEXT_EOS = ResponseStep(_TARGET_TEXT, ResponseControl.EOS)
_TARGET_AUDIO_STEP = ResponseStep(_TARGET_AUDIO, ResponseControl.AUDIO)


def _response(
    name: str,
    *steps: ResponseStep,
    prediction: PredictionModality,
    layout: ResponseLayout = ResponseLayout.SEQUENTIAL,
) -> ResponseSpec:
    return ResponseSpec(
        name=name,
        steps=steps,
        prediction=prediction,
        layout=layout,
    )


PROGRAMS: dict[Task, TaskProgram] = {
    Task.AUDIO_AR: TaskProgram(
        context=(),
        responses=(
            _response(DIRECT, _TARGET_AUDIO_STEP, prediction=PredictionModality.AUDIO),
        ),
        supports_pretraining=True,
    ),
    Task.ASR: TaskProgram(
        context=(_TARGET_AUDIO,),
        responses=(
            _response(DIRECT, _TARGET_ASR, prediction=PredictionModality.TEXT),
        ),
    ),
    Task.INTERLEAVED_AR: TaskProgram(
        context=(),
        responses=(
            _response(
                DIRECT,
                _TARGET_TEXT_EOS,
                _TARGET_AUDIO_STEP,
                prediction=PredictionModality.INTERLEAVED,
                layout=ResponseLayout.INTERLEAVED,
            ),
        ),
    ),
    Task.MASKED_AR: TaskProgram(
        context=(_TARGET_TEXT, _TARGET_AUDIO),
        responses=(
            _response(
                DIRECT,
                _TARGET_TEXT_EOS,
                _TARGET_AUDIO_STEP,
                prediction=PredictionModality.PARALLEL,
                layout=ResponseLayout.MASKED,
            ),
            _response(
                "interleaved",
                _TARGET_TEXT_EOS,
                _TARGET_AUDIO_STEP,
                prediction=PredictionModality.INTERLEAVED,
                layout=ResponseLayout.MASKED,
            ),
        ),
        objective=TaskObjective.RECONSTRUCTION,
    ),
    Task.MT: TaskProgram(
        context=(_SOURCE_TEXT,),
        responses=(
            _response(DIRECT, _TARGET_MT, prediction=PredictionModality.TEXT),
        ),
    ),
    Task.PARALLEL_AR: TaskProgram(
        context=(),
        responses=(
            _response(
                DIRECT,
                _TARGET_TEXT_EOS,
                _TARGET_AUDIO_STEP,
                prediction=PredictionModality.PARALLEL,
                layout=ResponseLayout.BLOCKWISE,
            ),
        ),
    ),
    Task.S2ST: TaskProgram(
        context=(_SOURCE_AUDIO,),
        responses=(
            _response(DIRECT, _TARGET_AUDIO_STEP, prediction=PredictionModality.AUDIO),
            _response(
                TARGET_COT,
                _TARGET_MT,
                _TARGET_AUDIO_STEP,
                prediction=PredictionModality.PARALLEL,
                layout=ResponseLayout.BLOCKWISE,
            ),
            _response(
                FULL_COT,
                _SOURCE_ASR,
                _TARGET_MT,
                _TARGET_AUDIO_STEP,
                prediction=PredictionModality.PARALLEL,
                layout=ResponseLayout.BLOCKWISE,
            ),
        ),
    ),
    Task.S2TT: TaskProgram(
        context=(_SOURCE_AUDIO,),
        responses=(
            _response(DIRECT, _TARGET_MT, prediction=PredictionModality.TEXT),
            _response(
                FULL_COT,
                _SOURCE_ASR,
                _TARGET_MT,
                prediction=PredictionModality.TEXT,
            ),
        ),
    ),
    Task.TEXT_AR: TaskProgram(
        context=(),
        responses=(
            _response(DIRECT, _TARGET_TEXT_EOS, prediction=PredictionModality.TEXT),
        ),
        supports_pretraining=True,
    ),
    Task.T2ST: TaskProgram(
        context=(_SOURCE_TEXT,),
        responses=(
            _response(DIRECT, _TARGET_AUDIO_STEP, prediction=PredictionModality.AUDIO),
            _response(
                TARGET_COT,
                _TARGET_MT,
                _TARGET_AUDIO_STEP,
                prediction=PredictionModality.PARALLEL,
                layout=ResponseLayout.BLOCKWISE,
            ),
        ),
    ),
    Task.T2TT: TaskProgram(
        context=(_SOURCE_TEXT,),
        responses=(
            _response(DIRECT, _TARGET_MT, prediction=PredictionModality.TEXT),
        ),
    ),
    Task.TTS: TaskProgram(
        context=(_TARGET_TEXT,),
        responses=(
            _response(DIRECT, _TARGET_AUDIO_STEP, prediction=PredictionModality.AUDIO),
        ),
    ),
}


def program_for(task: Task) -> TaskProgram:
    if not isinstance(task, Task):
        raise TypeError("task program lookup requires a Task.")
    try:
        return PROGRAMS[task]
    except KeyError as error:  # pragma: no cover - exhaustive mapping invariant
        raise KeyError(f"missing task program for {task.value}.") from error


if set(PROGRAMS) != set(Task):
    missing = sorted(task.value for task in set(Task) - set(PROGRAMS))
    extra = sorted(task.value for task in set(PROGRAMS) - set(Task))
    raise AssertionError(f"task program mapping mismatch; missing={missing}, extra={extra}.")


__all__ = [
    "DIRECT",
    "FULL_COT",
    "PROGRAMS",
    "TARGET_COT",
    "program_for",
]
