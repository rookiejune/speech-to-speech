"""Publish bounded deterministic streams for restart probes.

The training subprocess controller supplies stream identity through
``S2S_SYNTHESIS_*`` environment variables.  This producer materializes small,
valid canonical stores without loading a synthesis model.  The default probe
is the historical LongCat coupled stream; the composed ``glm4 -> bicodec``
variant publishes separate input/output stores using the same two-batch plan.
The first batch is durable before the optional delay, so terminating the first
invocation during that delay and starting the same command again exercises the
real resume boundary.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from anydataset.types import (
    AudioItem,
    AudioView,
    Lang,
    Modality,
    Role,
    Sample,
    TextItem,
    TextMeta,
    TextView,
)

from speech_to_speech.datamodule.streaming import StreamStatus, WorkspaceSnapshotLoader
from speech_to_speech.synthesis.publisher import (
    SnapshotPublisher,
    TranslationReference,
)
from speech_to_speech.synthesis.telemetry import emit_event, stage


_DELAY_ENV = "S2S_SYNTHESIS_PROBE_DELAY_SECONDS"
_DEFAULT_DELAY_SECONDS = 60.0
_LONGCAT_CODEBOOK_SIZES = (8192, 8100, 8100, 8100)
_GLM4_CODEBOOK_SIZE = 16_384
_BICODEC_SEMANTIC_CODEBOOK_SIZE = 8_192
_BICODEC_GLOBAL_CODEBOOK_SIZE = 4_096
_BICODEC_GLOBAL_UNIT_LENGTH = 32
_BICODEC_SEMANTIC_FRAMES = 4
_GLM4_SEMANTIC_FRAMES = 2


@dataclass(frozen=True)
class ProbeConfig:
    root: Path
    stream_id: str
    expected_samples: int
    codec: str
    input_codec: str
    split: str
    delay_seconds: float

    @classmethod
    def from_environment(cls, environment: Mapping[str, str]) -> ProbeConfig:
        codec = _required(environment, "S2S_SYNTHESIS_CODEC")
        input_codec = environment.get("S2S_SYNTHESIS_INPUT_CODEC", codec)
        output_codec = environment.get("S2S_SYNTHESIS_OUTPUT_CODEC", codec)
        if not input_codec:
            raise ValueError("S2S_SYNTHESIS_INPUT_CODEC must be a non-empty string.")
        if not output_codec:
            raise ValueError("S2S_SYNTHESIS_OUTPUT_CODEC must be a non-empty string.")
        if output_codec != codec:
            raise ValueError(
                "the bounded streaming probe requires "
                "S2S_SYNTHESIS_OUTPUT_CODEC to match S2S_SYNTHESIS_CODEC."
            )
        if (input_codec, codec) not in {
            ("longcat", "longcat"),
            ("glm4", "bicodec"),
        }:
            raise ValueError(
                "the bounded streaming probe only supports codec='longcat' "
                "or input_codec='glm4' with codec='bicodec'."
            )
        expected_samples = _positive_int(
            _required(environment, "S2S_SYNTHESIS_EXPECTED_SAMPLES"),
            "S2S_SYNTHESIS_EXPECTED_SAMPLES",
        )
        if expected_samples % 2 != 0:
            raise ValueError(
                "S2S_SYNTHESIS_EXPECTED_SAMPLES must be even for bidirectional 2N data."
            )
        return cls(
            root=Path(_required(environment, "S2S_SYNTHESIS_ROOT")).expanduser().resolve(),
            stream_id=_required(environment, "S2S_SYNTHESIS_STREAM_ID"),
            expected_samples=expected_samples,
            codec=codec,
            input_codec=input_codec,
            split=_required(environment, "S2S_SYNTHESIS_SPLIT"),
            delay_seconds=_nonnegative_seconds(
                environment.get(_DELAY_ENV, str(_DEFAULT_DELAY_SECONDS)),
                _DELAY_ENV,
            ),
        )


@dataclass(frozen=True)
class _Batch:
    snapshot_id: str
    indices: tuple[int, ...]


def run(
    environment: Mapping[str, str] | None = None,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> StreamStatus:
    """Publish or resume the bounded stream and return its sealed status."""

    config = ProbeConfig.from_environment(os.environ if environment is None else environment)
    batches = _batches(config.expected_samples)
    publisher = SnapshotPublisher(
        config.root,
        stream_id=config.stream_id,
        expected_samples=config.expected_samples,
        codec=config.codec,
        input_codec=config.input_codec,
        split=config.split,
        loader=WorkspaceSnapshotLoader(
            codec=config.codec,
            input_codec=config.input_codec,
            split=config.split,
        ),
    )
    initial = publisher.feed.status()
    _validate_prefix(initial, batches)
    first_was_published = bool(initial.catalog.snapshots)
    _event(
        "start",
        stream_id=config.stream_id,
        published_samples=initial.catalog.sample_count,
        expected_samples=config.expected_samples,
        sealed=initial.seal is not None,
    )

    for position, batch in enumerate(batches):
        with stage(
            "snapshot_publish",
            sample_count=len(batch.indices),
            device="cpu",
        ):
            publisher.publish(
                snapshot_id=batch.snapshot_id,
                sample_indices=batch.indices,
                translation_references=[
                    TranslationReference(index, _reference_translation(index))
                    for index in batch.indices
                ],
                base_samples=[_sample(index, AudioView.WAVEFORM) for index in batch.indices],
                codec_samples=[
                    _sample(index, _output_view(config)) for index in batch.indices
                ],
                input_codec_samples=(
                    [_input_sample(index, config.input_codec) for index in batch.indices]
                    if config.input_codec != config.codec
                    else None
                ),
            )
        _event(
            "published",
            snapshot_id=batch.snapshot_id,
            first_index=batch.indices[0],
            last_index=batch.indices[-1],
            sample_count=len(batch.indices),
        )
        if position == 0 and not first_was_published and config.delay_seconds > 0:
            _event("delay", seconds=config.delay_seconds)
            with stage("resume_probe_delay", sample_count=0, device="cpu"):
                sleep(config.delay_seconds)

    with stage(
        "stream_seal_validation",
        sample_count=config.expected_samples,
        device="cpu",
    ):
        status = publisher.feed.status()
        _validate_prefix(status, batches)
        if status.seal is None:
            raise RuntimeError("bounded streaming probe finished without sealing the stream.")
    _event(
        "sealed",
        stream_id=config.stream_id,
        snapshot_count=len(status.catalog.snapshots),
        sample_count=status.catalog.sample_count,
        catalog_sha256=status.catalog.sha256,
    )
    return status


def main() -> None:
    run()


def _batches(expected_samples: int) -> tuple[_Batch, _Batch]:
    midpoint = expected_samples // 2
    return (
        _Batch("probe-first-half", tuple(range(midpoint))),
        _Batch("probe-second-half", tuple(range(midpoint, expected_samples))),
    )


def _validate_prefix(status: StreamStatus, batches: Sequence[_Batch]) -> None:
    snapshots = status.catalog.snapshots
    if len(snapshots) > len(batches):
        raise RuntimeError("bounded streaming probe root contains unexpected snapshots.")
    for position, snapshot in enumerate(snapshots):
        expected = batches[position]
        if (
            snapshot.snapshot_id != expected.snapshot_id
            or snapshot.sample_indices != expected.indices
        ):
            raise RuntimeError(
                "bounded streaming probe root is not a prefix of its two-batch plan."
            )


def _output_view(config: ProbeConfig) -> AudioView:
    if config.codec == "longcat":
        return AudioView.LONGCAT
    if config.codec == "bicodec":
        return AudioView.BICODEC
    raise AssertionError(f"unsupported bounded probe output codec: {config.codec!r}.")


def _sample(index: int, view: AudioView) -> Sample:
    source_text, source_lang, target_text, target_lang = _translation(index)
    return cast(
        Sample,
        {
            (Role.SOURCE, Modality.TEXT): TextItem(
                views={TextView.TEXT: source_text},
                meta={TextMeta.LANG: source_lang},
            ),
            (Role.TARGET, Modality.TEXT): TextItem(
                views={TextView.TEXT: target_text},
                meta={TextMeta.LANG: target_lang},
            ),
            (Role.SOURCE, Modality.AUDIO): AudioItem(
                views={view: _audio(index, target=False, view=view)}
            ),
            (Role.TARGET, Modality.AUDIO): AudioItem(
                views={view: _audio(index, target=True, view=view)}
            ),
        },
    )


def _input_sample(index: int, codec: str) -> Sample:
    if codec != "glm4":
        raise AssertionError(f"unsupported bounded probe input codec: {codec!r}.")
    source_text, source_lang, target_text, target_lang = _translation(index)
    return cast(
        Sample,
        {
            (Role.SOURCE, Modality.TEXT): TextItem(
                views={TextView.TEXT: source_text},
                meta={TextMeta.LANG: source_lang},
            ),
            (Role.TARGET, Modality.TEXT): TextItem(
                views={TextView.TEXT: target_text},
                meta={TextMeta.LANG: target_lang},
            ),
            (Role.SOURCE, Modality.AUDIO): AudioItem(
                views={
                    AudioView.GLM4: _audio(
                        index,
                        target=False,
                        view=AudioView.GLM4,
                    )
                }
            ),
        },
    )


def _translation(index: int) -> tuple[str, Lang, str, Lang]:
    pair = index // 2
    if index % 2 == 0:
        return (
            f"流式探针源句 {pair}",
            Lang.ZH,
            f"streaming probe generated translation {pair}",
            Lang.EN,
        )
    return (
        f"streaming probe source sentence {pair}",
        Lang.EN,
        f"流式探针生成译文 {pair}",
        Lang.ZH,
    )


def _reference_translation(index: int) -> str:
    pair = index // 2
    if index % 2 == 0:
        return f"streaming probe dataset translation {pair}"
    return f"流式探针数据集译文 {pair}"


def _audio(index: int, *, target: bool, view: AudioView) -> object:
    seed = index * 2 + int(target)
    if view is AudioView.WAVEFORM:
        amplitude = float((seed % 11) + 1) / 100.0
        return torch.full((1, 320), amplitude, dtype=torch.float32), 16_000
    if view is AudioView.LONGCAT:
        frames = 4
        columns = [
            (torch.arange(frames, dtype=torch.long) + seed * 17 + codebook * 29) % size
            for codebook, size in enumerate(_LONGCAT_CODEBOOK_SIZES)
        ]
        return torch.stack(columns, dim=1)
    if view is AudioView.GLM4:
        steps = torch.arange(_GLM4_SEMANTIC_FRAMES, dtype=torch.long)
        return ((steps + seed * 31) % _GLM4_CODEBOOK_SIZE).unsqueeze(1)
    if view is AudioView.BICODEC:
        semantic_steps = torch.arange(_BICODEC_SEMANTIC_FRAMES, dtype=torch.long)
        global_steps = torch.arange(_BICODEC_GLOBAL_UNIT_LENGTH, dtype=torch.long)
        return {
            "semantic": (
                (semantic_steps + seed * 37) % _BICODEC_SEMANTIC_CODEBOOK_SIZE
            ).unsqueeze(1),
            "global": (
                (global_steps + seed * 41) % _BICODEC_GLOBAL_CODEBOOK_SIZE
            ).unsqueeze(1),
        }
    raise ValueError(f"unsupported bounded probe audio view: {view.value}.")


def _required(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name)
    if value is None or not value:
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _positive_int(value: str, name: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer.") from error
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return result


def _nonnegative_seconds(value: str, name: str) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a finite non-negative number.") from error
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite non-negative number.")
    return result


def _event(event: str, **values: object) -> None:
    emit_event(event, **values)


if __name__ == "__main__":
    main()
