"""Batch orchestration for resumable bidirectional streaming synthesis.

The pipeline owns durable snapshot boundaries and dependency timing.  Concrete
WMT19, translation, TTS, and codec model loading stays in workspace adapters.
Only model-generated target text is placed in the published training sample;
the original dataset translation is passed to :class:`SnapshotPublisher` as a
diagnostic sidecar.
"""

from __future__ import annotations

from collections.abc import Sequence, Sized
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol, TypeVar, cast

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

from speech_to_speech.datamodule.streaming import StreamStatus

from .cache import SynthesisStageCache
from .publisher import SnapshotPublisher, TranslationReference
from .telemetry import SynthesisTelemetry


_T = TypeVar("_T")


class TranslationSeedDataset(Sized, Protocol):
    """Map-style 2N seed view with a label-isolated diagnostic reference."""

    def __getitem__(self, index: int) -> Sample: ...

    def reference_translation(self, index: int) -> TextItem: ...


class SourceTTS(Protocol):
    def __call__(self, texts: Sequence[TextItem]) -> Sequence[AudioItem]: ...


class Translation(Protocol):
    def __call__(
        self,
        texts: Sequence[TextItem],
        target_languages: Sequence[Lang],
    ) -> Sequence[TextItem]: ...


class TargetTTS(Protocol):
    def __call__(
        self,
        texts: Sequence[TextItem],
        references: Sequence[AudioItem],
    ) -> Sequence[AudioItem]: ...


@dataclass(frozen=True)
class CodecPair:
    source: AudioItem
    target: AudioItem


class Codec(Protocol):
    def __call__(
        self,
        sources: Sequence[AudioItem],
        targets: Sequence[AudioItem],
    ) -> Sequence[CodecPair]: ...


@dataclass(frozen=True)
class Components:
    source_tts: SourceTTS
    translation: Translation
    target_tts: TargetTTS
    codec: Codec


@dataclass(frozen=True)
class StagePlacement:
    device: str | None = None
    gpu_ids: tuple[int | str, ...] = ()

    def __post_init__(self) -> None:
        if self.device is not None and (
            not isinstance(self.device, str) or not self.device
        ):
            raise ValueError("synthesis stage device must be non-empty when set.")
        for gpu_id in self.gpu_ids:
            if isinstance(gpu_id, bool) or not isinstance(gpu_id, (int, str)):
                raise TypeError("synthesis stage GPU ids must be integers or strings.")
            if isinstance(gpu_id, int) and gpu_id < 0:
                raise ValueError("synthesis stage GPU ids must be non-negative.")
            if isinstance(gpu_id, str) and not gpu_id:
                raise ValueError("synthesis stage GPU ids must be non-empty.")


@dataclass(frozen=True)
class PipelineConfig:
    batch_size: int
    source_tts: StagePlacement = StagePlacement()
    translation: StagePlacement = StagePlacement()
    target_tts: StagePlacement = StagePlacement()
    codec: StagePlacement = StagePlacement()

    def __post_init__(self) -> None:
        if type(self.batch_size) is not int or self.batch_size <= 0:
            raise ValueError("streaming synthesis batch_size must be positive.")


class StreamingSynthesisPipeline:
    """Resume at immutable batch snapshots until the complete 2N stream seals."""

    def __init__(
        self,
        dataset: TranslationSeedDataset,
        components: Components,
        publisher: SnapshotPublisher,
        config: PipelineConfig,
        cache: SynthesisStageCache | None = None,
    ) -> None:
        if not isinstance(dataset, Sized):
            raise TypeError("streaming synthesis seed dataset must expose __len__().")
        if len(dataset) != publisher.expected_samples:
            raise ValueError(
                "streaming synthesis seed count must match expected_samples: "
                f"{len(dataset)} != {publisher.expected_samples}."
            )
        if publisher.input_codec != publisher.codec:
            raise ValueError(
                "the standard streaming synthesis pipeline requires one codec "
                "for both source and target audio."
            )
        self.dataset = dataset
        self.components = components
        self.publisher = publisher
        self.config = config
        self.cache = cache

    def run(self, telemetry: SynthesisTelemetry) -> StreamStatus:
        """Continue after the published prefix and return the final seal."""

        status = self.publisher.feed.status()
        start = _published_prefix(status)
        telemetry.event(
            "pipeline_started",
            published_samples=start,
            expected_samples=self.publisher.expected_samples,
            batch_size=self.config.batch_size,
            sealed=status.seal is not None,
        )
        if status.seal is not None:
            return status
        if self.cache is not None:
            self.cache.discard_through(start)

        with ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="streaming-synthesis-front",
        ) as executor:
            while start < self.publisher.expected_samples:
                stop = min(start + self.config.batch_size, self.publisher.expected_samples)
                indices = tuple(range(start, stop))
                telemetry.event(
                    "batch_started",
                    first_index=start,
                    last_index=stop - 1,
                    sample_count=len(indices),
                )
                seeds = [self.dataset[index] for index in indices]
                source_texts = [_text_item(seed, Role.SOURCE, require_text=True) for seed in seeds]
                target_seeds = [_text_item(seed, Role.TARGET, require_text=False) for seed in seeds]
                target_languages = [_language(item, Role.TARGET) for item in target_seeds]
                reference_texts = [
                    _reference_text(
                        self.dataset.reference_translation(index),
                        language,
                    )
                    for index, language in zip(indices, target_languages)
                ]

                snapshot_id = f"samples-{start:012d}-{stop:012d}"
                cached_base = self._cache_load(
                    telemetry,
                    snapshot_id,
                    "target_tts",
                    indices,
                )
                if cached_base is not None:
                    source_audio, target_texts, target_audio, base_samples = (
                        _cached_base_batch(
                            cached_base,
                            source_texts=source_texts,
                            target_seeds=target_seeds,
                            target_languages=target_languages,
                        )
                    )
                else:
                    source_future = executor.submit(
                        self._source_tts,
                        telemetry,
                        snapshot_id,
                        indices,
                        source_texts,
                    )
                    translation_future = executor.submit(
                        self._translation,
                        telemetry,
                        snapshot_id,
                        indices,
                        source_texts,
                        target_seeds,
                        target_languages,
                    )
                    source_audio = self._join(
                        telemetry,
                        "source_tts_join",
                        source_future,
                        len(indices),
                    )
                    target_texts = self._join(
                        telemetry,
                        "translation_join",
                        translation_future,
                        len(indices),
                    )

                    with telemetry.stage(
                        "target_tts",
                        sample_count=len(indices),
                        device=self.config.target_tts.device,
                        gpu_ids=self.config.target_tts.gpu_ids,
                    ):
                        generated_audio = self.components.target_tts(
                            target_texts,
                            source_audio,
                        )
                    target_audio = _audio_batch(
                        generated_audio,
                        expected=len(indices),
                        name="target_tts",
                        waveform=True,
                    )
                    base_samples = [
                        _sample(source_text, target_text, source, target)
                        for source_text, target_text, source, target in zip(
                            source_texts,
                            target_texts,
                            source_audio,
                            target_audio,
                        )
                    ]
                    self._cache_save(
                        telemetry,
                        snapshot_id,
                        "target_tts",
                        indices,
                        base_samples,
                    )

                cached_codec = self._cache_load(
                    telemetry,
                    snapshot_id,
                    "codec",
                    indices,
                )
                if cached_codec is None:
                    with telemetry.stage(
                        "codec",
                        sample_count=2 * len(indices),
                        device=self.config.codec.device,
                        gpu_ids=self.config.codec.gpu_ids,
                    ):
                        encoded = self.components.codec(source_audio, target_audio)
                    pairs = _codec_batch(encoded, expected=len(indices))
                    codec_samples = [
                        _sample(source_text, target_text, pair.source, pair.target)
                        for source_text, target_text, pair in zip(
                            source_texts,
                            target_texts,
                            pairs,
                        )
                    ]
                    self._cache_save(
                        telemetry,
                        snapshot_id,
                        "codec",
                        indices,
                        codec_samples,
                    )
                else:
                    codec_samples = _cached_codec_batch(
                        cached_codec,
                        source_texts=source_texts,
                        target_texts=target_texts,
                    )
                references = [
                    TranslationReference(index, reference)
                    for index, reference in zip(indices, reference_texts)
                ]
                with telemetry.stage(
                    "snapshot_publish",
                    sample_count=len(indices),
                    device="cpu",
                ):
                    self.publisher.publish(
                        snapshot_id=snapshot_id,
                        sample_indices=indices,
                        translation_references=references,
                        base_samples=base_samples,
                        codec_samples=codec_samples,
                    )
                if self.cache is not None:
                    self.cache.discard(snapshot_id)
                telemetry.event(
                    "batch_published",
                    snapshot_id=snapshot_id,
                    first_index=start,
                    last_index=stop - 1,
                    sample_count=len(indices),
                )
                start = stop

        status = self.publisher.feed.status()
        if status.seal is None:
            raise RuntimeError("streaming synthesis finished without sealing the stream.")
        telemetry.event(
            "pipeline_sealed",
            snapshot_count=len(status.catalog.snapshots),
            sample_count=status.catalog.sample_count,
            catalog_sha256=status.catalog.sha256,
        )
        return status

    def _source_tts(
        self,
        telemetry: SynthesisTelemetry,
        snapshot_id: str,
        indices: Sequence[int],
        texts: Sequence[TextItem],
    ) -> Sequence[AudioItem]:
        cached = self._cache_load(
            telemetry,
            snapshot_id,
            "source_tts",
            indices,
        )
        if cached is not None:
            return _cached_source_batch(cached, source_texts=texts)
        with telemetry.stage(
            "source_tts",
            sample_count=len(texts),
            device=self.config.source_tts.device,
            gpu_ids=self.config.source_tts.gpu_ids,
        ):
            generated = self.components.source_tts(texts)
        audio = _audio_batch(
            generated,
            expected=len(texts),
            name="source_tts",
            waveform=True,
        )
        self._cache_save(
            telemetry,
            snapshot_id,
            "source_tts",
            indices,
            [
                {
                    (Role.SOURCE, Modality.TEXT): text,
                    (Role.SOURCE, Modality.AUDIO): item,
                }
                for text, item in zip(texts, audio)
            ],
        )
        return audio

    def _translation(
        self,
        telemetry: SynthesisTelemetry,
        snapshot_id: str,
        indices: Sequence[int],
        texts: Sequence[TextItem],
        target_seeds: Sequence[TextItem],
        languages: Sequence[Lang],
    ) -> Sequence[TextItem]:
        cached = self._cache_load(
            telemetry,
            snapshot_id,
            "translation",
            indices,
        )
        if cached is not None:
            return _cached_translation_batch(
                cached,
                source_texts=texts,
                target_seeds=target_seeds,
                target_languages=languages,
            )
        with telemetry.stage(
            "translation",
            sample_count=len(texts),
            device=self.config.translation.device,
            gpu_ids=self.config.translation.gpu_ids,
        ):
            generated = self.components.translation(texts, languages)
        targets = [
            _generated_target(item, seed, language)
            for item, seed, language in zip(
                _text_batch(generated, expected=len(texts), name="translation"),
                target_seeds,
                languages,
            )
        ]
        self._cache_save(
            telemetry,
            snapshot_id,
            "translation",
            indices,
            [
                {
                    (Role.SOURCE, Modality.TEXT): source,
                    (Role.TARGET, Modality.TEXT): target,
                }
                for source, target in zip(texts, targets)
            ],
        )
        return targets

    def _cache_load(
        self,
        telemetry: SynthesisTelemetry,
        snapshot_id: str,
        stage: str,
        indices: Sequence[int],
    ) -> list[Sample] | None:
        cache = self.cache
        if cache is None:
            return None
        with telemetry.stage(
            f"{stage}_cache_load",
            sample_count=len(indices),
            device="cpu",
        ):
            samples = cache.load(snapshot_id, stage, indices)
        if samples is not None:
            telemetry.event(
                "stage_cache_hit",
                stage=stage,
                snapshot_id=snapshot_id,
                sample_count=len(indices),
            )
        return samples

    def _cache_save(
        self,
        telemetry: SynthesisTelemetry,
        snapshot_id: str,
        stage: str,
        indices: Sequence[int],
        samples: Sequence[Sample],
    ) -> None:
        cache = self.cache
        if cache is None:
            return
        with telemetry.stage(
            f"{stage}_cache_write",
            sample_count=len(indices),
            device="cpu",
        ):
            cache.save(snapshot_id, stage, indices, samples)
        telemetry.event(
            "stage_cached",
            stage=stage,
            snapshot_id=snapshot_id,
            sample_count=len(indices),
        )

    @staticmethod
    def _join(
        telemetry: SynthesisTelemetry,
        name: str,
        future: Future[Sequence[_T]],
        sample_count: int,
    ) -> Sequence[_T]:
        with telemetry.wait(name, sample_count=sample_count):
            return future.result()


def _published_prefix(status: StreamStatus) -> int:
    count = status.catalog.sample_count
    if set(status.catalog.locations) != set(range(count)):
        raise RuntimeError(
            "streaming synthesis can only resume from a contiguous published prefix."
        )
    return count


def _text_item(sample: Sample, role: Role, *, require_text: bool) -> TextItem:
    reference = (role, Modality.TEXT)
    try:
        item = sample[reference]
    except KeyError as error:
        raise KeyError(f"streaming synthesis seed is missing {role.value} text.") from error
    if not isinstance(item, TextItem):
        raise TypeError(f"streaming synthesis seed {role.value} text must be a TextItem.")
    _language(item, role)
    if require_text:
        _text_value(item, role.value)
    elif TextView.TEXT in item.views:
        raise ValueError(
            "streaming synthesis target seed must not expose the dataset reference "
            "as TextView.TEXT."
        )
    return item


def _language(item: TextItem, role: Role) -> Lang:
    language = item.meta.get(TextMeta.LANG)
    if not isinstance(language, Lang):
        raise TypeError(
            f"streaming synthesis {role.value} text must declare one Lang value."
        )
    return language


def _text_value(item: TextItem, name: str) -> str:
    value = item.views.get(TextView.TEXT)
    if not isinstance(value, str) or not value:
        raise ValueError(f"streaming synthesis {name} text must be non-empty.")
    return value


def _reference_text(item: TextItem, language: Lang) -> str:
    reference_language = item.meta.get(TextMeta.LANG)
    if reference_language is not language:
        raise ValueError(
            "dataset reference language does not match the translation seed."
        )
    return _text_value(item, "dataset reference")


def _generated_target(item: TextItem, seed: TextItem, language: Lang) -> TextItem:
    text = _text_value(item, "generated target")
    generated_language = item.meta.get(TextMeta.LANG, language)
    if generated_language is not language:
        raise ValueError("generated target language does not match the translation seed.")
    meta = dict(seed.meta)
    for key, value in item.meta.items():
        previous = meta.get(key)
        if previous is not None and previous != value:
            raise ValueError(f"generated target metadata disagrees on {key.value!r}.")
        meta[key] = value
    meta[TextMeta.LANG] = language
    return TextItem(views={TextView.TEXT: text}, meta=meta)


def _cached_source_batch(
    samples: Sequence[Sample],
    *,
    source_texts: Sequence[TextItem],
) -> list[AudioItem]:
    values = _sample_batch(samples, expected=len(source_texts), name="source_tts")
    result: list[AudioItem] = []
    for sample, source_text in zip(values, source_texts):
        if _text_item(sample, Role.SOURCE, require_text=True) != source_text:
            raise ValueError("cached source TTS text does not match the seed.")
        result.append(_audio_item(sample, Role.SOURCE, waveform=True))
    return result


def _cached_translation_batch(
    samples: Sequence[Sample],
    *,
    source_texts: Sequence[TextItem],
    target_seeds: Sequence[TextItem],
    target_languages: Sequence[Lang],
) -> list[TextItem]:
    values = _sample_batch(samples, expected=len(source_texts), name="translation")
    result: list[TextItem] = []
    for sample, source_text, target_seed, language in zip(
        values,
        source_texts,
        target_seeds,
        target_languages,
    ):
        if _text_item(sample, Role.SOURCE, require_text=True) != source_text:
            raise ValueError("cached translation source text does not match the seed.")
        result.append(
            _generated_target(
                _text_item(sample, Role.TARGET, require_text=True),
                target_seed,
                language,
            )
        )
    return result


def _cached_base_batch(
    samples: Sequence[Sample],
    *,
    source_texts: Sequence[TextItem],
    target_seeds: Sequence[TextItem],
    target_languages: Sequence[Lang],
) -> tuple[list[AudioItem], list[TextItem], list[AudioItem], list[Sample]]:
    values = _sample_batch(samples, expected=len(source_texts), name="target_tts")
    sources: list[AudioItem] = []
    targets: list[TextItem] = []
    target_audio: list[AudioItem] = []
    normalized: list[Sample] = []
    for sample, source_text, target_seed, language in zip(
        values,
        source_texts,
        target_seeds,
        target_languages,
    ):
        if _text_item(sample, Role.SOURCE, require_text=True) != source_text:
            raise ValueError("cached target TTS source text does not match the seed.")
        target_text = _generated_target(
            _text_item(sample, Role.TARGET, require_text=True),
            target_seed,
            language,
        )
        source_audio = _audio_item(sample, Role.SOURCE, waveform=True)
        generated_audio = _audio_item(sample, Role.TARGET, waveform=True)
        sources.append(source_audio)
        targets.append(target_text)
        target_audio.append(generated_audio)
        normalized.append(
            _sample(source_text, target_text, source_audio, generated_audio)
        )
    return sources, targets, target_audio, normalized


def _cached_codec_batch(
    samples: Sequence[Sample],
    *,
    source_texts: Sequence[TextItem],
    target_texts: Sequence[TextItem],
) -> list[Sample]:
    values = _sample_batch(samples, expected=len(source_texts), name="codec")
    result: list[Sample] = []
    for sample, source_text, target_text in zip(
        values,
        source_texts,
        target_texts,
    ):
        if _text_item(sample, Role.SOURCE, require_text=True) != source_text:
            raise ValueError("cached codec source text does not match the seed.")
        if _text_item(sample, Role.TARGET, require_text=True) != target_text:
            raise ValueError("cached codec target text does not match target TTS.")
        result.append(
            _sample(
                source_text,
                target_text,
                _audio_item(sample, Role.SOURCE, waveform=False),
                _audio_item(sample, Role.TARGET, waveform=False),
            )
        )
    return result


def _sample_batch(
    values: Sequence[Sample],
    *,
    expected: int,
    name: str,
) -> list[Sample]:
    result = list(values)
    if len(result) != expected:
        raise ValueError(f"cached {name} sample count does not match the batch.")
    return result


def _audio_item(sample: Sample, role: Role, *, waveform: bool) -> AudioItem:
    reference = (role, Modality.AUDIO)
    try:
        item = sample[reference]
    except KeyError as error:
        raise KeyError(f"cached synthesis sample is missing {role.value} audio.") from error
    if not isinstance(item, AudioItem):
        raise TypeError(f"cached synthesis {role.value} audio must be an AudioItem.")
    if waveform and AudioView.WAVEFORM not in item.views:
        raise ValueError(f"cached synthesis {role.value} audio must contain waveform.")
    if not item.views:
        raise ValueError(f"cached synthesis {role.value} audio must contain one view.")
    return item


def _text_batch(
    values: Sequence[object],
    *,
    expected: int,
    name: str,
) -> list[TextItem]:
    result = list(values)
    if len(result) != expected:
        raise ValueError(f"{name} output count must match its input batch.")
    if any(not isinstance(value, TextItem) for value in result):
        raise TypeError(f"{name} outputs must be TextItem values.")
    return cast(list[TextItem], result)


def _audio_batch(
    values: Sequence[object],
    *,
    expected: int,
    name: str,
    waveform: bool,
) -> list[AudioItem]:
    result = list(values)
    if len(result) != expected:
        raise ValueError(f"{name} output count must match its input batch.")
    if any(not isinstance(value, AudioItem) for value in result):
        raise TypeError(f"{name} outputs must be AudioItem values.")
    items = cast(list[AudioItem], result)
    if waveform and any(AudioView.WAVEFORM not in item.views for item in items):
        raise ValueError(f"{name} outputs must contain AudioView.WAVEFORM.")
    return items


def _codec_batch(values: Sequence[CodecPair], *, expected: int) -> list[CodecPair]:
    result = list(values)
    if len(result) != expected:
        raise ValueError("codec output count must match its input batch.")
    if any(not isinstance(value, CodecPair) for value in result):
        raise TypeError("codec outputs must be CodecPair values.")
    if any(not value.source.views or not value.target.views for value in result):
        raise ValueError("codec outputs must contain source and target audio views.")
    return result


def _sample(
    source_text: TextItem,
    target_text: TextItem,
    source_audio: AudioItem,
    target_audio: AudioItem,
) -> Sample:
    return {
        (Role.SOURCE, Modality.TEXT): source_text,
        (Role.TARGET, Modality.TEXT): target_text,
        (Role.SOURCE, Modality.AUDIO): source_audio,
        (Role.TARGET, Modality.AUDIO): target_audio,
    }


__all__ = [
    "Codec",
    "CodecPair",
    "Components",
    "PipelineConfig",
    "SourceTTS",
    "StagePlacement",
    "StreamingSynthesisPipeline",
    "TargetTTS",
    "Translation",
    "TranslationSeedDataset",
]
