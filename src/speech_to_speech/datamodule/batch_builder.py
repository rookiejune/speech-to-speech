from collections.abc import Sequence

import torch
from anytrain.idspace import IdSpaceEmbedding, Modality
from torch import LongTensor, Tensor

from ..runtime import qwen3_tokenizer
from ..types import (
    IGNORE_INDEX,
    AutoregressionExample,
    AudioBoundary,
    CausalLMBatch,
    GenerationBatch,
    SpecialToken,
    TranslationExample,
)

SOURCE_AUDIO_PLACEHOLDER = "<<<SPEECH_TO_SPEECH_SOURCE_AUDIO>>>"


def _autoregression_prompt() -> str:
    return "Continue the speech."


def _translation_prompt() -> str:
    return "Translate the source speech."


class CausalLMBatchBuilder:
    def __init__(
        self,
        embedding: IdSpaceEmbedding,
        tokenizer: object | None = None,
    ) -> None:
        self.embedding = embedding
        self.tokenizer = tokenizer or qwen3_tokenizer()

        self.to_global = self.embedding.space.to_global
        self.special_token_id = self.embedding.space.special_token_id
        audio_vocab_size = self.embedding.space.modality_block(Modality.AUDIO).vocab_size
        self.audio_vocab_size = audio_vocab_size
        self.boa_id = self.special_token_id(AudioBoundary.BOA)
        self.eoa_id = self.special_token_id(AudioBoundary.EOA)

        self._autoregression_prompt_ids = self._chat_prompt_ids(_autoregression_prompt())
        self._translation_prompt_parts = self._chat_prompt_parts(_translation_prompt())

    def autoregression(
        self,
        examples: AutoregressionExample | Sequence[AutoregressionExample],
    ) -> CausalLMBatch:
        rows = [
            self._autoregression_row(example.audio_ids)
            for example in _normalize_examples(examples, AutoregressionExample)
        ]
        return self._collate(rows)

    def autoregression_generation(
        self,
        prefix_ids: Tensor | None = None,
    ) -> GenerationBatch:
        prompt_ids = [
            *self._autoregression_prompt_ids,
            self.boa_id,
        ]
        if prefix_ids is None:
            device = None
        else:
            prefix = _normalize_id_tensor(prefix_ids)
            prompt_ids.extend(self.to_global(Modality.AUDIO, _tensor_to_list(prefix)))
            device = prefix.device
        return self._generation_batch(
            torch.tensor(prompt_ids, dtype=torch.long, device=device)
        )

    def _autoregression_row(self, audio_ids: LongTensor) -> tuple[LongTensor, LongTensor]:
        prefix = torch.tensor(
            self._autoregression_prompt_ids,
            dtype=torch.long,
            device=audio_ids.device,
        )
        return self._causal_row(prefix, self._audio_global_ids(audio_ids))

    def _collate(self, rows: Sequence[tuple[LongTensor, LongTensor]]) -> CausalLMBatch:
        if not rows:
            raise ValueError("rows must not be empty.")

        device = rows[0][0].device
        pad_id = self.special_token_id(SpecialToken.PAD)
        max_length = max(input_ids.numel() for input_ids, _ in rows)
        input_ids = torch.full(
            (len(rows), max_length),
            pad_id,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros(
            (len(rows), max_length),
            dtype=torch.long,
            device=device,
        )
        labels = torch.full(
            (len(rows), max_length),
            IGNORE_INDEX,
            dtype=torch.long,
            device=device,
        )

        for index, (row_ids, row_labels) in enumerate(rows):
            if row_ids.device != device or row_labels.device != device:
                raise ValueError("all rows must be on the same device.")
            length = row_ids.numel()
            input_ids[index, :length] = row_ids
            attention_mask[index, :length] = 1
            labels[index, :length] = row_labels

        return CausalLMBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            logits_to_keep=int(labels.ne(IGNORE_INDEX).sum(dim=1).max().item()),
        )

    def translation(
        self,
        examples: TranslationExample | Sequence[TranslationExample],
    ) -> CausalLMBatch:
        rows = [
            self._translation_row(example.source_ids, example.target_ids)
            for example in _normalize_examples(examples, TranslationExample)
        ]
        return self._collate(rows)

    def mixed(
        self,
        examples: Sequence[AutoregressionExample | TranslationExample],
    ) -> CausalLMBatch:
        if not examples:
            raise ValueError("examples must not be empty.")

        rows: list[tuple[LongTensor, LongTensor]] = []
        for example in examples:
            if isinstance(example, AutoregressionExample):
                rows.append(self._autoregression_row(_normalize_id_tensor(example.audio_ids)))
                continue
            if isinstance(example, TranslationExample):
                rows.append(self._translation_row(example.source_ids, example.target_ids))
                continue
            raise TypeError("examples must contain task example values.")
        return self._collate(rows)

    def translation_generation(self, source_ids: Tensor) -> GenerationBatch:
        source = _normalize_id_tensor(source_ids)
        prompt_ids = [
            *self._translation_prompt_ids(source),
            self.boa_id,
        ]
        return self._generation_batch(
            torch.tensor(prompt_ids, dtype=torch.long, device=source.device)
        )

    def _translation_row(
        self,
        source_ids: Tensor,
        target_ids: Tensor,
    ) -> tuple[LongTensor, LongTensor]:
        source_ids = _normalize_id_tensor(source_ids)
        target_ids = _normalize_id_tensor(target_ids)
        prefix_ids = self._translation_prompt_ids(source_ids)
        prefix = torch.tensor(prefix_ids, dtype=torch.long, device=source_ids.device)
        return self._causal_row(prefix, self._audio_global_ids(target_ids))

    @staticmethod
    def _causal_row(
        prefix: LongTensor,
        target_global_ids: LongTensor,
    ) -> tuple[LongTensor, LongTensor]:
        input_ids = torch.cat((prefix, target_global_ids[:-1]))
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        labels[prefix.numel() - 1 :] = target_global_ids
        return input_ids, labels

    def _audio_global_ids(self, audio_ids: LongTensor) -> LongTensor:
        audio_ids = _normalize_id_tensor(audio_ids)
        global_ids = [
            self.boa_id,
            *self.to_global(Modality.AUDIO, _tensor_to_list(audio_ids)),
            self.eoa_id,
        ]
        return torch.tensor(
            global_ids,
            dtype=torch.long,
            device=audio_ids.device,
        )

    def _translation_prompt_ids(self, source_audio_ids: LongTensor) -> list[int]:
        prefix_ids, suffix_ids = self._translation_prompt_parts
        source_global_ids = self.to_global(
            Modality.AUDIO,
            _tensor_to_list(source_audio_ids),
        )
        return [*prefix_ids, self.boa_id, *source_global_ids, self.eoa_id, *suffix_ids]

    def _chat_prompt_ids(self, prompt: str) -> list[int]:
        ids = _apply_chat_template(self.tokenizer, prompt)
        global_ids = _to_global_text_ids(self.embedding, ids)
        return global_ids

    def _chat_prompt_parts(self, prompt: str) -> tuple[list[int], list[int]]:
        ids = _apply_chat_template(self.tokenizer, f"{prompt}\n{SOURCE_AUDIO_PLACEHOLDER}")
        global_ids = _to_global_text_ids(self.embedding, ids)
        placeholder_ids = _to_global_text_ids(
            self.embedding,
            _encode_text(self.tokenizer, SOURCE_AUDIO_PLACEHOLDER),
        )
        return _split_subsequence(
            global_ids,
            placeholder_ids,
        )

    @staticmethod
    def _generation_batch(input_ids: LongTensor) -> GenerationBatch:
        if input_ids.dim() != 1:
            raise ValueError("generation input ids must be 1D.")
        return GenerationBatch(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=torch.ones(
                (1, input_ids.numel()),
                dtype=torch.long,
                device=input_ids.device,
            ),
        )


def _normalize_examples[ExampleT](
    examples: ExampleT | Sequence[ExampleT],
    example_type: type[ExampleT],
) -> list[ExampleT]:
    if isinstance(examples, example_type):
        return [examples]
    if not isinstance(examples, Sequence) or isinstance(examples, str | bytes):
        raise TypeError("examples must be an example or a sequence of examples.")
    if not examples:
        raise ValueError("examples must not be empty.")
    for example in examples:
        if not isinstance(example, example_type):
            raise TypeError(f"examples must contain {example_type.__name__} values.")
    return list(examples)


def _normalize_id_tensor(ids: Tensor) -> LongTensor:
    if ids.dim() != 1:
        raise ValueError("each audio id sequence must be 1D.")
    if ids.numel() == 0:
        raise ValueError("audio id sequences must not be empty.")
    if ids.dtype == torch.bool or torch.is_floating_point(ids) or torch.is_complex(ids):
        raise TypeError("audio ids must contain integer ids.")
    return ids.to(dtype=torch.long)


def _tensor_to_list(ids: Tensor) -> list[int]:
    return [int(token_id) for token_id in ids.detach().cpu().tolist()]


def _apply_chat_template(tokenizer: object, content: str) -> list[int]:
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=False,
    )
    if not isinstance(ids, list) or not all(isinstance(token_id, int) for token_id in ids):
        raise TypeError("chat template must return a flat token id list.")
    return ids


def _encode_text(tokenizer: object, text: str) -> list[int]:
    return [int(token_id) for token_id in tokenizer.encode(text, add_special_tokens=False)]


def _to_global_text_ids(embedding: IdSpaceEmbedding, ids: Sequence[int]) -> list[int]:
    global_ids: list[int] = []
    for token_id in ids:
        if embedding.space.is_special_token_id(int(token_id)):
            global_ids.append(int(token_id))
            continue
        global_ids.extend(embedding.space.to_global(Modality.TEXT, [int(token_id)]))
    return global_ids


def _split_subsequence(
    values: Sequence[int],
    old: Sequence[int],
) -> tuple[list[int], list[int]]:
    if not old:
        raise ValueError("old subsequence must not be empty.")
    limit = len(values) - len(old) + 1
    for start in range(limit):
        if list(values[start : start + len(old)]) == list(old):
            return (
                list(values[:start]),
                list(values[start + len(old) :]),
            )
    raise ValueError("source audio placeholder was not found in chat template ids.")
