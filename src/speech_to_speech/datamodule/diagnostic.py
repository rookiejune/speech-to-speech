from __future__ import annotations

from typing import Union

from anydataset import types

from .._compat import StrEnum, auto
from ..task import Task

SampleItem = Union[types.AudioItem, types.TextItem]
SampleRef = tuple[types.Role, SampleItem]


class SampleSplit(StrEnum):
    TRAIN = auto()
    VALIDATION = auto()


def source_item(sample: types.Sample, task: Task) -> SampleRef | None:
    modality = task.source_modality
    if modality is None:
        return None
    pair_role = types.Role.SOURCE if task.uses_source_role else types.Role.TARGET
    return _item(sample, _role(sample, pair_role), modality)


def target_item(sample: types.Sample, task: Task) -> SampleRef:
    modality = task.target_modality
    if modality is None:
        modality = types.Modality.AUDIO
    return _item(sample, _role(sample, types.Role.TARGET), modality)


def _role(sample: types.Sample, pair_role: types.Role) -> types.Role:
    roles = {role for role, _ in sample}
    single = types.Role.DEFAULT in roles
    pair = bool(roles & {types.Role.SOURCE, types.Role.TARGET})
    if single and pair:
        raise ValueError("diagnostic samples cannot mix default and source/target roles.")
    return types.Role.DEFAULT if single else pair_role


def _item(
    sample: types.Sample,
    role: types.Role,
    modality: types.Modality,
) -> SampleRef:
    try:
        item = sample[(role, modality)]
    except KeyError as error:
        raise ValueError(
            f"diagnostic sample is missing {role.value}/{modality.value}."
        ) from error
    expected = types.AudioItem if modality is types.Modality.AUDIO else types.TextItem
    if not isinstance(item, expected):
        raise TypeError(
            f"diagnostic {role.value}/{modality.value} must contain {expected.__name__}."
        )
    return role, item


__all__ = [
    "SampleItem",
    "SampleRef",
    "SampleSplit",
    "source_item",
    "target_item",
]
