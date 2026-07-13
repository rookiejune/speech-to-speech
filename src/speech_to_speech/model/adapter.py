from __future__ import annotations

from torch import Tensor, nn

from .._compat import StrEnum, auto


class AdapterType(StrEnum):
    LINEAR = auto()
    MLP = auto()


class MLPAdapter(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()

        intermediate_size = int(round((8.0 / 3.0) * in_features))

        self.gate_proj = nn.Linear(in_features, intermediate_size, bias=False)
        self.up_proj = nn.Linear(in_features, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, out_features, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def create_adapter(
    adapter_type: AdapterType | None, in_features: int, out_features: int
) -> nn.Module:
    if adapter_type is None:
        if in_features != out_features:
            raise ValueError("identity adapter requires matching feature dimensions.")
        return nn.Identity()
    if adapter_type is AdapterType.LINEAR:
        return nn.Linear(in_features=in_features, out_features=out_features)
    if adapter_type is AdapterType.MLP:
        return MLPAdapter(in_features=in_features, out_features=out_features)
    raise AssertionError(f"unsupported adapter type: {adapter_type}")
