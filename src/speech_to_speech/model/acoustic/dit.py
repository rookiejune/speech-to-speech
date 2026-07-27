from __future__ import annotations

from anytrain.module.dit import DiT, DiTBlock, DiTConditionType, TimeEmbedding


class AcousticDiT(DiT):
    """Frame-conditioned acoustic decoder built from the shared anytrain DiT."""

    def __init__(
        self,
        condition_dim: int,
        latent_dim: int,
        *,
        hidden_dim: int | None = None,
        layers: int = 8,
        heads: int = 8,
        ffn_ratio: int = 4,
        repa_feature_dim: int | None = None,
        repa_student_layer: int | None = None,
    ) -> None:
        super().__init__(
            input_dim=latent_dim,
            output_dim=latent_dim,
            hidden_dim=hidden_dim,
            layers=layers,
            heads=heads,
            ffn_ratio=ffn_ratio,
            condition_dim=condition_dim,
            condition_type=DiTConditionType.FRAME_FILM,
            feature_dim=repa_feature_dim,
            feature_layer=repa_student_layer,
        )

__all__ = ["AcousticDiT", "DiTBlock", "TimeEmbedding"]
