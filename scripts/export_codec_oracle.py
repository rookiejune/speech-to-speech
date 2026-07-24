from __future__ import annotations

import argparse

from semantic_acoustic_codec.config import (
    AdapterType,
    DecoderConfig,
    Initialization,
    Route,
)
from semantic_acoustic_codec.export import export_legacy_s2s_oracle
from semantic_acoustic_codec.runtime import SemanticCodecConfig


def main() -> None:
    args = _args()
    export_legacy_s2s_oracle(
        args.checkpoint,
        args.output_dir,
        SemanticCodecConfig(
            route=Route(args.route),
            condition_dim=args.condition_dim,
            decoder=DecoderConfig(
                hidden_dim=args.hidden_dim,
                layers=args.layers,
                heads=args.heads,
                ffn_ratio=args.ffn_ratio,
            ),
            adapter=None if args.adapter == "none" else AdapterType(args.adapter),
            initialization=Initialization(args.initialization),
            seed=args.seed,
        ),
        device=args.device,
        strict=not args.non_strict,
    )


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a legacy speech-to-speech codec oracle checkpoint to a "
            "semantic-acoustic-codec artifact."
        ),
    )
    parser.add_argument("checkpoint")
    parser.add_argument("output_dir")
    parser.add_argument("--route", choices=["dit_flow", "rvq"], required=True)
    parser.add_argument("--condition-dim", type=int, required=True)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ffn-ratio", type=int, default=4)
    parser.add_argument("--adapter", choices=["linear", "mlp", "none"], default="linear")
    parser.add_argument("--initialization", choices=["codec", "random"], default="codec")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("--non-strict", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
