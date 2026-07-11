from __future__ import annotations

import torch
from torch import Tensor, nn

from ..base import SemanticModel


class AcousticFlowDecoder(nn.Module):
    """Frame-level conditional velocity model for continuous acoustic latents."""

    def __init__(self, condition_dim: int, latent_dim: int) -> None:
        super().__init__()
        if condition_dim <= 0 or latent_dim <= 0:
            raise ValueError("condition_dim and latent_dim must be positive.")
        self.latent_dim = latent_dim
        hidden_dim = max(condition_dim, latent_dim)
        self.time = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.condition = nn.Linear(condition_dim, hidden_dim)
        self.input = nn.Linear(latent_dim, hidden_dim)
        self.output = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, latent_dim))

    def forward(self, x_t: Tensor, t: Tensor, *, condition: Tensor) -> Tensor:
        if x_t.shape != condition.shape[:-1] + (self.latent_dim,):
            raise ValueError("acoustic latent and condition shapes do not align.")
        if t.shape != (x_t.size(0),):
            raise ValueError("flow time must have shape [batch].")
        time = self.time(t[:, None].to(dtype=x_t.dtype))[:, None]
        hidden = self.input(x_t) + self.condition(condition) + time
        return self.output(hidden)


class SpeechToSpeechFlowModel(SemanticModel):
    """Speech-to-speech composition using a flow-matching decoder."""

    def __init__(self, config=None, runtime_snapshot=None) -> None:
        super().__init__(config=config, runtime_snapshot=runtime_snapshot)
        backbone_weight = self.backbone.embed_tokens.weight
        self.acoustic_decoder = AcousticFlowDecoder(
            self.backbone.config.hidden_size,
            self.runtime.codec.acoustic_feature_dim,
        ).to(device=backbone_weight.device, dtype=backbone_weight.dtype)

    def acoustic_target_latent(self, acoustic_labels: Tensor) -> Tensor:
        if acoustic_labels.dim() != 3:
            raise ValueError("acoustic labels must have shape [B, F, N].")
        safe_labels = acoustic_labels.clamp_min(0)
        codebooks = self.config.acoustic_codebooks
        if codebooks is not None:
            if codebooks <= 0 or codebooks > safe_labels.size(-1):
                raise ValueError(
                    "acoustic_codebooks must select valid target codebooks."
                )
            safe_labels = safe_labels[..., :codebooks]
        features = self.runtime.codec.acoustic_codes_to_features(safe_labels)
        return features.masked_fill((acoustic_labels < 0).all(dim=-1)[..., None], 0)

    @torch.no_grad()
    def sample_acoustic(self, condition: Tensor) -> Tensor:
        from anytrain.framework.flow_matching import ODESampler

        latent = torch.randn(
            (*condition.shape[:2], self.acoustic_decoder.latent_dim),
            device=condition.device,
            dtype=condition.dtype,
        )
        return (
            ODESampler(return_intermediates=False)
            .sample(
                self.acoustic_decoder,
                latent,
                condition=condition,
            )
            .final
        )

    @torch.no_grad()
    def generate_audio(
        self,
        prompt_ids: Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        acoustic_input_ids: Tensor | None = None,
        acoustic_input_positions: Tensor | None = None,
        acoustic_input_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        generated = self.generate_semantic(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            acoustic_input_ids=acoustic_input_ids,
            acoustic_input_positions=acoustic_input_positions,
            acoustic_input_mask=acoustic_input_mask,
            stop_token_id=self.runtime.eoa_token_id,
            token_range=self.runtime.layout.blocks["audio"],
        )
        audio_ids = generated[:, prompt_ids.size(1) :]
        audio_ids = audio_ids[:, audio_ids.ne(self.runtime.eoa_token_id).all(dim=0)]
        local_start, local_end = self.runtime.layout.blocks["audio"]
        valid = audio_ids.ge(local_start) & audio_ids.lt(local_end)
        if not bool(valid.all()):
            raise ValueError(
                "semantic generation produced non-audio tokens in the response."
            )
        positions: list[Tensor] = []
        spans: list[Tensor] = []
        for row in audio_ids:
            local = row - local_start
            _, counts = self.runtime.audio_tokenizer.expand_with_counts(local)
            row_spans = torch.as_tensor(counts, device=row.device, dtype=torch.long)
            positions.append(
                torch.repeat_interleave(
                    torch.arange(
                        prompt_ids.size(1),
                        prompt_ids.size(1) + row.numel(),
                        device=row.device,
                    ),
                    row_spans,
                )
            )
            spans.append(row_spans)
        if not positions or len({value.numel() for value in positions}) != 1:
            raise ValueError(
                "generated audio rows must expand to the same frame count."
            )
        output = self(generated, output_hidden_states=True)
        if output.hidden_states is None:
            raise RuntimeError("model did not return hidden states.")
        condition = self.target_frame_condition(
            output.hidden_states[-1], torch.stack(positions)
        )
        return generated, self.sample_acoustic(condition), torch.stack(spans)
