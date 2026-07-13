from __future__ import annotations

from typing import Protocol, cast

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch import Tensor

from ...loss.types import LossItem, Outputs


class _LossOutputProvider(Protocol):
    def current_loss_outputs(self) -> Outputs: ...


class GradLogger(Callback):
    def __init__(
        self,
        loss_pair: tuple[str, str],
        parameter_name: str,
        every_n_steps: int = 5_000,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()

        if every_n_steps < 1:
            raise ValueError("every_n_steps must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")

        self.loss_pair = loss_pair
        self.parameter_name = parameter_name
        self.every_n_steps = every_n_steps
        self.eps = eps

    def on_before_backward(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        loss: Tensor,
    ) -> None:
        del loss
        if trainer.global_step % self.every_n_steps != 0:
            return

        parameter = dict(pl_module.named_parameters()).get(self.parameter_name)
        if parameter is None:
            raise KeyError(
                f"unknown parameter {self.parameter_name!r}; "
                "use a name from pl_module.named_parameters()"
            )

        provider = cast(_LossOutputProvider, cast(object, pl_module))
        outputs = provider.current_loss_outputs()
        loss_a = _loss_value(outputs, self.loss_pair[0])
        loss_b = _loss_value(outputs, self.loss_pair[1])

        grad_a = torch.autograd.grad(
            loss_a,
            parameter,
            retain_graph=True,
            allow_unused=True,
        )[0]
        grad_b = torch.autograd.grad(
            loss_b,
            parameter,
            retain_graph=True,
            allow_unused=True,
        )[0]

        norm_a = _gradient_norm(grad_a, parameter)
        norm_b = _gradient_norm(grad_b, parameter)
        log_ratio = torch.log(norm_a.clamp_min(self.eps)) - torch.log(
            norm_b.clamp_min(self.eps)
        )
        cosine = _gradient_cosine(grad_a, grad_b, self.eps, parameter)

        prefix = f"grad/{self.loss_pair[0]}_{self.loss_pair[1]}"
        for name, value in (
            (f"norm/{self.loss_pair[0]}", norm_a),
            (f"norm/{self.loss_pair[1]}", norm_b),
            ("log_ratio", log_ratio),
            ("cosine", cosine),
        ):
            pl_module.log(
                f"{prefix}/{name}",
                value.detach(),
                on_step=True,
                logger=True,
                sync_dist=True,
            )


class GradNormLogger(Callback):
    def __init__(self, tag: str = "train/grad_norm") -> None:
        super().__init__()
        self.tag = tag

    def on_before_optimizer_step(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        del trainer, optimizer
        gradients = [
            parameter.grad.detach().norm(2)
            for parameter in pl_module.parameters()
            if parameter.grad is not None
        ]
        if gradients:
            pl_module.log(
                self.tag,
                torch.stack(gradients).norm(2),
                on_step=True,
                sync_dist=True,
            )


def _gradient_norm(grad: Tensor | None, parameter: Tensor) -> Tensor:
    if grad is None:
        return parameter.new_zeros(())
    return grad.detach().norm()


def _loss_value(outputs: Outputs, name: str) -> Tensor:
    value = outputs.get(name)
    if not isinstance(value, LossItem):
        raise KeyError(f"output {name!r} is not a LossItem")
    return value.loss.mean()


def _gradient_cosine(
    grad_a: Tensor | None,
    grad_b: Tensor | None,
    eps: float,
    parameter: Tensor,
) -> Tensor:
    if grad_a is None or grad_b is None:
        return parameter.new_tensor(float("nan"))
    flat_a = grad_a.detach().flatten()
    flat_b = grad_b.detach().flatten()
    denominator = flat_a.norm() * flat_b.norm()
    if denominator < eps:
        return parameter.new_tensor(float("nan"))
    return torch.dot(flat_a, flat_b) / denominator
