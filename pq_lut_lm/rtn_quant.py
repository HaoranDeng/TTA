from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .modeling import DEFAULT_TARGET_REGEX, _set_submodule, iter_target_linears


@dataclass
class RTNConfig:
    bits: int = 8
    granularity: str = "per_channel"
    group_size: int = 128
    target_regex: str = DEFAULT_TARGET_REGEX
    include_lm_head: bool = False
    max_linears: int | None = None


class RTNLinear(nn.Module):
    """Dense Linear with round-to-nearest dequantized weights.

    This is an accuracy scaffold for RTN/PTQ. It stores dequantized weights and
    runs the regular dense matmul, so runtime is not an int8-kernel benchmark.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        source_name: str,
        config: RTNConfig,
        stats: dict[str, Any],
    ) -> None:
        super().__init__()
        self.in_features = int(weight.shape[1])
        self.out_features = int(weight.shape[0])
        self.source_name = source_name
        self.config = config
        self.register_buffer("weight", weight.detach().clone(), persistent=True)
        if bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", bias.detach().clone(), persistent=True)
        self._stats = stats

    @staticmethod
    def _quant_dequant_symmetric(
        weight: torch.Tensor,
        config: RTNConfig,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if config.bits < 2 or config.bits > 16:
            raise ValueError("--bits must be between 2 and 16")
        qmax = float(2 ** (config.bits - 1) - 1)
        fp = weight.float()
        eps = torch.finfo(torch.float32).eps

        if config.granularity == "per_tensor":
            scale = fp.abs().amax().clamp_min(eps) / qmax
            q = torch.round(fp / scale).clamp(-qmax, qmax)
            deq = q * scale
            scale_count = 1
        elif config.granularity == "per_channel":
            scale = fp.abs().amax(dim=1, keepdim=True).clamp_min(eps) / qmax
            q = torch.round(fp / scale).clamp(-qmax, qmax)
            deq = q * scale
            scale_count = int(fp.shape[0])
        elif config.granularity == "per_group":
            group = config.group_size
            if group <= 0:
                raise ValueError("--group-size must be positive for per_group RTN")
            out_features, in_features = fp.shape
            padded = ((in_features + group - 1) // group) * group
            if padded != in_features:
                pad = torch.zeros((out_features, padded - in_features), device=fp.device, dtype=fp.dtype)
                fp_padded = torch.cat([fp, pad], dim=1)
            else:
                fp_padded = fp
            view = fp_padded.view(out_features, padded // group, group)
            scale = view.abs().amax(dim=2, keepdim=True).clamp_min(eps) / qmax
            q = torch.round(view / scale).clamp(-qmax, qmax)
            deq = (q * scale).view(out_features, padded)[:, :in_features]
            scale_count = int(out_features * (padded // group))
        else:
            raise ValueError(f"Unsupported RTN granularity: {config.granularity}")

        err = (deq - fp).float()
        stats = {
            "weight_mse": float(torch.mean(err * err).item()),
            "weight_max_abs_error": float(err.abs().amax().item()),
            "scale_count": scale_count,
        }
        return deq.to(dtype=weight.dtype), stats

    @classmethod
    @torch.no_grad()
    def from_linear(cls, linear: nn.Linear, config: RTNConfig, source_name: str) -> "RTNLinear":
        deq_weight, err_stats = cls._quant_dequant_symmetric(linear.weight.detach(), config)
        stats = cls._hardware_stats_for(
            name=source_name,
            in_features=int(linear.in_features),
            out_features=int(linear.out_features),
            config=config,
            err_stats=err_stats,
        )
        return cls(deq_weight, linear.bias, source_name, config, stats)

    @staticmethod
    def _hardware_stats_for(
        name: str,
        in_features: int,
        out_features: int,
        config: RTNConfig,
        err_stats: dict[str, Any],
    ) -> dict[str, Any]:
        weight_count = in_features * out_features
        bias_values = out_features
        scale_count = int(err_stats["scale_count"])
        return {
            "name": name,
            "method": "rtn_weight_only",
            "bits": config.bits,
            "granularity": config.granularity,
            "group_size": config.group_size if config.granularity == "per_group" else 0,
            "in_features": in_features,
            "out_features": out_features,
            "weight_count": weight_count,
            "weight_bits": weight_count * config.bits,
            "weight_mib_packed": weight_count * config.bits / 8 / 2**20,
            "scale_count": scale_count,
            "scale_mib_fp16": scale_count * 2 / 2**20,
            "bias_values": bias_values,
            "dense_mac_per_token": weight_count,
            "lookups_per_token": 0,
            "adds_per_token": 0,
            **err_stats,
        }

    def hardware_stats(self) -> dict[str, Any]:
        return dict(self._stats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight.to(dtype=x.dtype), self.bias)


def _aggregate_rtn_stats(module_stats: list[dict[str, Any]]) -> dict[str, Any]:
    weight_count = sum(int(s["weight_count"]) for s in module_stats)
    weight_bits = sum(float(s["weight_bits"]) for s in module_stats)
    scale_count = sum(int(s["scale_count"]) for s in module_stats)
    dense_mac = sum(float(s["dense_mac_per_token"]) for s in module_stats)
    weighted_mse = sum(float(s["weight_mse"]) * int(s["weight_count"]) for s in module_stats)
    max_err = max((float(s["weight_max_abs_error"]) for s in module_stats), default=0.0)
    return {
        "quantized_linears": len(module_stats),
        "weight_count": weight_count,
        "weight_bits": weight_bits,
        "weight_mib_packed": weight_bits / 8 / 2**20,
        "scale_count": scale_count,
        "scale_mib_fp16": scale_count * 2 / 2**20,
        "dense_mac_per_token": dense_mac,
        "lookups_per_token": 0.0,
        "adds_per_token": 0.0,
        "weighted_weight_mse": weighted_mse / max(weight_count, 1),
        "weight_max_abs_error": max_err,
    }


@torch.no_grad()
def replace_with_rtn_quant(
    model: nn.Module,
    config: RTNConfig,
) -> dict[str, Any]:
    target_linears = iter_target_linears(
        model,
        config.target_regex,
        include_lm_head=config.include_lm_head,
        max_linears=config.max_linears,
    )
    if not target_linears:
        raise RuntimeError("No target Linear modules matched")
    modules = dict(model.named_modules())
    stats = []
    if next(model.parameters()).is_cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    for idx, (name, _linear) in enumerate(target_linears, start=1):
        current = modules[name]
        if not isinstance(current, nn.Linear):
            raise TypeError(f"{name} is no longer nn.Linear")
        wrapped = RTNLinear.from_linear(current, config, name)
        _set_submodule(model, name, wrapped)
        stats.append(wrapped.hardware_stats())
        modules = dict(model.named_modules())
        print(f"RTN quantized {idx}/{len(target_linears)}: {name}", flush=True)
    if next(model.parameters()).is_cuda:
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    return {
        "modules": stats,
        "aggregate": _aggregate_rtn_stats(stats),
        "quantization_seconds": seconds,
    }
