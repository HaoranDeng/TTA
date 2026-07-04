from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch import nn

from .modeling import DEFAULT_TARGET_REGEX, _set_submodule, collect_calibration_inputs, iter_target_linears
from .rtn_quant import RTNConfig, RTNLinear


@dataclass
class W8A8Config:
    weight_bits: int = 8
    activation_bits: int = 8
    weight_granularity: str = "per_channel"
    weight_group_size: int = 128
    activation_granularity: str = "dynamic_per_token"
    activation_percentile: float = 1.0
    smoothquant_alpha: float = -1.0
    smoothquant_min_scale: float = 1e-5
    smoothquant_max_scale: float = 1e5
    target_regex: str = DEFAULT_TARGET_REGEX
    include_lm_head: bool = False
    max_linears: int | None = None


class W8A8Linear(nn.Module):
    """Dense Linear with RTN weights and fake-quantized input activations.

    This is an accuracy scaffold for W8A8/PTQ. It quantizes and dequantizes
    activations in the forward pass and then runs a regular dense matmul.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        act_scale: torch.Tensor | None,
        smooth_scale: torch.Tensor | None,
        source_name: str,
        config: W8A8Config,
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
        if act_scale is None:
            self.act_scale = None
        else:
            self.register_buffer("act_scale", act_scale.detach().clone(), persistent=True)
        if smooth_scale is None:
            self.smooth_scale = None
        else:
            self.register_buffer("smooth_scale", smooth_scale.detach().clone(), persistent=True)
        self._stats = stats

    @staticmethod
    def _qmax(bits: int) -> float:
        if bits < 2 or bits > 16:
            raise ValueError("quantization bits must be between 2 and 16")
        return float(2 ** (bits - 1) - 1)

    @staticmethod
    def _static_activation_scale(
        calibration: torch.Tensor,
        config: W8A8Config,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        qmax = W8A8Linear._qmax(config.activation_bits)
        eps = torch.finfo(torch.float32).eps
        fp = calibration.float().abs()
        percentile = float(config.activation_percentile)
        if not (0.0 < percentile <= 1.0):
            raise ValueError("--activation-percentile must be in (0, 1]")

        if config.activation_granularity == "static_per_tensor":
            if percentile < 1.0:
                amax = torch.quantile(fp.reshape(-1), percentile)
            else:
                amax = fp.amax()
            scale = amax.clamp_min(eps) / qmax
            scale_count = 1
        elif config.activation_granularity == "static_per_feature":
            if percentile < 1.0:
                amax = torch.quantile(fp, percentile, dim=0)
            else:
                amax = fp.amax(dim=0)
            scale = amax.clamp_min(eps) / qmax
            scale_count = int(scale.numel())
        else:
            raise ValueError(f"Unsupported static activation granularity: {config.activation_granularity}")

        stats = {
            "activation_static_scale_count": scale_count,
            "activation_calib_absmax": float(fp.amax().item()),
            "activation_scale_min": float(scale.amin().item()),
            "activation_scale_max": float(scale.amax().item()),
        }
        return scale.to(dtype=torch.float32), stats

    @staticmethod
    def _smoothquant_scale(
        weight: torch.Tensor,
        calibration: torch.Tensor,
        config: W8A8Config,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        alpha = float(config.smoothquant_alpha)
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("--smoothquant-alpha must be in [0, 1], or negative to disable")
        eps = torch.finfo(torch.float32).eps
        act_absmax = calibration.float().abs().amax(dim=0).cpu().clamp_min(eps)
        weight_absmax = weight.detach().float().abs().amax(dim=0).cpu().clamp_min(eps)
        scale = act_absmax.pow(alpha) / weight_absmax.pow(1.0 - alpha)
        scale = scale.clamp(config.smoothquant_min_scale, config.smoothquant_max_scale)
        stats = {
            "smoothquant_alpha": alpha,
            "smoothquant_scale_count": int(scale.numel()),
            "smoothquant_scale_min": float(scale.amin().item()),
            "smoothquant_scale_max": float(scale.amax().item()),
        }
        return scale.to(dtype=torch.float32), stats

    @classmethod
    @torch.no_grad()
    def from_linear(
        cls,
        linear: nn.Linear,
        config: W8A8Config,
        source_name: str,
        calibration: torch.Tensor | None = None,
    ) -> "W8A8Linear":
        rtn_config = RTNConfig(
            bits=config.weight_bits,
            granularity=config.weight_granularity,
            group_size=config.weight_group_size,
            target_regex=config.target_regex,
            include_lm_head=config.include_lm_head,
            max_linears=config.max_linears,
        )
        original_weight = linear.weight.detach()
        source_weight = original_weight.float().cpu()
        smooth_scale = None
        smooth_stats: dict[str, Any] = {
            "smoothquant_alpha": config.smoothquant_alpha,
            "smoothquant_scale_count": 0,
            "smoothquant_scale_min": 0.0,
            "smoothquant_scale_max": 0.0,
        }
        calibration_for_act = calibration
        if config.smoothquant_alpha >= 0.0:
            if calibration is None:
                raise ValueError("--smoothquant-alpha requires calibration activations")
            smooth_scale, smooth_stats = cls._smoothquant_scale(source_weight, calibration, config)
            source_weight = source_weight * smooth_scale.view(1, -1)
            calibration_for_act = calibration.float() / smooth_scale.view(1, -1)

        deq_weight, weight_err_stats = RTNLinear._quant_dequant_symmetric(source_weight, rtn_config)
        deq_weight = deq_weight.to(device=original_weight.device, dtype=original_weight.dtype)

        act_scale = None
        act_stats: dict[str, Any] = {
            "activation_static_scale_count": 0,
            "activation_calib_absmax": 0.0,
            "activation_scale_min": 0.0,
            "activation_scale_max": 0.0,
        }
        if config.activation_granularity.startswith("static_"):
            if calibration_for_act is None:
                raise ValueError(f"{config.activation_granularity} requires calibration activations")
            act_scale, act_stats = cls._static_activation_scale(calibration_for_act, config)

        stats = cls._hardware_stats_for(
            name=source_name,
            in_features=int(linear.in_features),
            out_features=int(linear.out_features),
            config=config,
            weight_err_stats=weight_err_stats,
            act_stats=act_stats,
            smooth_stats=smooth_stats,
        )
        return cls(deq_weight, linear.bias, act_scale, smooth_scale, source_name, config, stats)

    @staticmethod
    def _hardware_stats_for(
        name: str,
        in_features: int,
        out_features: int,
        config: W8A8Config,
        weight_err_stats: dict[str, Any],
        act_stats: dict[str, Any],
        smooth_stats: dict[str, Any],
    ) -> dict[str, Any]:
        weight_count = in_features * out_features
        weight_scale_count = int(weight_err_stats["scale_count"])
        dynamic_scale_per_token = 1 if config.activation_granularity.startswith("dynamic_") else 0
        activation_static_scale_count = int(act_stats["activation_static_scale_count"])
        activation_bits_per_token = in_features * config.activation_bits
        return {
            "name": name,
            "method": "w8a8_fake_quant",
            "weight_bits": config.weight_bits,
            "activation_bits": config.activation_bits,
            "weight_granularity": config.weight_granularity,
            "weight_group_size": config.weight_group_size if config.weight_granularity == "per_group" else 0,
            "activation_granularity": config.activation_granularity,
            "activation_percentile": config.activation_percentile,
            "in_features": in_features,
            "out_features": out_features,
            "weight_count": weight_count,
            "weight_bits_total": weight_count * config.weight_bits,
            "weight_mib_packed": weight_count * config.weight_bits / 8 / 2**20,
            "weight_scale_count": weight_scale_count,
            "weight_scale_mib_fp16": weight_scale_count * 2 / 2**20,
            "activation_values_per_token": in_features,
            "activation_bits_per_token": activation_bits_per_token,
            "activation_mib_per_1k_tokens_packed": activation_bits_per_token * 1000 / 8 / 2**20,
            "activation_dynamic_scale_count_per_token": dynamic_scale_per_token,
            "activation_static_scale_count": activation_static_scale_count,
            "activation_static_scale_mib_fp16": activation_static_scale_count * 2 / 2**20,
            "smoothquant_alpha": config.smoothquant_alpha,
            "smoothquant_scale_count": int(smooth_stats["smoothquant_scale_count"]),
            "smoothquant_scale_mib_fp16": int(smooth_stats["smoothquant_scale_count"]) * 2 / 2**20,
            "dense_mac_per_token": weight_count,
            "int8_mac_per_token": weight_count,
            "lookups_per_token": 0,
            "adds_per_token": 0,
            **weight_err_stats,
            **act_stats,
            **smooth_stats,
        }

    def hardware_stats(self) -> dict[str, Any]:
        return dict(self._stats)

    def _quant_dequant_activation(self, x: torch.Tensor) -> torch.Tensor:
        qmax = self._qmax(self.config.activation_bits)
        eps = torch.finfo(torch.float32).eps
        fp = x.float()
        if self.smooth_scale is not None:
            fp = fp / self.smooth_scale.to(device=x.device, dtype=torch.float32)
        if self.config.activation_granularity == "dynamic_per_token":
            scale = fp.abs().amax(dim=-1, keepdim=True).clamp_min(eps) / qmax
        elif self.config.activation_granularity == "dynamic_per_tensor":
            scale = fp.abs().amax().clamp_min(eps) / qmax
        elif self.config.activation_granularity in {"static_per_tensor", "static_per_feature"}:
            if self.act_scale is None:
                raise RuntimeError("Static activation scale is missing")
            scale = self.act_scale.to(device=x.device, dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported activation granularity: {self.config.activation_granularity}")
        q = torch.round(fp / scale).clamp(-qmax, qmax)
        return (q * scale).to(dtype=x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = self._quant_dequant_activation(x)
        return torch.nn.functional.linear(x_q, self.weight.to(dtype=x.dtype), self.bias)


def _aggregate_w8a8_stats(module_stats: list[dict[str, Any]]) -> dict[str, Any]:
    weight_count = sum(int(s["weight_count"]) for s in module_stats)
    weight_bits = sum(float(s["weight_bits_total"]) for s in module_stats)
    weight_scale_count = sum(int(s["weight_scale_count"]) for s in module_stats)
    act_values_per_token = sum(int(s["activation_values_per_token"]) for s in module_stats)
    act_bits_per_token = sum(float(s["activation_bits_per_token"]) for s in module_stats)
    dynamic_act_scales_per_token = sum(int(s["activation_dynamic_scale_count_per_token"]) for s in module_stats)
    static_act_scale_count = sum(int(s["activation_static_scale_count"]) for s in module_stats)
    smooth_scale_count = sum(int(s["smoothquant_scale_count"]) for s in module_stats)
    dense_mac = sum(float(s["dense_mac_per_token"]) for s in module_stats)
    weighted_mse = sum(float(s["weight_mse"]) * int(s["weight_count"]) for s in module_stats)
    max_err = max((float(s["weight_max_abs_error"]) for s in module_stats), default=0.0)
    return {
        "quantized_linears": len(module_stats),
        "weight_count": weight_count,
        "weight_bits_total": weight_bits,
        "weight_mib_packed": weight_bits / 8 / 2**20,
        "weight_scale_count": weight_scale_count,
        "weight_scale_mib_fp16": weight_scale_count * 2 / 2**20,
        "activation_values_per_token": act_values_per_token,
        "activation_bits_per_token": act_bits_per_token,
        "activation_mib_per_1k_tokens_packed": act_bits_per_token * 1000 / 8 / 2**20,
        "activation_dynamic_scale_count_per_token": dynamic_act_scales_per_token,
        "activation_static_scale_count": static_act_scale_count,
        "activation_static_scale_mib_fp16": static_act_scale_count * 2 / 2**20,
        "smoothquant_scale_count": smooth_scale_count,
        "smoothquant_scale_mib_fp16": smooth_scale_count * 2 / 2**20,
        "dense_mac_per_token": dense_mac,
        "int8_mac_per_token": dense_mac,
        "lookups_per_token": 0.0,
        "adds_per_token": 0.0,
        "weighted_weight_mse": weighted_mse / max(weight_count, 1),
        "weight_max_abs_error": max_err,
    }


@torch.no_grad()
def replace_with_w8a8_quant(
    model: nn.Module,
    config: W8A8Config,
    calibration_batches: Iterable[dict[str, torch.Tensor]] | None = None,
    max_vectors_per_layer: int = 1024,
    device: torch.device | None = None,
) -> dict[str, Any]:
    if device is None:
        device = next(model.parameters()).device
    target_linears = iter_target_linears(
        model,
        config.target_regex,
        include_lm_head=config.include_lm_head,
        max_linears=config.max_linears,
    )
    if not target_linears:
        raise RuntimeError("No target Linear modules matched")
    target_names = [name for name, _ in target_linears]

    calibration_inputs: dict[str, torch.Tensor] | None = None
    calibration_seconds = 0.0
    needs_calibration = config.activation_granularity.startswith("static_") or config.smoothquant_alpha >= 0.0
    if needs_calibration:
        if calibration_batches is None:
            raise ValueError("Static activation quantization or SmoothQuant requires calibration batches")
        calibration_inputs, calibration_seconds = collect_calibration_inputs(
            model,
            calibration_batches,
            target_names,
            max_vectors_per_layer=max_vectors_per_layer,
            device=device,
        )

    modules = dict(model.named_modules())
    stats = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for idx, (name, _linear) in enumerate(target_linears, start=1):
        current = modules[name]
        if not isinstance(current, nn.Linear):
            raise TypeError(f"{name} is no longer nn.Linear")
        calibration = calibration_inputs[name] if calibration_inputs is not None else None
        wrapped = W8A8Linear.from_linear(current, config, name, calibration=calibration)
        _set_submodule(model, name, wrapped)
        stats.append(wrapped.hardware_stats())
        modules = dict(model.named_modules())
        print(f"W8A8 quantized {idx}/{len(target_linears)}: {name}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    quantization_seconds = time.perf_counter() - start
    return {
        "modules": stats,
        "aggregate": _aggregate_w8a8_stats(stats),
        "calibration_seconds": calibration_seconds,
        "quantization_seconds": quantization_seconds,
    }
