from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch import nn

from .modeling import (
    DEFAULT_TARGET_REGEX,
    QuantizationReport,
    _aggregate_stats,
    _set_submodule,
    collect_calibration_inputs,
    iter_target_linears,
)
from .pq_linear import PQConfig, PQLUTLinear, encode_activation, kmeans_padded


class STEActivationQuantLinear(nn.Module):
    """Dense Linear with LUT-LLM-style activation vector quantization and STE."""

    def __init__(
        self,
        linear: nn.Linear,
        act_centers: torch.Tensor,
        config: PQConfig,
        source_name: str,
    ) -> None:
        super().__init__()
        if linear.in_features % config.subdim != 0:
            raise ValueError(f"{source_name}: in_features must be divisible by subdim={config.subdim}")
        self.linear = linear
        self.config = config
        self.source_name = source_name
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.act_centers = nn.Parameter(act_centers.detach().clone().float())

    def quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, self.in_features)
        codes = encode_activation(flat, self.act_centers, distance=self.config.distance)
        xv = flat.view(flat.shape[0], -1, self.config.subdim)
        qv = self.act_centers[torch.arange(self.act_centers.shape[0], device=flat.device)[None, :], codes]
        q = qv.reshape_as(flat)
        # Forward value is quantized; gradients flow to both centers and upstream activations.
        ste = q + (flat - flat.detach())
        return ste.reshape(shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.quantize_activation(x).to(dtype=x.dtype))

    def hardware_stats(self) -> dict[str, Any]:
        m = self.in_features // self.config.subdim
        return {
            "name": self.source_name,
            "method": "act_ste",
            "in_features": self.in_features,
            "out_features": self.out_features,
            "subdim": self.config.subdim,
            "M": m,
            "weight_groups": 0,
            "weight_group_size": 0,
            "Ka": self.config.ka,
            "Kw": 0,
            "distance": self.config.distance,
            "lut_quant_bits": 0,
            "lut_storage": "dense-weight",
            "output_correction": "none",
            "act_center_values": m * self.config.ka * self.config.subdim,
            "weight_center_values": 0,
            "base_lut_entries": m * self.config.ka * self.out_features,
            "base_lut_bits": m * self.config.ka * self.out_features * 16,
            "expanded_lut_entries": m * self.config.ka * self.out_features,
            "weight_code_count": 0,
            "weight_code_bits": 0,
            "act_code_bits_per_token": m * (self.config.ka.bit_length() - 1),
            "lookups_per_token": m * self.out_features,
            "adds_per_token": max(m - 1, 0) * self.out_features,
            "centroid_distance_vectors_per_token": m * self.config.ka,
            "centroid_distance_scalar_ops_per_token": m * self.config.ka * self.config.subdim,
            "dense_mac_per_token": self.in_features * self.out_features,
            "lut_dtype": "float16",
            "train_seconds": 0.0,
        }


@dataclass
class ActQuantReport:
    module_stats: list[dict[str, Any]]
    aggregate: dict[str, Any]
    calibration_seconds: float
    initialization_seconds: float


@torch.no_grad()
def replace_with_ste_act_quant(
    model: nn.Module,
    calibration_batches: Iterable[dict[str, torch.Tensor]],
    config: PQConfig,
    target_regex: str = DEFAULT_TARGET_REGEX,
    include_lm_head: bool = False,
    max_linears: int | None = None,
    max_vectors_per_layer: int = 1024,
    device: torch.device | None = None,
) -> ActQuantReport:
    if device is None:
        device = next(model.parameters()).device
    target_linears = iter_target_linears(model, target_regex, include_lm_head, max_linears)
    target_names = [name for name, _ in target_linears]
    calibration_inputs, calibration_seconds = collect_calibration_inputs(
        model,
        calibration_batches,
        target_names,
        max_vectors_per_layer=max_vectors_per_layer,
        device=device,
    )
    modules = dict(model.named_modules())
    module_stats: list[dict[str, Any]] = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for name in target_names:
        old = modules[name]
        if not isinstance(old, nn.Linear):
            raise TypeError(f"{name} is no longer nn.Linear")
        m = old.in_features // config.subdim
        centers = torch.empty((m, config.ka, config.subdim), device=device, dtype=torch.float32)
        calib = calibration_inputs[name].to(device=device, dtype=old.weight.dtype)
        for mi in range(m):
            lo = mi * config.subdim
            hi = lo + config.subdim
            centers[mi] = kmeans_padded(
                calib[:, lo:hi],
                config.ka,
                config.kmeans_iters,
                config.seed + 1009 * mi,
                config.sample_limit,
                config.encode_chunk,
                config.distance,
            )
        wrapped = STEActivationQuantLinear(old, centers, config, name)
        _set_submodule(model, name, wrapped)
        module_stats.append(wrapped.hardware_stats())
        modules = dict(model.named_modules())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return ActQuantReport(
        module_stats=module_stats,
        aggregate=_aggregate_stats(module_stats),
        calibration_seconds=calibration_seconds,
        initialization_seconds=time.perf_counter() - start,
    )


def trainable_act_center_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [m.act_centers for m in model.modules() if isinstance(m, STEActivationQuantLinear)]


@torch.no_grad()
def convert_ste_act_quant_to_lut(
    model: nn.Module,
    calibration_batches: Iterable[dict[str, torch.Tensor]],
    config: PQConfig,
    target_regex: str = DEFAULT_TARGET_REGEX,
    include_lm_head: bool = False,
    max_linears: int | None = None,
    max_vectors_per_layer: int = 1024,
    device: torch.device | None = None,
) -> QuantizationReport:
    if device is None:
        device = next(model.parameters()).device
    target = []
    for name, module in model.named_modules():
        if isinstance(module, STEActivationQuantLinear):
            target.append((name, module))
            if max_linears is not None and len(target) >= max_linears:
                break
    if not target:
        raise RuntimeError("No STEActivationQuantLinear modules found")

    calibration_inputs, calibration_seconds = collect_calibration_inputs(
        model,
        calibration_batches,
        [name for name, _ in target],
        max_vectors_per_layer=max_vectors_per_layer,
        device=device,
    )
    module_stats: list[dict[str, Any]] = []
    modules = dict(model.named_modules())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for name, wrapped in target:
        current = modules[name]
        if not isinstance(current, STEActivationQuantLinear):
            continue
        pq = PQLUTLinear.from_linear(
            current.linear,
            calibration_inputs[name],
            config,
            current.source_name,
            act_centers_override=current.act_centers.detach(),
        )
        _set_submodule(model, name, pq)
        module_stats.append(pq.hardware_stats())
        modules = dict(model.named_modules())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    quantization_seconds = time.perf_counter() - start
    return QuantizationReport(
        module_stats=module_stats,
        aggregate=_aggregate_stats(module_stats),
        calibration_seconds=calibration_seconds,
        quantization_seconds=quantization_seconds,
    )
