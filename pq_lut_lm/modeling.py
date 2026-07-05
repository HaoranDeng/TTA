from __future__ import annotations

import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch import nn

from .pq_linear import PQConfig, PQLUTLinear


DEFAULT_TARGET_REGEX = r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"


@dataclass
class QuantizationReport:
    module_stats: list[dict[str, Any]]
    aggregate: dict[str, Any]
    calibration_seconds: float
    quantization_seconds: float
    calibration_inputs: dict[str, torch.Tensor] | None = None


class _ActivationCollector:
    def __init__(self, max_vectors: int) -> None:
        self.max_vectors = max_vectors
        self.parts: list[torch.Tensor] = []
        self.count = 0

    def add(self, x: torch.Tensor) -> None:
        if self.count >= self.max_vectors:
            return
        flat = x.detach().reshape(-1, x.shape[-1])
        take = min(flat.shape[0], self.max_vectors - self.count)
        if take <= 0:
            return
        self.parts.append(flat[:take].to(device="cpu", dtype=torch.float16))
        self.count += take

    def tensor(self) -> torch.Tensor:
        if not self.parts:
            raise RuntimeError("No calibration activations captured")
        return torch.cat(self.parts, dim=0)


def iter_target_linears(
    model: nn.Module,
    target_regex: str = DEFAULT_TARGET_REGEX,
    include_lm_head: bool = False,
    max_linears: int | None = None,
) -> list[tuple[str, nn.Linear]]:
    pattern = re.compile(target_regex)
    out: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name == "lm_head" and not include_lm_head:
            continue
        if pattern.search(name) or (include_lm_head and name == "lm_head"):
            out.append((name, module))
            if max_linears is not None and len(out) >= max_linears:
                break
    return out


def _set_submodule(root: nn.Module, name: str, module: nn.Module) -> None:
    if "." not in name:
        setattr(root, name, module)
        return
    parent_name, child_name = name.rsplit(".", 1)
    parent = root.get_submodule(parent_name)
    setattr(parent, child_name, module)


def _aggregate_stats(module_stats: Iterable[dict[str, Any]]) -> dict[str, Any]:
    stats = list(module_stats)
    numeric_keys = [
        "act_center_values",
        "weight_center_values",
        "base_lut_entries",
        "base_lut_bits",
        "expanded_lut_entries",
        "weight_code_count",
        "weight_code_bits",
        "act_code_bits_per_token",
        "lookups_per_token",
        "adds_per_token",
        "centroid_distance_vectors_per_token",
        "centroid_distance_scalar_ops_per_token",
        "dense_mac_per_token",
        "train_seconds",
    ]
    agg: dict[str, Any] = {"quantized_linears": len(stats)}
    for key in numeric_keys:
        agg[key] = sum(float(s[key]) for s in stats)
    agg["base_lut_mib_fp16"] = agg["base_lut_entries"] * 2 / 2**20
    agg["base_lut_mib_fp32"] = agg["base_lut_entries"] * 4 / 2**20
    agg["base_lut_mib_quantized"] = agg["base_lut_bits"] / 8 / 2**20
    agg["expanded_lut_mib_fp16"] = agg["expanded_lut_entries"] * 2 / 2**20
    agg["weight_codes_mib_packed"] = agg["weight_code_bits"] / 8 / 2**20
    return agg


@torch.no_grad()
def collect_calibration_inputs(
    model: nn.Module,
    batches: Iterable[dict[str, torch.Tensor]],
    target_names: list[str],
    max_vectors_per_layer: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], float]:
    modules = dict(model.named_modules())
    collectors: OrderedDict[str, _ActivationCollector] = OrderedDict(
        (name, _ActivationCollector(max_vectors_per_layer)) for name in target_names
    )
    handles = []
    for name in target_names:
        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...], layer_name: str = name) -> None:
            collectors[layer_name].add(inputs[0])

        handles.append(modules[name].register_forward_pre_hook(hook))

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    model.eval()
    for batch in batches:
        batch = {k: v.to(device) for k, v in batch.items()}
        model(**batch)
        if all(collector.count >= collector.max_vectors for collector in collectors.values()):
            break
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    for handle in handles:
        handle.remove()
    return {name: collector.tensor() for name, collector in collectors.items()}, elapsed


@torch.no_grad()
def quantize_model_linears(
    model: nn.Module,
    calibration_batches: Iterable[dict[str, torch.Tensor]],
    pq_config: PQConfig,
    target_regex: str = DEFAULT_TARGET_REGEX,
    include_lm_head: bool = False,
    max_linears: int | None = None,
    max_vectors_per_layer: int = 1024,
    device: torch.device | None = None,
) -> QuantizationReport:
    if device is None:
        device = next(model.parameters()).device
    target_linears = iter_target_linears(model, target_regex, include_lm_head, max_linears)
    target_names = [name for name, _ in target_linears]
    if not target_names:
        raise RuntimeError("No target Linear modules matched")

    calibration_inputs, calibration_seconds = collect_calibration_inputs(
        model,
        calibration_batches,
        target_names,
        max_vectors_per_layer=max_vectors_per_layer,
        device=device,
    )

    module_stats: list[dict[str, Any]] = []
    modules = dict(model.named_modules())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for name in target_names:
        old = modules[name]
        if not isinstance(old, nn.Linear):
            raise TypeError(f"{name} is no longer nn.Linear")
        pq = PQLUTLinear.from_linear(old, calibration_inputs[name], pq_config, name)
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
