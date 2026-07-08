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
from .pq_linear import PQConfig, PQLUTLinear, encode_activation, kmeans_padded_batched


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
        self.reconstruction_loss_enabled = False
        self.last_reconstruction_loss: torch.Tensor | None = None

    def quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, self.in_features)
        centers = self.act_centers.float()
        m = centers.shape[0]
        ka = centers.shape[1]
        max_dist_elements = int(self.config.act_quant_max_dist_elements)
        if max_dist_elements > 0:
            row_chunk = max(1, min(flat.shape[0], max_dist_elements // max(m * ka, 1)))
        else:
            row_chunk = flat.shape[0]

        chunks = []
        center_index = torch.arange(m, device=flat.device)[None, :]
        for start in range(0, flat.shape[0], row_chunk):
            flat_chunk = flat[start : start + row_chunk]
            xv = flat_chunk.view(flat_chunk.shape[0], -1, self.config.subdim).float()
            if self.config.distance == "chebyshev":
                dist = (xv[:, :, None, :] - centers[None, :, :, :]).abs().amax(dim=3)
            elif self.config.distance == "l2":
                x_norm = (xv * xv).sum(dim=2, keepdim=True)
                c_norm = (centers * centers).sum(dim=2).unsqueeze(0)
                dot = torch.einsum("nms,mks->nmk", xv, centers)
                dist = x_norm + c_norm - 2.0 * dot
            else:
                raise ValueError(f"Unsupported distance metric: {self.config.distance}")
            codes = dist.argmin(dim=2)
            qv = centers[center_index, codes]
            if self.training and self.config.act_train_mode != "hard":
                temp = max(float(self.config.act_softmax_temperature), 1e-6)
                probs = torch.softmax(-dist / temp, dim=2)
                soft_qv = torch.einsum("nmk,mks->nms", probs, centers)
                if self.config.act_train_mode == "soft":
                    qv = soft_qv
                elif self.config.act_train_mode == "soft_hard":
                    qv = qv + (soft_qv - soft_qv.detach())
                else:
                    raise ValueError(f"Unsupported activation train mode: {self.config.act_train_mode}")
            center_scale = float(self.config.act_ste_center_scale)
            if center_scale != 1.0:
                qv = qv.detach() + center_scale * (qv - qv.detach())
            q = qv.reshape_as(flat_chunk)
            # Forward value is quantized; gradients flow to both centers and upstream activations.
            chunks.append(q + float(self.config.act_ste_input_scale) * (flat_chunk - flat_chunk.detach()))
        return torch.cat(chunks, dim=0).reshape(shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_reconstruction_loss = None
        quantized = self.quantize_activation(x).to(dtype=x.dtype)
        quantized_out = self.linear(quantized)
        if self.training and self.reconstruction_loss_enabled:
            with torch.no_grad():
                dense_out = self.linear(x)
            self.last_reconstruction_loss = torch.nn.functional.mse_loss(
                quantized_out.float(),
                dense_out.float(),
            )
        return quantized_out

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
            "act_train_mode": self.config.act_train_mode,
            "act_softmax_temperature": self.config.act_softmax_temperature,
            "act_ste_input_scale": self.config.act_ste_input_scale,
            "act_ste_center_scale": self.config.act_ste_center_scale,
            "act_quant_max_dist_elements": self.config.act_quant_max_dist_elements,
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


class ActivationLUTLinear(nn.Module):
    """Activation-VQ LUT linear layer with directly trained lookup-table values."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: torch.Tensor | None,
        act_centers: torch.Tensor,
        expanded_lut: torch.Tensor,
        config: PQConfig,
        source_name: str,
        train_seconds: float,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        self.source_name = source_name
        self.train_seconds = train_seconds
        self.register_buffer("act_centers", act_centers.detach().clone().float(), persistent=True)
        self.register_buffer("expanded_lut", expanded_lut.detach().clone(), persistent=True)
        if bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", bias.detach().clone(), persistent=True)

    @classmethod
    @torch.no_grad()
    def initialize_from_linear(
        cls,
        linear: nn.Linear,
        calibration_inputs: torch.Tensor,
        config: PQConfig,
        source_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if linear.in_features % config.subdim != 0:
            raise ValueError(f"{source_name}: in_features must be divisible by subdim={config.subdim}")
        device = linear.weight.device
        weight = linear.weight.detach()
        calib = calibration_inputs.to(device=device, dtype=weight.dtype, non_blocking=True)
        out_features, in_features = weight.shape
        m = in_features // config.subdim
        centers = kmeans_padded_batched(
            calib,
            config.ka,
            config.kmeans_iters,
            config.seed,
            config.sample_limit,
            config.encode_chunk,
            config.distance,
            config.subdim,
        )
        lut = torch.empty((m, config.ka, out_features), device=device, dtype=weight.dtype)
        weight_view = weight.view(out_features, m, config.subdim).permute(1, 2, 0).contiguous()
        lut.copy_(torch.bmm(centers.to(dtype=weight.dtype), weight_view).to(dtype=weight.dtype))
        return centers, lut

    @classmethod
    def fit_from_linear(
        cls,
        linear: nn.Linear,
        calibration_inputs: torch.Tensor,
        config: PQConfig,
        source_name: str,
        fit_steps: int,
        fit_lr: float,
        fit_batch_size: int,
        fit_lut_dtype: torch.dtype = torch.float32,
        fit_centers: bool = False,
        fit_center_lr: float | None = None,
        fit_temperature: float = 1.0,
    ) -> "ActivationLUTLinear":
        device = linear.weight.device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        centers, lut = cls.initialize_from_linear(linear, calibration_inputs, config, source_name)
        calib = calibration_inputs.to(device=device, dtype=linear.weight.dtype, non_blocking=True)
        with torch.no_grad():
            target = linear(calib).float()
            codes = encode_activation(calib, centers, distance=config.distance)

        if fit_steps > 0:
            lut_param = nn.Parameter(lut.to(dtype=fit_lut_dtype))
            params: list[dict[str, Any]] = [{"params": [lut_param], "lr": fit_lr}]
            centers_param: nn.Parameter | None = None
            if fit_centers:
                centers_param = nn.Parameter(centers.detach().clone().float())
                params.append({"params": [centers_param], "lr": fit_center_lr or fit_lr})
            opt = torch.optim.AdamW(params, weight_decay=0.0)
            gen = torch.Generator(device=device)
            gen.manual_seed(config.seed)
            n = calib.shape[0]
            for _ in range(fit_steps):
                if fit_batch_size > 0 and fit_batch_size < n:
                    idx = torch.randint(n, (fit_batch_size,), generator=gen, device=device)
                    batch_codes = codes[idx]
                    batch_target = target[idx]
                else:
                    batch_codes = codes
                    batch_target = target
                    batch_calib = calib
                if centers_param is None:
                    pred = _activation_lut_forward(batch_codes, lut_param)
                else:
                    batch_calib = calib[idx] if fit_batch_size > 0 and fit_batch_size < n else calib
                    pred = _activation_lut_forward_soft_hard(
                        batch_calib.float(),
                        centers_param,
                        lut_param,
                        distance=config.distance,
                        subdim=config.subdim,
                        temperature=fit_temperature,
                    )
                if linear.bias is not None:
                    pred = pred + linear.bias.float()
                loss = torch.nn.functional.mse_loss(pred.float(), batch_target)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            if centers_param is not None:
                centers = centers_param.detach()
            lut = lut_param.detach().to(dtype=linear.weight.dtype)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - start
        return cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias,
            act_centers=centers,
            expanded_lut=lut,
            config=config,
            source_name=source_name,
            train_seconds=train_seconds,
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape[:-1]
        flat = x.reshape(-1, self.in_features)
        codes = encode_activation(flat, self.act_centers, distance=self.config.distance)
        out = _activation_lut_forward(codes, self.expanded_lut)
        if self.bias is not None:
            out = out + self.bias.float()
        return out.to(dtype=x.dtype).reshape(*shape, self.out_features)

    def hardware_stats(self) -> dict[str, Any]:
        m = self.in_features // self.config.subdim
        act_code_bits = self.config.ka.bit_length() - 1
        entries = m * self.config.ka * self.out_features
        return {
            "name": self.source_name,
            "method": "act_lut_fit",
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
            "lut_storage": "expanded",
            "output_correction": "none",
            "act_center_values": m * self.config.ka * self.config.subdim,
            "weight_center_values": 0,
            "base_lut_entries": entries,
            "base_lut_bits": entries * 16,
            "expanded_lut_entries": entries,
            "weight_code_count": 0,
            "weight_code_bits": 0,
            "act_code_bits_per_token": m * act_code_bits,
            "lookups_per_token": m * self.out_features,
            "adds_per_token": max(m - 1, 0) * self.out_features,
            "centroid_distance_vectors_per_token": m * self.config.ka,
            "centroid_distance_scalar_ops_per_token": m * self.config.ka * self.config.subdim,
            "dense_mac_per_token": self.in_features * self.out_features,
            "lut_dtype": "float16",
            "train_seconds": self.train_seconds,
        }


def _activation_lut_forward(codes: torch.Tensor, lut: torch.Tensor, chunk_m: int = 32) -> torch.Tensor:
    out = torch.zeros((codes.shape[0], lut.shape[2]), device=codes.device, dtype=torch.float32)
    n, m = codes.shape
    ka = lut.shape[1]
    for start in range(0, m, chunk_m):
        end = min(start + chunk_m, m)
        offsets = torch.arange(end - start, device=codes.device, dtype=codes.dtype) * ka
        idx = (codes[:, start:end].t() + offsets[:, None]).reshape(-1)
        vals = lut[start:end].reshape(-1, lut.shape[2]).index_select(0, idx)
        out = out + vals.reshape(end - start, n, lut.shape[2]).float().sum(dim=0)
    return out


def _activation_lut_forward_soft_hard(
    x: torch.Tensor,
    centers: torch.Tensor,
    lut: torch.Tensor,
    distance: str,
    subdim: int,
    temperature: float = 1.0,
    chunk_m: int = 32,
) -> torch.Tensor:
    xv = x.reshape(x.shape[0], -1, subdim).float()
    centers = centers.float()
    lut = lut.float()
    out = torch.zeros((xv.shape[0], lut.shape[2]), device=x.device, dtype=torch.float32)
    temp = max(float(temperature), 1e-6)
    for start in range(0, centers.shape[0], chunk_m):
        end = min(start + chunk_m, centers.shape[0])
        xb = xv[:, start:end]
        cb = centers[start:end]
        if distance == "chebyshev":
            dist = (xb[:, :, None, :] - cb[None, :, :, :]).abs().amax(dim=3)
        elif distance == "l2":
            dist = ((xb[:, :, None, :] - cb[None, :, :, :]) ** 2).sum(dim=3)
        else:
            raise ValueError(f"Unsupported distance metric: {distance}")
        probs = torch.softmax(-dist / temp, dim=2)
        soft = torch.einsum("nmk,mko->no", probs, lut[start:end])
        codes = dist.argmin(dim=2)
        hard = _activation_lut_forward(codes, lut[start:end], chunk_m=chunk_m)
        out = out + hard + (soft - soft.detach())
    return out


@torch.no_grad()
def reconstruct_linear_from_activation_lut(
    module: ActivationLUTLinear,
    dtype: torch.dtype | None = None,
) -> nn.Linear:
    """Recover dense weights from a trained activation LUT by least squares."""

    device = module.expanded_lut.device
    dtype = dtype or module.expanded_lut.dtype
    linear = nn.Linear(
        module.in_features,
        module.out_features,
        bias=module.bias is not None,
        device=device,
        dtype=dtype,
    )
    centers = module.act_centers.float()
    table = module.expanded_lut.float()
    sub_weight = torch.bmm(torch.linalg.pinv(centers), table)
    weight = sub_weight.permute(2, 0, 1).reshape(module.out_features, module.in_features).to(dtype=dtype)
    linear.weight.copy_(weight)
    if module.bias is not None:
        linear.bias.copy_(module.bias.to(dtype=dtype))
    for param in linear.parameters():
        param.requires_grad_(False)
    return linear


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
    for idx, name in enumerate(target_names, start=1):
        old = modules[name]
        if not isinstance(old, nn.Linear):
            raise TypeError(f"{name} is no longer nn.Linear")
        calib = calibration_inputs[name].to(device=device, dtype=old.weight.dtype)
        centers = kmeans_padded_batched(
            calib,
            config.ka,
            config.kmeans_iters,
            config.seed + 1009 * idx,
            config.sample_limit,
            config.encode_chunk,
            config.distance,
            config.subdim,
        )
        wrapped = STEActivationQuantLinear(old, centers, config, name)
        _set_submodule(model, name, wrapped)
        module_stats.append(wrapped.hardware_stats())
        if idx == 1 or idx == len(target_names) or idx % 10 == 0:
            print(f"initialized STE activation quantizer {idx}/{len(target_names)}: {name}", flush=True)
        modules = dict(model.named_modules())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return ActQuantReport(
        module_stats=module_stats,
        aggregate=_aggregate_stats(module_stats),
        calibration_seconds=calibration_seconds,
        initialization_seconds=time.perf_counter() - start,
    )


def replace_with_fitted_activation_lut(
    model: nn.Module,
    calibration_batches: Iterable[dict[str, torch.Tensor]],
    config: PQConfig,
    target_regex: str = DEFAULT_TARGET_REGEX,
    include_lm_head: bool = False,
    max_linears: int | None = None,
    max_vectors_per_layer: int = 1024,
    fit_steps: int = 0,
    fit_lr: float = 1e-2,
    fit_batch_size: int = 0,
    fit_lut_dtype: torch.dtype = torch.float32,
    fit_centers: bool = False,
    fit_center_lr: float | None = None,
    fit_temperature: float = 1.0,
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
    modules = dict(model.named_modules())
    module_stats: list[dict[str, Any]] = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for idx, name in enumerate(target_names, start=1):
        old = modules[name]
        if not isinstance(old, nn.Linear):
            raise TypeError(f"{name} is no longer nn.Linear")
        fitted = ActivationLUTLinear.fit_from_linear(
            old,
            calibration_inputs[name],
            config,
            source_name=name,
            fit_steps=fit_steps,
            fit_lr=fit_lr,
            fit_batch_size=fit_batch_size,
            fit_lut_dtype=fit_lut_dtype,
            fit_centers=fit_centers,
            fit_center_lr=fit_center_lr,
            fit_temperature=fit_temperature,
        )
        _set_submodule(model, name, fitted)
        module_stats.append(fitted.hardware_stats())
        if idx == 1 or idx == len(target_names) or idx % 10 == 0:
            print(f"fitted activation LUT {idx}/{len(target_names)}: {name}", flush=True)
        modules = dict(model.named_modules())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return QuantizationReport(
        module_stats=module_stats,
        aggregate=_aggregate_stats(module_stats),
        calibration_seconds=calibration_seconds,
        quantization_seconds=time.perf_counter() - start,
        calibration_inputs=calibration_inputs,
    )


def trainable_act_center_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [m.act_centers for m in model.modules() if isinstance(m, STEActivationQuantLinear)]


def set_ste_reconstruction_loss_enabled(model: nn.Module, enabled: bool) -> None:
    for module in model.modules():
        if isinstance(module, STEActivationQuantLinear):
            module.reconstruction_loss_enabled = enabled
            module.last_reconstruction_loss = None


def ste_reconstruction_loss(model: nn.Module) -> torch.Tensor | None:
    losses = [
        module.last_reconstruction_loss
        for module in model.modules()
        if isinstance(module, STEActivationQuantLinear) and module.last_reconstruction_loss is not None
    ]
    if not losses:
        return None
    return torch.stack(losses).mean()


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
    for idx, (name, wrapped) in enumerate(target, start=1):
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
        if idx == 1 or idx == len(target) or idx % 10 == 0:
            print(f"converted STE activation quantizer {idx}/{len(target)} to LUT: {name}", flush=True)
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


@torch.no_grad()
def convert_activation_lut_to_pq_lut(
    model: nn.Module,
    calibration_batches: Iterable[dict[str, torch.Tensor]],
    config: PQConfig,
    max_linears: int | None = None,
    max_vectors_per_layer: int = 1024,
    device: torch.device | None = None,
    calibration_inputs_override: dict[str, torch.Tensor] | None = None,
) -> QuantizationReport:
    if device is None:
        device = next(model.parameters()).device
    target_names = []
    for name, module in model.named_modules():
        if isinstance(module, ActivationLUTLinear):
            target_names.append(name)
            if max_linears is not None and len(target_names) >= max_linears:
                break
    if not target_names:
        raise RuntimeError("No ActivationLUTLinear modules found")

    if calibration_inputs_override is None:
        calibration_inputs, calibration_seconds = collect_calibration_inputs(
            model,
            calibration_batches,
            target_names,
            max_vectors_per_layer=max_vectors_per_layer,
            device=device,
        )
    else:
        calibration_inputs = {name: calibration_inputs_override[name] for name in target_names}
        calibration_seconds = 0.0
    module_stats: list[dict[str, Any]] = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for idx, name in enumerate(target_names, start=1):
        modules = dict(model.named_modules())
        current = modules[name]
        if not isinstance(current, ActivationLUTLinear):
            continue
        reconstructed = reconstruct_linear_from_activation_lut(current, dtype=current.expanded_lut.dtype)
        pq = PQLUTLinear.from_linear(
            reconstructed,
            calibration_inputs[name],
            config,
            current.source_name,
            act_centers_override=current.act_centers.detach(),
        )
        _set_submodule(model, name, pq)
        module_stats.append(pq.hardware_stats())
        del modules, current, reconstructed
        if idx == 1 or idx == len(target_names) or idx % 10 == 0:
            print(f"converted fitted activation LUT {idx}/{len(target_names)} to final LUT: {name}", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    quantization_seconds = time.perf_counter() - start
    return QuantizationReport(
        module_stats=module_stats,
        aggregate=_aggregate_stats(module_stats),
        calibration_seconds=calibration_seconds,
        quantization_seconds=quantization_seconds,
    )
