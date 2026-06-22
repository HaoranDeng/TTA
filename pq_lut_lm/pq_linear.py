from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class PQConfig:
    method: str = "pq"
    subdim: int = 32
    ka: int = 8
    kw: int = 16
    kmeans_iters: int = 4
    sample_limit: int = 2048
    encode_chunk: int = 8192
    lut_dtype: str = "float16"
    distance: str = "l2"
    weight_group_size: int = 0
    lut_quant_bits: int = 0
    seed: int = 123

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _lut_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported LUT dtype: {name}")


@torch.no_grad()
def assign_to_centers(
    x: torch.Tensor,
    centers: torch.Tensor,
    chunk: int = 8192,
    distance: str = "l2",
) -> torch.Tensor:
    """Nearest-center assignment for x [N, D] and centers [K, D]."""
    x = x.float().contiguous()
    centers = centers.float().contiguous()
    if distance not in {"l2", "chebyshev"}:
        raise ValueError(f"Unsupported distance metric: {distance}")
    center_norm = (centers * centers).sum(dim=1).view(1, -1)
    codes = []
    for start in range(0, x.shape[0], chunk):
        xb = x[start : start + chunk]
        if distance == "chebyshev":
            dist = (xb[:, None, :] - centers[None, :, :]).abs().amax(dim=2)
        else:
            dist = (xb * xb).sum(dim=1, keepdim=True) + center_norm - 2.0 * xb @ centers.t()
        codes.append(dist.argmin(dim=1))
    return torch.cat(codes, dim=0)


@torch.no_grad()
def kmeans(
    x: torch.Tensor,
    k: int,
    iters: int,
    seed: int,
    sample_limit: int,
    chunk: int = 8192,
    distance: str = "l2",
) -> torch.Tensor:
    """Small deterministic k-means used for calibration."""
    x = x.float().contiguous()
    if sample_limit > 0 and x.shape[0] > sample_limit:
        gen = torch.Generator(device=x.device)
        gen.manual_seed(seed)
        idx = torch.randperm(x.shape[0], generator=gen, device=x.device)[:sample_limit]
        x = x[idx].contiguous()

    n, _ = x.shape
    if n < k:
        raise ValueError(f"k={k} exceeds calibration sample count n={n}")

    gen = torch.Generator(device=x.device)
    gen.manual_seed(seed)
    centers = x[torch.randperm(n, generator=gen, device=x.device)[:k]].clone()
    for _ in range(iters):
        codes = assign_to_centers(x, centers, chunk=chunk, distance=distance)
        new_centers = torch.zeros_like(centers)
        counts = torch.bincount(codes, minlength=k).to(x.dtype)
        new_centers.index_add_(0, codes, x)
        nonempty = counts > 0
        new_centers[nonempty] = new_centers[nonempty] / counts[nonempty, None]
        centers = torch.where(nonempty[:, None], new_centers, centers)
    return centers


@torch.no_grad()
def kmeans_padded(
    x: torch.Tensor,
    k: int,
    iters: int,
    seed: int,
    sample_limit: int,
    chunk: int = 8192,
    distance: str = "l2",
) -> torch.Tensor:
    """Run k-means and repeat centers when the sample count is smaller than k."""
    effective_k = min(k, x.shape[0])
    centers = kmeans(x, effective_k, iters, seed, sample_limit, chunk, distance)
    if effective_k == k:
        return centers
    pad_count = k - effective_k
    repeats = centers[torch.arange(pad_count, device=centers.device) % effective_k]
    return torch.cat([centers, repeats], dim=0)


@torch.no_grad()
def encode_activation(x: torch.Tensor, act_centers: torch.Tensor, distance: str = "l2") -> torch.Tensor:
    """Encode flattened activations [N, in_features] into PQ codes [N, M]."""
    n, in_features = x.shape
    m, ka, subdim = act_centers.shape
    if in_features != m * subdim:
        raise ValueError(f"Expected input dim {m * subdim}, got {in_features}")

    xv = x.view(n, m, subdim).float()
    centers = act_centers.float()
    if distance == "chebyshev":
        dist = (xv[:, :, None, :] - centers[None, :, :, :]).abs().amax(dim=3)
    elif distance == "l2":
        x_norm = (xv * xv).sum(dim=2, keepdim=True)
        c_norm = (centers * centers).sum(dim=2).unsqueeze(0)
        dot = torch.einsum("nms,mks->nmk", xv, centers)
        dist = x_norm + c_norm - 2.0 * dot
    else:
        raise ValueError(f"Unsupported distance metric: {distance}")
    return dist.argmin(dim=2)


def _num_weight_groups(out_features: int, weight_group_size: int) -> int:
    if weight_group_size <= 0:
        return 1
    return math.ceil(out_features / weight_group_size)


def _quantize_lut(lut: torch.Tensor, bits: int) -> tuple[torch.Tensor, float, float]:
    if bits <= 0:
        return lut, 1.0, 0.0
    qmax = float((1 << bits) - 1)
    min_val = float(lut.min().item())
    max_val = float(lut.max().item())
    if max_val <= min_val:
        return torch.zeros_like(lut), 1.0, 0.0
    scale = (max_val - min_val) / qmax
    zeropoint = -min_val / scale
    q = torch.round(lut / scale + zeropoint).clamp_(0, qmax)
    dequant = (q - zeropoint) * scale
    return dequant, scale, zeropoint


class PQLUTLinear(nn.Module):
    """Post-training PQ+LUT approximation for one Linear layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: torch.Tensor | None,
        act_centers: torch.Tensor,
        weight_centers: torch.Tensor,
        weight_codes: torch.Tensor,
        expanded_lut: torch.Tensor,
        lut_scales: torch.Tensor,
        lut_zeropoints: torch.Tensor,
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
        self.register_buffer("act_centers", act_centers, persistent=True)
        self.register_buffer("weight_centers", weight_centers, persistent=True)
        self.register_buffer("weight_codes", weight_codes, persistent=True)
        self.register_buffer("expanded_lut", expanded_lut, persistent=True)
        self.register_buffer("lut_scales", lut_scales, persistent=True)
        self.register_buffer("lut_zeropoints", lut_zeropoints, persistent=True)
        if bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", bias.detach().clone(), persistent=True)

    @classmethod
    @torch.no_grad()
    def from_linear(
        cls,
        linear: nn.Linear,
        calibration_inputs: torch.Tensor,
        config: PQConfig,
        source_name: str,
    ) -> "PQLUTLinear":
        if linear.in_features % config.subdim != 0:
            raise ValueError(
                f"{source_name}: in_features={linear.in_features} must be divisible by subdim={config.subdim}"
            )

        device = linear.weight.device
        weight = linear.weight.detach()
        calib = calibration_inputs.to(device=device, dtype=weight.dtype, non_blocking=True)
        out_features, in_features = weight.shape
        m = in_features // config.subdim
        group_count = _num_weight_groups(out_features, config.weight_group_size)
        lut_dtype = _lut_dtype(config.lut_dtype)

        act_centers = torch.empty((m, config.ka, config.subdim), device=device, dtype=torch.float32)
        weight_centers = torch.empty((m, group_count, config.kw, config.subdim), device=device, dtype=torch.float32)
        weight_codes = torch.empty((m, out_features), device=device, dtype=torch.long)
        expanded_lut = torch.empty((m, config.ka, out_features), device=device, dtype=lut_dtype)
        lut_scales = torch.empty((m, group_count), device=device, dtype=torch.float32)
        lut_zeropoints = torch.empty((m, group_count), device=device, dtype=torch.float32)

        _sync(device)
        start = time.perf_counter()
        for mi in range(m):
            lo = mi * config.subdim
            hi = lo + config.subdim
            ca = kmeans_padded(
                calib[:, lo:hi],
                config.ka,
                config.kmeans_iters,
                config.seed + 1009 * mi,
                config.sample_limit,
                config.encode_chunk,
                config.distance,
            )
            act_centers[mi] = ca

            for gi in range(group_count):
                g_lo = gi * config.weight_group_size if config.weight_group_size > 0 else 0
                g_hi = min((gi + 1) * config.weight_group_size, out_features) if config.weight_group_size > 0 else out_features
                ww = weight[g_lo:g_hi, lo:hi]
                cw = kmeans_padded(
                    ww,
                    config.kw,
                    config.kmeans_iters,
                    config.seed + 2003 * mi + 9176 * gi,
                    config.sample_limit,
                    config.encode_chunk,
                    config.distance,
                )
                wc = assign_to_centers(ww, cw, chunk=config.encode_chunk, distance=config.distance)
                lut = ca @ cw.t()
                lut, scale, zeropoint = _quantize_lut(lut, config.lut_quant_bits)
                weight_centers[mi, gi] = cw
                weight_codes[mi, g_lo:g_hi] = wc
                expanded_lut[mi, :, g_lo:g_hi] = lut[:, wc].to(lut_dtype)
                lut_scales[mi, gi] = scale
                lut_zeropoints[mi, gi] = zeropoint
        _sync(device)
        train_seconds = time.perf_counter() - start

        return cls(
            in_features=in_features,
            out_features=out_features,
            bias=linear.bias,
            act_centers=act_centers,
            weight_centers=weight_centers,
            weight_codes=weight_codes,
            expanded_lut=expanded_lut,
            lut_scales=lut_scales,
            lut_zeropoints=lut_zeropoints,
            config=config,
            source_name=source_name,
            train_seconds=train_seconds,
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape[:-1]
        flat = x.reshape(-1, self.in_features)
        codes = encode_activation(flat, self.act_centers, distance=self.config.distance)
        out = torch.zeros((flat.shape[0], self.out_features), device=flat.device, dtype=torch.float32)
        for mi in range(codes.shape[1]):
            out.add_(self.expanded_lut[mi, codes[:, mi], :].float())
        if self.bias is not None:
            out.add_(self.bias.float())
        return out.to(dtype=x.dtype).reshape(*original_shape, self.out_features)

    def hardware_stats(self) -> dict[str, Any]:
        m = self.in_features // self.config.subdim
        group_count = _num_weight_groups(self.out_features, self.config.weight_group_size)
        act_code_bits = math.ceil(math.log2(self.config.ka))
        weight_code_bits = math.ceil(math.log2(self.config.kw))
        lut_bits = self.config.lut_quant_bits if self.config.lut_quant_bits > 0 else self.expanded_lut.element_size() * 8
        base_lut_entries = m * group_count * self.config.ka * self.config.kw
        expanded_lut_entries = m * self.config.ka * self.out_features
        return {
            "name": self.source_name,
            "method": self.config.method,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "subdim": self.config.subdim,
            "M": m,
            "weight_groups": group_count,
            "weight_group_size": self.config.weight_group_size,
            "Ka": self.config.ka,
            "Kw": self.config.kw,
            "distance": self.config.distance,
            "lut_quant_bits": self.config.lut_quant_bits,
            "act_center_values": m * self.config.ka * self.config.subdim,
            "weight_center_values": m * group_count * self.config.kw * self.config.subdim,
            "base_lut_entries": base_lut_entries,
            "base_lut_bits": base_lut_entries * lut_bits,
            "expanded_lut_entries": expanded_lut_entries,
            "weight_code_count": m * self.out_features,
            "weight_code_bits": m * self.out_features * weight_code_bits,
            "act_code_bits_per_token": m * act_code_bits,
            "lookups_per_token": m * self.out_features,
            "adds_per_token": max(m - 1, 0) * self.out_features,
            "centroid_distance_vectors_per_token": m * self.config.ka,
            "centroid_distance_scalar_ops_per_token": m * self.config.ka * self.config.subdim,
            "dense_mac_per_token": self.in_features * self.out_features,
            "lut_dtype": self.config.lut_dtype,
            "train_seconds": self.train_seconds,
        }
