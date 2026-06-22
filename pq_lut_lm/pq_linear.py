from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class PQConfig:
    subdim: int = 32
    ka: int = 8
    kw: int = 16
    kmeans_iters: int = 4
    sample_limit: int = 2048
    encode_chunk: int = 8192
    lut_dtype: str = "float16"
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
def assign_to_centers(x: torch.Tensor, centers: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
    """Nearest-center assignment for x [N, D] and centers [K, D]."""
    x = x.float().contiguous()
    centers = centers.float().contiguous()
    center_norm = (centers * centers).sum(dim=1).view(1, -1)
    codes = []
    for start in range(0, x.shape[0], chunk):
        xb = x[start : start + chunk]
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
        codes = assign_to_centers(x, centers, chunk=chunk)
        new_centers = torch.zeros_like(centers)
        counts = torch.bincount(codes, minlength=k).to(x.dtype)
        new_centers.index_add_(0, codes, x)
        nonempty = counts > 0
        new_centers[nonempty] = new_centers[nonempty] / counts[nonempty, None]
        centers = torch.where(nonempty[:, None], new_centers, centers)
    return centers


@torch.no_grad()
def encode_activation(x: torch.Tensor, act_centers: torch.Tensor) -> torch.Tensor:
    """Encode flattened activations [N, in_features] into PQ codes [N, M]."""
    n, in_features = x.shape
    m, ka, subdim = act_centers.shape
    if in_features != m * subdim:
        raise ValueError(f"Expected input dim {m * subdim}, got {in_features}")

    xv = x.view(n, m, subdim).float()
    centers = act_centers.float()
    x_norm = (xv * xv).sum(dim=2, keepdim=True)
    c_norm = (centers * centers).sum(dim=2).unsqueeze(0)
    dot = torch.einsum("nms,mks->nmk", xv, centers)
    dist = x_norm + c_norm - 2.0 * dot
    return dist.argmin(dim=2)


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
        lut_dtype = _lut_dtype(config.lut_dtype)

        act_centers = torch.empty((m, config.ka, config.subdim), device=device, dtype=torch.float32)
        weight_centers = torch.empty((m, config.kw, config.subdim), device=device, dtype=torch.float32)
        weight_codes = torch.empty((m, out_features), device=device, dtype=torch.long)
        compact_lut = torch.empty((m, config.ka, config.kw), device=device, dtype=torch.float32)

        _sync(device)
        start = time.perf_counter()
        for mi in range(m):
            lo = mi * config.subdim
            hi = lo + config.subdim
            ca = kmeans(
                calib[:, lo:hi],
                config.ka,
                config.kmeans_iters,
                config.seed + 1009 * mi,
                config.sample_limit,
                config.encode_chunk,
            )
            cw = kmeans(
                weight[:, lo:hi],
                config.kw,
                config.kmeans_iters,
                config.seed + 2003 * mi,
                config.sample_limit,
                config.encode_chunk,
            )
            wc = assign_to_centers(weight[:, lo:hi], cw, chunk=config.encode_chunk)
            act_centers[mi] = ca
            weight_centers[mi] = cw
            weight_codes[mi] = wc
            compact_lut[mi] = ca @ cw.t()

        expanded_lut = torch.empty((m, config.ka, out_features), device=device, dtype=lut_dtype)
        for mi in range(m):
            expanded_lut[mi] = compact_lut[mi][:, weight_codes[mi]].to(lut_dtype)
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
            config=config,
            source_name=source_name,
            train_seconds=train_seconds,
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape[:-1]
        flat = x.reshape(-1, self.in_features)
        codes = encode_activation(flat, self.act_centers)
        out = torch.zeros((flat.shape[0], self.out_features), device=flat.device, dtype=torch.float32)
        for mi in range(codes.shape[1]):
            out.add_(self.expanded_lut[mi, codes[:, mi], :].float())
        if self.bias is not None:
            out.add_(self.bias.float())
        return out.to(dtype=x.dtype).reshape(*original_shape, self.out_features)

    def hardware_stats(self) -> dict[str, Any]:
        m = self.in_features // self.config.subdim
        act_code_bits = math.ceil(math.log2(self.config.ka))
        weight_code_bits = math.ceil(math.log2(self.config.kw))
        return {
            "name": self.source_name,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "subdim": self.config.subdim,
            "M": m,
            "Ka": self.config.ka,
            "Kw": self.config.kw,
            "act_center_values": m * self.config.ka * self.config.subdim,
            "weight_center_values": m * self.config.kw * self.config.subdim,
            "base_lut_entries": m * self.config.ka * self.config.kw,
            "expanded_lut_entries": m * self.config.ka * self.out_features,
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
