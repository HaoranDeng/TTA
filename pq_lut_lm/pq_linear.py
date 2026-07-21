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
    lut_storage: str = "expanded"
    distance: str = "l2"
    weight_group_size: int = 0
    lut_quant_bits: int = 0
    weight_code_reassign_iters: int = 0
    weight_center_refine_iters: int = 0
    weight_center_refine_reg: float = 1e-4
    weight_center_refine_blend: float = 1.0
    act_train_mode: str = "hard"
    act_softmax_temperature: float = 1.0
    act_ste_input_scale: float = 1.0
    act_ste_center_scale: float = 1.0
    act_quant_max_dist_elements: int = 0
    reconstruction_target: str = "current"
    act_smooth_alpha: float = -1.0
    act_smooth_min_scale: float = 1e-5
    act_smooth_max_scale: float = 1e5
    output_correction: str = "none"
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


def _lut_dtype_bits(name: str) -> int:
    return {"float16": 16, "bfloat16": 16, "float32": 32}[name]


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
def assign_to_centers_batched(
    x: torch.Tensor,
    centers: torch.Tensor,
    distance: str = "l2",
) -> torch.Tensor:
    """Nearest-center assignment for x [B, N, D] and centers [B, K, D]."""
    x = x.float().contiguous()
    centers = centers.float().contiguous()
    if distance == "chebyshev":
        dist = (x[:, :, None, :] - centers[:, None, :, :]).abs().amax(dim=3)
    elif distance == "l2":
        dist = ((x[:, :, None, :] - centers[:, None, :, :]) ** 2).sum(dim=3)
    else:
        raise ValueError(f"Unsupported distance metric: {distance}")
    return dist.argmin(dim=2)


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
    effective_sample_limit = max(sample_limit, k) if sample_limit > 0 else sample_limit
    if effective_sample_limit > 0 and x.shape[0] > effective_sample_limit:
        gen = torch.Generator(device=x.device)
        gen.manual_seed(seed)
        idx = torch.randperm(x.shape[0], generator=gen, device=x.device)[:effective_sample_limit]
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
def kmeans_padded_batched(
    x: torch.Tensor,
    k: int,
    iters: int,
    seed: int,
    sample_limit: int,
    chunk: int = 8192,
    distance: str = "l2",
    subdim: int = 2,
) -> torch.Tensor:
    """Run k-means for all activation subspaces in one Linear module.

    Input x is [N, in_features]. The output is [M, K, subdim], where
    M = in_features / subdim. This avoids hundreds of thousands of tiny
    Python k-means calls for full-model LUT-LLM runs.
    """
    x = x.float().contiguous()
    if x.shape[1] % subdim != 0:
        raise ValueError(f"in_features={x.shape[1]} is not divisible by subdim={subdim}")
    effective_sample_limit = max(sample_limit, k) if sample_limit > 0 else sample_limit
    if effective_sample_limit > 0 and x.shape[0] > effective_sample_limit:
        gen = torch.Generator(device=x.device)
        gen.manual_seed(seed)
        idx = torch.randperm(x.shape[0], generator=gen, device=x.device)[:effective_sample_limit]
        x = x[idx].contiguous()

    n = x.shape[0]
    if n <= 0:
        raise ValueError("No calibration samples for k-means")
    m = x.shape[1] // subdim
    xv = x.view(n, m, subdim)
    effective_k = min(k, n)

    gen = torch.Generator(device=x.device)
    gen.manual_seed(seed)
    init_idx = torch.randint(n, (m, effective_k), generator=gen, device=x.device)
    m_idx = torch.arange(m, device=x.device).view(m, 1)
    centers = xv[init_idx, m_idx].contiguous()

    # Bound the [N, M_chunk, K] distance tensor. The CLI encode_chunk is a row
    # chunk for other paths, so convert it into a conservative subspace chunk.
    m_chunk = max(1, min(m, max(16, chunk // max(effective_k, 1))))
    for _ in range(iters):
        for start in range(0, m, m_chunk):
            end = min(start + m_chunk, m)
            xb = xv[:, start:end, :]
            cb = centers[start:end]
            if distance == "chebyshev":
                dist = (xb[:, :, None, :] - cb[None, :, :, :]).abs().amax(dim=3)
            elif distance == "l2":
                dist = ((xb[:, :, None, :] - cb[None, :, :, :]) ** 2).sum(dim=3)
            else:
                raise ValueError(f"Unsupported distance metric: {distance}")
            codes = dist.argmin(dim=2)
            one_hot = torch.nn.functional.one_hot(codes, num_classes=effective_k).to(dtype=xb.dtype)
            counts = one_hot.sum(dim=0)
            sums = torch.einsum("nmk,nms->mks", one_hot, xb)
            nonempty = counts > 0
            updated = sums / counts.clamp_min(1.0).unsqueeze(2)
            centers[start:end] = torch.where(nonempty.unsqueeze(2), updated, cb)

    if effective_k == k:
        return centers
    pad_count = k - effective_k
    repeats = centers[:, torch.arange(pad_count, device=centers.device) % effective_k, :]
    return torch.cat([centers, repeats], dim=1)


@torch.no_grad()
def kmeans_padded_rows_batched(
    x: torch.Tensor,
    k: int,
    iters: int,
    seed: int,
    sample_limit: int,
    distance: str = "l2",
) -> torch.Tensor:
    """Run k-means independently for many small row sets.

    Input x is [B, N, D], output is [B, K, D]. This is used for LUT-LLM
    weight VQ, where each batch item is one (input subspace, output group).
    """
    x = x.float().contiguous()
    b, n, _ = x.shape
    if n <= 0:
        raise ValueError("No rows for batched k-means")
    effective_sample_limit = max(sample_limit, k) if sample_limit > 0 else sample_limit
    if effective_sample_limit > 0 and n > effective_sample_limit:
        gen = torch.Generator(device=x.device)
        gen.manual_seed(seed)
        idx = torch.randperm(n, generator=gen, device=x.device)[:effective_sample_limit]
        x = x[:, idx, :].contiguous()
        n = x.shape[1]

    effective_k = min(k, n)
    gen = torch.Generator(device=x.device)
    gen.manual_seed(seed)
    init_idx = torch.randint(n, (b, effective_k), generator=gen, device=x.device)
    b_idx = torch.arange(b, device=x.device).view(b, 1)
    centers = x[b_idx, init_idx].contiguous()

    for _ in range(iters):
        codes = assign_to_centers_batched(x, centers, distance=distance)
        one_hot = torch.nn.functional.one_hot(codes, num_classes=effective_k).to(dtype=x.dtype)
        counts = one_hot.sum(dim=1)
        sums = torch.einsum("bnk,bnd->bkd", one_hot, x)
        nonempty = counts > 0
        updated = sums / counts.clamp_min(1.0).unsqueeze(2)
        centers = torch.where(nonempty.unsqueeze(2), updated, centers)

    if effective_k == k:
        return centers
    pad_count = k - effective_k
    repeats = centers[:, torch.arange(pad_count, device=centers.device) % effective_k, :]
    return torch.cat([centers, repeats], dim=1)


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


@torch.no_grad()
def reassign_weight_codes_output_aware(
    calib: torch.Tensor,
    weight_group: torch.Tensor,
    act_codes: torch.Tensor,
    dequant_lut: torch.Tensor,
    init_codes: torch.Tensor,
    iters: int,
) -> torch.Tensor:
    """Coordinate-descent weight-code assignment using layer-output MSE."""
    if iters <= 0:
        return init_codes
    calib = calib.float().contiguous()
    weight_group = weight_group.float().contiguous()
    dequant_lut = dequant_lut.float().contiguous()
    codes = init_codes.clone().long()
    target = calib @ weight_group.t()
    pred = torch.zeros_like(target)
    n, out_features = target.shape
    m = codes.shape[0]

    for mi in range(m):
        contrib = dequant_lut[mi].index_select(0, act_codes[:, mi]).float()
        pred.add_(contrib.index_select(1, codes[mi]))

    for _ in range(iters):
        for mi in range(m):
            contrib = dequant_lut[mi].index_select(0, act_codes[:, mi]).float()
            old = contrib.index_select(1, codes[mi])
            pred.sub_(old)
            residual = target - pred
            scores = -2.0 * contrib.t().matmul(residual)
            scores.add_((contrib * contrib).sum(dim=0).view(-1, 1))
            new_codes = scores.argmin(dim=0)
            pred.add_(contrib.index_select(1, new_codes))
            codes[mi].copy_(new_codes)
    if pred.shape != (n, out_features):
        raise RuntimeError("Unexpected reconstruction shape while reassigning weight codes")
    return codes


@torch.no_grad()
def refine_weight_centers_output_aware(
    calib: torch.Tensor,
    weight_group: torch.Tensor,
    act_centers: torch.Tensor,
    act_codes: torch.Tensor,
    init_centers: torch.Tensor,
    init_codes: torch.Tensor,
    iters: int,
    reg: float,
    blend: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update weight centroids for fixed codes using calibration-output reconstruction."""
    if iters <= 0:
        return init_centers, init_codes
    calib = calib.float().contiguous()
    weight_group = weight_group.float().contiguous()
    act_centers = act_centers.float().contiguous()
    centers = init_centers.clone().float()
    codes = init_codes.clone().long()
    blend = max(0.0, min(float(blend), 1.0))
    target = calib @ weight_group.t()
    m, kw, subdim = centers.shape
    n, out_features = target.shape
    eye = torch.eye(subdim, device=calib.device, dtype=torch.float32)

    for _ in range(iters):
        lut = torch.einsum("mks,mws->mkw", act_centers, centers)
        pred = torch.zeros_like(target)
        for mi in range(m):
            contrib = lut[mi].index_select(0, act_codes[:, mi]).float()
            pred.add_(contrib.index_select(1, codes[mi]))

        for mi in range(m):
            contrib = lut[mi].index_select(0, act_codes[:, mi]).float()
            pred.sub_(contrib.index_select(1, codes[mi]))
            residual = target - pred
            x = act_centers[mi].index_select(0, act_codes[:, mi]).float()
            xtx = x.t().matmul(x)
            residual_by_code = torch.zeros((n, kw), device=calib.device, dtype=torch.float32)
            residual_by_code.index_add_(1, codes[mi], residual)
            rhs = residual_by_code.t().matmul(x)
            counts = torch.bincount(codes[mi], minlength=kw).float()
            nonempty = counts > 0
            lhs = counts[:, None, None] * xtx[None, :, :] + float(reg) * eye[None, :, :]
            updated = torch.linalg.solve(lhs[nonempty], rhs[nonempty].unsqueeze(2)).squeeze(2)
            centers[mi, nonempty] = (1.0 - blend) * centers[mi, nonempty] + blend * updated
            new_lut = act_centers[mi].matmul(centers[mi].t())
            pred.add_(new_lut.index_select(0, act_codes[:, mi]).index_select(1, codes[mi]))
    if pred.shape != (n, out_features):
        raise RuntimeError("Unexpected reconstruction shape while refining weight centers")
    return centers, codes


def _num_weight_groups(out_features: int, weight_group_size: int) -> int:
    if weight_group_size <= 0:
        return 1
    return math.ceil(out_features / weight_group_size)


def _quantize_lut(lut: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    if bits <= 0:
        return lut, lut, 1.0, 0.0
    qmax = float((1 << bits) - 1)
    min_val = float(lut.min().item())
    max_val = float(lut.max().item())
    if max_val <= min_val:
        q = torch.zeros_like(lut, dtype=torch.uint8)
        return q, torch.zeros_like(lut), 1.0, 0.0
    scale = (max_val - min_val) / qmax
    zeropoint = -min_val / scale
    q = torch.round(lut / scale + zeropoint).clamp_(0, qmax)
    dequant = (q - zeropoint) * scale
    if bits <= 8:
        q = q.to(torch.uint8)
    else:
        q = q.to(torch.int16)
    return q, dequant, scale, zeropoint


def _quantize_lut_batched(lut: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if bits <= 0:
        ones = torch.ones((lut.shape[0],), device=lut.device, dtype=torch.float32)
        zeros = torch.zeros((lut.shape[0],), device=lut.device, dtype=torch.float32)
        return lut, lut, ones, zeros
    qmax = float((1 << bits) - 1)
    min_val = lut.amin(dim=(1, 2))
    max_val = lut.amax(dim=(1, 2))
    valid = max_val > min_val
    scale = torch.where(valid, (max_val - min_val) / qmax, torch.ones_like(max_val))
    zeropoint = torch.where(valid, -min_val / scale, torch.zeros_like(max_val))
    q = torch.round(lut / scale[:, None, None] + zeropoint[:, None, None]).clamp_(0, qmax)
    q = torch.where(valid[:, None, None], q, torch.zeros_like(q))
    dequant = (q - zeropoint[:, None, None]) * scale[:, None, None]
    dequant = torch.where(valid[:, None, None], dequant, torch.zeros_like(dequant))
    if bits <= 8:
        q = q.to(torch.uint8)
    else:
        q = q.to(torch.int16)
    return q, dequant, scale.float(), zeropoint.float()


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
        compact_lut: torch.Tensor,
        lut_scales: torch.Tensor,
        lut_zeropoints: torch.Tensor,
        correction_scale: torch.Tensor,
        correction_bias: torch.Tensor,
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
        self.register_buffer("compact_lut", compact_lut, persistent=True)
        self.register_buffer("lut_scales", lut_scales, persistent=True)
        self.register_buffer("lut_zeropoints", lut_zeropoints, persistent=True)
        self.register_buffer("correction_scale", correction_scale, persistent=True)
        self.register_buffer("correction_bias", correction_bias, persistent=True)
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
        act_centers_override: torch.Tensor | None = None,
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
        if config.lut_storage not in {"expanded", "compact"}:
            raise ValueError(f"Unsupported LUT storage mode: {config.lut_storage}")

        act_centers = torch.empty((m, config.ka, config.subdim), device=device, dtype=torch.float32)
        weight_centers = torch.empty((m, group_count, config.kw, config.subdim), device=device, dtype=torch.float32)
        weight_codes = torch.empty((m, out_features), device=device, dtype=torch.long)
        if config.lut_storage == "expanded":
            expanded_lut = torch.empty((m, config.ka, out_features), device=device, dtype=lut_dtype)
            compact_lut = torch.empty((0,), device=device, dtype=torch.uint8)
        else:
            expanded_lut = torch.empty((0,), device=device, dtype=lut_dtype)
            compact_dtype = torch.uint8 if config.lut_quant_bits > 0 and config.lut_quant_bits <= 8 else lut_dtype
            compact_lut = torch.empty((m, group_count, config.ka, config.kw), device=device, dtype=compact_dtype)
        lut_scales = torch.empty((m, group_count), device=device, dtype=torch.float32)
        lut_zeropoints = torch.empty((m, group_count), device=device, dtype=torch.float32)

        _sync(device)
        start = time.perf_counter()
        if act_centers_override is None:
            act_centers.copy_(
                kmeans_padded_batched(
                    calib,
                    config.ka,
                    config.kmeans_iters,
                    config.seed,
                    config.sample_limit,
                    config.encode_chunk,
                    config.distance,
                    config.subdim,
                )
            )
        else:
            act_centers.copy_(act_centers_override.to(device=device, dtype=torch.float32))

        act_codes_for_reassign = None
        if config.weight_code_reassign_iters > 0:
            act_codes_for_reassign = encode_activation(calib.float(), act_centers, distance=config.distance)

        weight_by_subspace = weight.view(out_features, m, config.subdim).permute(1, 0, 2).contiguous()
        for gi in range(group_count):
            g_lo = gi * config.weight_group_size if config.weight_group_size > 0 else 0
            g_hi = min((gi + 1) * config.weight_group_size, out_features) if config.weight_group_size > 0 else out_features
            ww = weight_by_subspace[:, g_lo:g_hi, :]
            cw = kmeans_padded_rows_batched(
                ww,
                config.kw,
                config.kmeans_iters,
                config.seed + 9176 * gi,
                config.sample_limit,
                config.distance,
            )
            wc = assign_to_centers_batched(ww, cw, distance=config.distance)
            if act_codes_for_reassign is not None and config.weight_center_refine_iters > 0:
                cw, wc = refine_weight_centers_output_aware(
                    calib,
                    weight[g_lo:g_hi],
                    act_centers,
                    act_codes_for_reassign,
                    cw,
                    wc,
                    config.weight_center_refine_iters,
                    config.weight_center_refine_reg,
                    config.weight_center_refine_blend,
                )
            lut = torch.einsum("mks,mws->mkw", act_centers, cw)
            stored_lut, dequant_lut, scale, zeropoint = _quantize_lut_batched(lut, config.lut_quant_bits)
            if act_codes_for_reassign is not None:
                wc = reassign_weight_codes_output_aware(
                    calib,
                    weight[g_lo:g_hi],
                    act_codes_for_reassign,
                    dequant_lut,
                    wc,
                    config.weight_code_reassign_iters,
                )
            weight_centers[:, gi] = cw
            weight_codes[:, g_lo:g_hi] = wc
            if config.lut_storage == "expanded":
                gather_idx = wc[:, None, :].expand(-1, config.ka, -1)
                expanded_lut[:, :, g_lo:g_hi] = dequant_lut.gather(2, gather_idx).to(lut_dtype)
            else:
                compact_lut[:, gi] = stored_lut.to(compact_lut.dtype)
            lut_scales[:, gi] = scale
            lut_zeropoints[:, gi] = zeropoint
        _sync(device)
        train_seconds = time.perf_counter() - start

        pq = cls(
            in_features=in_features,
            out_features=out_features,
            bias=linear.bias,
            act_centers=act_centers,
            weight_centers=weight_centers,
            weight_codes=weight_codes,
            expanded_lut=expanded_lut,
            compact_lut=compact_lut,
            lut_scales=lut_scales,
            lut_zeropoints=lut_zeropoints,
            correction_scale=torch.ones((out_features,), device=device, dtype=torch.float32),
            correction_bias=torch.zeros((out_features,), device=device, dtype=torch.float32),
            config=config,
            source_name=source_name,
            train_seconds=train_seconds,
        )
        if config.output_correction != "none":
            pq._fit_output_correction(linear, calib)
        return pq

    @torch.no_grad()
    def _fit_output_correction(self, linear: nn.Linear, calib: torch.Tensor) -> None:
        if self.config.output_correction not in {"bias", "affine"}:
            raise ValueError(f"Unsupported output correction: {self.config.output_correction}")
        orig = linear(calib).float()
        approx = self(calib).float()
        if self.config.output_correction == "bias":
            self.correction_bias.copy_((orig - approx).mean(dim=0))
            return
        approx_mean = approx.mean(dim=0)
        orig_mean = orig.mean(dim=0)
        centered_approx = approx - approx_mean
        centered_orig = orig - orig_mean
        var = (centered_approx * centered_approx).mean(dim=0).clamp_min(1e-6)
        cov = (centered_approx * centered_orig).mean(dim=0)
        scale = (cov / var).clamp(-8.0, 8.0)
        bias = orig_mean - scale * approx_mean
        self.correction_scale.copy_(scale)
        self.correction_bias.copy_(bias)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape[:-1]
        flat = x.reshape(-1, self.in_features)
        codes = encode_activation(flat, self.act_centers, distance=self.config.distance)
        out = torch.zeros((flat.shape[0], self.out_features), device=flat.device, dtype=torch.float32)
        if self.config.lut_storage == "compact":
            group_count = _num_weight_groups(self.out_features, self.config.weight_group_size)
            n, m = codes.shape
            chunk_m = 16
            ka_kw = self.config.ka * self.config.kw
            for start in range(0, m, chunk_m):
                end = min(start + chunk_m, m)
                chunk_codes = codes[:, start:end].t()
                chunk_offsets = (
                    torch.arange(end - start, device=codes.device, dtype=codes.dtype).view(-1, 1, 1) * ka_kw
                )
                for gi in range(group_count):
                    g_lo = gi * self.config.weight_group_size if self.config.weight_group_size > 0 else 0
                    g_hi = min((gi + 1) * self.config.weight_group_size, self.out_features) if self.config.weight_group_size > 0 else self.out_features
                    weight_code = self.weight_codes[start:end, g_lo:g_hi]
                    idx = chunk_offsets + chunk_codes[:, :, None] * self.config.kw + weight_code[:, None, :]
                    vals = self.compact_lut[start:end, gi].reshape(-1).index_select(0, idx.reshape(-1))
                    vals = vals.reshape(end - start, n, g_hi - g_lo).float()
                    if self.config.lut_quant_bits > 0:
                        scales = self.lut_scales[start:end, gi].view(end - start, 1, 1)
                        zeropoints = self.lut_zeropoints[start:end, gi].view(end - start, 1, 1)
                        vals = (vals - zeropoints) * scales
                    out[:, g_lo:g_hi].add_(vals.sum(dim=0))
            if self.bias is not None:
                out.add_(self.bias.float())
            out.mul_(self.correction_scale).add_(self.correction_bias)
            return out.to(dtype=x.dtype).reshape(*original_shape, self.out_features)

        n, m = codes.shape
        chunk_m = 32
        for start in range(0, m, chunk_m):
            end = min(start + chunk_m, m)
            offsets = torch.arange(end - start, device=codes.device, dtype=codes.dtype) * self.config.ka
            idx = (codes[:, start:end].t() + offsets[:, None]).reshape(-1)
            vals = self.expanded_lut[start:end].reshape(-1, self.out_features).index_select(0, idx)
            out.add_(vals.reshape(end - start, n, self.out_features).float().sum(dim=0))
        if self.bias is not None:
            out.add_(self.bias.float())
        out.mul_(self.correction_scale).add_(self.correction_bias)
        return out.to(dtype=x.dtype).reshape(*original_shape, self.out_features)

    def hardware_stats(self) -> dict[str, Any]:
        m = self.in_features // self.config.subdim
        group_count = _num_weight_groups(self.out_features, self.config.weight_group_size)
        act_code_bits = math.ceil(math.log2(self.config.ka))
        weight_code_bits = math.ceil(math.log2(self.config.kw))
        lut_bits = self.config.lut_quant_bits if self.config.lut_quant_bits > 0 else _lut_dtype_bits(self.config.lut_dtype)
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
            "weight_code_reassign_iters": self.config.weight_code_reassign_iters,
            "weight_center_refine_iters": self.config.weight_center_refine_iters,
            "weight_center_refine_reg": self.config.weight_center_refine_reg,
            "weight_center_refine_blend": self.config.weight_center_refine_blend,
            "Ka": self.config.ka,
            "Kw": self.config.kw,
            "distance": self.config.distance,
            "lut_quant_bits": self.config.lut_quant_bits,
            "lut_storage": self.config.lut_storage,
            "output_correction": self.config.output_correction,
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
