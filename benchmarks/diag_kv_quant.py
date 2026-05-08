#  Copyright © 2026 Emmi AI GmbH. All rights reserved.
"""Diagnostic: polar KV-cache quantization (PolarQuant principle).

Tests whether perceiver attention KV tensors can be compressed to 3-4 bits
via JL rotation + polar quantization without meaningful accuracy loss.

Three tests:
  1. Distribution  — angles concentrate after a JL rotation (validates the premise)
  2. Reconstruction — cosine similarity of K/V after polar quantize → dequantize
  3. Attention fidelity — output error of F.scaled_dot_product_attention with
                          quantized vs original K/V

Run with:
    uv run python benchmarks/diag_kv_quant.py
"""

import math

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Core quantization primitives
# ---------------------------------------------------------------------------


def make_jl_rotation(dim: int, seed: int = 0) -> torch.Tensor:
    """Random orthogonal matrix via QR of a Gaussian draw.

    Returns shape (dim, dim). Fixed seed so compress/decompress agree.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(dim, dim, generator=g))
    return Q


def _to_complex(x: torch.Tensor) -> torch.Tensor:
    """View (..., D) real tensor as (..., D//2) complex by pairing adjacent dims."""
    return torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))


def _to_real(x_c: torch.Tensor) -> torch.Tensor:
    """View (..., D//2) complex tensor as (..., D) real."""
    return torch.view_as_real(x_c).flatten(-2)


def polar_quantize(
    x: torch.Tensor, n_angle_bits: int, n_radius_bits: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compress x via polar quantization.

    Args:
        x: (..., D) float tensor. D must be even.
        n_angle_bits: Bits used for angle quantization (per complex pair).
        n_radius_bits: Bits used for radius quantization (per complex pair).

    Returns:
        (angle_codes, radius_codes, radius_min, radius_scale) where:
        - angle_codes: int64 in [0, 2^n_angle_bits)
        - radius_codes: int64 in [0, 2^n_radius_bits)
        - radius_min, radius_scale: per-tensor scalars for radius dequant
    """
    assert x.shape[-1] % 2 == 0, "Last dim must be even for complex pairing"
    x_c = _to_complex(x)  # (..., D//2)

    # Angle: uniform quantization over (-pi, pi]
    angles = x_c.angle()  # (-pi, pi]
    n_a = 2**n_angle_bits
    angle_codes = ((angles + math.pi) / (2 * math.pi) * n_a).long().clamp(0, n_a - 1)

    # Radius: min-max quantization (all radii are non-negative)
    radii = x_c.abs()
    r_min = radii.min()
    r_max = radii.max()
    r_scale = (r_max - r_min).clamp(min=1e-8)
    n_r = 2**n_radius_bits
    radius_codes = ((radii - r_min) / r_scale * (n_r - 1)).round().long().clamp(0, n_r - 1)

    return angle_codes, radius_codes, r_min, r_scale


def polar_dequantize(
    angle_codes: torch.Tensor,
    radius_codes: torch.Tensor,
    r_min: torch.Tensor,
    r_scale: torch.Tensor,
    n_angle_bits: int,
    n_radius_bits: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Reconstruct x from polar-quantized codes."""
    n_a = 2**n_angle_bits
    angles = angle_codes.float() / n_a * 2 * math.pi - math.pi

    n_r = 2**n_radius_bits
    radii = radius_codes.float() / (n_r - 1) * r_scale + r_min

    x_c = torch.polar(radii, angles)
    return _to_real(x_c).to(dtype)


def compress(
    x: torch.Tensor, R: torch.Tensor, n_angle_bits: int, n_radius_bits: int
) -> tuple:
    """JL-rotate x then polar-quantize."""
    *leading, dim = x.shape
    x_rot = (x.float().reshape(-1, dim) @ R.T).reshape(*leading, dim)
    return polar_quantize(x_rot, n_angle_bits, n_radius_bits)


def decompress(codes: tuple, R: torch.Tensor, n_angle_bits: int, n_radius_bits: int, dtype: torch.dtype) -> torch.Tensor:
    """Polar-dequantize then invert JL rotation."""
    angle_codes, radius_codes, r_min, r_scale = codes
    x_rot = polar_dequantize(angle_codes, radius_codes, r_min, r_scale, n_angle_bits, n_radius_bits, torch.float32)
    *leading, dim = x_rot.shape
    x_rec = (x_rot.reshape(-1, dim) @ R).reshape(*leading, dim)
    return x_rec.to(dtype)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def mean_cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.reshape(-1, a.shape[-1]).float()
    b_f = b.reshape(-1, b.shape[-1]).float()
    return (F.normalize(a_f, dim=-1) * F.normalize(b_f, dim=-1)).sum(-1).mean().item()


def relative_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / a.float().norm()).item()


def angle_entropy_bits(angles: torch.Tensor, n_bins: int = 64) -> float:
    hist = torch.histc(angles.float(), bins=n_bins, min=-math.pi, max=math.pi)
    p = hist / hist.sum()
    p = p[p > 0]
    return -(p * p.log2()).sum().item()


def bits_per_element(n_angle_bits: int, n_radius_bits: int) -> float:
    """Storage bits per original real-valued element.

    One complex pair (2 reals) costs n_angle_bits (angle) + n_radius_bits (radius).
    """
    return (n_angle_bits + n_radius_bits) / 2


# ---------------------------------------------------------------------------
# Synthetic data that mimics trained perceiver KV activations
# ---------------------------------------------------------------------------


def make_kv(B: int, H: int, S: int, D: int, seed: int) -> torch.Tensor:
    """Gaussian base + low-rank structure to mimic trained model activations."""
    g = torch.Generator()
    g.manual_seed(seed)
    base = torch.randn(B, H, S, D, generator=g)
    # Low-rank component: a few dominant directions with higher variance
    lr = torch.randn(B, H, S, 8, generator=g) @ torch.randn(8, D, generator=g) * 2.0
    return base + lr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    torch.manual_seed(42)

    # Shapes matching a small AB-UPT perceiver: 2 batch, 8 heads, 256 anchors, 64 head_dim
    B, H, S, D = 2, 8, 256, 64
    Q_SEQ = 512  # query sequence length (mesh query points)

    K = make_kv(B, H, S, D, seed=1)
    V = make_kv(B, H, S, D, seed=2)
    Q = torch.randn(B, H, Q_SEQ, D)

    R = make_jl_rotation(D)

    # Configs to test: (n_angle_bits, n_radius_bits, label)
    configs = [
        (8, 8, "8a+8r  (baseline)"),
        (4, 8, "4a+8r"),
        (4, 4, "4a+4r"),
        (3, 4, "3a+4r"),
        (3, 3, "3a+3r"),
        (2, 3, "2a+3r"),
    ]

    # -----------------------------------------------------------------------
    # Test 1: Angle distribution
    # -----------------------------------------------------------------------
    print("=" * 65)
    print("TEST 1: Angle entropy before/after JL rotation")
    print("        (lower = more structured; uniform = max 6.0 bits)")
    print("=" * 65)

    K_flat = K.reshape(-1, D).float()

    angles_raw = _to_complex(K_flat).angle()
    entropy_raw = angle_entropy_bits(angles_raw)

    K_rot = K_flat @ R.T
    angles_rot = _to_complex(K_rot).angle()
    entropy_rot = angle_entropy_bits(angles_rot)

    print(f"  Before JL:  entropy = {entropy_raw:.3f} bits  (std = {angles_raw.std():.4f})")
    print(f"  After  JL:  entropy = {entropy_rot:.3f} bits  (std = {angles_rot.std():.4f})")
    print()
    if abs(entropy_rot - entropy_raw) < 0.1:
        print("  Note: distributions are similar — JL rotation preserves angular")
        print("  entropy for activations that are already approximately isotropic.")
    print()

    # -----------------------------------------------------------------------
    # Test 2: Round-trip fidelity
    # -----------------------------------------------------------------------
    print("=" * 65)
    print("TEST 2: Round-trip reconstruction fidelity")
    print("=" * 65)
    print(f"  {'Config':<18}  {'bpe':>5}  {'K cos-sim':>10}  {'V cos-sim':>10}  {'K rel-L2':>9}")
    print(f"  {'-' * 57}")

    for n_a, n_r, label in configs:
        ck = compress(K, R, n_a, n_r)
        cv = compress(V, R, n_a, n_r)
        K_rec = decompress(ck, R, n_a, n_r, K.dtype)
        V_rec = decompress(cv, R, n_a, n_r, V.dtype)

        bpe = bits_per_element(n_a, n_r)
        cos_k = mean_cosine_sim(K, K_rec)
        cos_v = mean_cosine_sim(V, V_rec)
        l2_k = relative_l2(K, K_rec)
        print(f"  {label:<18}  {bpe:>5.1f}  {cos_k:>10.6f}  {cos_v:>10.6f}  {l2_k:>9.6f}")

    print()

    # -----------------------------------------------------------------------
    # Test 3: Attention output fidelity
    # -----------------------------------------------------------------------
    print("=" * 65)
    print("TEST 3: Attention output fidelity")
    print("=" * 65)
    print(f"  {'Config':<18}  {'bpe':>5}  {'out cos-sim':>12}  {'out rel-L2':>11}")
    print(f"  {'-' * 53}")

    out_orig = F.scaled_dot_product_attention(Q, K, V)

    for n_a, n_r, label in configs:
        ck = compress(K, R, n_a, n_r)
        cv = compress(V, R, n_a, n_r)
        K_q = decompress(ck, R, n_a, n_r, K.dtype)
        V_q = decompress(cv, R, n_a, n_r, V.dtype)

        out_q = F.scaled_dot_product_attention(Q, K_q, V_q)
        cos_out = mean_cosine_sim(out_orig, out_q)
        l2_out = relative_l2(out_orig, out_q)
        print(f"  {label:<18}  {bits_per_element(n_a, n_r):>5.1f}  {cos_out:>12.6f}  {l2_out:>11.6f}")

    print()

    # -----------------------------------------------------------------------
    # Memory arithmetic
    # -----------------------------------------------------------------------
    print("=" * 65)
    print("MEMORY: KV cache per geometry (this tensor shape)")
    print("=" * 65)
    n_elements = K.numel() * 2  # K + V
    fp16_bytes = n_elements * 2

    print(f"  float16 baseline: {fp16_bytes / 1024:.1f} KB")
    for n_a, n_r, label in configs:
        bpe = bits_per_element(n_a, n_r)
        quant_bytes = n_elements * bpe / 8
        ratio = fp16_bytes / quant_bytes
        print(f"  {label:<18}  {quant_bytes / 1024:>6.1f} KB  ({ratio:.1f}x reduction vs float16)")

    print()
    print("Shape context: B=%d H=%d anchors=%d head_dim=%d -> %d KV elements" % (B, H, S, D, n_elements))


if __name__ == "__main__":
    main()
