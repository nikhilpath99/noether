#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import time

import pytest
import torch

from noether.data.pipeline.sample_processors import (
    FieldGradientWeightSampleProcessor,
    ImportancePointSamplingSampleProcessor,
    PointSamplingSampleProcessor,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def uniform_sample():
    """500 points, flat field — no gradient anywhere."""
    torch.manual_seed(0)
    N = 500
    pos = torch.rand(N, 3)
    field = torch.ones(N, 1)
    return {"volume_position": pos, "volume_pressure": field}


@pytest.fixture
def gradient_sample():
    """500 points split into two halves.
    Left half (x < 0.5): pressure = 0.
    Right half (x >= 0.5): pressure = 10.
    The boundary region (0.45–0.55) has a sharp gradient.
    """
    torch.manual_seed(1)
    N = 500
    pos = torch.rand(N, 3)
    pressure = torch.where(pos[:, 0:1] >= 0.5, torch.tensor(10.0), torch.tensor(0.0))
    return {"volume_position": pos, "volume_pressure": pressure}


# ---------------------------------------------------------------------------
# FieldGradientWeightSampleProcessor tests
# ---------------------------------------------------------------------------

class TestFieldGradientWeightSampleProcessor:

    def test_output_key_added(self, uniform_sample):
        proc = FieldGradientWeightSampleProcessor(
            position_item="volume_position",
            field_items=["volume_pressure"],
            weight_key="weights",
        )
        out = proc(uniform_sample)
        assert "weights" in out

    def test_weight_shape(self, uniform_sample):
        N = len(uniform_sample["volume_position"])
        proc = FieldGradientWeightSampleProcessor(
            position_item="volume_position",
            field_items=["volume_pressure"],
            weight_key="weights",
        )
        out = proc(uniform_sample)
        assert out["weights"].shape == (N,)

    def test_weights_non_negative(self, gradient_sample):
        proc = FieldGradientWeightSampleProcessor(
            position_item="volume_position",
            field_items=["volume_pressure"],
            weight_key="weights",
        )
        out = proc(gradient_sample)
        assert (out["weights"] >= 0).all()

    def test_high_gradient_region_gets_higher_weight(self, gradient_sample):
        """Points near the x=0.5 boundary should have higher weights."""
        proc = FieldGradientWeightSampleProcessor(
            position_item="volume_position",
            field_items=["volume_pressure"],
            weight_key="weights",
            k=8,
        )
        out = proc(gradient_sample)
        pos = gradient_sample["volume_position"]
        weights = out["weights"]

        near_boundary = (pos[:, 0] - 0.5).abs() < 0.1
        far_from_boundary = (pos[:, 0] - 0.5).abs() > 0.3

        mean_near = weights[near_boundary].mean()
        mean_far = weights[far_from_boundary].mean()
        assert mean_near > mean_far, (
            f"Expected higher weights near boundary ({mean_near:.4f}) "
            f"than far from it ({mean_far:.4f})"
        )

    def test_does_not_modify_input(self, uniform_sample):
        original_pos = uniform_sample["volume_position"].clone()
        proc = FieldGradientWeightSampleProcessor(
            position_item="volume_position",
            field_items=["volume_pressure"],
            weight_key="weights",
        )
        proc(uniform_sample)
        assert torch.equal(uniform_sample["volume_position"], original_pos)

    def test_invalid_k(self):
        with pytest.raises(ValueError):
            FieldGradientWeightSampleProcessor(
                position_item="pos", field_items=["f"], weight_key="w", k=0
            )


# ---------------------------------------------------------------------------
# ImportancePointSamplingSampleProcessor tests
# ---------------------------------------------------------------------------

class TestImportancePointSamplingSampleProcessor:

    def _make_sample_with_weights(self, N=200, seed=0):
        torch.manual_seed(seed)
        pos = torch.rand(N, 3)
        field = torch.rand(N, 1)
        # artificial: first half gets weight 1, second half gets weight 100
        weights = torch.ones(N)
        weights[N // 2 :] = 100.0
        return {"pos": pos, "field": field, "w": weights}

    def test_output_shape(self):
        sample = self._make_sample_with_weights()
        proc = ImportancePointSamplingSampleProcessor(
            items={"pos", "field"}, weight_item="w", num_points=50
        )
        out = proc(sample)
        assert out["pos"].shape == (50, 3)
        assert out["field"].shape == (50, 1)

    def test_sampled_points_exist_in_input(self):
        sample = self._make_sample_with_weights()
        proc = ImportancePointSamplingSampleProcessor(
            items={"pos"}, weight_item="w", num_points=50
        )
        out = proc(sample)
        for row in out["pos"]:
            assert any(torch.equal(row, r) for r in sample["pos"])

    def test_high_weight_region_oversampled(self):
        """Second half (weight=100) should dominate the sample."""
        torch.manual_seed(42)
        N = 400
        sample = self._make_sample_with_weights(N=N)
        proc = ImportancePointSamplingSampleProcessor(
            items={"pos"}, weight_item="w", num_points=100, temperature=1.0
        )
        out = proc(sample)
        # count how many sampled points came from the high-weight half
        original_pos = sample["pos"]
        high_weight_pos = original_pos[N // 2 :]
        count_high = sum(
            any(torch.equal(row, r) for r in high_weight_pos)
            for row in out["pos"]
        )
        # With weight ratio 100:1, virtually all samples should come from high-weight half
        assert count_high > 80, f"Expected >80 from high-weight half, got {count_high}"

    def test_deterministic_with_seed(self):
        sample = {**self._make_sample_with_weights(), "index": 7}
        proc = ImportancePointSamplingSampleProcessor(
            items={"pos"}, weight_item="w", num_points=50, seed=42
        )
        out1 = proc(sample)
        out2 = proc(sample)
        assert torch.equal(out1["pos"], out2["pos"])

    def test_nondeterministic_without_seed(self):
        torch.manual_seed(0)
        sample = self._make_sample_with_weights(N=500)
        proc = ImportancePointSamplingSampleProcessor(
            items={"pos"}, weight_item="w", num_points=50
        )
        out1 = proc(sample)
        out2 = proc(sample)
        assert not torch.equal(out1["pos"], out2["pos"])

    def test_missing_weight_key_raises(self):
        sample = {"pos": torch.rand(10, 3)}
        proc = ImportancePointSamplingSampleProcessor(
            items={"pos"}, weight_item="missing", num_points=5
        )
        with pytest.raises(KeyError, match="missing"):
            proc(sample)

    def test_invalid_num_points(self):
        with pytest.raises(ValueError):
            ImportancePointSamplingSampleProcessor(
                items={"pos"}, weight_item="w", num_points=0
            )

    def test_invalid_temperature(self):
        with pytest.raises(ValueError):
            ImportancePointSamplingSampleProcessor(
                items={"pos"}, weight_item="w", num_points=10, temperature=0.0
            )

    def test_temperature_effect(self):
        """Low temperature should be more concentrated than high temperature."""
        torch.manual_seed(0)
        N = 200
        sample = self._make_sample_with_weights(N=N)

        proc_sharp = ImportancePointSamplingSampleProcessor(
            items={"pos"}, weight_item="w", num_points=50, temperature=0.1
        )
        proc_flat = ImportancePointSamplingSampleProcessor(
            items={"pos"}, weight_item="w", num_points=50, temperature=10.0
        )

        original_pos = sample["pos"]
        high_weight_pos = original_pos[N // 2 :]

        def count_high(out):
            return sum(
                any(torch.equal(row, r) for r in high_weight_pos)
                for row in out["pos"]
            )

        sharp_count = count_high(proc_sharp(sample))
        flat_count = count_high(proc_flat(sample))
        assert sharp_count >= flat_count, (
            f"Sharp temperature should sample more from high-weight region: "
            f"sharp={sharp_count}, flat={flat_count}"
        )


# ---------------------------------------------------------------------------
# Runtime benchmark (not a pytest test — run directly)
# ---------------------------------------------------------------------------

def benchmark(n_points: int = 10_000, num_sample: int = 2048, k: int = 16, repeats: int = 10, device: str = "cpu"):
    torch.manual_seed(0)
    sample = {
        "volume_position": torch.rand(n_points, 3).to(device),
        "volume_pressure": torch.rand(n_points, 1).to(device),
        "volume_velocity": torch.rand(n_points, 3).to(device),
    }
    items = {"volume_position", "volume_pressure", "volume_velocity"}

    # --- Uniform baseline ---
    uniform_proc = PointSamplingSampleProcessor(items=items, num_points=num_sample)

    # warmup
    uniform_proc(sample)
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(repeats):
        uniform_proc(sample)
    if device == "cuda":
        torch.cuda.synchronize()
    uniform_ms = (time.perf_counter() - t0) / repeats * 1000

    # --- Importance sampling ---
    grad_proc = FieldGradientWeightSampleProcessor(
        position_item="volume_position",
        field_items=["volume_pressure", "volume_velocity"],
        weight_key="volume_weights",
        k=k,
    )
    imp_proc = ImportancePointSamplingSampleProcessor(
        items=items, weight_item="volume_weights", num_points=num_sample
    )

    # warmup
    s = grad_proc(sample)
    imp_proc(s)
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(repeats):
        s = grad_proc(sample)
        imp_proc(s)
    if device == "cuda":
        torch.cuda.synchronize()
    importance_ms = (time.perf_counter() - t0) / repeats * 1000

    print(f"  [{device.upper():4s}] N={n_points:>6,}  uniform={uniform_ms:7.2f}ms  importance={importance_ms:7.2f}ms  overhead={importance_ms / uniform_ms:6.1f}x")


if __name__ == "__main__":
    sizes = [1_000, 5_000, 10_000, 30_000]

    print(f"\n{'='*75}")
    print(f"Benchmark — sample=2048  k=16  repeats=10")
    print(f"{'='*75}")

    print("\nCPU:")
    for n in sizes:
        benchmark(n_points=n, device="cpu")

    if torch.cuda.is_available():
        print("\nCUDA:")
        for n in sizes:
            benchmark(n_points=n, device="cuda")
    else:
        print("\nCUDA not available.")
