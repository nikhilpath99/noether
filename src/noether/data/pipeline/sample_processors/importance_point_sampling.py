#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Any

import torch

from noether.data.pipeline.sample_processor import SampleProcessor


class ImportancePointSamplingSampleProcessor(SampleProcessor):
    """Samples points with probability proportional to per-point importance weights.

    Drop-in replacement for :class:`PointSamplingSampleProcessor` when importance
    weights are available in the sample dict (e.g. from
    :class:`FieldGradientWeightSampleProcessor` or a model-uncertainty callback).

    The ``temperature`` parameter controls sharpness of the sampling distribution:
    - ``temperature=1.0``  → sample proportional to raw weights
    - ``temperature→0``    → concentrate entirely on highest-weight points (greedy)
    - ``temperature→∞``    → approach uniform random sampling

    .. code-block:: python

        processor = ImportancePointSamplingSampleProcessor(
            items={"volume_position", "volume_pressure"},
            weight_item="volume_importance_weights",
            num_points=1024,
            temperature=0.5,
        )
    """

    def __init__(
        self,
        items: set[str],
        weight_item: str,
        num_points: int,
        temperature: float = 1.0,
        seed: int | None = None,
    ):
        """
        Args:
            items: Items to subsample with shared indices.
            weight_item: Key in the sample dict containing per-point weights (N,).
            num_points: Number of points to sample.
            temperature: Sharpness of the sampling distribution. Values < 1 concentrate
                sampling on high-weight points; values > 1 approach uniform.
            seed: Optional seed for deterministic sampling. Requires ``index`` key in sample.
        """
        if num_points <= 0:
            raise ValueError("num_points must be positive.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")

        self.items = items
        self.weight_item = weight_item
        self.num_points = num_points
        self.temperature = temperature
        self.seed = seed

    def __call__(self, input_sample: dict[str, Any]) -> dict[str, Any]:
        output_sample = self.save_copy(input_sample)

        if self.weight_item not in output_sample:
            raise KeyError(
                f"Weight item '{self.weight_item}' not found in sample. "
                f"Available keys: {list(output_sample.keys())}"
            )

        weights: torch.Tensor = output_sample[self.weight_item]
        if weights.ndim != 1:
            raise ValueError(
                f"Weight tensor must be 1D, got shape {weights.shape}. "
                "Use weight_item pointing to a (N,) tensor."
            )

        if self.seed is not None:
            if "index" not in output_sample:
                raise ValueError("Sample 'index' key required for deterministic sampling.")
            generator = torch.Generator().manual_seed(output_sample["index"] + self.seed)
        else:
            generator = None

        # Apply temperature scaling: w^(1/T), then renormalize.
        # Using log-space for numerical stability with extreme temperatures.
        log_weights = torch.log(weights.float().clamp(min=1e-10))
        scaled_weights = torch.softmax(log_weights / self.temperature, dim=0)

        num_available = len(weights)
        n = min(self.num_points, num_available)
        perm = torch.multinomial(scaled_weights, num_samples=n, replacement=False, generator=generator)

        for item in self.items:
            output_sample[item] = output_sample[item][perm]

        return output_sample
