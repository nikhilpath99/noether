#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Any

import torch

from noether.data.pipeline.sample_processor import SampleProcessor
from noether.modeling.functional.geometric import knn


class FieldGradientWeightSampleProcessor(SampleProcessor):
    """Computes per-point importance weights from local field gradient magnitude.

    For each point, estimates the local gradient by averaging the normalised
    field-difference to its k nearest neighbours:

        w_i = mean_j( ||f_j - f_i||_2 / (||x_j - x_i||_2 + eps) )

    Points in high-gradient regions (wakes, separation zones, shocks) receive
    higher weights and will be over-sampled by
    :class:`ImportancePointSamplingSampleProcessor`.

    .. code-block:: python

        # Chain with ImportancePointSamplingSampleProcessor:
        gradient_proc = FieldGradientWeightSampleProcessor(
            position_item="volume_position",
            field_items=["volume_velocity", "volume_pressure"],
            weight_key="volume_importance_weights",
            k=16,
        )
        sample_proc = ImportancePointSamplingSampleProcessor(
            items={"volume_position", "volume_velocity", "volume_pressure"},
            weight_item="volume_importance_weights",
            num_points=2048,
            temperature=0.5,
        )
    """

    def __init__(
        self,
        position_item: str,
        field_items: list[str],
        weight_key: str,
        k: int = 16,
        eps: float = 1e-8,
    ):
        """
        Args:
            position_item: Key for the (N, D_pos) position tensor.
            field_items: Keys for field tensors (N, D_field) to compute gradients from.
                Multiple fields are concatenated before gradient estimation.
            weight_key: Output key where the computed (N,) weight tensor is stored.
            k: Number of nearest neighbours used for gradient estimation.
            eps: Small value added to distances to avoid division by zero.
        """
        if k < 1:
            raise ValueError("k must be at least 1.")

        self.position_item = position_item
        self.field_items = field_items
        self.weight_key = weight_key
        self.k = k
        self.eps = eps

    def __call__(self, input_sample: dict[str, Any]) -> dict[str, Any]:
        output_sample = self.save_copy(input_sample)

        if self.weight_key in output_sample:
            return output_sample

        pos: torch.Tensor = output_sample[self.position_item].float()   # (N, D_pos)
        field = torch.cat(
            [output_sample[key].float() for key in self.field_items], dim=-1
        )  # (N, D_field)

        N = pos.size(0)
        k = min(self.k, N - 1)

        # k-NN graph: edges[0] = query indices (y), edges[1] = neighbor indices (x)
        # +1 so we can remove the self-loop that knn includes when x==y
        edges = knn(x=pos, y=pos, k=k + 1)
        query_idx, neighbor_idx = edges.unbind()  # same convention as supernode_pooling.py

        # Remove self-loops
        mask = query_idx != neighbor_idx
        query_idx, neighbor_idx = query_idx[mask], neighbor_idx[mask]

        field_diff = (field[query_idx] - field[neighbor_idx]).norm(dim=-1)  # (E,)
        pos_diff = (pos[query_idx] - pos[neighbor_idx]).norm(dim=-1)        # (E,)
        grad_mag = field_diff / (pos_diff + self.eps)                        # (E,)

        # Scatter mean gradient magnitude back to each query point
        weights = torch.zeros(N, device=pos.device)
        weights.scatter_reduce_(0, query_idx, grad_mag, reduce="mean", include_self=False)

        output_sample[self.weight_key] = weights
        return output_sample