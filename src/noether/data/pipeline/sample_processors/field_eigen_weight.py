#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Any

import torch

from noether.data.pipeline.sample_processor import SampleProcessor
from noether.modeling.functional.geometric import knn


class FieldEigenWeightSampleProcessor(SampleProcessor):
    """Computes per-point importance weights from the local structure tensor eigenvalues.

    For each point, builds the structure tensor from gradient vectors to its k
    nearest neighbours:

        g_{ij} = (f_j - f_i) / (||x_j - x_i||_2 + eps)   shape (D_field,)
        S_i    = (1/k) * sum_j  g_{ij} g_{ij}^T            shape (D_field, D_field)
        w_i    = sum of eigenvalues of S_i  (= trace = Frobenius^2 of gradient)

    Unlike scalar gradient magnitude, the structure tensor captures *anisotropy*:
    vortex cores (gradients in all directions, lambda_1 ~ lambda_2) are weighted
    differently from shocks/shear layers (one dominant direction).

    The weight ``mode`` controls how eigenvalues are combined:
    - ``"trace"``  : sum of all eigenvalues — equivalent to mean squared gradient magnitude.
    - ``"max"``    : largest eigenvalue only — dominant gradient direction.
    - ``"det"``    : product of eigenvalues — non-zero only when all directions are active.

    Args:
        position_item: Key for the (N, D_pos) position tensor.
        field_items: Keys for field tensors (N, D_field) to compute gradients from.
            Multiple fields are concatenated before gradient estimation.
        weight_key: Output key where the computed (N,) weight tensor is stored.
        k: Number of nearest neighbours used for gradient estimation.
        eps: Small value added to distances to avoid division by zero.
        mode: How to combine eigenvalues into a scalar weight.
            One of ``"trace"``, ``"max"``, ``"det"``.

    Example:

        .. testcode::

            import torch
            from noether.data.pipeline.sample_processors import FieldEigenWeightSampleProcessor

            proc = FieldEigenWeightSampleProcessor(
                position_item="volume_position",
                field_items=["volume_velocity", "volume_pressure"],
                weight_key="volume_importance_weights",
                k=16,
                mode="trace",
            )
            sample = {
                "volume_position": torch.rand(128, 2),
                "volume_velocity": torch.rand(128, 2),
                "volume_pressure": torch.rand(128, 1),
            }
            out = proc(sample)
            print(out["volume_importance_weights"].shape)

        .. testoutput::

            torch.Size([128])
    """

    def __init__(
        self,
        position_item: str,
        field_items: list[str],
        weight_key: str,
        k: int = 16,
        eps: float = 1e-8,
        mode: str = "trace",
    ):
        if k < 1:
            raise ValueError("k must be at least 1.")
        if mode not in ("trace", "max", "det"):
            raise ValueError("mode must be one of 'trace', 'max', 'det'.")

        self.position_item = position_item
        self.field_items = field_items
        self.weight_key = weight_key
        self.k = k
        self.eps = eps
        self.mode = mode

    def __call__(self, input_sample: dict[str, Any]) -> dict[str, Any]:
        output_sample = self.save_copy(input_sample)

        if self.weight_key in output_sample:
            return output_sample

        pos: torch.Tensor = output_sample[self.position_item].float()  # (N, D_pos)
        field = torch.cat(
            [output_sample[key].float() for key in self.field_items], dim=-1
        )  # (N, D_field)

        N, D = pos.size(0), field.size(1)
        k = min(self.k, N - 1)

        edges = knn(x=pos, y=pos, k=k + 1)
        query_idx, neighbor_idx = edges.unbind()

        mask = query_idx != neighbor_idx
        query_idx, neighbor_idx = query_idx[mask], neighbor_idx[mask]

        pos_diff = (pos[query_idx] - pos[neighbor_idx]).norm(dim=-1, keepdim=True)  # (E, 1)
        grad_vecs = (field[query_idx] - field[neighbor_idx]) / (pos_diff + self.eps)  # (E, D)

        # Structure tensor: outer product g g^T, then mean per query point → (N, D, D)
        outer = grad_vecs.unsqueeze(-1) * grad_vecs.unsqueeze(-2)  # (E, D, D)
        S = torch.zeros(N, D, D, device=pos.device)
        count = torch.zeros(N, device=pos.device)
        S.scatter_add_(0, query_idx.view(-1, 1, 1).expand_as(outer), outer)
        count.scatter_add_(0, query_idx, torch.ones(query_idx.size(0), device=pos.device))
        count = count.clamp(min=1).view(N, 1, 1)
        S = S / count  # (N, D, D)

        eigenvalues = torch.linalg.eigvalsh(S)  # (N, D), ascending

        if self.mode == "trace":
            weights = eigenvalues.sum(dim=-1)
        elif self.mode == "max":
            weights = eigenvalues[:, -1]
        else:  # det
            weights = eigenvalues.prod(dim=-1)

        output_sample[self.weight_key] = weights
        return output_sample
