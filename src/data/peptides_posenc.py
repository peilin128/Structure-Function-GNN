from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import (
    get_laplacian,
    is_undirected,
    scatter,
    to_dense_adj,
    to_scipy_sparse_matrix,
    to_undirected,
)


VALID_PE_MODES = {"rwse", "lappe"}


def _l2_normalize_eigenvectors(eigvecs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    denom = eigvecs.norm(p=2, dim=0, keepdim=True)
    return eigvecs / denom.clamp_min(eps).expand_as(eigvecs)


def _compute_lappe(data, max_freqs: int = 10):
    num_nodes = int(data.num_nodes)
    edge_index = data.edge_index
    if not is_undirected(edge_index, num_nodes=num_nodes):
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)

    lap_edge_index, lap_edge_weight = get_laplacian(
        edge_index,
        normalization=None,
        num_nodes=num_nodes,
    )
    laplacian = to_scipy_sparse_matrix(
        lap_edge_index,
        lap_edge_weight,
        num_nodes=num_nodes,
    )
    evals_np, evecs_np = np.linalg.eigh(laplacian.toarray())

    order = evals_np.argsort()[:max_freqs]
    evals = torch.from_numpy(np.real(evals_np[order])).float().clamp_min(0)
    evecs = torch.from_numpy(np.real(evecs_np[:, order])).float()
    evecs = _l2_normalize_eigenvectors(evecs)

    if num_nodes < max_freqs:
        pad = max_freqs - num_nodes
        eigvecs = F.pad(evecs, (0, pad), value=float("nan"))
        eigvals = F.pad(evals, (0, pad), value=float("nan"))
    else:
        eigvecs = evecs
        eigvals = evals

    data.EigVecs = eigvecs
    data.EigVals = eigvals.unsqueeze(0).repeat(num_nodes, 1).unsqueeze(2)
    return data


def _compute_rwse(data, steps: tuple[int, ...] = tuple(range(1, 21))):
    num_nodes = int(data.num_nodes)
    edge_index = data.edge_index
    edge_weight = torch.ones(edge_index.size(1), dtype=torch.float)

    source = edge_index[0]
    degree = scatter(
        edge_weight,
        source,
        dim=0,
        dim_size=num_nodes,
        reduce="sum",
    )
    degree_inv = degree.pow(-1.0)
    degree_inv.masked_fill_(degree_inv == float("inf"), 0)

    if edge_index.numel() == 0:
        transition = torch.zeros((1, num_nodes, num_nodes), dtype=torch.float)
    else:
        adjacency = to_dense_adj(edge_index, max_num_nodes=num_nodes).float()
        transition = torch.diag(degree_inv) @ adjacency

    min_step = min(steps)
    max_step = max(steps)
    if list(steps) != list(range(min_step, max_step + 1)):
        diagonals = [
            torch.diagonal(transition.matrix_power(step), dim1=-2, dim2=-1)
            for step in steps
        ]
    else:
        power = transition.clone().detach().matrix_power(min_step)
        diagonals = []
        for _step in range(min_step, max_step + 1):
            diagonals.append(torch.diagonal(power, dim1=-2, dim2=-1))
            power = power @ transition

    data.pestat_RWSE = torch.cat(diagonals, dim=0).transpose(0, 1).contiguous()
    return data


@dataclass
class PeptidesPositionEncodingTransform(BaseTransform):
    """Precompute the exact PE statistics required by one tuned Peptides task."""

    mode: str
    max_freqs: int = 10

    def __post_init__(self):
        self.mode = str(self.mode).strip().lower()
        if self.mode not in VALID_PE_MODES:
            raise ValueError(
                f"Invalid mode={self.mode!r}; expected one of {sorted(VALID_PE_MODES)}."
            )
        if int(self.max_freqs) != 10:
            raise ValueError("The tuned Peptides LapPE configuration requires max_freqs=10.")

    def __call__(self, data):
        if self.mode == "rwse":
            return _compute_rwse(data)
        return _compute_lappe(data, max_freqs=self.max_freqs)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(mode={self.mode!r}, "
            f"max_freqs={self.max_freqs})"
        )