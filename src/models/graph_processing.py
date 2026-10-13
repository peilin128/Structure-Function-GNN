from __future__ import annotations

import torch
import torch.nn.functional as F


_TRIU_CACHE: dict[tuple[int, str], torch.Tensor] = {}


def _get_triu_indices(num_nodes: int, device: torch.device) -> torch.Tensor:
    key = (num_nodes, str(device))
    if key not in _TRIU_CACHE:
        _TRIU_CACHE[key] = torch.triu_indices(
            num_nodes,
            num_nodes,
            offset=1,
            device=device,
        )
    return _TRIU_CACHE[key]


def pgh_output_dim(num_interp_nodes: int) -> int:
    """Return the fixed PGH vector length for an interpolation size."""
    num_interp_nodes = int(num_interp_nodes)
    if num_interp_nodes < 2:
        raise ValueError("num_interp_nodes must be at least 2.")
    return num_interp_nodes * (num_interp_nodes - 1) // 2


def _pearson_matrix(node_features: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute the Pearson correlation matrix of one graph's node embeddings."""
    if node_features.ndim != 2:
        raise ValueError(
            "node_features must have shape [num_nodes, hidden_dim], "
            f"got {tuple(node_features.shape)}."
        )

    num_nodes, num_features = node_features.shape
    if num_features <= 1:
        return node_features.new_zeros((num_nodes, num_nodes))

    centered = node_features - node_features.mean(dim=1, keepdim=True)
    scale = centered.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
    normalized = centered / scale
    return normalized @ normalized.transpose(0, 1) / num_features


def _resample_1d(values: torch.Tensor, output_length: int) -> torch.Tensor:
    if output_length < 0:
        raise ValueError("output_length must be non-negative.")
    if output_length == 0:
        return values.new_empty((0,))
    if values.numel() == output_length:
        return values
    if values.numel() == 0:
        return values.new_zeros(output_length)
    if values.numel() == 1:
        return values.expand(output_length)

    return F.interpolate(
        values.view(1, 1, -1),
        size=output_length,
        mode="linear",
        align_corners=True,
    ).view(-1)


def _gumbel_rank_scores(
    edge_weights: torch.Tensor,
    tau: float,
    training: bool,
) -> torch.Tensor:
    """Use Gumbel-perturbed ranking in training and exact ranking in evaluation."""
    logits = torch.log(edge_weights.clamp_min(1e-6))
    if not training:
        return logits

    tau = max(float(tau), 1e-6)
    uniform = torch.rand_like(logits).clamp_(1e-10, 1.0 - 1e-10)
    gumbel_noise = -torch.log(-torch.log(uniform))
    return logits / tau + gumbel_noise


def _pgh_vector_from_edges(
    edge_weights: torch.Tensor,
    num_nodes: int,
    num_interp_nodes: int,
    tau: float,
    training: bool,
) -> torch.Tensor:
    """Partition ranked edges into birth and death sets and resample them."""
    output_length = pgh_output_dim(num_interp_nodes)
    birth_length = num_interp_nodes - 1
    death_length = output_length - birth_length

    if num_nodes <= 1 or edge_weights.numel() == 0:
        return edge_weights.new_zeros(output_length)

    rank_scores = _gumbel_rank_scores(edge_weights, tau=tau, training=training)
    rank_order = torch.argsort(rank_scores, descending=True)
    ranked_weights = edge_weights[rank_order]

    split_index = min(num_nodes - 1, ranked_weights.numel())
    birth_values = torch.sort(
        ranked_weights[:split_index], descending=True
    ).values
    death_values = torch.sort(
        ranked_weights[split_index:], descending=True
    ).values

    birth_vector = _resample_1d(birth_values, birth_length)
    death_vector = _resample_1d(death_values, death_length)
    return torch.cat([birth_vector, death_vector], dim=0)


def connectivity_module(
    node_embeddings: torch.Tensor,
    batch: torch.Tensor,
    num_interp_nodes: int,
    tau: float = 0.5,
    training: bool = False,
) -> torch.Tensor:
    """Compute one signed-correlation PGH vector per graph in a mini-batch.

    Pearson correlations are shifted from ``[-1, 1]`` to ``[0, 1]``. During
    training, edges are ranked with Gumbel perturbations. Validation and test
    evaluation use the corresponding deterministic ranking.
    """
    output_length = pgh_output_dim(num_interp_nodes)
    if node_embeddings.ndim != 2:
        raise ValueError(
            "node_embeddings must have shape [total_nodes, hidden_dim], "
            f"got {tuple(node_embeddings.shape)}."
        )
    if batch.ndim != 1 or batch.numel() != node_embeddings.size(0):
        raise ValueError(
            "batch must be one-dimensional and match the number of nodes."
        )
    if batch.numel() == 0:
        return node_embeddings.new_zeros((0, output_length))

    num_graphs = int(batch.max().item()) + 1
    counts = torch.bincount(batch, minlength=num_graphs).tolist()
    graph_embeddings = node_embeddings.split(counts, dim=0)
    graph_vectors: list[torch.Tensor] = []

    for graph_features in graph_embeddings:
        num_nodes = graph_features.size(0)
        if num_nodes <= 1:
            graph_vectors.append(graph_features.new_zeros(output_length))
            continue

        correlations = _pearson_matrix(graph_features)
        upper = _get_triu_indices(num_nodes, graph_features.device)
        signed_edges = correlations[upper[0], upper[1]].clamp(-1.0, 1.0)
        edge_weights = (signed_edges + 1.0) * 0.5

        graph_vectors.append(
            _pgh_vector_from_edges(
                edge_weights=edge_weights,
                num_nodes=num_nodes,
                num_interp_nodes=num_interp_nodes,
                tau=tau,
                training=training,
            )
        )

    return torch.stack(graph_vectors, dim=0)
