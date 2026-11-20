"""Low-rank (truncated SVD) factorization of the MLP layers in GPT-2 blocks.

Each MLP projection (c_fc and c_proj) is replaced by two smaller layers whose
product approximates the original weight at a chosen rank.
"""

import torch
from torch import nn
from transformers.pytorch_utils import Conv1D

MLP_LAYERS = ("c_fc", "c_proj")


class LowRankLinear(nn.Module):
    """Two stacked linear maps, in -> rank -> out, standing in for one layer."""

    def __init__(self, first: nn.Linear, second: nn.Linear):
        super().__init__()
        self.first = first
        self.second = second

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.second(self.first(x))


def rank_for(rank_ratio: float, out_features: int, in_features: int) -> int:
    """Rank as a fraction of the largest possible rank, min(out, in)."""
    return max(1, int(round(rank_ratio * min(out_features, in_features))))


@torch.no_grad()
def factorize(layer: Conv1D, rank_ratio: float) -> LowRankLinear:
    """Replace one projection with its truncated-SVD approximation."""
    # Work in nn.Linear layout: weight is (out_features, in_features).
    out_features, in_features = layer.nf, layer.nx
    weight = layer.weight.reshape(out_features, in_features)

    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
    rank = rank_for(rank_ratio, out_features, in_features)

    first = nn.Linear(in_features, rank, bias=False)
    second = nn.Linear(rank, out_features, bias=True)
    first.weight.copy_(Vh[:rank])
    second.weight.copy_(U[:, :rank] * S[:rank])
    second.bias.copy_(layer.bias)
    return LowRankLinear(first, second)


def compress_model(model: nn.Module, rank_ratio: float) -> nn.Module:
    """Factorize c_fc and c_proj in every transformer block at rank_ratio."""
    for block in model.transformer.h:
        for name in MLP_LAYERS:
            layer = getattr(block.mlp, name)
            if isinstance(layer, LowRankLinear):
                continue  # already factorized
            setattr(block.mlp, name, factorize(layer, rank_ratio))
    return model


def block_params(model: nn.Module) -> int:
    """Parameters in the transformer blocks. Embeddings are not counted."""
    total = 0
    for block in model.transformer.h:
        for module in block.modules():
            if isinstance(module, (Conv1D, nn.LayerNorm)):
                total += module.weight.numel() + module.bias.numel()
            elif isinstance(module, LowRankLinear):
                # A rank-r approximation keeps r of the min(out, in) directions.
                rank = module.first.out_features
                total += rank * max(module.first.in_features, module.second.out_features)
    return total
