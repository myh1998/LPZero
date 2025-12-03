import math
from copy import deepcopy
from typing import Dict, Iterable

import torch
import torch.nn as nn
import transformers


class LoRAWrappedModule(nn.Module):
    """Lightweight LoRA wrapper that keeps the original module intact.

    The wrapper computes: base(x) + scaling * (x @ A^T @ B^T)
    and exposes the LoRA matrices so that zero-cost measures can pick up
    their weights and gradients.
    """

    def __init__(
        self,
        base_module: nn.Module,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.base_module = base_module
        if isinstance(base_module, transformers.Conv1D):
            in_features = base_module.weight.shape[0]
            out_features = base_module.weight.shape[1]
        else:
            if not isinstance(base_module, nn.Linear):
                raise TypeError("LoRA wrapper only supports Linear/Conv1D modules")
            in_features = base_module.in_features
            out_features = base_module.out_features

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / float(rank)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_module(x)
        lora_out = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return base_out + self.scaling * lora_out


def _iter_named_modules(model: nn.Module, targets: Iterable[str]):
    for name, module in model.named_modules():
        if any(name.endswith(target) for target in targets):
            yield name, module


def apply_lora_to_model(model: nn.Module, lora_config: Dict) -> nn.Module:
    """Attach LoRA modules to the given model in-place.

    Args:
        model: Base model instance.
        lora_config: Dictionary with the following optional keys:
            - rank: low-rank dimension (int)
            - alpha: scaling factor (float)
            - target_modules: list of module name suffixes to wrap (default: ["c_attn", "c_proj"])
            - dropout: dropout rate applied to LoRA branch

    Returns:
        The model instance with LoRA adapters attached.
    """

    cfg = deepcopy(lora_config) if lora_config else {}
    rank = int(cfg.pop("rank", 4))
    alpha = float(cfg.pop("alpha", 16))
    dropout = float(cfg.pop("dropout", 0.0))
    target_modules = cfg.pop("target_modules", ["c_attn", "c_proj"])

    for name, module in _iter_named_modules(model, target_modules):
        parent = model
        path_parts = name.split(".")
        for part in path_parts[:-1]:
            parent = getattr(parent, part)
        last_part = path_parts[-1]
        wrapped = LoRAWrappedModule(module, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, last_part, wrapped)

    return model


__all__ = ["apply_lora_to_model", "LoRAWrappedModule"]
