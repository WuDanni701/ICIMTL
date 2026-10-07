from __future__ import annotations
import torch
from torch import nn
from ICIMTL_model import sharedEncoder

class PooledPredictionHead(nn.Module):

    def __init__(self, shared_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(shared_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, shared_features: torch.Tensor) -> torch.Tensor:
        return self.classifier(shared_features)
class PooledICIMTL(nn.Module):

    def __init__(
        self,
        input_dim: int,
        expert_hidden_units: list[int],
        num_experts: int,
        gate_hidden_units: list[int],
        fusion_dim: int,
        l2_reg: float,
        dropout_rate: float,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.shared_encoder = sharedEncoder(
            input_dim=input_dim,
            expert_dnn_hidden_units=expert_hidden_units,
            num_experts=num_experts,
            gate_dnn_hidden_units=gate_hidden_units,
            num_tasks=1,
            l2_reg=l2_reg,
            dnn_activation="relu",
            dropout_rate=dropout_rate,
            use_bn=True,
            init_std=0.001,
            device=device,
        )
        self.prediction_head = PooledPredictionHead(
            shared_dim=expert_hidden_units[-1],
            hidden_dim=fusion_dim,
        )
        self.to(device)
    def forward(self, shared_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        task_features, regularization_loss = self.shared_encoder(shared_input)
        logits = self.prediction_head(task_features[0]).squeeze(-1)
        return logits, regularization_loss
