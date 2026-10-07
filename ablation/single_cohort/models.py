from __future__ import annotations

import torch
from torch import nn

from ICIMTL_model import TaskSpecificHead, sharedEncoder

class SingleCohortMMoE(nn.Module):
    def __init__(
        self,
        shared_input_dim: int,
        specific_dim: int,
        expert_hidden_units: list[int],
        num_experts: int,
        l2_reg: float,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.shared_encoder = sharedEncoder(
            input_dim=shared_input_dim,
            expert_dnn_hidden_units=expert_hidden_units,
            num_experts=num_experts,
            gate_dnn_hidden_units=[16],
            num_tasks=1,
            l2_reg=l2_reg,
            dnn_activation="relu",
            dropout_rate=0.4,
            use_bn=True,
            init_std=0.001,
            device=device,
        )
        self.task_head = TaskSpecificHead(
            shared_dim=expert_hidden_units[-1],
            cohort_feature_dim=specific_dim,
            fusion_dim=64,
            device=device,
        )
        self.to(device)

    def forward(self, shared_input: torch.Tensor, specific_input: torch.Tensor):
        shared_outputs, regularization_loss = self.shared_encoder(shared_input)
        logits = self.task_head(shared_outputs[0], specific_input).squeeze(-1)
        return logits, regularization_loss

class SingleCohortDNN(nn.Module):
    def __init__(self, shared_input_dim: int, specific_dim: int) -> None:
        super().__init__()
        input_dim = shared_input_dim + specific_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(16, 1),
        )
    def forward(self, shared_input: torch.Tensor, specific_input: torch.Tensor):
        logits = self.network(torch.cat([shared_input, specific_input], dim=1)).squeeze(-1)
        return logits, logits.new_zeros(())
