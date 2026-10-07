"""Train the only-MTL ablation from the repository root."""

from __future__ import annotations

import sys
from pathlib import Path

ABLATION_DIR = Path(__file__).resolve().parents[1]
if str(ABLATION_DIR) not in sys.path:
    sys.path.insert(0, str(ABLATION_DIR))

from experiment_common import (COHORTS, expert_dnn_hidden_units, gate_dnn_hidden_units, l2_reg,
                               num_experts, build_parser, run_experiment)
from model import OnlyMTL


def keep_shared_inputs(encoded_rna, shared, specific):
    """No masking is needed; the model deliberately ignores `specific`."""
    return encoded_rna, shared, specific


def model_factory(input_dim, task_dims, device):
    return OnlyMTL(
        input_dim=input_dim, expert_hidden_units=expert_dnn_hidden_units,
        num_experts=num_experts, gate_hidden_units=gate_dnn_hidden_units,
        num_tasks=len(COHORTS), fusion_dim=128, l2_reg=l2_reg,
        dropout_rate=0.4, device=device,
    )


if __name__ == "__main__":
    parser = build_parser("Only-MTL ablation without task-specific inputs", "ablation_only_mtl")
    run_experiment(parser.parse_args(), model_factory, keep_shared_inputs, "Only-MTL ablation")
