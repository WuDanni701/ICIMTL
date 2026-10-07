"""Feature masks and runner shared by the two omics ablations."""

from __future__ import annotations

import sys
from pathlib import Path

ABLATION_DIR = Path(__file__).resolve().parents[1]
if str(ABLATION_DIR) not in sys.path:
    sys.path.insert(0, str(ABLATION_DIR))

from experiment_common import (COHORTS, expert_dnn_hidden_units, gate_dnn_hidden_units, l2_reg,
                               num_experts, build_parser, run_experiment)
from ICIMTL_model import ICIMTL


LUNDTAX_COLUMNS = {
    f"t3__{name}" for name in (
        "Lund_MS1a", "Lund_MS1b", "Lund_MS2a1", "Lund_MS2a2",
        "Lund_MS2b1", "Lund_MS2b2.1", "Lund_MS2b2.2",
    )
}
GENOMIC_COLUMNS = {
    f"t{task_id}__{name}"
    for task_id, names in {
        0: ["FS_Counts", "NEO_Weak", "ITH", "homozygous_ANY_0", "homozygous_ANY_1",
            "Deletion_9p21.3_MUT", "Deletion_9p21.3_WT", "Deletion_11q23.1_MUT",
            "Deletion_11q23.1_WT", "Amplification_12q24.32_MUT",
            "Amplification_12q24.32_WT", "Amplification_12q24.32_WUT",
            "Amplification_6q21_MUT", "Amplification_6q21_WT"],
        1: ["CNA_prop", "heterogeneity", "Purity", "TMB_clonal", "TMB_subclonal"],
        2: ["Purity", "Neoantigens", "TMB_clonal", "TMB_subclonal"],
        3: ["Neoantigen_burden"],
    }.items()
    for name in names
}


def rna_only(encoded_rna, shared, specific):
    """Keep RNA embeddings and the IMvigor210 LundTax transcriptomic subtype."""
    encoded_rna, shared, specific = encoded_rna.copy(), shared.copy(), specific.copy()
    shared.loc[:, :] = 0.0
    missing = LUNDTAX_COLUMNS.difference(specific.columns)
    if missing:
        raise KeyError(f"Missing LundTax columns: {sorted(missing)}")
    specific.loc[:, [c for c in specific.columns if c not in LUNDTAX_COLUMNS]] = 0.0
    return encoded_rna, shared, specific


def genomic_only(encoded_rna, shared, specific):
    """Keep TMB and cohort genomic variables; remove RNA and clinical variables."""
    encoded_rna, shared, specific = encoded_rna.copy(), shared.copy(), specific.copy()
    encoded_rna.loc[:, :] = 0.0
    shared.loc[:, [column for column in shared.columns if column != "TMB"]] = 0.0
    missing = GENOMIC_COLUMNS.difference(specific.columns)
    if missing:
        raise KeyError(f"Missing genomic columns: {sorted(missing)}")
    specific.loc[:, [c for c in specific.columns if c not in GENOMIC_COLUMNS]] = 0.0
    return encoded_rna, shared, specific


def model_factory(input_dim, task_dims, device):
    return ICIMTL(
        shared_input_dim=input_dim, expert_dnn_hidden_units=expert_dnn_hidden_units,
        num_experts=num_experts, gate_dnn_hidden_units=gate_dnn_hidden_units,
        num_tasks=len(COHORTS), task_feature_dims=task_dims, fusion_dim=128,
        tower_dnn_hidden_units=[64, 32], l2_reg=l2_reg, dnn_activation="relu",
        dropout_rate=0.4, use_bn=True, init_std=0.001, seed=269, device=device,
    )


def run(mode: str) -> None:
    if mode == "rna":
        parser = build_parser("RNA-only MTL ablation", "ablation_rna_only")
        run_experiment(parser.parse_args(), model_factory, rna_only, "RNA-only MTL ablation")
    elif mode == "genomic":
        parser = build_parser("Genomic-features-only MTL ablation", "ablation_genomic_only")
        run_experiment(parser.parse_args(), model_factory, genomic_only, "Genomic-only MTL ablation")
    else:
        raise ValueError(f"Unknown omic ablation: {mode}")
