from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ICIMTL_DIR = REPOSITORY_ROOT / "ICIMTL"
if str(ICIMTL_DIR) not in sys.path:
    sys.path.insert(0, str(ICIMTL_DIR))

from autoencoder import Autoencoder  # noqa: E402
from config import (  # noqa: E402
    SEED,
    batch_size,
    expert_dnn_hidden_units,
    gate_dnn_hidden_units,
    gene_dropout,
    hidden_dims_list,
    l2_reg,
    latent_dim,
    learning_rate_train,
    logvar_max,
    logvar_min,
    noise_std,
    num_experts,
)
from ICIdataset import ICIDataset, ICIProcessor  # noqa: E402
from utils1 import EarlyStopping, ICITransferTrainer  # noqa: E402


COHORTS = ("DavidA", "DavidLiu", "Ravi", "IMvigor210")
SHARED_COLUMNS = ["Sex_F", "Sex_M", "TMB"]
TASK_SPECIFIC_COLUMNS = [
    [
        "Age", "FS_Counts", "NEO_Weak", "ITH", "MSKCC_FAVORABLE",
        "MSKCC_INTERMEDIATE", "MSKCC_POOR", "homozygous_ANY_0",
        "homozygous_ANY_1", "Deletion_9p21.3_MUT", "Deletion_9p21.3_WT",
        "Deletion_11q23.1_MUT", "Deletion_11q23.1_WT",
        "Amplification_12q24.32_MUT", "Amplification_12q24.32_WT",
        "Amplification_12q24.32_WUT", "Amplification_6q21_MUT",
        "Amplification_6q21_WT",
    ],
    [
        "CNA_prop", "heterogeneity", "Purity", "MHC-II", "ECOG_0",
        "ECOG_1", "ECOG_2", "ECOG_3", "Primary_Type_binary_0",
        "Primary_Type_binary_1", "TMB_clonal", "TMB_subclonal",
    ],
    [
        "Purity", "Smoking_Pack_Years", "Neoantigens", "Smoking_Status_0",
        "Smoking_Status_1", "Smoking_Status_2", "Histology_Harmonized_Adeno",
        "Histology_Harmonized_LC-NE", "Histology_Harmonized_Other",
        "Histology_Harmonized_Squamous", "Prior_Platinum_0",
        "Prior_Platinum_1", "TMB_clonal", "TMB_subclonal",
    ],
    [
        "Neoantigen_burden", "ECOG_0", "ECOG_1", "ECOG_2",
        "Smoking_Status_CURRENT", "Smoking_Status_NEVER",
        "Smoking_Status_PREVIOUS", "IC.Level_IC0", "IC.Level_IC1",
        "IC.Level_IC2+", "TC.Level_TC0", "TC.Level_TC1", "TC.Level_TC2+",
        "Lund_MS1a", "Lund_MS1b", "Lund_MS2a1", "Lund_MS2a2",
        "Lund_MS2b1", "Lund_MS2b2.1", "Lund_MS2b2.2", "Neo_missing_0",
        "Neo_missing_1",
    ],
]

TASK_CONTINUOUS_COLUMNS = {
    0: ["Age", "FS_Counts", "NEO_Weak", "ITH"],
    1: ["CNA_prop", "heterogeneity", "Purity", "MHC-II", "TMB_clonal", "TMB_subclonal"],
    2: ["Purity", "Smoking_Pack_Years", "Neoantigens", "TMB_clonal", "TMB_subclonal"],
    3: ["Neoantigen_burden"],
}
TASK_CATEGORICAL_GROUPS = {
    0: {
        "MSKCC": ["MSKCC_FAVORABLE", "MSKCC_INTERMEDIATE", "MSKCC_POOR"],
        "Deletion_9p21.3": ["Deletion_9p21.3_MUT", "Deletion_9p21.3_WT"],
        "Deletion_11q23.1": ["Deletion_11q23.1_MUT", "Deletion_11q23.1_WT"],
        "Amplification_12q24.32": ["Amplification_12q24.32_MUT", "Amplification_12q24.32_WT", "Amplification_12q24.32_WUT"],
        "Amplification_6q21": ["Amplification_6q21_MUT", "Amplification_6q21_WT"],
    },
    1: {"ECOG": ["ECOG_0", "ECOG_1", "ECOG_2", "ECOG_3"]},
}


def build_parser(description: str, output_name: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-dir", type=Path, default=REPOSITORY_ROOT / "icidata")
    parser.add_argument("--pretraining-dir", type=Path, default=REPOSITORY_ROOT / "TCGA_pretraining")
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY_ROOT / "outputs" / output_name)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=batch_size)
    parser.add_argument("--learning-rate", type=float, default=learning_rate_train)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--train-shared", action="store_true", help="Train shared experts instead of freezing transferred experts.")
    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(value)


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required files are missing:\n  " + "\n  ".join(missing))


def load_pretraining(pretraining_dir: Path):
    genes_path = pretraining_dir / "selected_genes.json"
    scaler_path = pretraining_dir / "std_scaler.joblib"
    encoder_path = pretraining_dir / "encoder.pth"
    shared_path = pretraining_dir / "shared_encoder.pth"
    require_files([genes_path, scaler_path, encoder_path, shared_path])
    with genes_path.open("r", encoding="utf-8") as handle:
        selected_genes = json.load(handle)
    return (
        selected_genes,
        joblib.load(scaler_path),
        torch.load(encoder_path, map_location="cpu", weights_only=True),
        shared_path,
    )


def load_data(data_dir: Path, pretraining_dir: Path, device: torch.device, encoding_batch_size: int):
    clinical_paths = [data_dir / "clinical" / f"{name}_clinical.csv" for name in COHORTS]
    rna_paths = [data_dir / "rna" / f"{name}_rna.csv" for name in COHORTS]
    require_files(clinical_paths + rna_paths)
    selected_genes, cohort_scalers, encoder_state, shared_path = load_pretraining(pretraining_dir)

    processor = ICIProcessor(
        [str(path) for path in rna_paths],
        [str(path) for path in clinical_paths],
        SHARED_COLUMNS,
        TASK_SPECIFIC_COLUMNS,
    )
    _, shared, specific, response, task_ids, os_time, os_event, tmb_original = processor.load_all()

    autoencoder = Autoencoder(
        input_dim=len(selected_genes), hidden_dims=hidden_dims_list,
        latent_dim=latent_dim, dropout=0.4, noise_std=noise_std,
        gene_dropout=gene_dropout, logvar_min=logvar_min, logvar_max=logvar_max,
    ).to(device)
    autoencoder.load_state_dict(encoder_state, strict=False)
    autoencoder.eval()

    latent_frames = []
    with torch.no_grad():
        for cohort, rna_path, clinical_path in zip(COHORTS, rna_paths, clinical_paths):
            rna = pd.read_csv(rna_path, index_col=0)
            clinical = pd.read_csv(clinical_path, index_col=0)
            rna.columns = rna.columns.astype(str).str.strip()
            common = clinical.index.intersection(rna.index).sort_values()
            rna = rna.loc[common]
            if cohort not in cohort_scalers or "scaler" not in cohort_scalers[cohort]:
                raise KeyError(f"No RNA scaler found for cohort {cohort}.")
            scaled = cohort_scalers[cohort]["scaler"].transform(rna)
            scaled = pd.DataFrame(scaled, index=rna.index, columns=rna.columns)
            expression = scaled.reindex(columns=selected_genes, fill_value=0.0).to_numpy(np.float32)
            chunks = []
            for start in range(0, len(expression), encoding_batch_size):
                batch = torch.as_tensor(expression[start:start + encoding_batch_size], device=device)
                chunks.append(autoencoder.encoder(batch).cpu().numpy())
            latent_frames.append(pd.DataFrame(np.concatenate(chunks), index=common))

    encoded_rna = pd.concat(latent_frames, axis=0)
    encoded_rna.columns = [f"z{i + 1}" for i in range(encoded_rna.shape[1])]
    if not encoded_rna.index.equals(shared.index):
        raise ValueError("RNA and clinical sample orders do not match.")
    return encoded_rna, shared, specific, response, task_ids, os_time, os_event, tmb_original, shared_path


def preprocess_fold(shared, specific, task_ids, train_idx, val_idx):
    shared_out, specific_out = shared.copy(), specific.copy()
    train_idx, val_idx = np.asarray(train_idx), np.asarray(val_idx)
    task_array = np.asarray(task_ids, dtype=int)

    imputer, scaler = SimpleImputer(strategy="median"), RobustScaler()
    train_values = imputer.fit_transform(shared.iloc[train_idx][["TMB"]])
    tmb_position = shared_out.columns.get_loc("TMB")
    shared_out.iloc[train_idx, tmb_position] = scaler.fit_transform(train_values).ravel()
    shared_out.iloc[val_idx, tmb_position] = scaler.transform(
        imputer.transform(shared.iloc[val_idx][["TMB"]])
    ).ravel()

    for task_id, groups in TASK_CATEGORICAL_GROUPS.items():
        task_train = train_idx[task_array[train_idx] == task_id]
        task_val = val_idx[task_array[val_idx] == task_id]
        for names in groups.values():
            columns = [f"t{task_id}__{name}" for name in names]
            train_group = specific.iloc[task_train][columns]
            train_missing = train_group.isna().all(axis=1)
            observed = train_group.loc[~train_missing]
            if observed.empty:
                raise ValueError(f"No observed training category for task {task_id}: {names}")
            mode_vector = np.zeros(len(columns), dtype=float)
            mode_vector[columns.index(observed.sum(axis=0).idxmax())] = 1.0
            positions = specific_out.columns.get_indexer(columns)
            specific_out.iloc[task_train[train_missing.to_numpy()], positions] = mode_vector
            if len(task_val):
                val_missing = specific.iloc[task_val][columns].isna().all(axis=1)
                specific_out.iloc[task_val[val_missing.to_numpy()], positions] = mode_vector

    for task_id, names in TASK_CONTINUOUS_COLUMNS.items():
        columns = [f"t{task_id}__{name}" for name in names]
        task_train = train_idx[task_array[train_idx] == task_id]
        task_val = val_idx[task_array[val_idx] == task_id]
        imputer, scaler = SimpleImputer(strategy="median"), RobustScaler()
        values = imputer.fit_transform(specific.iloc[task_train][columns])
        positions = specific_out.columns.get_indexer(columns)
        specific_out.iloc[task_train, positions] = scaler.fit_transform(values)
        if len(task_val):
            specific_out.iloc[task_val, positions] = scaler.transform(
                imputer.transform(specific.iloc[task_val][columns])
            )

    for frame in (shared_out.iloc[train_idx], shared_out.iloc[val_idx], specific_out.iloc[train_idx], specific_out.iloc[val_idx]):
        if frame.isna().any().any() or not np.isfinite(frame.to_numpy(float)).all():
            raise ValueError("NaN or non-finite value remains after fold preprocessing.")
    return shared_out, specific_out


def transfer_shared_encoder(model, checkpoint: Path, device: torch.device) -> tuple[int, int]:
    pretrained = torch.load(checkpoint, map_location=device, weights_only=True)
    current = model.shared_encoder.state_dict()
    gate_mapping = {0: [0], 3: [1], 2: [2], 4: [3]}
    experts, gates = 0, 0
    for name, value in pretrained.items():
        targets = []
        if name.startswith("expert_dnn."):
            targets = [name]
        elif name.startswith("gate_dnn.") or name.startswith("gate_dnn_final_layer."):
            match = re.match(r"(gate_dnn(?:_final_layer)?)\.(\d+)\.(.+)", name)
            if match and int(match.group(2)) in gate_mapping:
                targets = [f"{match.group(1)}.{target}.{match.group(3)}" for target in gate_mapping[int(match.group(2))]]
        for target in targets:
            if target in current and current[target].shape == value.shape:
                current[target] = value.clone()
                if name.startswith("expert_dnn."):
                    experts += 1
                else:
                    gates += 1
    model.shared_encoder.load_state_dict(current, strict=False)
    if experts == 0:
        raise RuntimeError(f"No compatible expert tensors found in {checkpoint}.")
    return experts, gates


def run_experiment(
    args: argparse.Namespace,
    model_factory: Callable,
    feature_ablation: Callable,
    experiment_name: str,
) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_data(args.data_dir, args.pretraining_dir, device, args.batch_size)
    encoded_rna, shared, specific, labels, task_ids, os_time, os_event, tmb_original, shared_path = data
    encoded_rna, shared, specific = feature_ablation(encoded_rna, shared, specific)

    y, tasks = np.asarray(labels, int), np.asarray(task_ids, int)
    stratification = np.asarray([f"t{task}_y{label}" for task, label in zip(tasks, y)])
    counts = pd.Series(stratification).value_counts()
    if counts.min() < args.folds:
        raise ValueError(f"Smallest cohort/label group has {counts.min()} samples; cannot form {args.folds} folds.")
    task_dims = [len(columns) for columns in TASK_SPECIFIC_COLUMNS]
    splitter = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
    metric_rows, prediction_rows = [], []

    for fold, (train_idx, val_idx) in enumerate(splitter.split(y, stratification), start=1):
        set_seed(args.seed + fold)
        fold_shared, fold_specific = preprocess_fold(shared, specific, task_ids, train_idx, val_idx)
        train_dataset = ICIDataset(
            encoded_rna.iloc[train_idx], fold_shared.iloc[train_idx], fold_specific.iloc[train_idx],
            labels.iloc[train_idx], task_ids.iloc[train_idx], os_time.iloc[train_idx],
            os_event.iloc[train_idx], tmb_original.iloc[train_idx], task_dims,
        )
        val_dataset = ICIDataset(
            encoded_rna.iloc[val_idx], fold_shared.iloc[val_idx], fold_specific.iloc[val_idx],
            labels.iloc[val_idx], task_ids.iloc[val_idx], os_time.iloc[val_idx],
            os_event.iloc[val_idx], tmb_original.iloc[val_idx], task_dims,
        )
        generator = torch.Generator().manual_seed(args.seed + fold)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        model = model_factory(encoded_rna.shape[1] + shared.shape[1], task_dims, device)
        copied_experts, copied_gates = transfer_shared_encoder(model, shared_path, device)
        print(f"Fold {fold}: transferred experts={copied_experts}, gates={copied_gates}")
        freeze_shared = not args.train_shared
        trainer = ICITransferTrainer(model, device, learning_rate=args.learning_rate, freeze_shared=freeze_shared)
        trainer.update_pos_weights_from_loader(train_loader)
        early_stop = EarlyStopping(args.patience, 1e-4, "auc", "max")
        best_path = args.output_dir / f"best_fold_{fold}.pth"

        for epoch in range(1, args.epochs + 1):
            train_loss, _ = trainer.train_epoch(train_loader)
            val_loss, metrics = trainer.evaluate(val_loader, return_loss=True)
            aucs = [value["auc"] for value in metrics.values() if "auc" in value]
            mean_auc = float(np.mean(aucs)) if aucs else 0.0
            if early_stop.step({"auc": mean_auc}, epoch):
                torch.save(model.state_dict(), best_path)
            elif early_stop.should_stop():
                break
            trainer.scheduler.step(epoch)
            if epoch == 1 or epoch % 10 == 0:
                print(f"Fold {fold} epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}, mean AUC={mean_auc:.4f}")

        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        _, metrics = trainer.evaluate(val_loader, return_loss=True)
        for task_id in range(len(COHORTS)):
            task_key = f"task_{task_id}"
            if task_key not in metrics:
                continue
            result = metrics[task_key]
            metric_rows.append({
                "fold": fold, "cohort": COHORTS[task_id],
                "auc": result["auc"], "auprc": result["auprc"],
                "accuracy": result["accuracy"], "f1": result["f1"],
            })
            task_val_idx = np.asarray(val_idx)[tasks[val_idx] == task_id]
            for global_idx, probability in zip(task_val_idx, result["preds"]):
                prediction_rows.append({
                    "fold": fold, "cohort": COHORTS[task_id],
                    "patient_id": str(encoded_rna.index[global_idx]),
                    "label": int(labels.iloc[global_idx]), "probability": float(probability),
                })

    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(args.output_dir / "out_of_fold_predictions.csv", index=False)
    print(f"\n{experiment_name}")
    print(metrics_frame.groupby("cohort")[["auc", "auprc", "accuracy", "f1"]].agg(["mean", "std"]))
