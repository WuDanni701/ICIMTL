"""Independent single-cohort cross-validation for MMoE and DNN models.

Examples from the repository root:
    python ablation/single_cohort/train.py --model mmoe --cohort all
    python ablation/single_cohort/train.py --model dnn --cohort DavidA
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from torch import nn
from torch.utils.data import DataLoader


ABLATION_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ABLATION_DIR.parent
ICIMTL_DIR = REPOSITORY_ROOT / "ICIMTL"
for path in (ABLATION_DIR, ICIMTL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiment_common import (  # noqa: E402
    COHORTS,
    SHARED_COLUMNS,
    TASK_CATEGORICAL_GROUPS,
    TASK_CONTINUOUS_COLUMNS,
    TASK_SPECIFIC_COLUMNS,
    batch_size,
    expert_dnn_hidden_units,
    gene_dropout,
    hidden_dims_list,
    l2_reg,
    latent_dim,
    learning_rate_train,
    load_pretraining,
    logvar_max,
    logvar_min,
    noise_std,
    num_experts,
)
from autoencoder import Autoencoder  # noqa: E402
from ICIdataset import ICIDataset, ICIProcessor  # noqa: E402
from models import SingleCohortDNN, SingleCohortMMoE  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independent single-cohort ablation")
    parser.add_argument("--model", choices=("mmoe", "dnn"), default="mmoe")
    parser.add_argument("--cohort", choices=(*COHORTS, "all"), default="all")
    parser.add_argument("--data-dir", type=Path, default=REPOSITORY_ROOT / "icidata")
    parser.add_argument("--pretraining-dir", type=Path, default=REPOSITORY_ROOT / "TCGA_pretraining")
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY_ROOT / "outputs" / "single_cohort")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=batch_size)
    parser.add_argument("--learning-rate", type=float, default=learning_rate_train)
    parser.add_argument("--seed", type=int, default=269)
    parser.add_argument(
        "--use-tcga-experts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Transfer TCGA experts into the MMoE model; disabled for the strict baseline.",
    )
    parser.add_argument(
        "--freeze-experts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze transferred experts (only relevant with --use-tcga-experts).",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
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
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(value)


def load_one_cohort(
    cohort: str,
    data_dir: Path,
    autoencoder: Autoencoder,
    selected_genes: list[str],
    cohort_scalers,
    device: torch.device,
    encoding_batch_size: int,
):
    """Load one cohort only; no sample from another cohort enters this run."""
    cohort_index = COHORTS.index(cohort)
    clinical_path = data_dir / "clinical" / f"{cohort}_clinical.csv"
    rna_path = data_dir / "rna" / f"{cohort}_rna.csv"
    for path in (clinical_path, rna_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    processor = ICIProcessor(
        [str(rna_path)],
        [str(clinical_path)],
        SHARED_COLUMNS,
        [TASK_SPECIFIC_COLUMNS[cohort_index]],
    )
    rna, shared, specific, response, task_ids, os_time, os_event, tmb = processor.load_all()
    if set(np.asarray(task_ids, dtype=int)) != {0}:
        raise RuntimeError("Single-cohort isolation failed: unexpected task id.")
    if cohort not in cohort_scalers or "scaler" not in cohort_scalers[cohort]:
        raise KeyError(f"No RNA scaler found for cohort {cohort}.")

    scaled = cohort_scalers[cohort]["scaler"].transform(rna)
    scaled = pd.DataFrame(scaled, index=rna.index, columns=rna.columns)
    expression = scaled.reindex(columns=selected_genes, fill_value=0.0).to_numpy(np.float32)
    latent_batches = []
    autoencoder.eval()
    with torch.no_grad():
        for start in range(0, len(expression), encoding_batch_size):
            batch = torch.as_tensor(expression[start:start + encoding_batch_size], device=device)
            latent_batches.append(autoencoder.encoder(batch).cpu().numpy())
    encoded = pd.DataFrame(
        np.concatenate(latent_batches),
        index=rna.index,
        columns=[f"z{i + 1}" for i in range(latent_dim)],
    )
    return encoded, shared, specific, response, task_ids, os_time, os_event, tmb


def preprocess_fold(shared, specific, cohort: str, train_idx, val_idx):
    """Fit imputation and scaling on the current cohort's training fold only."""
    cohort_index = COHORTS.index(cohort)
    shared_out, specific_out = shared.copy(), specific.copy()

    imputer, scaler = SimpleImputer(strategy="median"), RobustScaler()
    train_tmb = imputer.fit_transform(shared.iloc[train_idx][["TMB"]])
    tmb_position = shared.columns.get_loc("TMB")
    shared_out.iloc[train_idx, tmb_position] = scaler.fit_transform(train_tmb).ravel()
    shared_out.iloc[val_idx, tmb_position] = scaler.transform(
        imputer.transform(shared.iloc[val_idx][["TMB"]])
    ).ravel()

    groups = TASK_CATEGORICAL_GROUPS.get(cohort_index, {})
    for names in groups.values():
        columns = [f"t0__{name}" for name in names]
        train_group = specific.iloc[train_idx][columns]
        train_missing = train_group.isna().all(axis=1)
        observed = train_group.loc[~train_missing]
        if observed.empty:
            raise ValueError(f"No observed category in {cohort}: {names}")
        mode = np.zeros(len(columns), dtype=float)
        mode[columns.index(observed.sum(axis=0).idxmax())] = 1.0
        positions = specific_out.columns.get_indexer(columns)
        specific_out.iloc[np.asarray(train_idx)[train_missing.to_numpy()], positions] = mode
        val_missing = specific.iloc[val_idx][columns].isna().all(axis=1)
        specific_out.iloc[np.asarray(val_idx)[val_missing.to_numpy()], positions] = mode

    continuous = [f"t0__{name}" for name in TASK_CONTINUOUS_COLUMNS[cohort_index]]
    imputer, scaler = SimpleImputer(strategy="median"), RobustScaler()
    values = imputer.fit_transform(specific.iloc[train_idx][continuous])
    positions = specific_out.columns.get_indexer(continuous)
    specific_out.iloc[train_idx, positions] = scaler.fit_transform(values)
    specific_out.iloc[val_idx, positions] = scaler.transform(
        imputer.transform(specific.iloc[val_idx][continuous])
    )

    for frame in (shared_out.iloc[train_idx], shared_out.iloc[val_idx], specific_out.iloc[train_idx], specific_out.iloc[val_idx]):
        if frame.isna().any().any() or not np.isfinite(frame.to_numpy(float)).all():
            raise ValueError(f"NaN or non-finite value remains after preprocessing {cohort}.")
    return shared_out, specific_out


def make_dataset(data, indices):
    encoded, shared, specific, response, task_ids, os_time, os_event, tmb = data
    return ICIDataset(
        encoded.iloc[indices], shared.iloc[indices], specific.iloc[indices],
        response.iloc[indices], task_ids.iloc[indices], os_time.iloc[indices],
        os_event.iloc[indices], tmb.iloc[indices], [specific.shape[1]],
    )


def transfer_experts(model: SingleCohortMMoE, checkpoint: Path, device: torch.device) -> int:
    source = torch.load(checkpoint, map_location=device, weights_only=True)
    target = model.shared_encoder.state_dict()
    copied = 0
    for name, value in source.items():
        if name.startswith("expert_dnn.") and name in target and target[name].shape == value.shape:
            target[name] = value.clone()
            copied += 1
    if copied == 0:
        raise RuntimeError(f"No compatible expert tensors found in {checkpoint}.")
    model.shared_encoder.load_state_dict(target, strict=False)
    return copied


def calculate_metrics(labels, probabilities) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= 0.5).astype(int)
    result = {
        "auc": float(roc_auc_score(labels, probabilities)) if np.unique(labels).size == 2 else np.nan,
        "auprc": float(average_precision_score(labels, probabilities)) if np.unique(labels).size == 2 else np.nan,
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
    }
    if np.unique(labels).size == 2:
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        result.update({
            "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
            "specificity": tn / (tn + fp) if tn + fp else np.nan,
        })
    return result


def run_epoch(model, loader, device, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, probabilities, labels = 0.0, [], []
    for batch in loader:
        if training:
            optimizer.zero_grad()
        shared_input = torch.cat([batch["rna"], batch["shared"]], dim=1).to(device)
        specific_input = batch["specific"].to(device)
        target = batch["label"].float().to(device)
        with torch.set_grad_enabled(training):
            logits, regularization_loss = model(shared_input, specific_input)
            loss = criterion(logits, target) + 0.02 * regularization_loss
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        total_loss += float(loss.item())
        probabilities.extend(torch.sigmoid(logits).detach().cpu().numpy())
        labels.extend(target.detach().cpu().numpy())
    probabilities = np.asarray(probabilities)
    return total_loss / max(len(loader), 1), calculate_metrics(labels, probabilities), probabilities


def run_cohort(cohort, args, artifacts, device):
    selected_genes, cohort_scalers, encoder_state, expert_checkpoint = artifacts
    autoencoder = Autoencoder(
        input_dim=len(selected_genes), hidden_dims=hidden_dims_list,
        latent_dim=latent_dim, dropout=0.4, noise_std=noise_std,
        gene_dropout=gene_dropout, logvar_min=logvar_min, logvar_max=logvar_max,
    ).to(device)
    autoencoder.load_state_dict(encoder_state, strict=False)
    data = load_one_cohort(
        cohort, args.data_dir, autoencoder, selected_genes, cohort_scalers,
        device, args.batch_size,
    )
    labels = np.asarray(data[3], dtype=int)
    class_counts = np.bincount(labels, minlength=2)
    n_splits = min(args.folds, int(class_counts.min()))
    if n_splits < 2:
        raise ValueError(f"{cohort} has insufficient samples for stratified CV: {class_counts}")

    output_dir = args.output_dir / args.model / cohort
    output_dir.mkdir(parents=True, exist_ok=True)
    splitter = StratifiedKFold(n_splits, shuffle=True, random_state=args.seed)
    records, metric_rows = [], []

    for fold, (train_idx, val_idx) in enumerate(splitter.split(labels, labels), start=1):
        seed_everything(args.seed + fold)
        fold_shared, fold_specific = preprocess_fold(data[1], data[2], cohort, train_idx, val_idx)
        fold_data = (data[0], fold_shared, fold_specific, *data[3:])
        train_set, val_set = make_dataset(fold_data, train_idx), make_dataset(fold_data, val_idx)
        train_batch_size = min(args.batch_size, len(train_set))
        drop_last = len(train_set) > train_batch_size and len(train_set) % train_batch_size == 1
        generator = torch.Generator().manual_seed(args.seed + fold)
        train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, drop_last=drop_last, generator=generator)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

        shared_dim = data[0].shape[1] + data[1].shape[1]
        if args.model == "mmoe":
            model = SingleCohortMMoE(
                shared_dim, data[2].shape[1], expert_dnn_hidden_units,
                num_experts, l2_reg, device,
            ).to(device)
            copied = transfer_experts(model, expert_checkpoint, device) if args.use_tcga_experts else 0
            if args.use_tcga_experts and args.freeze_experts:
                for parameter in model.shared_encoder.expert_dnn.parameters():
                    parameter.requires_grad = False
        else:
            model = SingleCohortDNN(shared_dim, data[2].shape[1]).to(device)
            copied = 0

        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=args.learning_rate, weight_decay=2e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        positives = int(labels[train_idx].sum())
        negatives = len(train_idx) - positives
        weight = max(1.0, min(8.0, negatives / max(positives, 1)))
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight], device=device))
        best_auc, remaining_patience = -np.inf, args.patience
        best_path = output_dir / f"best_fold_{fold}.pth"

        for epoch in range(1, args.epochs + 1):
            train_loss, _, _ = run_epoch(model, train_loader, device, criterion, optimizer)
            val_loss, val_metrics, _ = run_epoch(model, val_loader, device, criterion)
            score = float(np.nan_to_num(val_metrics["auc"], nan=0.0))
            if score > best_auc + 1e-4:
                best_auc, remaining_patience = score, args.patience
                torch.save(model.state_dict(), best_path)
            else:
                remaining_patience -= 1
            scheduler.step()
            if epoch == 1 or epoch % 10 == 0:
                print(f"{cohort} fold={fold} epoch={epoch} train={train_loss:.4f} val={val_loss:.4f} AUC={score:.4f}")
            if remaining_patience <= 0:
                break

        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        _, metrics, probabilities = run_epoch(model, val_loader, device, criterion)
        metric_rows.append({
            "cohort": cohort, "fold": fold, "n_train": len(train_idx),
            "n_validation": len(val_idx), "expert_tensors_loaded": copied, **metrics,
        })
        for position, global_index in enumerate(val_idx):
            records.append({
                "fold": fold, "cohort": cohort,
                "patient_id": str(data[0].index[global_index]),
                "label": int(data[3].iloc[global_index]),
                "probability": float(probabilities[position]),
            })

    pd.DataFrame(records).to_csv(output_dir / "out_of_fold_predictions.csv", index=False)
    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(output_dir / "fold_metrics.csv", index=False)
    return pd.DataFrame(records), metrics_frame


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    if args.model == "dnn" and args.use_tcga_experts:
        raise ValueError("--use-tcga-experts is only valid with --model mmoe.")
    artifacts = load_pretraining(args.pretraining_dir)
    cohorts = list(COHORTS) if args.cohort == "all" else [args.cohort]
    all_predictions, all_metrics = [], []
    for cohort in cohorts:
        predictions, metrics = run_cohort(cohort, args, artifacts, device)
        all_predictions.append(predictions)
        all_metrics.append(metrics)

    combined_dir = args.output_dir / args.model
    combined_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(all_predictions, ignore_index=True).to_csv(combined_dir / "all_predictions.csv", index=False)
    pd.concat(all_metrics, ignore_index=True).to_csv(combined_dir / "all_fold_metrics.csv", index=False)
    config = vars(args).copy()
    config.update({key: str(value) for key, value in config.items() if isinstance(value, Path)})
    with (combined_dir / "experiment_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
