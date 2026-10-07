from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ICIMTL_DIR = REPOSITORY_ROOT / "ICIMTL"
if str(ICIMTL_DIR) not in sys.path:
    sys.path.insert(0, str(ICIMTL_DIR))

from autoencoder import Autoencoder
from config import (
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

try:
    from .model import PooledICIMTL
except ImportError:
    from model import PooledICIMTL


COHORTS = ("DavidA", "DavidLiu", "Ravi", "IMvigor210")
SHARED_COLUMNS = ("Sex_F", "Sex_M", "TMB")


class PooledDataset(Dataset):
    def __init__(
        self,
        encoded_rna: np.ndarray,
        shared_features: np.ndarray,
        labels: np.ndarray,
        cohort_ids: np.ndarray,
    ) -> None:
        self.rna = torch.as_tensor(encoded_rna, dtype=torch.float32)
        self.shared = torch.as_tensor(shared_features, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)
        self.cohort_ids = torch.as_tensor(cohort_ids, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "rna": self.rna[index],
            "shared": self.shared[index],
            "label": self.labels[index],
            "cohort_id": self.cohort_ids[index],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pooled no-cohort ablation")
    parser.add_argument("--data-dir", type=Path, default=REPOSITORY_ROOT / "icidata")
    parser.add_argument(
        "--pretraining-dir",
        type=Path,
        default=REPOSITORY_ROOT / "TCGA_pretraining",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "outputs" / "ablation_pooled",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=batch_size)
    parser.add_argument("--learning-rate", type=float, default=learning_rate_train)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--train-experts",
        action="store_true",
        help="Fine-tune experts instead of freezing TCGA-pretrained expert layers.",
    )
    return parser.parse_args()


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
    experts_path = pretraining_dir / "shared_encoder.pth"
    require_files([genes_path, scaler_path, encoder_path, experts_path])

    with genes_path.open("r", encoding="utf-8") as handle:
        selected_genes = json.load(handle)
    cohort_scalers = joblib.load(scaler_path)
    encoder_state = torch.load(encoder_path, map_location="cpu", weights_only=True)
    return selected_genes, cohort_scalers, encoder_state, experts_path


def load_and_encode_data(
    data_dir: Path,
    selected_genes: list[str],
    cohort_scalers,
    autoencoder: Autoencoder,
    device: torch.device,
    encoding_batch_size: int,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    encoded_inputs = []
    shared_frames = []
    labels = []
    cohort_ids = []
    sample_ids: list[str] = []

    for cohort_id, cohort in enumerate(COHORTS):
        rna_path = data_dir / "rna" / f"{cohort}_rna.csv"
        clinical_path = data_dir / "clinical" / f"{cohort}_clinical.csv"
        require_files([rna_path, clinical_path])

        rna = pd.read_csv(rna_path, index_col=0)
        clinical = pd.read_csv(clinical_path, index_col=0)
        rna.columns = rna.columns.astype(str).str.strip()
        clinical.columns = clinical.columns.astype(str).str.strip()

        common = clinical.index.intersection(rna.index).sort_values()
        if common.empty:
            raise ValueError(f"No matched RNA/clinical samples for {cohort}.")
        rna = rna.loc[common]
        clinical = clinical.loc[common]

        missing_columns = [name for name in (*SHARED_COLUMNS, "response") if name not in clinical]
        if missing_columns:
            raise ValueError(f"{cohort} clinical data lacks columns: {missing_columns}")
        if cohort not in cohort_scalers or "scaler" not in cohort_scalers[cohort]:
            raise KeyError(f"No RNA scaler found for cohort {cohort}.")

        scaler = cohort_scalers[cohort]["scaler"]
        scaled = scaler.transform(rna)
        scaled_rna = pd.DataFrame(scaled, index=rna.index, columns=rna.columns)
        model_input = scaled_rna.reindex(columns=selected_genes, fill_value=0.0)

        encoded_inputs.append(model_input.to_numpy(dtype=np.float32))
        shared_frames.append(clinical.loc[:, SHARED_COLUMNS].astype(float))
        labels.append(clinical["response"].to_numpy(dtype=np.int64))
        cohort_ids.append(np.full(len(common), cohort_id, dtype=np.int64))
        sample_ids.extend([f"{cohort}:{sample_id}" for sample_id in common.astype(str)])

    expression = np.concatenate(encoded_inputs, axis=0)
    autoencoder.eval()
    latent_batches = []
    with torch.no_grad():
        for start in range(0, len(expression), encoding_batch_size):
            batch = torch.as_tensor(
                expression[start : start + encoding_batch_size],
                dtype=torch.float32,
                device=device,
            )
            latent_batches.append(autoencoder.encoder(batch).cpu().numpy())

    return (
        np.concatenate(latent_batches, axis=0),
        pd.concat(shared_frames, axis=0),
        np.concatenate(labels),
        np.concatenate(cohort_ids),
        sample_ids,
    )


def preprocess_shared_fold(
    shared: pd.DataFrame,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit TMB imputation/scaling on the training fold only."""
    train = shared.iloc[train_indices].copy()
    validation = shared.iloc[validation_indices].copy()

    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    train_tmb = imputer.fit_transform(train[["TMB"]])
    train.loc[:, "TMB"] = scaler.fit_transform(train_tmb).ravel()
    validation.loc[:, "TMB"] = scaler.transform(
        imputer.transform(validation[["TMB"]])
    ).ravel()

    for name, frame in (("training", train), ("validation", validation)):
        values = frame.to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite shared feature remains in {name} fold.")
    return train.to_numpy(dtype=np.float32), validation.to_numpy(dtype=np.float32)


def copy_pretrained_experts(model: PooledICIMTL, checkpoint: Path, device: torch.device) -> int:
    pretrained = torch.load(checkpoint, map_location=device, weights_only=True)
    current = model.shared_encoder.state_dict()
    copied = 0
    for name, value in pretrained.items():
        if name.startswith("expert_dnn.") and name in current and current[name].shape == value.shape:
            current[name] = value.clone()
            copied += 1
    model.shared_encoder.load_state_dict(current, strict=False)
    if copied == 0:
        raise RuntimeError(f"No compatible expert parameters found in {checkpoint}.")
    return copied


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
    }


class Trainer:
    def __init__(
        self,
        model: PooledICIMTL,
        device: torch.device,
        learning_rate: float,
        freeze_experts: bool,
    ) -> None:
        self.model = model
        self.device = device
        self.freeze_experts = freeze_experts
        expert_parameters, other_parameters = [], []
        for name, parameter in model.named_parameters():
            if name.startswith("shared_encoder.expert_dnn."):
                expert_parameters.append(parameter)
            else:
                other_parameters.append(parameter)
        if freeze_experts:
            for parameter in expert_parameters:
                parameter.requires_grad = False
        groups = [{"params": other_parameters, "lr": learning_rate}]
        if not freeze_experts:
            groups.append({"params": expert_parameters, "lr": learning_rate * 0.5})
        self.optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
        self.criterion: nn.Module = nn.BCEWithLogitsLoss()

    def set_class_weight(self, loader: DataLoader) -> None:
        labels = torch.cat([batch["label"] for batch in loader])
        positives = int((labels == 1).sum())
        negatives = int((labels == 0).sum())
        weight = min(max(negatives / positives, 1.0), 6.0) if positives else 1.0
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([weight], device=self.device)
        )

    def _forward_loss(self, batch: dict[str, torch.Tensor]):
        inputs = torch.cat(
            [batch["rna"].to(self.device), batch["shared"].to(self.device)], dim=1
        )
        labels = batch["label"].to(self.device)
        logits, regularization_loss = self.model(inputs)
        loss = self.criterion(logits, labels) + 0.02 * regularization_loss
        return loss, logits, labels

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        if self.freeze_experts:
            for expert in self.model.shared_encoder.expert_dnn:
                expert.eval()
        total = 0.0
        for batch in loader:
            self.optimizer.zero_grad()
            loss, _, _ = self._forward_loss(batch)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total += float(loss.item())
        return total / max(len(loader), 1)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader):
        self.model.eval()
        probabilities, labels, cohort_ids = [], [], []
        for batch in loader:
            _, logits, batch_labels = self._forward_loss(batch)
            probabilities.extend(torch.sigmoid(logits).cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())
            cohort_ids.extend(batch["cohort_id"].numpy())
        probabilities_array = np.asarray(probabilities)
        labels_array = np.asarray(labels, dtype=int)
        return binary_metrics(labels_array, probabilities_array), probabilities_array, np.asarray(cohort_ids)


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Repository: {REPOSITORY_ROOT}")
    print(f"Device: {device}")

    selected_genes, cohort_scalers, encoder_state, experts_path = load_pretraining(
        args.pretraining_dir
    )
    autoencoder = Autoencoder(
        input_dim=len(selected_genes),
        hidden_dims=hidden_dims_list,
        latent_dim=latent_dim,
        dropout=0.4,
        noise_std=noise_std,
        gene_dropout=gene_dropout,
        logvar_min=logvar_min,
        logvar_max=logvar_max,
    ).to(device)
    autoencoder.load_state_dict(encoder_state, strict=False)

    encoded_rna, shared, labels, cohort_ids, sample_ids = load_and_encode_data(
        args.data_dir,
        selected_genes,
        cohort_scalers,
        autoencoder,
        device,
        args.batch_size,
    )
    stratification = np.asarray(
        [f"{cohort}:{label}" for cohort, label in zip(cohort_ids, labels)]
    )
    counts = pd.Series(stratification).value_counts()
    if counts.min() < args.folds:
        raise ValueError(
            f"The smallest cohort/label group has {counts.min()} samples; "
            f"cannot create {args.folds} stratified folds."
        )

    splitter = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
    fold_rows, prediction_rows = [], []
    freeze_experts = not args.train_experts

    for fold, (train_indices, validation_indices) in enumerate(
        splitter.split(labels, stratification), start=1
    ):
        set_seed(args.seed + fold)
        train_shared, validation_shared = preprocess_shared_fold(
            shared, train_indices, validation_indices
        )
        train_data = PooledDataset(
            encoded_rna[train_indices], train_shared, labels[train_indices], cohort_ids[train_indices]
        )
        validation_data = PooledDataset(
            encoded_rna[validation_indices],
            validation_shared,
            labels[validation_indices],
            cohort_ids[validation_indices],
        )
        generator = torch.Generator().manual_seed(args.seed + fold)
        train_loader = DataLoader(
            train_data,
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
        )
        validation_loader = DataLoader(
            validation_data, batch_size=args.batch_size, shuffle=False
        )

        model = PooledICIMTL(
            input_dim=encoded_rna.shape[1] + len(SHARED_COLUMNS),
            expert_hidden_units=expert_dnn_hidden_units,
            num_experts=num_experts,
            gate_hidden_units=gate_dnn_hidden_units,
            fusion_dim=128,
            l2_reg=l2_reg,
            dropout_rate=0.4,
            device=device,
        )
        copied = copy_pretrained_experts(model, experts_path, device)
        print(f"Fold {fold}: loaded {copied} pretrained expert tensors")

        trainer = Trainer(model, device, args.learning_rate, freeze_experts)
        trainer.set_class_weight(train_loader)
        best_auc, stale_epochs = -np.inf, 0
        best_path = args.output_dir / f"best_fold_{fold}.pth"

        for epoch in range(1, args.epochs + 1):
            train_loss = trainer.train_epoch(train_loader)
            metrics, _, _ = trainer.evaluate(validation_loader)
            if metrics["auc"] > best_auc + 1e-4:
                best_auc, stale_epochs = metrics["auc"], 0
                torch.save(model.state_dict(), best_path)
            else:
                stale_epochs += 1
            if epoch == 1 or epoch % 10 == 0:
                print(
                    f"Fold {fold} epoch {epoch}: loss={train_loss:.4f}, "
                    f"AUC={metrics['auc']:.4f}, AUPRC={metrics['auprc']:.4f}"
                )
            if stale_epochs >= args.patience:
                break

        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        metrics, probabilities, validation_cohorts = trainer.evaluate(validation_loader)
        fold_rows.append({"fold": fold, **metrics})
        for local_index, global_index in enumerate(validation_indices):
            prediction_rows.append(
                {
                    "fold": fold,
                    "sample_id": sample_ids[global_index],
                    "cohort": COHORTS[int(validation_cohorts[local_index])],
                    "label": int(labels[global_index]),
                    "probability": float(probabilities[local_index]),
                }
            )
        print(f"Fold {fold} result: {metrics}")

    fold_frame = pd.DataFrame(fold_rows)
    fold_frame.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(
        args.output_dir / "out_of_fold_predictions.csv", index=False
    )
    print("\nCross-validation summary (mean ± std)")
    for metric in ("auc", "auprc", "accuracy", "f1"):
        print(f"{metric}: {fold_frame[metric].mean():.4f} ± {fold_frame[metric].std(ddof=0):.4f}")


if __name__ == "__main__":
    run(parse_args())
