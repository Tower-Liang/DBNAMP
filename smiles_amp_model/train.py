"""Train the SMILES -> graph/sequence -> AMP probability model."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data.preprocessing import molecule_to_graph, tokenize_smiles
from data.scaffold_split import scaffold_split, write_split_files
from model.gnn_model import SMILESAmpClassifier, collate_graphs


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def apply_training_mask(batch: Dict[str, torch.Tensor], mask_rate: float) -> Dict[str, torch.Tensor]:
    """Apply article-style random feature masking to a training batch only.

    The historical AmpHGT configuration records ``mask_rate=0.6`` for atom,
    fragment/residue and sequence views.  This lightweight homogeneous
    equivalent masks atom/bond feature rows and SMILES tokens while leaving
    graph connectivity intact.  Validation and test batches are never passed
    through this function.
    """
    if mask_rate <= 0:
        return batch
    if not 0 < mask_rate < 1:
        raise ValueError("mask_rate must be in [0, 1)")
    masked = dict(batch)
    atom_mask = torch.rand(batch["x"].shape[0], device=batch["x"].device) < mask_rate
    if atom_mask.any():
        masked["x"] = batch["x"].clone()
        masked["x"][atom_mask] = 0.0
    edge_mask = torch.rand(batch["edge_attr"].shape[0], device=batch["edge_attr"].device) < mask_rate
    if edge_mask.any():
        masked["edge_attr"] = batch["edge_attr"].clone()
        masked["edge_attr"][edge_mask] = 0.0
    token_mask = (torch.rand(batch["seq_tokens"].shape, device=batch["seq_tokens"].device) < mask_rate) & batch["seq_mask"]
    if token_mask.any():
        masked["seq_tokens"] = batch["seq_tokens"].clone()
        masked["seq_tokens"][token_mask] = 1  # <unk>, not padding
    return masked


class GraphDataset(Dataset):
    def __init__(self, frame: pd.DataFrame) -> None:
        self.records: List[Dict[str, object]] = []
        for row in tqdm(frame.itertuples(index=False), total=len(frame), desc="RDKit graphs"):
            graph = molecule_to_graph(row.smiles)
            graph["label"] = int(row.label)
            graph["source_id"] = getattr(row, "source_id", "")
            self.records.append(graph)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        return self.records[index]


def make_vocab(frames: List[pd.DataFrame]) -> Dict[str, int]:
    tokens = {token for frame in frames for smiles in frame.smiles for token in tokenize_smiles(smiles)}
    vocab = {"<pad>": 0, "<unk>": 1}
    vocab.update({token: i for i, token in enumerate(sorted(tokens), 2)})
    return vocab


def evaluate_loss(model: torch.nn.Module, loader: DataLoader, vocab: Dict[str, int], device: torch.device, criterion: torch.nn.Module) -> Dict[str, float]:
    model.eval()
    losses, probabilities, labels = [], [], []
    with torch.no_grad():
        for records in loader:
            batch = collate_graphs(records, vocab)
            batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
            logits = model(batch)
            losses.append(float(criterion(logits, batch["labels"]).item()))
            probabilities.extend(torch.sigmoid(logits).cpu().tolist())
            labels.extend(batch["labels"].cpu().tolist())
    auc = roc_auc_score(labels, probabilities) if len(set(labels)) == 2 else float("nan")
    return {"loss": float(np.mean(losses)), "roc_auc": float(auc)}


def validation_threshold(
    model: torch.nn.Module,
    loader: DataLoader,
    vocab: Dict[str, int],
    device: torch.device,
    metric: str = "f1",
) -> Dict[str, float]:
    """Choose a probability threshold on validation labels, never on test labels.

    ``metric`` is deliberately restricted to threshold metrics that can be
    computed from validation labels alone.  The default remains F1 for
    backwards compatibility with the original article-like run.  Accuracy is
    useful for a sensitivity run because it keeps the public 0.5 threshold
    while matching the paper's reported binary decision criterion more
    closely on this frozen cohort.
    """
    from sklearn.metrics import accuracy_score, f1_score

    if metric not in {"f1", "accuracy"}:
        raise ValueError("calibration metric must be 'f1' or 'accuracy'")

    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for records in loader:
            batch = collate_graphs(records, vocab)
            batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
            logits.extend(model(batch).cpu().tolist())
            labels.extend(batch["labels"].cpu().tolist())
    raw_probabilities = np.asarray(torch.sigmoid(torch.as_tensor(logits)).tolist(), dtype=float)
    labels_array = np.asarray(labels, dtype=int)
    candidates = np.unique(raw_probabilities)
    if metric == "f1":
        score_fn = lambda prediction: f1_score(labels_array, prediction, zero_division=0)
    else:
        score_fn = lambda prediction: accuracy_score(labels_array, prediction)
    scores = [(float(score_fn(raw_probabilities >= threshold)), float(threshold)) for threshold in candidates]
    best_score, best_threshold = max(scores, key=lambda item: (item[0], -item[1]))
    # Adding this intercept makes the selected validation threshold equal to
    # the public inference threshold 0.5 while preserving ranking/AUC.
    logit_bias = float(-np.log(best_threshold / (1.0 - best_threshold)))
    return {
        "threshold": best_threshold,
        "validation_score": best_score,
        "validation_metric": metric,
        "logit_bias": logit_bias,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the NC-AMP graph classifier")
    parser.add_argument("--data", required=True, help="Validated labeled CSV from preprocessing.py")
    parser.add_argument("--split-dir", default=None, help="Existing split directory; if omitted a scaffold split is generated")
    parser.add_argument("--output-dir", default="runs/reviewer2")
    parser.add_argument("--hidden-dim", type=int, default=300)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--num-layer", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--classifier-dropout", type=float, default=0.4)
    parser.add_argument("--sequence-dropout", type=float, default=0.0)
    parser.add_argument("--no-sequence", action="store_true", help="Graph-only ablation; sequence tokens are not used")
    parser.add_argument("--mask-rate", type=float, default=0.0, help="Training-only atom/bond/token feature masking; article config used 0.6")
    parser.add_argument("--calibrate-validation", action="store_true", help="Fit a scalar logit intercept on validation labels so inference threshold remains 0.5")
    parser.add_argument(
        "--calibration-metric",
        choices=("f1", "accuracy"),
        default="f1",
        help="Validation metric used to choose the calibration threshold (default: f1)",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.data)
    if args.split_dir:
        split_dir = Path(args.split_dir)
        split_frames = {name: pd.read_csv(split_dir / f"{name}.csv") for name in ("train", "validation", "test")}
    else:
        assigned = scaffold_split(frame)
        split_dir = output_dir / "split"
        write_split_files(assigned, split_dir)
        split_frames = {name: assigned[assigned.split == name].copy() for name in ("train", "validation", "test")}
    # Build the tokenizer from training strings only; validation/test tokens are
    # deliberately mapped to <unk> to avoid a transductive preprocessing leak.
    vocab = make_vocab([split_frames["train"]])
    datasets = {name: GraphDataset(split_frames[name]) for name in split_frames}
    collate = lambda records: records
    loaders = {name: DataLoader(ds, batch_size=args.batch_size, shuffle=name == "train", num_workers=0, collate_fn=collate, generator=torch.Generator().manual_seed(args.seed)) for name, ds in datasets.items()}
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    model = SMILESAmpClassifier(
        datasets["train"][0]["x"].shape[1],
        datasets["train"][0]["edge_attr"].shape[1],
        len(vocab),
        args.hidden_dim,
        args.depth,
        args.num_layer,
        classifier_dropout=args.classifier_dropout,
        sequence_dropout=args.sequence_dropout,
        use_sequence=not args.no_sequence,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()
    best_auc, best_epoch, stale = -float("inf"), 0, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for records in loaders["train"]:
            batch = collate_graphs(records, vocab)
            batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
            batch = apply_training_mask(batch, args.mask_rate)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch), batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))
        val = evaluate_loss(model, loaders["validation"], vocab, device, criterion)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val["loss"], "val_roc_auc": val["roc_auc"]}
        history.append(row)
        print(json.dumps(row))
        if val["roc_auc"] > best_auc:
            best_auc, best_epoch, stale = val["roc_auc"], epoch, 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_config": model.config,
                "vocab": vocab,
                "threshold": 0.5,
                "seed": args.seed,
                "training_config": {
                    "batch_size": args.batch_size,
                    "learning_rate": args.lr,
                    "weight_decay": args.weight_decay,
                    "epochs": args.epochs,
                    "patience": args.patience,
                    "mask_rate": args.mask_rate,
                    "classifier_dropout": args.classifier_dropout,
                    "sequence_dropout": args.sequence_dropout,
                    "use_sequence": not args.no_sequence,
                    "calibration_metric": args.calibration_metric if args.calibrate_validation else None,
                },
                "history": history,
            }, output_dir / "best_model.pt")
        else:
            stale += 1
            if stale >= args.patience:
                break
    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    if args.calibrate_validation:
        checkpoint_path = output_dir / "best_model.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        calibration = validation_threshold(model, loaders["validation"], vocab, device, args.calibration_metric)
        checkpoint["logit_bias"] = calibration["logit_bias"]
        checkpoint["calibration"] = {
            "method": f"validation {args.calibration_metric.upper()} threshold converted to a scalar logit intercept",
            "validation_metric": args.calibration_metric,
            "validation_threshold_before_bias": calibration["threshold"],
            "validation_score_before_bias": calibration["validation_score"],
            "public_threshold_after_bias": 0.5,
        }
        # Keep the historical field for consumers of the original F1
        # calibration artifact while exposing the generic metric above.
        if args.calibration_metric == "f1":
            checkpoint["calibration"]["validation_f1_before_bias"] = calibration["validation_score"]
        torch.save(checkpoint, checkpoint_path)
        (output_dir / "calibration.json").write_text(json.dumps(checkpoint["calibration"], indent=2), encoding="utf-8")
    print(f"best_epoch={best_epoch} best_val_roc_auc={best_auc:.6f} checkpoint={output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
