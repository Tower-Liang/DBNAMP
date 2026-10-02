"""Run SMILES -> AMP probability inference with a frozen checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from data.preprocessing import canonicalize_smiles, molecule_to_graph
from model.gnn_model import SMILESAmpClassifier, collate_graphs


def read_inputs(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".smi", ".txt"}:
        rows = []
        for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = raw.strip()
            if not line or line.lower().startswith("smiles"):
                continue
            parts = line.split("!^!")
            rows.append({"id": parts[1] if len(parts) > 1 else f"row_{number}", "smiles": parts[0]})
        return pd.DataFrame(rows)
    frame = pd.read_csv(path)
    column = next((c for c in ("smiles", "canonical_isomeric_smiles", "canonical_smiles", "best_smiles", "smiles_if_available") if c in frame.columns), None)
    if column is None:
        raise ValueError("Input must contain a SMILES column")
    result = pd.DataFrame({"id": frame.get("id", frame.get("name", pd.Series(range(len(frame))))), "smiles": frame[column]})
    return result


def predict(input_path: str | Path, checkpoint_path: str | Path, threshold: float = 0.5, device: str = "auto") -> pd.DataFrame:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["model_config"]
    vocab = checkpoint["vocab"]
    logit_bias = float(checkpoint.get("logit_bias", 0.0))
    model = SMILESAmpClassifier(**config)
    model.load_state_dict(checkpoint["model_state_dict"])
    selected_device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device)
    model.to(selected_device).eval()
    inputs = read_inputs(input_path)
    rows, valid = [], []
    for order, row in enumerate(inputs.itertuples(index=False)):
        smiles = canonicalize_smiles(row.smiles)
        if smiles is None:
            rows.append({"_order": order, "id": row.id, "smiles": row.smiles, "probability": None, "predicted_label": None, "error": "RDKit_parse_failed"})
            continue
        graph = molecule_to_graph(smiles)
        graph["label"] = 0
        valid.append((order, row.id, smiles, graph))
    with torch.no_grad():
        for start in range(0, len(valid), 32):
            chunk = valid[start:start + 32]
            batch = collate_graphs([item[3] for item in chunk], vocab)
            batch = {key: value.to(selected_device) if torch.is_tensor(value) else value for key, value in batch.items()}
            probabilities = torch.sigmoid(model(batch) + logit_bias).cpu().tolist()
            for (order, identifier, smiles, _), probability in zip(chunk, probabilities):
                rows.append({"_order": order, "id": identifier, "smiles": smiles, "probability": float(probability), "predicted_label": int(probability >= threshold), "error": None})
    return pd.DataFrame(rows).sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict AMP probability from SMILES")
    parser.add_argument("--input", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--output", default="predictions.csv")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    result = predict(args.input, args.checkpoint, args.threshold, args.device)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    Path(str(args.output) + ".json").write_text(json.dumps(result.to_dict(orient="records"), indent=2), encoding="utf-8")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
