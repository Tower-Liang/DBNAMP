"""Materialize the archived AmpHGT article-scale train/validation/test protocol.

The historical AmpHGT run consumed six preassigned SMILES files.  This module
records those files verbatim as an auditable dataset manifest; it deliberately
does not randomize, deduplicate, or replace the archived split assignments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from data.preprocessing import canonicalize_smiles


ARTICLE_COUNTS = {
    "train": {1: 2828, 0: 2828},
    "validation": {1: 606, 0: 606},
    "test": {1: 606, 0: 606},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_smi(path: Path, label: int, split: str) -> tuple[pd.DataFrame, Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    rejected = 0
    rejected_lines: List[int] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        text = raw.strip()
        if not text or text.lower().startswith("smiles"):
            continue
        if "!^!" in text:
            parts = text.split("!^!", 1)
        elif "\t" in text:
            parts = text.split("\t", 1)
        else:
            parts = [text]
        smiles_raw = parts[0].strip()
        source_id = parts[1].strip() if len(parts) > 1 else f"{path.stem}_{line_no:06d}"
        smiles = canonicalize_smiles(smiles_raw)
        if smiles is None:
            rejected += 1
            rejected_lines.append(line_no)
            continue
        rows.append(
            {
                "smiles": smiles,
                "smiles_raw": smiles_raw,
                "label": int(label),
                "split": split,
                "source": "AmpHGT_article_archive",
                "source_id": source_id,
                "source_file": str(path),
                "source_line": line_no,
            }
        )
    return pd.DataFrame(rows), {
        "raw_rows": len(rows) + rejected,
        "valid_rows": len(rows),
        "rejected_rows": rejected,
        "rejected_line_numbers": rejected_lines,
    }


def build_article_protocol(source_dir: str | Path, output_dir: str | Path, strict_counts: bool = True) -> pd.DataFrame:
    source_dir = Path(source_dir).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_map = {
        "train": (("train_positive.smi", 1), ("train_negative.smi", 0)),
        "validation": (("valid_positive.smi", 1), ("valid_negative.smi", 0)),
        "test": (("test_positive.smi", 1), ("test_negative.smi", 0)),
    }
    frames: List[pd.DataFrame] = []
    source_manifest: List[Dict[str, object]] = []
    observed: Dict[str, Dict[int, int]] = {}
    observed_valid: Dict[str, Dict[int, int]] = {}
    for split, files in file_map.items():
        split_frames = []
        for filename, label in files:
            path = source_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"Missing archived article split: {path}")
            frame, stats = _read_smi(path, label, split)
            split_frames.append(frame)
            source_manifest.append({"path": str(path), "sha256": sha256(path), "label": label, "split": split, **stats})
        split_frame = pd.concat(split_frames, ignore_index=True)
        # The archived AmpHGT loader counted raw rows before dropping records
        # for which RDKit returned ``None``.  Validate that historical contract
        # against raw rows, while keeping only valid rows in model inputs.
        raw_counts = {
            label: int(sum(item["raw_rows"] for item in source_manifest
                           if item["split"] == split and item["label"] == label))
            for label in (0, 1)
        }
        counts = split_frame["label"].value_counts().to_dict()
        observed[split] = {int(k): int(v) for k, v in raw_counts.items()}
        observed_valid[split] = {int(k): int(v) for k, v in counts.items()}
        if strict_counts and observed[split] != ARTICLE_COUNTS[split]:
            raise ValueError(f"Archived {split} raw counts changed: observed={observed[split]}, expected={ARTICLE_COUNTS[split]}")
        split_frame.to_csv(output_dir / f"{split}.csv", index=False)
        frames.append(split_frame)
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(output_dir / "article_dataset.csv", index=False)
    manifest = {
        "protocol": "AmpHGT historical article-scale archived split",
        "source_dir": str(source_dir),
        "split_method": "preassigned source split; no randomization or deduplication",
        "counts": observed,
        "valid_counts": observed_valid,
        "total_rows": int(len(result)),
        "raw_total_rows": int(sum(sum(value.values()) for value in observed.values())),
        "source_files": source_manifest,
        "notes": [
            "These files are the historical AmpHGT run inputs, not the Reviewer 2 2,842-row DBNAMP/UniProt reconstruction.",
            "Rows are retained even when canonical SMILES repeat because the archived protocol is entry-based.",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "split_summary.json").write_text(json.dumps(observed, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize the archived AmpHGT article-scale split")
    parser.add_argument("--source-dir", default="/home/liangtian/AmpHGT/data_work/splits")
    parser.add_argument("--output-dir", default="artifacts/article_protocol")
    parser.add_argument("--allow-count-drift", action="store_true", help="Do not enforce the historical 2828/606/606 counts")
    args = parser.parse_args()
    frame = build_article_protocol(args.source_dir, args.output_dir, strict_counts=not args.allow_count_drift)
    print(frame.groupby(["split", "label"]).size().to_string())


if __name__ == "__main__":
    main()
