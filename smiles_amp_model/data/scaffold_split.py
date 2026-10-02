"""Deterministic Bemis--Murcko scaffold split with auditable assignments."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.optimize import LinearConstraint, milp
import numpy as np


TARGET_COUNTS = {
    "train": {1: 1129, 0: 1145},
    "validation": {1: 147, 0: 137},
    "test": {1: 145, 0: 139},
}


def scaffold_key(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES in scaffold split: {smiles}")
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=True)
    # Linear peptides have an empty Bemis--Murcko scaffold.  A canonical
    # fallback keeps them as independent chemical groups instead of creating
    # one impossible-to-split mega-group.
    return scaffold or "ACYCLIC::" + Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def scaffold_split(frame: pd.DataFrame, targets: Dict[str, Dict[int, int]] | None = None) -> pd.DataFrame:
    required = {"smiles", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    targets = targets or TARGET_COUNTS
    work = frame.reset_index(drop=True).copy()
    work["scaffold"] = work["smiles"].map(scaffold_key)
    groups: Dict[str, List[int]] = defaultdict(list)
    for idx, key in enumerate(work["scaffold"]):
        groups[key].append(idx)
    # Largest groups first, ties broken by scaffold text for stable output.
    ordered = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
    split_names = ["train", "validation", "test"]
    # Solve the small binary group-assignment problem directly.  Each scaffold
    # is one indivisible group, and the six class/split equalities encode the
    # requested sizes.  scipy.milp is deterministic for this fixed matrix and
    # avoids the class skew of a capacity-only greedy heuristic.
    n_groups = len(ordered)
    n_vars = n_groups * len(split_names)
    # Distinct deterministic costs select one solution among potentially many
    # feasible partitions.  Costs are large enough to survive MILP feasibility
    # tolerances, but tiny relative to the hard equality constraints.
    objective = np.zeros(n_vars, dtype=float)
    for group_index, (key, _) in enumerate(ordered):
        for split_index in range(len(split_names)):
            objective[group_index * 3 + split_index] = ((group_index + 1) * 2654435761 + (split_index + 1) * 97531) % 1000003
    rows, lower, upper = [], [], []
    for group_index in range(n_groups):
        row = np.zeros(n_vars)
        row[group_index * 3:(group_index + 1) * 3] = 1.0
        rows.append(row); lower.append(1.0); upper.append(1.0)
    for split_index, split in enumerate(split_names):
        for label in (0, 1):
            row = np.zeros(n_vars)
            for group_index, (_, indices) in enumerate(ordered):
                row[group_index * 3 + split_index] = sum(int(work.loc[i, "label"]) == label for i in indices)
            rows.append(row); lower.append(targets[split][label]); upper.append(targets[split][label])
    result = milp(objective, integrality=np.ones(n_vars), bounds=(np.zeros(n_vars), np.ones(n_vars)), constraints=LinearConstraint(np.asarray(rows), np.asarray(lower), np.asarray(upper)), options={"presolve": True})
    if not result.success:
        raise ValueError("Requested per-class scaffold counts are infeasible for this dataset: " + str(result.message))
    assignments: Dict[int, str] = {}
    counts = {s: {0: 0, 1: 0} for s in split_names}
    for group_index, (_, indices) in enumerate(ordered):
        split = split_names[int(np.argmax(result.x[group_index * 3:(group_index + 1) * 3]))]
        for idx in indices:
            assignments[idx] = split
        for label in (0, 1):
            counts[split][label] += sum(int(work.loc[i, "label"]) == label for i in indices)
    work["split"] = [assignments[i] for i in range(len(work))]
    work.attrs["split_counts"] = counts
    work.attrs["target_counts"] = targets
    return work


def write_split_files(frame: pd.DataFrame, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "split_assignments.csv", index=False)
    summary = {split: {str(k): int(v) for k, v in frame.loc[frame.split == split, "label"].value_counts().to_dict().items()} for split in ("train", "validation", "test")}
    (output_dir / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for split in ("train", "validation", "test"):
        frame.loc[frame.split == split, ["smiles", "label", "source", "source_id", "scaffold"]].to_csv(output_dir / f"{split}.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a deterministic scaffold split")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    assigned = scaffold_split(frame)
    write_split_files(assigned, args.output_dir)
    print(assigned.groupby(["split", "label"]).size().to_string())


if __name__ == "__main__":
    main()
