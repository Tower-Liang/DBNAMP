"""RDKit preprocessing for the SMILES based NC-AMP classifier.

The module deliberately keeps labels and provenance in the tabular dataset.  A
SMILES is parsed exactly once with RDKit and converted to a homogeneous graph;
no fingerprints or similarity scores are used by the model.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem


ATOM_NUMBERS = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53, 11, 12, 19, 20, 26, 29, 30, 33, 34, 0]
BOND_TYPES = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]
SMILES_TOKEN_RE = re.compile(r"Cl|Br|Si|Na|Ca|Li|Al|[A-Z][a-z]?|[cnospb]|\[[^\]]+\]|[()=#@+\\/\-:.%0-9]")


def one_hot(value: int, values: Sequence[int]) -> List[float]:
    out = [0.0] * (len(values) + 1)
    try:
        out[values.index(value)] = 1.0
    except ValueError:
        out[-1] = 1.0
    return out


def atom_features(atom: Chem.Atom) -> List[float]:
    """Atom identity, valence, charge, chirality, aromaticity and degree."""
    features: List[float] = []
    features.extend(one_hot(atom.GetAtomicNum(), ATOM_NUMBERS))
    features.extend(one_hot(int(atom.GetTotalValence()), list(range(0, 7))))
    features.extend(one_hot(int(atom.GetFormalCharge()), [-2, -1, 0, 1, 2]))
    features.extend(one_hot(int(atom.GetChiralTag()), list(range(0, 4))))
    features.append(float(atom.GetIsAromatic()))
    features.extend(one_hot(int(atom.GetDegree()), list(range(0, 6))))
    features.append(float(atom.GetMass()) / 200.0)
    return features


def bond_features(bond: Chem.Bond) -> List[float]:
    out = [float(bond.GetBondType() == kind) for kind in BOND_TYPES]
    out.extend([float(bond.GetIsConjugated()), float(bond.IsInRing())])
    out.extend(one_hot(int(bond.GetStereo()), list(range(0, 6))))
    return out


def tokenize_smiles(smiles: str) -> List[str]:
    tokens = SMILES_TOKEN_RE.findall(smiles)
    return tokens if tokens else ["<unk>"]


def canonicalize_smiles(smiles: object) -> str | None:
    if smiles is None or (isinstance(smiles, float) and np.isnan(smiles)):
        return None
    text = str(smiles).strip()
    if not text:
        return None
    molecule = Chem.MolFromSmiles(text)
    if molecule is None or molecule.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def molecule_to_graph(smiles: str) -> Dict[str, object]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"RDKit failed to parse SMILES: {smiles}")
    atoms = np.asarray([atom_features(a) for a in molecule.GetAtoms()], dtype=np.float32)
    edges: List[Tuple[int, int]] = []
    edge_attr: List[List[float]] = []
    for bond in molecule.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feature = bond_features(bond)
        edges.extend([(i, j), (j, i)])
        edge_attr.extend([feature, feature])
    if not edges:
        edges = [(0, 0)]
        edge_attr = [[0.0] * 13]
    return {
        "x": atoms,
        "edge_index": np.asarray(edges, dtype=np.int64).T,
        "edge_attr": np.asarray(edge_attr, dtype=np.float32),
        "tokens": tokenize_smiles(smiles),
    }


def _find_smiles_column(frame: pd.DataFrame) -> str:
    for candidate in ("canonical_isomeric_smiles", "canonical_smiles", "smiles", "best_smiles", "smiles_if_available"):
        if candidate in frame.columns:
            return candidate
    raise ValueError("Input table must contain one of canonical_isomeric_smiles, canonical_smiles, smiles, best_smiles, smiles_if_available")


def read_source(path: str | Path, label: int, source_name: str | None = None) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".smi", ".txt"}:
        rows = []
        for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = raw.strip()
            if not line or line.lower().startswith("smiles"):
                continue
            parts = line.split("!^!")
            rows.append({"smiles": parts[0].strip(), "name": parts[1].strip() if len(parts) > 1 else f"{path.stem}_{line_no}"})
        frame = pd.DataFrame(rows)
    else:
        frame = pd.read_csv(path)
    column = _find_smiles_column(frame)
    out = pd.DataFrame({"smiles_raw": frame[column], "label": int(label)})
    out["smiles"] = out["smiles_raw"].map(canonicalize_smiles)
    out["source"] = source_name or path.stem
    if "sequence_id" in frame.columns:
        out["source_id"] = frame["sequence_id"].astype(str)
    elif "candidate_id" in frame.columns:
        out["source_id"] = frame["candidate_id"].astype(str)
    elif "name" in frame.columns:
        out["source_id"] = frame["name"].astype(str)
    else:
        out["source_id"] = [f"{path.stem}_{i:06d}" for i in range(len(out))]
    out = out[out["smiles"].notna()].copy()
    out = out.drop_duplicates(subset=["smiles", "label"], keep="first")
    return out[["smiles", "label", "source", "source_id"]].reset_index(drop=True)


def build_labeled_dataset(positive_paths: Iterable[str | Path], negative_paths: Iterable[str | Path], output: str | Path | None = None) -> pd.DataFrame:
    frames = [read_source(p, 1, "DBNAMP") for p in positive_paths]
    frames.extend(read_source(p, 0, "UniProt") for p in negative_paths)
    if not frames:
        raise ValueError("At least one positive and one negative source are required")
    frame = pd.concat(frames, ignore_index=True)
    conflicts = frame.groupby("smiles")["label"].nunique()
    if (conflicts > 1).any():
        raise ValueError(f"Found {(conflicts > 1).sum()} SMILES with conflicting labels")
    frame = frame.drop_duplicates(subset=["smiles"], keep="first").reset_index(drop=True)
    counts = frame["label"].value_counts().to_dict()
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, index=False)
        Path(str(output) + ".summary.json").write_text(json.dumps({"rows": len(frame), "label_counts": counts}, indent=2), encoding="utf-8")
    return frame


def build_reviewer2_dataset(
    dbnamp_database: str | Path,
    dbnamp_audit: str | Path,
    uniprot_candidates: str | Path,
    output: str | Path,
    positive_count: int = 1421,
    negative_count: int = 1421,
) -> pd.DataFrame:
    """Materialize the frozen Reviewer 2 cohort from the local source exports.

    The original DBNAMP audit has exactly 1,421 records with a PDB and valid
    canonical SMILES.  UniProt candidates are presumed negatives (absence of a
    target annotation), which is retained in ``label_evidence`` rather than
    silently presented as experimentally inactive compounds.
    """
    db = pd.read_csv(dbnamp_database)
    audit = pd.read_csv(dbnamp_audit)
    positive_ids = set(audit.loc[(audit["pdb_present"].astype(str).str.lower() == "true") & (audit["smiles_parse_status"] == "valid"), "sequence_id"])
    pos = db[db["sequence_id"].isin(positive_ids)].copy()
    pos["smiles"] = pos["canonical_smiles"].where(pos["canonical_smiles"].notna(), pos["smiles"])
    pos = pos[["sequence_id", "smiles"]].rename(columns={"sequence_id": "source_id"})
    pos["smiles"] = pos["smiles"].map(canonicalize_smiles)
    # The unit in the paper is a DBNAMP entry/PDB-linked sequence, not a
    # unique molecular graph.  Retain repeated canonical SMILES for distinct
    # audited entries so the requested 1,421 rows remain reproducible.
    pos = pos[pos.smiles.notna()].sort_values("source_id")
    if len(pos) < positive_count:
        raise ValueError(f"DBNAMP manifest contains {len(pos)} valid structures, fewer than requested {positive_count}")
    pos = pos.head(positive_count)
    pos["label"], pos["source"], pos["label_evidence"] = 1, "DBNAMP", "experimentally_supported_structurally_resolved"

    neg = pd.read_csv(uniprot_candidates)
    if "canonical_isomeric_smiles" in neg.columns:
        neg["smiles"] = neg["canonical_isomeric_smiles"].where(neg["canonical_isomeric_smiles"].notna(), neg.get("smiles"))
    if "training_use" in neg.columns:
        neg = neg[neg["training_use"].astype(str).str.contains("eligible", na=False)].copy()
    neg["smiles"] = neg["smiles"].map(canonicalize_smiles)
    neg = neg[neg.smiles.notna()].sort_values("candidate_id")
    if len(neg) < negative_count:
        raise ValueError(f"UniProt candidate pool contains {len(neg)} valid structures, fewer than requested {negative_count}")
    neg = neg.head(negative_count)
    neg = neg.rename(columns={"candidate_id": "source_id"})
    neg["label"], neg["source"], neg["label_evidence"] = 0, "UniProtKB", "presumed_negative_no_target_annotation"
    result = pd.concat([pos[["smiles", "label", "source", "source_id", "label_evidence"]], neg[["smiles", "label", "source", "source_id", "label_evidence"]]], ignore_index=True)
    result = result.sample(frac=1.0, random_state=42).reset_index(drop=True)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    Path(str(output) + ".summary.json").write_text(json.dumps({"rows": len(result), "positive": int((result.label == 1).sum()), "negative": int((result.label == 0).sum()), "selection": "DBNAMP PDB-linked audit IDs + lexicographically first eligible UniProt reconstructed candidates", "negative_semantics": "presumed-negative, not experimentally confirmed inactive"}, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and merge DBNAMP/UniProt SMILES tables")
    parser.add_argument("--reviewer2-defaults", action="store_true", help="Use the local frozen DBNAMP/UniProt exports and create the 2,842-row cohort")
    parser.add_argument("--positive", action="append", help="DBNAMP CSV or .smi; repeatable")
    parser.add_argument("--negative", action="append", help="UniProt CSV or .smi; repeatable")
    parser.add_argument("--output", required=True)
    repo_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--dbnamp-database", default=str(repo_root / "data/final2/database/database_final.csv"))
    parser.add_argument("--dbnamp-audit", default=str(repo_root / "DBNAMP_JCIM_Major_Revision_R1/DBNAMP_original_full_record_audit.csv"))
    parser.add_argument("--uniprot-candidates", default="/home/liangtian/NCAPepDB/data/ml/negative_candidate_inventory_v1/uniprot_reconstructed_peptide_candidates.csv")
    args = parser.parse_args()
    if args.reviewer2_defaults:
        frame = build_reviewer2_dataset(args.dbnamp_database, args.dbnamp_audit, args.uniprot_candidates, args.output)
    else:
        if not args.positive or not args.negative:
            parser.error("--positive and --negative are required unless --reviewer2-defaults is used")
        frame = build_labeled_dataset(args.positive, args.negative, args.output)
    print(frame.groupby("label").size().to_string())


if __name__ == "__main__":
    main()
