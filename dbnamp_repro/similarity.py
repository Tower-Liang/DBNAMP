from __future__ import annotations

from typing import Any

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

RADIUS = 2
FP_SIZE = 2048


def fingerprint(smiles: str, *, include_chirality: bool = True):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=FP_SIZE,
        includeChirality=include_chirality, useBondTypes=True,
        onlyNonzeroInvariants=False)
    return generator.GetFingerprint(molecule)


def tanimoto(query_smiles: str, candidate_smiles: str, *, include_chirality: bool = True) -> float:
    return float(DataStructs.TanimotoSimilarity(fingerprint(query_smiles, include_chirality=include_chirality),
                                                fingerprint(candidate_smiles, include_chirality=include_chirality)))


def search(query_smiles: str, candidates: list[dict[str, Any]], *, include_chirality: bool = True) -> list[dict[str, Any]]:
    rows = []
    for row in candidates:
        try:
            score = tanimoto(query_smiles, str(row.get("smiles", "")), include_chirality=include_chirality)
        except ValueError:
            continue
        item = dict(row); item["tanimoto"] = score; rows.append(item)
    return sorted(rows, key=lambda r: (-float(r["tanimoto"]), str(r.get("id", ""))))
