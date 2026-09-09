from __future__ import annotations

import re
import unicodedata
from typing import Any

from rdkit import Chem


TOKEN_RE = re.compile(r"\[[^\]]+\]|\{[^}]+\}|[A-Za-z]")


def normalize_sequence(sequence: Any) -> str:
    """Normalize presentation while preserving D-residue and NNAA semantics."""
    text = unicodedata.normalize("NFKC", "" if sequence is None else str(sequence)).strip()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", "", text)
    return text


def sequence_identity_key(sequence: Any) -> str:
    """Create a conservative exact-match key for overlap analysis.

    Lowercase one-letter residues are intentionally retained.  Bracketed and
    braced labels are normalized only by outer delimiters and case, not mapped
    to guessed chemistry.
    """
    text = normalize_sequence(sequence)
    text = text.replace("{", "[").replace("}", "]")
    text = re.sub(r"-{2,}", "-", text)
    tokens = TOKEN_RE.findall(text)
    return "|".join(tok.strip("[]").upper() if tok.startswith(("[", "{")) else tok for tok in tokens)


def standardize_smiles(smiles: Any, *, isomeric: bool = True) -> str:
    """Sanitize a SMILES string and return canonical (optionally isomeric) SMILES."""
    text = "" if smiles is None else str(smiles).strip()
    if not text or text.lower() == "nan":
        return ""
    molecule = Chem.MolFromSmiles(text)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {text}")
    Chem.SanitizeMol(molecule)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=isomeric)


def standardize_record(sequence: Any, smiles: Any = "") -> dict[str, str]:
    seq = normalize_sequence(sequence)
    result = {"sequence": seq, "sequence_identity_key": sequence_identity_key(seq), "isomeric_smiles": ""}
    if str(smiles or "").strip():
        result["isomeric_smiles"] = standardize_smiles(smiles, isomeric=True)
    return result
