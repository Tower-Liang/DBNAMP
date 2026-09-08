from __future__ import annotations

import math
import re
from typing import Any

from Bio.SeqUtils.ProtParam import ProteinAnalysis

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
HYDRO = {"A": .62, "C": .29, "D": -.90, "E": -.74, "F": 1.19, "G": .48,
         "H": -.40, "I": 1.38, "K": -1.50, "L": 1.06, "M": .64, "N": -.78,
         "P": .12, "Q": -.85, "R": -2.53, "S": -.18, "T": -.05, "V": 1.08,
         "W": .81, "Y": .26}


def _tokens(sequence: str) -> list[str]:
    return re.findall(r"\[[^\]]+\]|\{[^}]+\}|[A-Za-z]", str(sequence or ""))


def _standard_sequence(sequence: str) -> str:
    return "".join(tok.upper() for tok in _tokens(sequence) if len(tok) == 1 and tok.upper() in STANDARD_AA)


def compute_descriptors(sequence: str) -> dict[str, float | int]:
    tokens = _tokens(sequence)
    std = _standard_sequence(sequence)
    n = len(tokens)
    analysis = ProteinAnalysis(std) if std else None
    aa = {a: (std.count(a) / len(std) if std else 0.0) for a in sorted(STANDARD_AA)}
    charged = sum(aa.get(a, 0.0) for a in "KRHDE")
    aromatic = sum(aa.get(a, 0.0) for a in "FWY")
    d_count = sum(1 for t in tokens if len(t) == 1 and t.islower())
    nonstandard = sum(1 for t in tokens if not (len(t) == 1 and t.upper() in STANDARD_AA))
    return {
        "length": n,
        "length_standard": len(std),
        "charge_ph7": float(analysis.charge_at_pH(7.0)) if analysis else 0.0,
        "net_charge": float(analysis.charge_at_pH(7.0)) if analysis else 0.0,
        "hydrophobicity_eisenberg": sum(HYDRO.get(a, 0.0) for a in std) / len(std) if std else 0.0,
        "isoelectric_point": float(analysis.isoelectric_point()) if analysis else 0.0,
        "molecular_weight": float(analysis.molecular_weight()) if analysis else 0.0,
        "boman_index": 0.0,
        "instability_index": float(analysis.instability_index()) if analysis else 0.0,
        "gravy": float(analysis.gravy()) if analysis else 0.0,
        "charged_aa_fraction": charged,
        "aromatic_aa_fraction": aromatic,
        "nnaa_fraction": nonstandard / n if n else 0.0,
        "d_aa_fraction": d_count / n if n else 0.0,
        "d_amino_acid_count": d_count,
        "modified_residue_count": nonstandard,
        "has_nonstandard": int(nonstandard > 0),
    }
