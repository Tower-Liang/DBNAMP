#!/usr/bin/env python3
"""Small controlled test showing why includeChirality must be explicit."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dbnamp_repro.similarity import tanimoto
from dbnamp_repro.standardize import standardize_smiles

def main():
    p = argparse.ArgumentParser(description="Compare an enantiomeric pair with and without chirality encoding.")
    p.add_argument("--l-smiles", default="N[C@H](C)C(=O)O")
    p.add_argument("--d-smiles", default="N[C@@H](C)C(=O)O")
    p.add_argument("--output")
    a = p.parse_args()
    l, d = standardize_smiles(a.l_smiles), standardize_smiles(a.d_smiles)
    result = {"l_smiles": l, "d_smiles": d,
              "tanimoto_without_chirality": tanimoto(l, d, include_chirality=False),
              "tanimoto_with_chirality": tanimoto(l, d, include_chirality=True)}
    text = json.dumps(result, indent=2)
    if a.output:
        with open(a.output, "w", encoding="utf-8") as handle: handle.write(text + "\n")
    print(text)

if __name__ == "__main__": main()
