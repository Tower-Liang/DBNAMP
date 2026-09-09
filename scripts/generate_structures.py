#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from dbnamp_repro.standardize import standardize_smiles
from dbnamp_repro.structures import generate_conformer, write_structure_files

def main():
    p = argparse.ArgumentParser(description="Generate ETKDGv3 conformers and SDF/PDB files.")
    p.add_argument("input_csv"); p.add_argument("output_csv"); p.add_argument("--output-dir", required=True)
    p.add_argument("--id-column", default="id"); p.add_argument("--smiles-column", default="smiles")
    a = p.parse_args(); df = pd.read_csv(a.input_csv); out = []
    for i, row in df.iterrows():
        ident = str(row.get(a.id_column, f"row_{i}")); raw = str(row.get(a.smiles_column, ""))
        record = {a.id_column: ident, "smiles": raw, "status": "failed"}
        try:
            smi = standardize_smiles(raw); mol, metrics = generate_conformer(smi)
            paths = write_structure_files(mol, a.output_dir, ident)
            record.update({"smiles": smi, "status": "structure_generated", **metrics, **paths})
        except Exception as exc:
            record["error"] = str(exc)
        out.append(record)
    pd.DataFrame(out).to_csv(a.output_csv, index=False)

if __name__ == "__main__": main()
