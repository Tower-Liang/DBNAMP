#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from dbnamp_repro.standardize import standardize_record

def main():
    p = argparse.ArgumentParser(description="Standardize DBNAMP sequence and optional SMILES columns.")
    p.add_argument("input_csv"); p.add_argument("output_csv")
    p.add_argument("--sequence-column", default="sequence")
    p.add_argument("--smiles-column", default="smiles")
    a = p.parse_args(); df = pd.read_csv(a.input_csv)
    if a.sequence_column not in df: raise SystemExit(f"Missing column: {a.sequence_column}")
    rows = []
    for _, row in df.iterrows():
        item = dict(row); item.update(standardize_record(row[a.sequence_column], row.get(a.smiles_column, "")))
        rows.append(item)
    pd.DataFrame(rows).to_csv(a.output_csv, index=False)

if __name__ == "__main__": main()
