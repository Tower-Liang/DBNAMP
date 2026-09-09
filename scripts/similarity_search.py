#!/usr/bin/env python3
import argparse, json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from dbnamp_repro.similarity import search

def main():
    p = argparse.ArgumentParser(description="Morgan/Tanimoto similarity search.")
    p.add_argument("query_csv"); p.add_argument("candidate_csv"); p.add_argument("output_csv")
    p.add_argument("--query-smiles-column", default="smiles"); p.add_argument("--candidate-smiles-column", default="smiles")
    p.add_argument("--no-chirality", action="store_true")
    a = p.parse_args(); q = pd.read_csv(a.query_csv).iloc[0]; c = pd.read_csv(a.candidate_csv)
    candidates = [{"id": row.get("id", i), "smiles": row.get(a.candidate_smiles_column, ""), **row.to_dict()} for i, (_, row) in enumerate(c.iterrows())]
    rows = search(str(q[a.query_smiles_column]), candidates, include_chirality=not a.no_chirality)
    pd.DataFrame(rows).to_csv(a.output_csv, index=False)

if __name__ == "__main__": main()
