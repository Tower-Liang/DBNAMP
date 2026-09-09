#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from dbnamp_repro.descriptors import compute_descriptors

def main():
    p = argparse.ArgumentParser(description="Calculate sequence-aware DBNAMP descriptors.")
    p.add_argument("input_csv"); p.add_argument("output_csv"); p.add_argument("--sequence-column", default="sequence")
    a = p.parse_args(); df = pd.read_csv(a.input_csv)
    if a.sequence_column not in df: raise SystemExit(f"Missing column: {a.sequence_column}")
    desc = pd.DataFrame([compute_descriptors(s) for s in df[a.sequence_column].fillna("")])
    pd.concat([df.reset_index(drop=True), desc], axis=1).to_csv(a.output_csv, index=False)

if __name__ == "__main__": main()
