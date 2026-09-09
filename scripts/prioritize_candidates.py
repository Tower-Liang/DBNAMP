#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from dbnamp_repro.prioritize import rank

def main():
    p = argparse.ArgumentParser(description="Apply deterministic candidate prioritization.")
    p.add_argument("input_csv"); p.add_argument("output_csv"); a = p.parse_args()
    df = pd.read_csv(a.input_csv)
    pd.DataFrame(rank(df.to_dict(orient="records"))).to_csv(a.output_csv, index=False)

if __name__ == "__main__": main()
