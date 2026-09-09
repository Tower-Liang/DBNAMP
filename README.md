# DBNAMP reproducibility workflow

This directory is the public, portable reference implementation accompanying the
DBNAMP revision.  It exposes the operations used in the manuscript as small
command-line programs:

1. sequence and isomeric-SMILES standardization;
2. sequence-aware physicochemical descriptor calculation;
3. RDKit ETKDGv3 conformer generation followed by MMFF94 (UFF fallback);
4. Morgan/Tanimoto structure similarity search with explicit chirality control;
5. deterministic candidate prioritization.

The package is deliberately independent of the authors' workstation paths.  It
does not redistribute records downloaded from dbAMP, DRAMP, DBAASP, SATPdb or
APD.  Those source datasets remain subject to their original licences.  Only
small synthetic/example inputs are included here.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

RDKit wheels are available for common CPython versions.  If a platform does not
provide a wheel, install RDKit through conda and then install the remaining
requirements with pip.

## Quick start

```bash
python scripts/run_standardization.py examples/input_sequences.csv /tmp/standardized.csv
python scripts/calculate_descriptors.py /tmp/standardized.csv /tmp/descriptors.csv
python scripts/generate_structures.py examples/candidate_structures.csv /tmp/structures.csv --output-dir /tmp/dbnamp_structures
python scripts/similarity_search.py examples/query.csv examples/candidate_structures.csv /tmp/similarity.csv
python scripts/prioritize_candidates.py /tmp/similarity.csv /tmp/ranked.csv
python scripts/validate_chirality.py
```

Input conventions are documented in each script's `--help` output.  Sequence
standardization preserves lowercase residues as D-residue signals and keeps
bracketed noncanonical tokens as labels.  The versioned monomer dictionary and
token-recovery contracts are included under `configs/` for auditability.
Structure standardization retains
tetrahedral stereochemistry by writing isomeric canonical SMILES.

## Similarity parameters

The default implementation uses RDKit's `MorganGenerator` with radius 2, 2,048
bits, binary bit-vector representation, default connectivity atom invariants,
default bond-type invariants, `useFeatures=False`, and `includeChirality=True`.
The latter can be disabled explicitly for a connectivity-only comparison.  The
similarity score is the RDKit Tanimoto coefficient.

## Reproducibility scope

The code reproduces the algorithmic transformations and parameter choices.  It
does not guarantee byte-identical results across RDKit releases, because
toolkit sanitization and force-field implementations can change.  The tested
environment for the revision was RDKit 2024.03.2.  Amber/tleap and xTB routes
used for selected downstream structures are optional and are not required by
the default open workflow.

## Licence and data provenance

Code in this directory is released under the MIT licence.  The monomer mapping
file is a configuration artifact derived from the DBNAMP preparation workflow;
external database contents and literature-derived records are not relicensed by
this repository.  Users should obtain and cite each upstream database directly.

## Citation

Please cite the DBNAMP manuscript and the versioned GitHub release associated
with the manuscript revision.  A machine-readable `CITATION.cff` is included.

The correspondence between the internal revision scripts and this public
reference implementation is listed in `WORKFLOW_MAPPING.md`.
