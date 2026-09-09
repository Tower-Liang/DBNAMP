# Workflow mapping

| Manuscript operation | Revision repository source | Public implementation |
|---|---|---|
| Sequence representation and token recovery | `src/data/qc.py`, `src/data/token_recovery.py`, `configs/monomer_dictionary_v0.2.json` | `dbnamp_repro/standardize.py`, `configs/monomer_dictionary_v0.2.json` |
| Physicochemical descriptors | `src/features/descriptors.py` | `dbnamp_repro/descriptors.py` |
| Conformer generation | `src/data/structure_conformer_db.py` | `dbnamp_repro/structures.py` |
| Morgan/Tanimoto similarity | `scripts/validate_chirality_aware_similarity.py` and retrieval wrapper | `dbnamp_repro/similarity.py`, `scripts/similarity_search.py` |
| Candidate prioritization | analysis-specific case-selection code | `dbnamp_repro/prioritize.py`, `scripts/prioritize_candidates.py` |

The public directory is a portable reference implementation.  It removes
workstation-specific paths and optional Amber/tleap/xTB integrations while
keeping the documented algorithmic choices and defaults explicit.  The source
repository remains the authoritative record for the complete database assembly
and source-specific recovery rules.
