# DBNAMP reproducibility resources

This repository contains the reproducibility resources accompanying the DBNAMP manuscript. It provides two complementary computational components:

1. portable utilities for sequence and molecular standardization, descriptor calculation, conformer generation, molecular-similarity search, and deterministic ranking; and
2. the released SMILES-based antimicrobial peptide classification model, including the labeled cohort, fixed scaffold split, model implementation, checkpoint, inference workflow, and independent-test predictions.

The two components serve different purposes. In particular, `dbnamp_repro/prioritize.py` is a deterministic sorting utility and is separate from the learned SMILES-based classification model. It is not used for model training, inference, threshold selection, or evaluation.

## Repository contents

### Portable molecular-processing utilities

- `dbnamp_repro/standardize.py`: sequence-key and isomeric-SMILES standardization
- `dbnamp_repro/descriptors.py`: physicochemical descriptor calculation
- `dbnamp_repro/structures.py`: RDKit conformer generation
- `dbnamp_repro/similarity.py`: chirality-aware Morgan/Tanimoto similarity
- `dbnamp_repro/prioritize.py`: deterministic ranking by similarity, structure confidence, evidence score, and identifier
- `scripts/`: command-line wrappers for the portable utilities
- `examples/`: small example inputs

### SMILES-based AMP classification model

- `smiles_amp_model/data/preprocessing.py`: SMILES validation, canonicalization, feature generation, and cohort preparation
- `smiles_amp_model/data/scaffold_split.py`: deterministic scaffold-group assignment
- `smiles_amp_model/model/gnn_model.py`: pure-PyTorch graph and SMILES-token classifier
- `smiles_amp_model/train.py`: model training, validation-based checkpoint selection, and threshold adjustment
- `smiles_amp_model/inference.py`: inference from a released checkpoint
- `smiles_amp_model/evaluate.py`: independent-test metrics and stratified bootstrap confidence intervals
- `smiles_amp_model/artifacts/reviewer2_dataset.csv`: released 2,842-record labeled cohort
- `smiles_amp_model/artifacts/split/`: fixed training, validation, and test assignments
- `smiles_amp_model/checkpoints/best_model.pt`: released checkpoint
- `smiles_amp_model/checkpoints/test_predictions_labeled.csv`: independent-test predictions used for the reported metrics

## Environment

Python 3.10 or later is required. Python 3.11 is recommended.

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "biopython>=1.81" "numpy>=1.24" "pandas>=2.0" "rdkit>=2024.03.2" "torch>=2.0" "scikit-learn>=1.3" "scipy>=1.9" "tqdm>=4.65"
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install "biopython>=1.81" "numpy>=1.24" "pandas>=2.0" "rdkit>=2024.03.2" "torch>=2.0" "scikit-learn>=1.3" "scipy>=1.9" "tqdm>=4.65"
```

RDKit wheels are available for common Python versions. Conda can be used when an RDKit wheel is unavailable for the target platform.

## Portable workflow

Run these commands from the repository root:

```bash
python scripts/run_standardization.py examples/input_sequences.csv standardized.csv
python scripts/calculate_descriptors.py standardized.csv descriptors.csv
python scripts/generate_structures.py examples/candidate_structures.csv structures.csv --output-dir generated_structures
python scripts/similarity_search.py examples/query.csv examples/candidate_structures.csv similarity.csv
python scripts/prioritize_candidates.py similarity.csv ranked.csv
python scripts/validate_chirality.py
```

The similarity implementation uses binary Morgan fingerprints with radius 2, 2,048 bits, RDKit default connectivity-based atom invariants, default bond-type invariants, `useFeatures=False`, and `includeChirality=True`. Similarity is calculated using the Tanimoto coefficient.

## Frozen model cohort and split

The released cohort contains 2,842 records: 1,421 positive samples and 1,421 presumed-negative background samples without an annotated antimicrobial target. Background samples are used as a non-AMP-annotated comparison class and should not be interpreted as experimentally confirmed inactive peptides.

| Split | Positive | Background | Total |
| --- | ---: | ---: | ---: |
| Training | 1,129 | 1,145 | 2,274 |
| Validation | 147 | 137 | 284 |
| Independent test | 145 | 139 | 284 |
| Total | 1,421 | 1,421 | 2,842 |

The fixed assignments are stored in:

```text
smiles_amp_model/artifacts/split/split_assignments.csv
smiles_amp_model/artifacts/split/train.csv
smiles_amp_model/artifacts/split/validation.csv
smiles_amp_model/artifacts/split/test.csv
```

Bemis-Murcko scaffolds are generated with chirality enabled. When RDKit returns an empty scaffold for an acyclic molecule, the canonical isomeric SMILES is used as the grouping key. No released scaffold group or identical canonical SMILES occurs in more than one split. Use the fixed split files when reproducing the released checkpoint protocol.

## Released checkpoint configuration

The following values are stored in `smiles_amp_model/checkpoints/best_model.pt`:

| Setting | Released value |
| --- | ---: |
| Atom feature dimension | 51 |
| Bond feature dimension | 13 |
| SMILES-token vocabulary size | 38 |
| Hidden dimension | 40 |
| Message-passing depth | 5 |
| Graph network layers | 5 |
| Sequence encoder | GRU |
| Sequence encoding enabled | Yes |
| Fusion | Learned gate |
| Graph pooling | Mean pooling |
| Classifier dropout | 0.6 |
| Sequence dropout | 0.5 |
| Training feature-mask rate | 0.8 |
| Batch size | 16 |
| Learning rate | 1 x 10^-4 |
| Weight decay | 0 |
| Optimizer | AdamW |
| Loss | `BCEWithLogitsLoss` |
| Maximum training epochs | 4 |
| Early-stopping patience | 4 |
| Random seed | 42 |
| Checkpoint criterion | Highest validation ROC-AUC |
| Validation decision metric | Accuracy |
| Raw validation threshold | 0.5274683237075806 |
| Validation accuracy at selected threshold | 0.7429577464788732 |
| Additive logit bias | -0.10998402924465986 |
| Public inference threshold after bias | 0.5 |

The SMILES-token vocabulary is built from the training split. Validation and test tokens absent from the training vocabulary are mapped to `<unk>`. Random masking of atom features, bond features, and SMILES tokens is applied only to training batches; graph connectivity is retained, and validation/test inputs are not masked.

## Validation-derived decision rule

The epoch with the highest validation ROC-AUC is selected first. After checkpoint selection, candidate raw sigmoid thresholds are evaluated using validation labels only. The threshold that maximizes validation accuracy is `0.5274683237075806`, with validation accuracy `0.7429577464788732`.

The selected threshold is converted to an additive logit intercept:

```text
logit_bias = -log(threshold / (1 - threshold))
           = -0.10998402924465986
```

Inference uses:

```text
released_score = sigmoid(raw_model_logit + logit_bias)
released_label = released_score >= 0.5
```

The public threshold of 0.5 is equivalent to applying the raw validation-selected threshold before the intercept shift. Test labels are not used for checkpoint or threshold selection. This operation aligns the binary decision rule; it is not a probability-calibration analysis. The output should be interpreted as a model-derived candidate-prioritization score.

## Recompute the reported test metrics

The most direct way to reproduce the manuscript metrics is to evaluate the released independent-test prediction table:

```bash
python smiles_amp_model/evaluate.py --predictions smiles_amp_model/checkpoints/test_predictions_labeled.csv --output evaluation_recomputed.json --threshold 0.5 --n-boot 10000 --seed 42
```

Expected results:

| Metric | Estimate | 95% confidence interval |
| --- | ---: | ---: |
| Accuracy | 0.7606 | 0.7113-0.8063 |
| F1-score | 0.7792 | 0.7335-0.8220 |
| ROC-AUC | 0.8357 | 0.7868-0.8815 |
| MCC | 0.5239 | 0.4241-0.6181 |

The corresponding confusion matrix is:

```text
               Predicted 0   Predicted 1
Observed 0          96            43
Observed 1          25           120
```

Accuracy, F1-score, and MCC use released scores thresholded at 0.5, with AMP as the positive class. ROC-AUC uses continuous released scores. Confidence intervals are 10,000 class-stratified percentile bootstrap resamples of the fixed independent-test predictions with random seed 42; the model is not retrained during bootstrap resampling.

## Retrain using the released configuration

```bash
python smiles_amp_model/train.py --data smiles_amp_model/artifacts/reviewer2_dataset.csv --split-dir smiles_amp_model/artifacts/split --output-dir runs/reviewer2_retrain --hidden-dim 40 --depth 5 --num-layer 5 --batch-size 16 --lr 1e-4 --weight-decay 0 --epochs 4 --patience 4 --seed 42 --classifier-dropout 0.6 --sequence-dropout 0.5 --mask-rate 0.8 --calibrate-validation --calibration-metric accuracy --device auto
```

The resulting checkpoint is written to `runs/reviewer2_retrain/best_model.pt`. The checkpoint records the model configuration, training configuration, vocabulary, epoch history, random seed, validation-selected threshold, calibration method, and additive logit bias.

Fixed seeds and fixed splits support controlled repetition. Floating-point operations and model weights may differ across PyTorch, RDKit, CUDA, and hardware versions. Use the released prediction table when exact verification of the reported metrics and confidence intervals is required.

## Run inference

```bash
python smiles_amp_model/inference.py --input examples/candidate_structures.csv --checkpoint smiles_amp_model/checkpoints/best_model.pt --output model_predictions.csv --threshold 0.5 --device auto
```

The output contains the input identifier, canonical isomeric SMILES, model-derived prioritization score after the stored logit adjustment, binary model class at the requested threshold, and an error field for structures that RDKit cannot parse.

To evaluate predictions from a newly trained checkpoint:

```bash
python smiles_amp_model/inference.py --input smiles_amp_model/artifacts/split/test.csv --checkpoint runs/reviewer2_retrain/best_model.pt --output runs/reviewer2_retrain/test_predictions.csv --threshold 0.5 --device auto

python -c "import pandas as pd; p=pd.read_csv('runs/reviewer2_retrain/test_predictions.csv'); t=pd.read_csv('smiles_amp_model/artifacts/split/test.csv'); p['label']=t['label'].to_numpy(); p.to_csv('runs/reviewer2_retrain/test_predictions_labeled.csv', index=False)"

python smiles_amp_model/evaluate.py --predictions runs/reviewer2_retrain/test_predictions_labeled.csv --output runs/reviewer2_retrain/evaluation.json --threshold 0.5 --n-boot 10000 --seed 42
```

## Deterministic ranking and learned classification

`dbnamp_repro/prioritize.py` and `scripts/prioritize_candidates.py` implement a transparent deterministic ranking utility. Records are ordered by descending Tanimoto similarity, descending structure-confidence value, descending evidence score, and identifier as a stable tie-breaker.

This utility has no learned parameters. It does not construct molecular graphs, train the PyTorch classifier, generate model scores, adjust the decision threshold, or contribute to the reported classification metrics. The learned classifier is located exclusively under `smiles_amp_model/`.

## Sequence-key standardization

Generate standardized sequence keys with:

```bash
python scripts/run_standardization.py input_sequences.csv standardized_sequences.csv
```

The implementation applies Unicode NFKC normalization, removes whitespace, normalizes Unicode dash variants, converts braced residue labels to bracketed labels, preserves lowercase unbracketed residues as D-residue signals, and normalizes bracketed noncanonical-residue labels for deterministic exact comparison.

## Artifact checksums

The following SHA-256 checksums identify the released model artifacts:

```text
c4670c621d2359a41382d3c8c4e27f8f9c385e07797197d5df80e42853a1db44  smiles_amp_model/artifacts/reviewer2_dataset.csv
629f23b19dda7ebaaa2fab48d6d6ef362ae43abe29a7738a51f9e1da874c467b  smiles_amp_model/artifacts/split/split_assignments.csv
d1ec99b1b0c6f16be71119feec9b4559517ab7b7af222514eb4345d7fe6079e9  smiles_amp_model/checkpoints/best_model.pt
0b84e24c613027c3cc4ae1b60cc20c55db8da4e946995e4800fdeb8dddf3b681  smiles_amp_model/checkpoints/test_predictions_labeled.csv
```

Record the repository revision used for an analysis with `git rev-parse HEAD`.

## License and citation

Repository code is released under the MIT License. Cite the DBNAMP manuscript when using these resources and record the exact Git commit used for the analysis.
