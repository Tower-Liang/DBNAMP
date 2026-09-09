import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dbnamp_repro.standardize import sequence_identity_key, standardize_smiles
from dbnamp_repro.descriptors import compute_descriptors
from dbnamp_repro.similarity import tanimoto
from dbnamp_repro.prioritize import rank

def test_sequence_preserves_d_signal_and_tokens():
    assert sequence_identity_key("GIGk-[Orn]") == "G|I|G|k|ORN"

def test_smiles_and_descriptors():
    smi = standardize_smiles("C[C@H](N)C(=O)O")
    assert "@" in smi
    assert compute_descriptors("GIGk")["d_amino_acid_count"] == 1

def test_similarity_and_ranking():
    smi = standardize_smiles("C[C@H](N)C(=O)O")
    assert tanimoto(smi, smi) == 1.0
    assert rank([{"id":"b","tanimoto":.9},{"id":"a","tanimoto":.9}])[0]["id"] == "a"
