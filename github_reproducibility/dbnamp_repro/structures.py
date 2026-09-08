from __future__ import annotations

from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase
from rdkit.Chem import AllChem


def generate_conformer(smiles: str, *, seed: int = 0x5F3759DF) -> tuple[Chem.Mol, dict[str, Any]]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    molecule = Chem.AddHs(molecule)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.useSmallRingTorsions = True
    params.useMacrocycleTorsions = True
    status = int(AllChem.EmbedMolecule(molecule, params))
    if status != 0:
        raise RuntimeError("ETKDGv3 embedding failed")
    props = AllChem.MMFFGetMoleculeProperties(molecule, mmffVariant="MMFF94")
    if props is not None:
        opt_status = int(AllChem.MMFFOptimizeMolecule(molecule, mmffVariant="MMFF94"))
        force_field = AllChem.MMFFGetMoleculeForceField(molecule, props)
        method = "MMFF94"
    else:
        opt_status = int(AllChem.UFFOptimizeMolecule(molecule))
        force_field = AllChem.UFFGetMoleculeForceField(molecule)
        method = "UFF"
    return molecule, {"rdkit_version": rdBase.rdkitVersion, "embedding_method": "ETKDGv3",
                      "optimization_method": method, "optimization_status": opt_status,
                      "final_energy": float(force_field.CalcEnergy()) if force_field else None}


def write_structure_files(molecule: Chem.Mol, output_dir: str | Path, structure_id: str) -> dict[str, str]:
    out = Path(output_dir)
    (out / "sdf").mkdir(parents=True, exist_ok=True)
    (out / "pdb").mkdir(parents=True, exist_ok=True)
    sdf = out / "sdf" / f"{structure_id}.sdf"
    pdb = out / "pdb" / f"{structure_id}.pdb"
    writer = Chem.SDWriter(str(sdf)); writer.write(molecule); writer.close()
    pdb.write_text(Chem.MolToPDBBlock(molecule), encoding="utf-8")
    return {"sdf": str(sdf), "pdb": str(pdb)}
