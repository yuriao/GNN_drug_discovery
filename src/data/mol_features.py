from __future__ import annotations
"""
Global molecular feature computation via RDKit.

Provides 12 physicochemical descriptors that capture drug-likeness properties
the GNN must otherwise infer purely from graph topology. Injecting these
directly into the classification head saves model capacity and provides
features that are provably correlated with bioactivity.

Features chosen based on Lipinski's Rule of Five + ADMET relevance:
  - MolWt        : molecular weight — larger molecules harder to absorb
  - LogP         : lipophilicity — membrane permeability
  - TPSA         : topological polar surface area — cell penetration
  - NumHDonors   : H-bond donors — protein binding
  - NumHAcceptors: H-bond acceptors — protein binding
  - NumRotBonds  : rotatable bonds — conformational flexibility
  - NumArRings   : aromatic ring count — π-stacking interactions
  - NumRings     : total ring count — rigidity
  - FractionCSP3 : fraction of sp3 carbons — 3D character
  - QED          : drug-likeness composite score
  - MaxPartCharge: max partial charge (electrostatics)
  - MinPartCharge: min partial charge (electrostatics)
"""

import numpy as np
from typing import Optional

# Feature dimension
NUM_GLOBAL_FEATURES = 12
_FEAT_NAMES = [
    "MolWt", "LogP", "TPSA", "NumHDonors", "NumHAcceptors",
    "NumRotBonds", "NumArRings", "NumRings",
    "FractionCSP3", "QED", "MaxPartCharge", "MinPartCharge",
]

# Normalisation constants (mean / std estimated from PubChem drug-like subset)
_NORM_MEAN = np.array([
    350.0,   2.5,  80.0, 2.0, 5.0,
    5.0,     2.0,  3.0,
    0.35,    0.55, 0.2,  -0.3,
], dtype=np.float32)

_NORM_STD = np.array([
    150.0,   2.5,  50.0, 2.0, 3.0,
    3.0,     1.5,  2.0,
    0.25,    0.20, 0.2,   0.2,
], dtype=np.float32)


def smiles_to_global_features(smiles: str) -> Optional[np.ndarray]:
    """
    Compute 12 normalised global molecular features for a SMILES string.
    Returns None if RDKit fails to parse the SMILES.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors, QED
        from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        ComputeGasteigerCharges(mol)
        charges = [a.GetDoubleProp("_GasteigerCharge") for a in mol.GetAtoms()]
        max_charge = float(max(charges)) if charges else 0.0
        min_charge = float(min(charges)) if charges else 0.0

        feats = np.array([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            rdMolDescriptors.CalcTPSA(mol),
            rdMolDescriptors.CalcNumHBD(mol),
            rdMolDescriptors.CalcNumHBA(mol),
            rdMolDescriptors.CalcNumRotatableBonds(mol),
            rdMolDescriptors.CalcNumAromaticRings(mol),
            rdMolDescriptors.CalcNumRings(mol),
            rdMolDescriptors.CalcFractionCSP3(mol),
            QED.qed(mol),
            max_charge,
            min_charge,
        ], dtype=np.float32)

        # Normalise (z-score with pre-computed constants)
        feats = (feats - _NORM_MEAN) / (_NORM_STD + 1e-8)
        # Clip extreme values
        feats = np.clip(feats, -5.0, 5.0)
        return feats

    except Exception:
        return None


def batch_smiles_to_features(smiles_list: list[str]) -> np.ndarray:
    """
    Compute global features for a list of SMILES.
    Invalid molecules get a zero vector.
    """
    result = np.zeros((len(smiles_list), NUM_GLOBAL_FEATURES), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        feat = smiles_to_global_features(smi)
        if feat is not None:
            result[i] = feat
    return result


def add_global_features_to_dataset(dataset, smiles_list: list[str]) -> list:
    """
    Attach precomputed global features to each PyG Data object.
    Returns augmented list with data.global_feat set.
    """
    import torch
    augmented = []
    for i, (data, smi) in enumerate(zip(dataset, smiles_list)):
        feat = smiles_to_global_features(smi)
        if feat is None:
            feat = np.zeros(NUM_GLOBAL_FEATURES, dtype=np.float32)
        data.global_feat = torch.from_numpy(feat).unsqueeze(0)  # [1, 12]
        augmented.append(data)
    return augmented
