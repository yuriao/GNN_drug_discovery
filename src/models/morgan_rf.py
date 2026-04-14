from __future__ import annotations

"""
Morgan Fingerprint + Random Forest classifier for HIV inhibition prediction.

Uses RDKit to generate ECFP4 (Morgan radius-2, 2048-bit) fingerprints —
a completely different molecular representation from graph neural networks.
While GNNs learn from local atom neighbourhoods via message passing,
Morgan fingerprints encode the presence/absence of circular substructures
at fixed radii. The two representations make complementary errors, which
is why combining them in an ensemble consistently outperforms either alone.

OGB leaderboard #9: Morgan FP + Random Forest → 0.8208 test ROC-AUC (CPU!)
Reference: Rogers & Hahn, J. Chem. Inf. Model. 2010
"""

import numpy as np
from pathlib import Path


MORGAN_RADIUS  = 2       # ECFP4
MORGAN_NBITS   = 2048    # fingerprint length


def smiles_to_fp(smiles: str) -> np.ndarray | None:
    """Convert a SMILES string to a Morgan fingerprint numpy array."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_NBITS)
        return np.array(fp, dtype=np.float32)
    except Exception:
        return None


def smiles_list_to_fps(smiles_list: list[str]) -> tuple[np.ndarray, list[int]]:
    """
    Convert a list of SMILES to a fingerprint matrix.
    Returns (X, valid_indices) where valid_indices are the indices of
    successfully converted SMILES in the original list.
    """
    fps, valid_idx = [], []
    for i, smi in enumerate(smiles_list):
        fp = smiles_to_fp(smi)
        if fp is not None:
            fps.append(fp)
            valid_idx.append(i)
    if not fps:
        return np.empty((0, MORGAN_NBITS), dtype=np.float32), []
    return np.stack(fps), valid_idx


class MorganRFClassifier:
    """
    Thin wrapper around sklearn RandomForestClassifier that:
    - Converts SMILES → Morgan fingerprints
    - Trains / predicts on the fingerprint matrix
    - Saves / loads the sklearn model via joblib
    """

    def __init__(self, n_estimators: int = 500, random_state: int = 42):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced",   # handles ~3.5% positive imbalance
            random_state=random_state,
            n_jobs=-1,
            max_features="sqrt",
        )
        self._is_fitted = False

    def fit(self, smiles_list: list[str], labels: list[int]) -> "MorganRFClassifier":
        X, valid_idx = smiles_list_to_fps(smiles_list)
        y = np.array([labels[i] for i in valid_idx], dtype=int)
        print(f"Training RF on {len(X)} molecules ({int(y.sum())} positive, {int((1-y).sum())} negative)")
        self.model.fit(X, y)
        self._is_fitted = True
        return self

    def predict_proba_smiles(self, smiles_list: list[str]) -> np.ndarray:
        """
        Returns probability of class 1 (HIV active) for each SMILES.
        Invalid SMILES get probability 0.5 (uncertain).
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        n = len(smiles_list)
        result = np.full(n, 0.5, dtype=np.float32)
        X, valid_idx = smiles_list_to_fps(smiles_list)
        if len(X) > 0:
            probs = self.model.predict_proba(X)[:, 1]
            for i, vi in enumerate(valid_idx):
                result[vi] = probs[i]
        return result

    def predict_proba_single(self, smiles: str) -> float:
        """Predict HIV inhibition probability for a single SMILES."""
        return float(self.predict_proba_smiles([smiles])[0])

    def save(self, path: str | Path) -> None:
        import joblib
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        print(f"RF model saved → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "MorganRFClassifier":
        import joblib
        obj = cls.__new__(cls)
        obj.model = joblib.load(path)
        obj._is_fitted = True
        return obj
