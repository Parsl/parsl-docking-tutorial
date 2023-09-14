from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from functools import partial
import numpy as np


from concurrent.futures import ProcessPoolExecutor
_pool = ProcessPoolExecutor(max_workers=1)

def compute_morgan_fingerprints(smiles: str, fingerprint_length: int, fingerprint_radius: int):
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    """Get Morgan Fingerprint of a specific SMILES string.
    Adapted from: <https://github.com/google-research/google-research/blob/
    dfac4178ccf521e8d6eae45f7b0a33a6a5b691ee/mol_dqn/chemgraph/dqn/deep_q_networks.py#L750>
    Args:
      graph (str): The molecule as a SMILES string
      fingerprint_length (int): Bit-length of fingerprint
      fingerprint_radius (int): Radius used to compute fingerprint
    Returns:
      np.array. shape = [hparams, fingerprint_length]. The Morgan fingerprint.
    """
    # Parse the molecule
    molecule = Chem.MolFromSmiles(smiles)

    # Compute the fingerprint
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(
        molecule, fingerprint_radius, fingerprint_length)
    arr = np.zeros((1,), dtype=bool)

    # ConvertToNumpyArray takes ~ 0.19 ms, while
    # np.asarray takes ~ 4.69 ms
    DataStructs.ConvertToNumpyArray(fingerprint, arr)
    return arr


class MorganFingerprintTransformer(BaseEstimator, TransformerMixin):
    """Class that converts SMILES strings to fingerprint vectors"""

    def __init__(self, length: int = 256, radius: int = 4):
        self.length = length
        self.radius = radius

    def fit(self, X, y=None):
        return self  # Do need to do anything

    def transform(self, X, y=None):
        """Compute the fingerprints
        
        Args:
            X: List of SMILES strings
        Returns:
            Array of fingerprints
        """
        
        fps = []
        for x in X: 
            fps.append(compute_morgan_fingerprints(x, self.length, self.radius))
            
        return fps     

def train_model(training_data):
    """Train a machine learning model using Morgan Fingerprints.
    
    Args:
        train_data: Dataframe with a 'smiles' and 'score' column
            that contains molecule structure and docking score, respectfully.
    Returns:
        A trained model
    """
    
    # Imports for python functions run remotely must be defined inside the function
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import Pipeline
    
    
    model = Pipeline([
        ('fingerprint', MorganFingerprintTransformer()),
        ('knn', KNeighborsRegressor(n_neighbors=4, weights='distance', metric='jaccard', n_jobs=-1))  # n_jobs = -1 lets the model run all available processors
    ])
    
    return model.fit(training_data['smiles'], training_data['score'])

def run_model(model, smiles):
    """Run a model on a list of smiles strings
    
    Args:
        model: Trained model that takes SMILES strings as inputs
        smiles: List of molecules to evaluate
    Returns:
        A dataframe with the molecules and their predicted outputs
    """
    import pandas as pd
    pred_y = model.predict(smiles)
    return pd.DataFrame({'smiles': smiles, 'score': pred_y})