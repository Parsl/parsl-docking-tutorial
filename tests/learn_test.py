from __future__ import annotations

import pandas
from numpy import ndarray
from sklearn.pipeline import Pipeline

from parsldock.learn import compute_morgan_fingerprints
from parsldock.learn import MorganFingerprintTransformer
from parsldock.learn import run_model
from parsldock.learn import train_model


def test_compute_morgan_fingerprints(smiles):
    fingerprints = compute_morgan_fingerprints(
        smiles=smiles, fingerprint_length=256, fingerprint_radius=4
    )
    assert isinstance(fingerprints, ndarray)


def test_morgantransformer(smiles):
    mft = MorganFingerprintTransformer()
    mft.fit(1, 2)

    other_mft = MorganFingerprintTransformer()
    assert mft.length == other_mft.length
    assert mft.radius == other_mft.radius

    orig_fingerprints = compute_morgan_fingerprints(
        smiles=smiles, fingerprint_length=256, fingerprint_radius=4
    )

    transformed_fingerprints = mft.transform(X=[smiles])

    assert (orig_fingerprints == transformed_fingerprints).all()


def test_trainmodel(smiles):
    df = pandas.DataFrame({'smiles': [smiles], 'score': [-1]})
    trained = train_model(df)
    assert isinstance(trained, Pipeline)


def test_runmodel(smiles):
    df = pandas.DataFrame(
        {'smiles': [smiles, smiles, smiles, smiles], 'score': [-1, 1, 0.5, 0]}
    )
    trained = train_model(df)
    expected = run_model(trained, [smiles])

    assert expected.values.tolist()[0][1] == 0.125
