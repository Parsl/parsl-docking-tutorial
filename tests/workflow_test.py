from __future__ import annotations

import pandas

from parsldock.workflow import run


def test_run(ligand, receptor):
    df = run(
        smi_file_name_ligand='data/dataset_orz_original_1k.csv',
        receptor=receptor,
        initial_count=1,
        num_loops=1,
        batch_size=1,
    )

    assert isinstance(df, pandas.DataFrame)
