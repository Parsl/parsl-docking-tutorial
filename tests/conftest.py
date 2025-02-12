from __future__ import annotations

from pathlib import Path

import parsl
import pytest
from parsl import Config
from parsl import HighThroughputExecutor

from tests import parallel_test


@pytest.fixture(scope='module', autouse=True)
def conf_parsl(request):
    if request.module is not parallel_test:
        return
    config = Config(
        executors=[
            HighThroughputExecutor(
                max_workers_per_node=1,
                cpu_affinity='block',
            )
        ]
    )
    parsl.clear()
    parsl.load(config)


@pytest.fixture(scope='session')
def smiles():
    return (
        'CC1(C2C1C(N(C2)C(=O)C(C(C)(C)C)'
        'NC(=O)C(F)(F)F)C(=O)NC(CC3CCNC3=O)C#N)C'
    )


@pytest.fixture(scope='session')
def pdb():
    return Path('paxalovid-molecule.pdb')


@pytest.fixture(scope='session')
def pdbcoords():
    return Path('paxalovid-molecule-coords.pdb')


@pytest.fixture(scope='session')
def pdbqt():
    return Path('paxalovid-molecule-coords.pdbqt')


@pytest.fixture(scope='session')
def vina_config():
    return Path('paxalovid-config.txt')


@pytest.fixture(scope='session')
def receptor():
    return 'data/1iep_receptor.pdbqt'


@pytest.fixture(scope='session')
def ligand():
    return 'paxalovid-molecule-coords.pdbqt'


@pytest.fixture(scope='session')
def vina_ligand():
    return Path('paxalovid-molecule-out.pdb')


@pytest.fixture(scope='session')
def center():
    return (15.614, 53.380, 15.455)


@pytest.fixture(scope='session')
def size():
    return (20, 20, 20)


@pytest.fixture(scope='module', autouse=True)
def cleanup(pdb, pdbcoords, pdbqt, vina_config, vina_ligand):
    yield

    if pdb.exists():
        pdb.unlink()
    if pdbcoords.exists():
        pdbcoords.unlink()
    if pdbqt.exists():
        pdbqt.unlink()
    if vina_config.exists():
        vina_config.unlink()
    if vina_ligand.exists():
        vina_ligand.unlink()
