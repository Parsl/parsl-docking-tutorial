from __future__ import annotations

from pathlib import Path

from parsldock.docking.sequential import autodock_vina
from parsldock.docking.sequential import make_autodock_vina_config
from parsldock.docking.sequential import pdb_to_pdbqt
from parsldock.docking.sequential import set_element
from parsldock.docking.sequential import smi_txt_to_pdb

smi = 'CC1(C2C1C(N(C2)C(=O)C(C(C)(C)C)NC(=O)C(F)(F)F)C(=O)NC(CC3CCNC3=O)C#N)C'
pdb_file = 'paxalovid-molecule.pdb'
pdbcoords_file = 'paxalovid-molecule-coords.pdb'
pdbqt_file = 'paxalovid-molecule-coords.pdbqt'
config_file = 'paxalovid-config.txt'


def test_smi_to_pdb():
    smi_txt_to_pdb(smi, pdb_file)

    assert Path(pdb_file).exists()


def test_set_element():
    set_element(pdb_file, pdbcoords_file)

    assert Path(pdbcoords_file).exists()


def test_pdb_to_pdbqt():
    pdb_to_pdbqt(pdb_file=pdbcoords_file, pdbqt_file=pdbqt_file)

    assert Path(pdbqt_file).exists()


def test_vina_config():
    receptor = 'data/1iep_receptor.pdbqt'
    ligand = 'paxalovid-molecule-coords.pdbqt'

    exhaustiveness = 1
    # specific to 1iep receptor
    cx, cy, cz = 15.614, 53.380, 15.455
    sx, sy, sz = 20, 20, 20

    make_autodock_vina_config(
        receptor,
        ligand,
        config_file,
        ligand,
        (cx, cy, cz),
        (sx, sy, sz),
        exhaustiveness,
    )

    Path(config_file).exists()


def test_autodock_vina():
    score = autodock_vina(config_file=config_file, num_cpu=1)
    assert isinstance(score, float)
