from __future__ import annotations

import subprocess

from parsldock.docking.sequential import autodock_vina
from parsldock.docking.sequential import make_autodock_vina_config
from parsldock.docking.sequential import pdb_to_pdbqt
from parsldock.docking.sequential import set_element
from parsldock.docking.sequential import smi_txt_to_pdb


def test_smi_to_pdb(smiles, pdb):
    smi_txt_to_pdb(smiles, pdb)

    assert pdb.exists()


def test_set_element(pdb, pdbcoords):
    set_element(pdb, pdbcoords)

    assert pdbcoords.exists()


def test_pdb_to_pdbqt(pdbcoords, pdbqt):
    pdb_to_pdbqt(pdb_file=pdbcoords, pdbqt_file=pdbqt)

    assert pdbqt.exists()


def test_vina_config(receptor, ligand, vina_ligand, center, size, vina_config):
    exhaustiveness = 1

    make_autodock_vina_config(
        receptor,
        ligand,
        vina_config,
        vina_ligand,
        center,
        size,
        exhaustiveness,
    )

    assert vina_config.exists()


def test_autodock_vina(vina_config, monkeypatch):
    score = autodock_vina(config_file=vina_config, num_cpu=1)
    assert isinstance(score, float)

    # test when configfile does not exist
    missing_conf = autodock_vina(config_file='test', num_cpu=1)
    assert missing_conf is None

    # missing test for Exception
    def mockreturn(*args, **kwargs):
        return 'somerandomstring'

    monkeypatch.setattr(subprocess, 'check_output', mockreturn)
    bad_output = autodock_vina(config_file=vina_config, num_cpu=1)
    assert bad_output is None
