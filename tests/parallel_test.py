from __future__ import annotations

from parsl.data_provider.files import File as PFile

from parsldock.docking.parallel import cleanup
from parsldock.docking.parallel import parsl_autodock_vina
from parsldock.docking.parallel import parsl_make_autodock_config
from parsldock.docking.parallel import parsl_pdb_to_pdbqt
from parsldock.docking.parallel import parsl_set_element
from parsldock.docking.parallel import parsl_smi_to_pdb


def test_parsl_smi_to_pdb(smiles, pdb):
    future = parsl_smi_to_pdb(smiles, outputs=[PFile(pdb)])
    out = future.result()

    assert out and pdb.exists()


def test_parsl_set_element(pdb, pdbcoords):
    future = parsl_set_element(pdb, outputs=[PFile(pdbcoords)])
    out = future.result()

    assert out == 0 and pdbcoords.exists()


def test_parsl_pdb_to_pdbqt(pdbcoords, pdbqt):
    future = parsl_pdb_to_pdbqt(pdbcoords, outputs=[PFile(pdbqt)])
    out = future.result()

    assert out == 0 and pdbqt.exists()


def test_parsl_make_autodock_config(
    receptor, ligand, vina_config, vina_ligand
):
    future = parsl_make_autodock_config(
        PFile(receptor), ligand, vina_ligand, outputs=[PFile(vina_config)]
    )
    out = future.result()

    assert out and vina_config.exists()


def test_parsl_autodock_vina(vina_config, smiles):
    future = parsl_autodock_vina(vina_config, smiles)
    _, score = future.result()

    assert isinstance(score, float)


def test_cleanup(pdb, pdbcoords, pdbqt, vina_config, vina_ligand):
    future = cleanup(None, pdb, pdbcoords, pdbqt, vina_config, vina_ligand)
    future.result()

    assert not pdb.exists()
    assert not pdbcoords.exists()
    assert not pdbqt.exists()
    assert not vina_config.exists()
    assert not vina_ligand.exists()
