from __future__ import annotations

from parsl import bash_app
from parsl import python_app


@python_app
def parsl_smi_to_pdb(smiles: str, pdb_file: str) -> bool:
    from parsldock.docking.sequential import smi_txt_to_pdb

    smi_txt_to_pdb(smiles=smiles, pdb_file=pdb_file)
    return True


@bash_app
def parsl_set_element(input_pdb: str, outputs: list = []) -> str:
    tcl_script = 'set_element.tcl'
    command = (
        f'vmd -dispdev text -e {tcl_script} -args {input_pdb} {outputs[0]}'
    )
    return command


@bash_app
def parsl_pdb_to_pdbqt(
    input_pdb: str, outputs: list[str] = [], ligand: bool = True
):
    import os
    from pathlib import Path

    autodocktools_path = os.getenv('MGLTOOLS_HOME')

    # Select the correct settings for ligand or receptor preparation
    script, flag = (
        ('prepare_ligand4.py', 'l')
        if ligand
        else ('prepare_receptor4.py', 'r')
    )

    script_path = (
        Path(autodocktools_path)
        / 'MGLToolsPckgs/AutoDockTools/Utilities24'
        / script
    )

    command = (
        f"{'python2.7'}"
        f" {script_path}"
        f" -{flag} {input_pdb}"
        f" -o {outputs[0]}"
        f" -U nphs_lps_waters"
    )
    return command


@python_app
def parsl_make_autodock_config(
    input_receptor: str,
    input_ligand: str,
    output_pdbq: str,
    outputs: list[str] = [],
    center: tuple[float, float, float] = (15.614, 53.380, 15.455),
    size: tuple[int, int, int] = (20, 20, 20),
    exhaustiveness: int = 1,
    num_modes: int = 20,
    energy_range: int = 10,
):
    from parsldock.docking.sequential import make_autodock_vina_config

    make_autodock_vina_config(
        input_receptor_pdbqt_file=input_receptor,
        input_ligand_pdbqt_file=input_ligand,
        output_conf_file=outputs[0].filepath,
        output_ligand_pdbqt_file=output_pdbq,
        center=center,
        size=size,
        exhaustiveness=exhaustiveness,
        num_modes=num_modes,
        energy_range=energy_range,
    )

    return True


@python_app
def parsl_autodock_vina(input_config, smiles, num_cpu=1):
    import subprocess

    autodock_vina_exe = 'vina'
    try:
        command = (
            f'{autodock_vina_exe} --config {input_config} --cpu {num_cpu}'
        )
        # print(command)
        result = subprocess.check_output(command.split(), encoding='utf-8')

        # find the last row of the table and extract the affinity score
        result_list = result.split('\n')
        last_row = result_list[-3]
        score = last_row.split()
        return (smiles, float(score[1]))
    except subprocess.CalledProcessError as e:
        return (
            f"Command '{e.cmd}' returned non-zero exit status {e.returncode}"
        )
    except Exception as e:
        return f'Error: {e}'


@python_app
def cleanup(dock_future, pdb, pdb_coords, pdb_qt, autodoc_config, docking):
    import os

    os.remove(pdb)
    os.remove(pdb_coords)
    os.remove(pdb_qt)
    os.remove(autodoc_config)
    os.remove(docking)

    return dock_future
