from __future__ import annotations

from parsl import bash_app
from parsl import python_app


@python_app
def parsl_smi_to_pdb(smiles: str, outputs: list[str] = []) -> bool:
    """Convert SMILES string to PDB representation.

    The conversion to PDB file will contain atomic coordinates
    that will be used for docking.

    Args:
        smiles: Molecule representation in SMILES format.
        outputs: A list containing the Path of the PDB
            file to create.

    Returns:
        True to indicate that the operation was successful.
    """
    from parsldock.docking.sequential import smi_txt_to_pdb

    smi_txt_to_pdb(smiles=smiles, pdb_file=outputs[0].filepath)
    return True


@bash_app
def parsl_set_element(input_pdb: str, outputs: list[str] = []) -> str:
    """Add coordinated to the PDB file using VMD.

    Args:
        input_pdb: Path of input PDB file.
        outputs: list containing path to PDB file with atomic coordinates.

    Returns:
        The command execution output.
    """

    tcl_script = 'scripts/set_element.tcl'
    command = (
        f'vmd -dispdev text -e {tcl_script}'
        f' -args {input_pdb} {outputs[0].filepath}'
    )
    return command


@bash_app
def parsl_pdb_to_pdbqt(
    input_pdb: str, outputs: list[str] = [], ligand: bool = True
):
    """Convert PDB file to PDBQT format.

    PDBQT files are similar to the PDB format, but also includes connectivity
    information.

    Args:
        input_pdb: input PDB file to convert.
        outputs: list containing output converted PDBQT file.
        ligand: If the molecule is a ligand or not.

    Returns:
        Command execution output.
    """

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
        f' {script_path}'
        f' -{flag} {input_pdb}'
        f' -o {outputs[0]}'
        f' -U nphs_lps_waters'
    )
    return command


@python_app
def parsl_make_autodock_config(
    input_receptor: str,
    input_ligand: str,
    output_pdbqt: str,
    outputs: list[str] = [],
    center: tuple[float, float, float] = (15.614, 53.380, 15.455),
    size: tuple[int, int, int] = (20, 20, 20),
    exhaustiveness: int = 1,
    num_modes: int = 20,
    energy_range: int = 10,
) -> bool:
    """Create configuration for AutoDock Vina.

    Create a configuration file for AutoDock Vina by describing
    the target receptor and setting coordinate bounds for the
    docking experiment.

    Args:
        input_receptor: Target receptor PDBQT file.
        input_ligand: Target ligand PDBQT file.
        output_pdbqt: Output ligand PDBQT file path.
        outputs: List containing the generated Vina conf file.
        center: Center coordinates.
        size: Size of the search space.
        exhaustiveness: Number of monte carlo simulations.
        num_modes: Number of binding modes.
        energy_range: Maximum energy difference between
            the best binding mode and the worst one displayed (kcal/mol).

    Returns:
        True to indicate successful execution
    """
    from parsldock.docking.sequential import make_autodock_vina_config

    make_autodock_vina_config(
        input_receptor_pdbqt_file=input_receptor,
        input_ligand_pdbqt_file=input_ligand,
        output_conf_file=outputs[0].filepath,
        output_ligand_pdbqt_file=output_pdbqt,
        center=center,
        size=size,
        exhaustiveness=exhaustiveness,
        num_modes=num_modes,
        energy_range=energy_range,
    )

    return True


@python_app
def parsl_autodock_vina(
    input_config: str, smiles: str, num_cpu: int = 1
) -> tuple[str, float] | str:
    """Compute the docking score.

    The docking score captures the potential energy change when the protein
    and ligand are docked. A strong binding is represented by a negative score,
    weaker (or no) binders are represented by positive scores.

    Args:
        input_config: Vina configuration file.
        smiles: The SMILES string of molecule.
        num_cpu: Number of CPUs to use.

    Returns:
        A tuple containing the SMILES string and score or None to
            indicate error.
    """
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
def cleanup(
    dock_future: str,
    pdb: str,
    pdb_coords: str,
    pdb_qt: str,
    autodock_config: str,
    docking: str,
) -> str:
    """Cleanup all generated files.

    Args:
        dock_future: The output of the Autodock Vina execution.
            Used to link steps together.
        pdb: Pdb file generated from the SMILES string.
        pdb_coords: Pdb file with coordinated attached.
        pdb_qt: Pdbqt file generated from the pdb_coords file.
        autodock_config: Config file used to run Autodock-Vina.
        docking: Autodock-Vina output ligand pdbqt file.
    """
    import os

    os.remove(pdb)
    os.remove(pdb_coords)
    os.remove(pdb_qt)
    os.remove(autodock_config)
    os.remove(docking)

    return dock_future
