from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Tuple


autodocktools_path = os.getenv('MGLTOOLS_HOME')


def smi_txt_to_pdb(smiles: str, pdb_file: str) -> None:
    """Convert SMILES string to PDB representation.

    The conversion to PDB file will contain atomic coordinates
    that will be used for docking.

    Args:
        smiles: Molecule representation in SMILES format.
        pdb_file: Path of the PDB
            file to create.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    # Convert SMILES to RDKit molecule object
    mol = Chem.MolFromSmiles(smiles)
    # Add hydrogens to the molecule
    mol = Chem.AddHs(mol)
    # Generate a 3D conformation for the molecule
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol)

    # Write the molecule to a PDB file
    writer = Chem.PDBWriter(pdb_file)
    writer.write(mol)
    writer.close()


def set_element(input_pdb_file: str, output_pdb_file: str) -> bytes:
    """Add coordinated to the PDB file using VMD.

    Args:
        input_pdb: Path of input PDB file.
        output_pdb_file: Path to PDB file with atomic coordinates.

    Returns:
        The command execution output.
    """
    tcl_script = 'scripts/set_element.tcl'
    command = (
        f'vmd -dispdev text -e {tcl_script}'
        f' -args {input_pdb_file} {output_pdb_file}'
    )

    result = subprocess.check_output(command.split())
    return result


def pdb_to_pdbqt(pdb_file: str, pdbqt_file: str, ligand: bool = True) -> bytes:
    """Convert PDB file to PDBQT format.

    PDBQT files are similar to the PDB format, but also includes connectivity
    information.

    Args:
        input_pdb: input PDB file to convert.
        pdbqt_file: Converted output PDBQT file.
        ligand: If the molecule is a ligand or not.

    Returns:
        Command execution output.
    """

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
        f' -{flag} {pdb_file}'
        f' -o {pdbqt_file}'
        f' -U nphs_lps_waters'
    )
    result = subprocess.check_output(command.split(), encoding='utf-8')
    return result


def make_autodock_vina_config(
    input_receptor_pdbqt_file: str,
    input_ligand_pdbqt_file: str,
    output_conf_file: str,
    output_ligand_pdbqt_file: str,
    center: Tuple[float, float, float],
    size: Tuple[int, int, int],
    exhaustiveness: int = 20,
    num_modes: int = 20,
    energy_range: int = 10,
) -> None:
    """Create configuration for AutoDock Vina.

    Create a configuration file for AutoDock Vina by describing
    the target receptor and setting coordinate bounds for the
    docking experiment.

    Args:
        input_receptor_pdbqt_file: Target receptor PDBQT file.
        input_ligand_pdbqt_file: Target ligand PDBQT file.
        output_conf_file: Generated Vina conf file.
        output_ligand_pdbqt_file: Output ligand PDBQT file path.
        center: Center coordinates.
        size: Size of the search space.
        exhaustiveness: Number of monte carlo simulations.
        num_modes: Number of binding modes.
        energy_range: Maximum energy difference between
            the best binding mode and the worst one displayed (kcal/mol).
    """
    # Format configuration file
    file_contents = (
        f'receptor = {input_receptor_pdbqt_file}\n'
        f'ligand = {input_ligand_pdbqt_file}\n'
        f'center_x = {center[0]}\n'
        f'center_y = {center[1]}\n'
        f'center_z = {center[2]}\n'
        f'size_x = {size[0]}\n'
        f'size_y = {size[1]}\n'
        f'size_z = {size[2]}\n'
        f'exhaustiveness = {exhaustiveness}\n'
        f'num_modes = {num_modes}\n'
        f'energy_range = {energy_range}\n'
        f'out = {output_ligand_pdbqt_file}\n'
    )
    # Write configuration file
    with open(output_conf_file, 'w') as f:
        f.write(file_contents)


def autodock_vina(config_file: str, num_cpu: int = 8) -> float | None:
    """Compute the docking score.

    The docking score captures the potential energy change when the protein
    and ligand are docked. A strong binding is represented by a negative score,
    weaker (or no) binders are represented by positive scores.

    Args:
        config_file: Vina configuration file.
        num_cpu: Number of CPUs to use.

    Returns:
        Docking score or None to indicate error.
    """
    autodock_vina_exe = 'vina'
    try:
        command = f'{autodock_vina_exe} --config {config_file} --cpu {num_cpu}'
        result = subprocess.check_output(command.split(), encoding='utf-8')

        # find the last row of the table and extract the affinity score
        result_list = result.split('\n')
        last_row = result_list[-3]
        score = last_row.split()
        return float(score[1])
    except subprocess.CalledProcessError as e:
        print(
            f"Command '{e.cmd}' returned non-zero exit status {e.returncode}"
        )
        return None
    except Exception as e:
        print(f'Error: {e}')
        return None
