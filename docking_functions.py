import os
import subprocess
from pathlib import Path
from typing import Tuple

autodocktools_path = os.getenv('MGLTOOLS_HOME')

def smi_txt_to_pdb(smiles, pdb_file):
   
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

def set_element(input_pdb_file, output_pdb_file):
    tcl_script = "set_element.tcl"
    command = (
        f"vmd -dispdev text -e {tcl_script} -args {input_pdb_file} {output_pdb_file}"
    )

    result = subprocess.check_output(command.split())
    return result


def pdb_to_pdbqt(pdb_file, pdbqt_file, ligand= True):

    script, flag = (
        ("prepare_ligand4.py", "l") if ligand else ("prepare_receptor4.py", "r")
    )

    command = (
        f"{'python2.7'}"
        f" {Path(autodocktools_path) / 'MGLToolsPckgs/AutoDockTools/Utilities24' / script}"
        f" -{flag} {pdb_file}"
        f" -o {pdbqt_file}"
        f" -U nphs_lps_waters"
    )
    result = subprocess.check_output(command.split(), encoding="utf-8")


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
):
    
    # Format configuration file
    file_contents = (
        f"receptor = {input_receptor_pdbqt_file}\n"
        f"ligand = {input_ligand_pdbqt_file}\n"
        f"center_x = {center[0]}\n"
        f"center_y = {center[1]}\n"
        f"center_z = {center[2]}\n"
        f"size_x = {size[0]}\n"
        f"size_y = {size[1]}\n"
        f"size_z = {size[2]}\n"
        f"exhaustiveness = {exhaustiveness}\n"
        f"num_modes = {num_modes}\n"
        f"energy_range = {energy_range}\n"
        f"out = {output_ligand_pdbqt_file}\n"
    )
    # Write configuration file
    with open(output_conf_file, "w") as f:
        f.write(file_contents)


def autodock_vina(config_file, num_cpu = 8):
    autodock_vina_exe = "vina"
    try:
        command = f"{autodock_vina_exe} --config {config_file} --cpu {num_cpu}"
        result = subprocess.check_output(command.split(), encoding="utf-8")

        # find the last row of the table and extract the affinity score
        result_list = result.split('\n')
        last_row = result_list[-3]
        score = last_row.split()
        return float(score[1])
    except subprocess.CalledProcessError as e:
        print(f"Command '{e.cmd}' returned non-zero exit status {e.returncode}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None
    