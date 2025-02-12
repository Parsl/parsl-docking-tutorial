from __future__ import annotations

import argparse
import uuid
from concurrent.futures import as_completed
from time import monotonic

import pandas as pd
import parsl
from parsl.config import Config
from parsl.data_provider.files import File as PFile
from parsl.executors import HighThroughputExecutor

from parsldock.docking.parallel import cleanup
from parsldock.docking.parallel import parsl_autodock_vina
from parsldock.docking.parallel import parsl_make_autodock_config
from parsldock.docking.parallel import parsl_pdb_to_pdbqt
from parsldock.docking.parallel import parsl_set_element
from parsldock.docking.parallel import parsl_smi_to_pdb
from parsldock.learn import run_model
from parsldock.learn import train_model


def run(
    smi_file_name_ligand: str,
    receptor: str,
    initial_count: int,
    num_loops: int,
    batch_size: int,
) -> None:
    """Protein docking with Parsl application.

    Args:
        smi_file_name_ligand: Path to ligand SMILES string.
        receptor: Path to target receptor PDBQT file.
        initial_count: Initial number of simulations to perform.
        num_loops: Number of infer-simulate-train loops to perform.
        batch_size: Number of simulations per iteration.
    """
    futures = []
    train_data = []

    smiles_simulated = []

    search_space = pd.read_csv(smi_file_name_ligand)
    search_space = search_space[['TITLE', 'SMILES']]

    while len(futures) < 5:
        selected = search_space.sample(1).iloc[0]
        _, smiles = selected['TITLE'], selected['SMILES']

        # workflow
        fname = uuid.uuid4().hex

        smi_future = parsl_smi_to_pdb(
            smiles, outputs=[PFile('%s.pdb' % fname)]
        )
        element_future = parsl_set_element(
            smi_future.outputs[0], outputs=[PFile('%s-coords.pdb' % fname)]
        )
        pdbqt_future = parsl_pdb_to_pdbqt(
            element_future.outputs[0],
            outputs=[PFile('%s-coords.pdbqt' % fname)],
        )
        config_future = parsl_make_autodock_config(
            PFile(receptor),
            pdbqt_future.outputs[0],
            '%s-out.pdb' % fname,
            outputs=[PFile('%s-config.txt' % fname)],
        )
        dock_future = parsl_autodock_vina(config_future.outputs[0], smiles)
        cleanup(
            dock_future,
            smi_future.outputs[0],
            element_future.outputs[0],
            pdbqt_future.outputs[0],
            config_future.outputs[0],
            PFile('%s-out.pdb' % fname),
        )

        futures.append(dock_future)

    while len(futures) > 0:
        future = next(as_completed(futures))
        smiles, score = future.result()
        futures.remove(future)

        print(f'Computation for {smiles} succeeded: {score}')

        train_data.append(
            {'smiles': smiles, 'score': score, 'time': monotonic()}
        )

    training_df = pd.DataFrame(train_data)
    m = train_model(training_df)
    predictions = run_model(m, search_space['SMILES'])
    predictions.sort_values('score', ascending=True).head(5)

    # start with an initial set of random smiles
    for i in range(initial_count):
        selected = search_space.sample(1).iloc[0]
        _, smiles = selected['TITLE'], selected['SMILES']

        # workflow
        fname = uuid.uuid4().hex

        smi_future = parsl_smi_to_pdb(
            smiles, outputs=[PFile('%s.pdb' % fname)]
        )
        element_future = parsl_set_element(
            smi_future.outputs[0], outputs=[PFile('%s-coords.pdb' % fname)]
        )
        pdbqt_future = parsl_pdb_to_pdbqt(
            element_future.outputs[0],
            outputs=[PFile('%s-coords.pdbqt' % fname)],
        )
        config_future = parsl_make_autodock_config(
            PFile(receptor),
            pdbqt_future.outputs[0],
            '%s-out.pdb' % fname,
            outputs=[PFile('%s-config.txt' % fname)],
        )
        dock_future = parsl_autodock_vina(config_future.outputs[0], smiles)
        cleanup(
            dock_future,
            smi_future.outputs[0],
            element_future.outputs[0],
            pdbqt_future.outputs[0],
            config_future.outputs[0],
            PFile('%s-out.pdb' % fname),
        )

        futures.append(dock_future)

    # wait for all the futures to finish
    while len(futures) > 0:
        future = next(as_completed(futures))
        smiles, score = future.result()
        futures.remove(future)

        print(f'Computation for {smiles} succeeded: {score}')

        train_data.append(
            {'smiles': smiles, 'score': score, 'time': monotonic()}
        )
        smiles_simulated.append(smiles)

    # train model, run inference, and run more simulations
    for i in range(num_loops):
        print(f'\nStarting batch {i}')
        m = train_model(training_df)
        predictions = run_model(m, search_space['SMILES'])
        predictions.sort_values(
            'score', ascending=True, inplace=True
        )  # .head(5)

        train_data = []
        futures = []
        batch_count = 0
        for smiles in predictions['smiles']:
            if smiles not in smiles_simulated:
                fname = uuid.uuid4().hex

                smi_future = parsl_smi_to_pdb(
                    smiles, outputs=[PFile('%s.pdb' % fname)]
                )
                element_future = parsl_set_element(
                    smi_future.outputs[0],
                    outputs=[PFile('%s-coords.pdb' % fname)],
                )
                pdbqt_future = parsl_pdb_to_pdbqt(
                    element_future.outputs[0],
                    outputs=[PFile('%s-coords.pdbqt' % fname)],
                )
                config_future = parsl_make_autodock_config(
                    PFile(receptor),
                    pdbqt_future.outputs[0],
                    '%s-out.pdb' % fname,
                    outputs=[PFile('%s-config.txt' % fname)],
                )
                dock_future = parsl_autodock_vina(
                    config_future.outputs[0], smiles
                )
                cleanup(
                    dock_future,
                    smi_future.outputs[0],
                    element_future.outputs[0],
                    pdbqt_future.outputs[0],
                    config_future.outputs[0],
                    PFile('%s-out.pdb' % fname),
                )

                futures.append(dock_future)

                batch_count += 1

            if batch_count > batch_size:
                break

        # wait for all the workflows to complete
        while len(futures) > 0:
            future = next(as_completed(futures))
            smiles, score = future.result()
            futures.remove(future)

            print(f'Computation for {smiles} succeeded: {score}')

            train_data.append(
                {'smiles': smiles, 'score': score, 'time': monotonic()}
            )
            smiles_simulated.append(smiles)

        training_df = pd.concat(
            (training_df, pd.DataFrame(train_data)), ignore_index=True
        )

    return training_df


def main():  # pragma: no cover
    parser = argparse.ArgumentParser(
        prog='ParslDock',
        description='Distributed protein docking with Parsl',
    )

    parser.add_argument(
        '-c',
        '--initial-count',
        type=int,
        default=5,
        help='Initial count of SMILES computations',
    )
    parser.add_argument(
        '-l',
        '--num-loops',
        type=int,
        default=3,
        help='Number of train, inference and simulation loops',
    )

    parser.add_argument(
        '-b',
        '--batch-size',
        type=int,
        default=3,
        help='Number of SMILES strings to simulate within a batch',
    )

    parser.add_argument(
        '-s',
        '--smi-ligand',
        type=str,
        default='data/dataset_orz_original_1k.csv',
        help='File containing list of ligands to consider',
    )

    parser.add_argument(
        '-r',
        '--receptor',
        type=str,
        default='data/1iep_receptor.pdbqt',
        help='File containing list of ligands to consider',
    )

    parser.add_argument(
        '-w',
        '--max-workers',
        type=int,
        default=4,
        help='Maximum number of Parsl workers per node',
    )
    args = parser.parse_args()

    # Configure Parsl to limit max_workers and
    # prevent workers from using the same cores
    config = Config(
        executors=[
            HighThroughputExecutor(
                max_workers_per_node=args.max_workers,
                cpu_affinity='block',
            )
        ]
    )
    parsl.clear()
    parsl.load(config)

    run(
        smi_file_name_ligand=args.smi_ligand,
        receptor=args.receptor,
        initial_count=args.initial_count,
        num_loops=args.num_loops,
        batch_size=args.batch_size,
    )


if __name__ == '__main__':  # pragma: no cover
    main()
