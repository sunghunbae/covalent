from covalent import Reaction, Geometry, FukuiIndex
from covalent import prune, hartree2kcalmol

from rdkit import Chem
from pathlib import Path

import psi4
import json
import time
import logging


psi4.set_num_threads(8)
psi4.set_memory('8 GB')

infile = Path("Danilack_et_al_2024_table_S2.csv")

workdir = Path("./S2")
workdir.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

file_handler = logging.FileHandler('run_s2.log')
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)

logger.addHandler(file_handler)

data = {}
with open(infile, "r") as f:
    for line in f:
        if line.startswith("Name"):
            continue

        name, smiles, GSH_half_life_min = line.strip().split(",")
        logger.info(f"Processing {name} with GSH half-life {GSH_half_life_min} min...")
        
        # we need to prune the original molecule for QM calculations
        # fast check fukui index for truncation candidates using xtb 
        rxn = Reaction(smiles, thiol_smiles="SC", verbose=False)
        jk = (rxn.alpha_idx, rxn.beta_idx)
        mol = Chem.MolFromSmiles(smiles)
        
        logger.info(f"Original SMILES: {smiles}")
        logger.info(f"Original number of atoms: {mol.GetNumAtoms()}")
        logger.info(f"Original Alpha index: {jk[0]}, Beta index: {jk[1]}")
        
        frag = prune(mol, center=jk, cap_dummy_atom=1, verbose=False)
        na, frag_mol, frag_center = sorted(frag, key=lambda x: x[0])[0] # smallest fragment
        
        logger.info(f"Pruned Fragment SMILES: {Chem.MolToSmiles(Chem.RemoveHs(frag_mol))}")
        logger.info(f"Pruned Number of atoms in fragment: {na}")
        logger.info(f"Pruned Fragment center: {frag_center}")

        rxn_min = Reaction(frag_mol, thiol_smiles="SC", verbose=False)
        jk_min = (rxn_min.alpha_idx, rxn_min.beta_idx)
        logger.info(f"Pruned Alpha index: {jk_min[0]}, Beta index: {jk_min[1]}")

        for (rc, g) in [
            ('reactant', Geometry(smiles=rxn_min.reactant_smiles)),
            ('intermediate', Geometry(rdmol=rxn_min.carbanion_rdmol, charge=-1)), 
            ('product', Geometry(smiles=rxn_min.product_smiles)),
            ]:
        
            title = f"{name}_{rc}"

            logger.info(f"Processing {title}...")
            logger.info(f"Original SMILES: {smiles}")
            logger.info(f"Fragment SMILES: {Chem.MolToSmiles(Chem.RemoveHs(frag_mol))}")

            psi4.core.set_output_file((workdir / f"{title}.log").as_posix(), True)
            start_time = time.perf_counter()

            try:
                g.xtb_optimize()
                g.optimize()
            except:
                try:
                    g.optimize()
                except:
                    continue

            end_time = time.perf_counter()

            g.write_xyz(workdir / f"{title}.xyz")
            
            data[title] = {
                'SMILES': g.smiles,
                'solvent': 'water',
                'solvent_model': 'pcm',
                'functional': 'wb97x-d',
                'basis': '6-311+G(d,p)',
                'time(optimize)': end_time - start_time, # (sec)
                }
            logger.info(f"Optimization time: {data[title]['time(optimize)']:.2f} seconds")

            start_time = time.perf_counter()

            E = g.single_point_energy() # hartree

            end_time = time.perf_counter()

            data[title].update({
                'E(kcal/mol)': E * hartree2kcalmol,
                'time(SP)': end_time - start_time, # (sec)
                })
            
            logger.info(f"Energy: {data[title]['E(kcal/mol)']:.2f} kcal/mol")
            logger.info(f"SP time: {data[title]['time(SP)']:.2f} seconds")

with open(workdir / "results.json", "w") as f:
    json.dump(data, f, indent=4)
