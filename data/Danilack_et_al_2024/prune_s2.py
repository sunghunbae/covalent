from covalent import Reaction, Geometry, FukuiIndex
from covalent import prune, hartree2kcalmol

from rdkit import Chem
from pathlib import Path

import psi4
import json
import time


infile = Path("Danilack_et_al_2024_table_S2.csv")

with open(infile, "r") as f:
    for line in f:
        if line.startswith("Name"):
            continue

        name, smiles, GSH_half_life_min = line.strip().split(",")
        print(f"Processing {name} with GSH half-life {GSH_half_life_min} min...")
        
        # we need to prune the original molecule for QM calculations
        # fast check fukui index for truncation candidates using xtb 
        rxn = Reaction(smiles, thiol_smiles="SC", verbose=False)
        jk = (rxn.alpha_idx, rxn.beta_idx)
        mol = Chem.MolFromSmiles(smiles)
        
        print(f"Original SMILES: {smiles}")
        print(f"Original number of atoms: {mol.GetNumAtoms()}")
        print(f"Original Alpha index: {jk[0]}, Beta index: {jk[1]}")
        
        frag = prune(mol, center=jk, cap_dummy_atom=1, verbose=False)
        na, frag_mol, frag_center = sorted(frag, key=lambda x: x[0])[0] # smallest fragment
        
        print(f"Pruned Fragment SMILES: {Chem.MolToSmiles(Chem.RemoveHs(frag_mol))}")
        print(f"Pruned Number of atoms in fragment: {na}")
        print(f"Pruned Fragment center: {frag_center}")

        rxn_min = Reaction(frag_mol, thiol_smiles="SC", verbose=False)
        jk_min = (rxn_min.alpha_idx, rxn_min.beta_idx)
        print(f"Pruned Alpha index: {jk_min[0]}, Beta index: {jk_min[1]}")
