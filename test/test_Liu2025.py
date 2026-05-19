from covalent import Reaction, Geometry, FukuiIndex
from covalent import truncate
from rdkit import Chem
from pathlib import Path

import psi4

E_thiolate = -438.24228526784225 # hartree

infile = Path("./Danilack_et_al_2024_table_S2.csv")
outfile = Path("./carbanion_S2.csv")

with open(infile, "r") as f, open(outfile, "w") as g:
    for line in f:
        if line.startswith("Name"):
            continue
        name, smiles, GSH_half_life_min = line.strip().split(",")
        
        # we need to truncate the original molecule for QM calculations
        # fast check fukui index for truncation candidates using xtb 
        
        rxn = Reaction(smiles, thiol_smiles="SC", verbose=False)
        jk = (rxn.alpha_idx, rxn.beta_idx)
        
        reactant = Geometry(smiles)
        reactant.xtb_optimize()

        print("jk=", jk)
        print("truncated=", truncate(reactant.rdmolH, jk))
        print(name, smiles, rxn.carbanion_smiles, rxn.product_smiles)
        print()
        continue
        
        # g.write(f"{name},{smiles},{reactant.carbanion_smiles},{reactant.product_smiles}")

        # reactant = Geometry(smiles)
        # reactant.xtb_optimize()
        # reactant.optimize(functional='wb97x-d', 
        #                 basis='6-311+G(d,p)',
        #                 num_threads=10)
        
        # E_r = reactant.single_point_energy(functional='wb97x-d',
        #                                 basis='6-311+G(d,p)',
        #                                 solvent='water',
        #                                 num_threads=10)
        
        # intermediate = Geometry(reactant.carbanion_smiles, charge=-1)
        # intermediate.xtb_optimize()
        # intermediate.optimize(functional='wb97x-d', 
        #                 basis='6-311+G(d,p)',
        #                 num_threads=10)
        
        # E_i = intermediate.single_point_energy(functional='wb97x-d',
        #                                 basis='6-311+G(d,p)',
        #                                 solvent='water',
        #                                 num_threads=10)
        
        # product = Geometry(reactant.product_smiles, charge=0)
        # product.xtb_optimize()
        # product.optimize(functional='wb97x-d', 
        #                 basis='6-311+G(d,p)',
        #                 num_threads=10)
        
        # E_p = product.single_point_energy(functional='wb97x-d',
        #                                 basis='6-311+G(d,p)',
        #                                 solvent='water',
        #                                 num_threads=10)
        
        # dE_c = psi4.constants.hartree2kcalmol * (E_i - E_r - E_thiolate)
        # dE_p = psi4.constants.hartree2kcalmol * (E_i - E_p)

        # g.write(f",{E_thiolate},{E_r},{E_i},{E_p},{dE_c},{dE_p}\n")