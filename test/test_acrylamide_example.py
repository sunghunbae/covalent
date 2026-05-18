from covalent import Geometry, Reaction, hartree2kcalmol
from covalent import E_methyl_thiol, E_methyl_thiolate, G_methyl_thiol, G_methyl_thiolate
from pathlib import Path
import json
import time
import psi4


num_threads = 24
psi4.set_num_threads(num_threads)

rxn = Reaction(reactant_smiles="C=CC(=O)N", thiol_smiles="SC", verbose=True)

print(f"Reactant SMILES: {rxn.reactant_smiles}")
print(f"Intermediate SMILES: {rxn.carbanion_smiles}")
print(f"Product SMILES: {rxn.product_smiles}")

workdir = Path("./acrylamide")
workdir.mkdir(parents=True, exist_ok=True)

data = {}
for (name, g) in [
    ('reactant', Geometry(smiles=rxn.reactant_smiles)),
    ('intermediate', Geometry(rdmol=rxn.carbanion_rdmol, charge=-1)), 
    ('product', Geometry(smiles=rxn.product_smiles)),
    ]:
    start_time = time.perf_counter()
    psi4.core.set_output_file((workdir / f"{name}.log").as_posix(), True)
    g.xtb_optimize()
    g.optimize(solvent='water', max_iter=500, num_threads=num_threads)
    end_time = time.perf_counter()
    data[name] = {
        'SMILES': g.smiles,
        'solvent': 'water',
        'solvent_model': 'pcm',
        'functional': 'wb97x-d',
        'basis': '6-311+G(d,p)',
        'time(optimize)': end_time - start_time, # (sec)
        }
    g.wirte_xyz(workdir / f"{name}.xyz")

    start_time = time.perf_counter()
    E = g.single_point_energy(num_threads=num_threads) # hartree
    end_time = time.perf_counter()
    data[name].update({
        'time(SP)': end_time - start_time, # (sec)
        'E(kcal/mol)': E * hartree2kcalmol,
        })
    
    start_time = time.perf_counter()
    G = g.gibbs_free_energy(num_threads=num_threads) # hartree
    end_time = time.perf_counter()
    data[name].update({
        'time(Gibbs)': end_time - start_time, # (sec)
        'G(kcal/mol)': G * hartree2kcalmol, 
        })

with open(workdir / 'results.json', 'w') as f:
    f.write(json.dumps(data))