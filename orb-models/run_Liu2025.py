# docker run -it --rm --gpus all -v .:/home/appuser orb_models:bitnami bash

import ase
import json
import time

from pathlib import Path

from ase.optimize import BFGS
from ase.io import read, write
from ase import units

from orb_models.forcefield import pretrained
from orb_models.forcefield.inference.calculator import ORBCalculator

device="cuda" # or device="cuda"

# or choose another model using ORB_PRETRAINED_MODELS[model_name]()
orbff, atoms_adapter = pretrained.orb_v3_conservative_inf_omat(device=device)
calc = ORBCalculator(orbff, atoms_adapter=atoms_adapter, device=device)


workdir = Path(f"../data/Liu_et_al_2025/si/")

outdir = Path(f"../data/Liu_et_al_2025/orb_opt/")
outdir.mkdir(exist_ok=True, parents=True)
outfile = outdir / "energies.json"

data = {}

for xyz_file in sorted(workdir.glob("*.xyz")):
    print(f"Reading XYZ file: {xyz_file}")
    # ase_atoms = read_xyz_to_ase_atoms(xyz_file)
    ase_atoms = read(xyz_file) # read with ASE to get the correct format for the calculator
    if xyz_file.stem.endswith("TS") or xyz_file.stem.endswith("carbanion"):
        charge = -1
        spin = 1
    else:
        charge = 0
        spin = 1
    
    ase_atoms.info["charge"] = charge
    ase_atoms.info["spin"] = spin
    ase_atoms.calc = calc

    energy_eV_ini = ase_atoms.get_potential_energy() # eV
    print(f"Initial Energy: {energy_eV_ini:.4f} eV")

    # Rattle the atoms to get them out of the minimum energy configuration
    # Randomly displace atoms.
    # This method adds random displacements to the atomic positions, 
    # taking a possible constraint into account. The random numbers 
    # are drawn from a normal distribution of standard deviation stdev.
    ase_atoms.rattle(stdev=0.01)
    energy_eV_rattled = ase_atoms.get_potential_energy()
    print(f"Rattled Energy(stdev=0.01): {energy_eV_rattled:.4f} eV")
    
    start_time = time.perf_counter()
    dyn = BFGS(ase_atoms)
    dyn.run(fmax=0.01)
    energy_eV_opt = ase_atoms.get_potential_energy() # eV
    end_time = time.perf_counter()

    print(f"Optimized Energy: {energy_eV_opt:.4f} eV")
    print(f"Optimization Time: {end_time - start_time:.2f} seconds\n")

    # write optimized geometry to xyz file
    print(f"Writing optimized geometry to {outdir / f'{xyz_file.stem}.xyz'}")
    write(outdir / f"{xyz_file.stem}.xyz", ase_atoms) 
    
    data[xyz_file.stem] = {
        'E(initial, kcal/mol)': energy_eV_ini * (units.mol / units.kcal),
        'E(rattled, kcal/mol)': energy_eV_rattled * (units.mol / units.kcal),
        'E(opt, kcal/mol)': energy_eV_opt * (units.mol / units.kcal),
        'time(opt, sec)': end_time - start_time, # (sec)
        }

with open(outfile, "w") as f:
    json.dump(data, f, indent=4)