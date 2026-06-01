# docker run -it --rm --gpus all -v .:/home/appuser orb_models:bitnami bash

import ase
import json
import time

from pathlib import Path

from ase.optimize import BFGS
from ase.io import read, write
from ase import units

from aimnet.calculators import AIMNet2Calculator, AIMNet2ASE

# isayevlab/aimnet2-wb97m-d3	aimnet2	        General organic chemistry
# isayevlab/aimnet2-2025	    aimnet2-2025	Improved intermolecular interactions
# isayevlab/aimnet2-nse	        aimnet2-nse	    Open-shell chemistry

for model_name in [
    # "isayevlab/aimnet2-nse",
    "isayevlab/aimnet2-wb97m-d3",
    "isayevlab/aimnet2-2025",
    ]:

    model_name_alias = f"{model_name.split('/')[-1]}" # e.g. "aimnet2-nse"

    device="cuda" # or device="cuda"
    calc_nse = AIMNet2Calculator(model_name, compile_model=True, device=device) 
    # compile_model=True is recommended for GPU usage

    for dataset in [
        '../data/Liu_et_al_2023/si',
        '../data/Danilack_et_al_2024/si/S2', 
        '../data/Danilack_et_al_2024/si/S3', 
        '../data/Danilack_et_al_2024/si/S4', 
        '../data/Danilack_et_al_2024/si/S5']:
        workdir = Path(dataset)
        dataset_name = dataset.split("/")[-1]

        if dataset_name != 'si':
            outdir = workdir.parent.parent / f"{model_name_alias}_opt/{dataset_name}"
        else:
            outdir = workdir.parent / f"{model_name_alias}_opt"

        print(f"Processing dataset: {dataset}")
        print(f"Input directory: {workdir}")
        print(f"Output directory: {outdir}\n")

        outdir.mkdir(exist_ok=True, parents=True)
        outfile = outdir / "energies.json"

        data = {}

        for xyz_file in sorted(workdir.glob("*.xyz")):
            print(f"Reading XYZ file: {xyz_file}")
            # ase_atoms = read_xyz_to_ase_atoms(xyz_file)
            ase_atoms = read(xyz_file) # read with ASE to get the correct format for the calculator
            if xyz_file.stem.endswith("intermediate"):
                charge = -1
                spin = 1
            else:
                charge = 0
                spin = 1
            
            ase_atoms.calc = AIMNet2ASE(calc_nse, charge=charge, mult=spin)

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
