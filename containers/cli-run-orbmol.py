# docker run -it --rm --gpus all -v .:/home/appuser orb_models:bitnami bash

import csv
import sqlite3
import argparse
import time

from io import StringIO
from pathlib import Path

from ase.optimize import BFGS
from ase.io import read, write
from ase import units

from orb_models import __version__
from orb_models.forcefield import pretrained
from orb_models.forcefield.inference.calculator import ORBCalculator


print (f"Orb models version {__version__}")


parser = argparse.ArgumentParser(description="Run Covalent Warhead Workflow")
parser.add_argument("db", type=str, help="input sqlite3 .db file")
args = parser.parse_args()

with sqlite3.connect(args.db) as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, rc, smiles, charge, alpha, beta, xyz FROM reaction")
    workload = cursor.fetchall()
conn.close()


model_alias = 'orbmol-v2'

prefix = Path(args.db).stem
o_dbfile = Path(f"{prefix}_{model_alias}.db")
o_csvfile = Path(f"{prefix}_{model_alias}.csv")


device="cuda" # or device="cuda"

# or choose another model using ORB_PRETRAINED_MODELS[model_name]()
# The orb_v3_conservative_inf_omat checkpoint is a universal crystal materials model 
# trained strictly on bulk solid-state data (the OMat24 dataset). 
# Unlike the molecular-focused model variations, it does not contain a charge prediction head 
# in its neural network architecture. 
# If you query atoms.calc.results.get("charges") or search the direct graph prediction dictionary, 
# the "charges" key will simply return None or raise a KeyError.

# orbff, atoms_adapter = pretrained.orb_v3_conservative_inf_omat(device=device)

# OrbMol-v2: Adds a CoulombModule explicitly designed to handle long-range electrostatics alongside local bonding. 
# It features a LatentChargeHead that predicts individual partial atomic charges constrained 
# to sum directly to your system's net total charge. 
# It shifts between bare \(1/r\) direct Coulomb sums for isolated gaseous configurations 
# and Particle Mesh Ewald (PME) wrappers for periodic, bulk molecular arrays.

# As of version 0.7.0, `charges` are not supported.
# results keys= ['energy', 'grad_forces', 'grad_stress', 'rotational_grad', 'confidence']

orbff, atoms_adapter = pretrained.orbmol_v2(device=device, precision="float32-high")
# weights_path: str = "https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orbmol-v2-teqabfhg-20260523.ckpt"
calculator = ORBCalculator(orbff, atoms_adapter=atoms_adapter, device=device)


with sqlite3.connect(o_dbfile) as conn:
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS optimized (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            rc TEXT,
            smiles TEXT,
            charge INTEGER,
            alpha INTEGER,
            beta INTEGER,
            xyz TEXT
        )
    """)
    conn.commit()

    with open(o_csvfile, "w") as out_csvfile:
        writer = csv.DictWriter(out_csvfile,
                                fieldnames=['Name', 'RC', 'SMILES', 'charge',
                                            'QCa', 'QCb', 'E_HOMO', 'E_LUMO',
                                            'f_plus_Cb', 'f_minus_Cb', 'f_zero_Cb',
                                            'f_plus_Ca', 'f_minus_Ca', 'f_zero_Ca',
                                            'E(opt, kcal/mol)', 'time(opt, sec)',
                                            'E(rattled, kcal/mol)', 'E(initial, kcal/mol)'])
        writer.writeheader()

        prev_name = ""
        for row in workload:
            row_id, name, rc, smiles, charge, alpha_idx, beta_idx, xyz = row
            print(f"Processing {name} ...")

            ase_atoms = read(StringIO(xyz), format='xyz')
            # read with ASE to get the correct format for the calculator
            ase_atoms.info["charge"] = charge
            ase_atoms.info["spin"] = 1
            ase_atoms.calc = calculator

            energy_eV_ini = ase_atoms.get_potential_energy() # eV
            
            # Rattle the atoms to get them out of the minimum energy configuration
            # Randomly displace atoms.
            # This method adds random displacements to the atomic positions,
            # taking a possible constraint into account. The random numbers
            # are drawn from a normal distribution of standard deviation stdev.
            ase_atoms.rattle(stdev=0.01)
            energy_eV_rattled = ase_atoms.get_potential_energy()

            start_time = time.perf_counter()

            dyn = BFGS(ase_atoms, logfile=None)
            dyn.run(fmax=0.01) # could be 0.03 or 0.05
            energy_eV_opt = ase_atoms.get_potential_energy() # eV

            
            try:
                partial_charges = ase_atoms.calc.results["charges"]
                QCa = partial_charges[alpha_idx]
                QCb = partial_charges[beta_idx]

                # calculate cation energy +1 on the same geometry (vertical IP)
                ase_atoms.info["charge"] = charge + 1
                ase_atoms.info["spin"] = 1
                ase_atoms.calc = calculator
                energy_eV_plus1 = ase_atoms.get_potential_energy() # eV
                partial_charges_plus1 = ase_atoms.calc.results["charges"]

                # calculate cation energy -1 on the same geometry (vertical EA)
                ase_atoms.info["charge"] = charge - 1
                ase_atoms.info["spin"] = 1
                ase_atoms.calc = calculator
                energy_eV_minus1 = ase_atoms.get_potential_energy() # eV
                partial_charges_minus1 = ase_atoms.calc.results["charges"]
                
                # HOMO and LUMO
                IP = energy_eV_plus1 - energy_eV_opt
                EA = energy_eV_opt - energy_eV_minus1
                E_HOMO = -IP
                E_LUMO = -EA

                # Condensed Fukui Functions
                f_plus = partial_charges - partial_charges_minus1 # nucleophilic
                f_minus = partial_charges_plus1 - partial_charges # electrophilic
                f_zero = 0.5 * (partial_charges_plus1 - partial_charges_minus1) # radical
                f_plus_Cb = f_plus[beta_idx]
                f_minus_Cb = f_minus[beta_idx]
                f_zero_Cb = f_zero[beta_idx]
                f_plus_Ca = f_plus[alpha_idx]
                f_minus_Ca = f_minus[alpha_idx]
                f_zero_Ca = f_zero[alpha_idx]

            except KeyError:
                QCa = 0.
                QCb = 0.
                E_HOMO = 0.
                E_LUMO = 0.
                f_plus_Cb = 0.
                f_minus_Cb = 0.
                f_zero_Cb = 0.
                f_plus_Ca = 0.
                f_minus_Ca = 0.
                f_zero_Ca = 0.
            
            end_time = time.perf_counter()

            string_buffer = StringIO()
            write(string_buffer, ase_atoms, format='xyz')

            writer.writerow({'Name': name,
                            'RC': rc,
                            'charge': charge,
                            'SMILES': smiles,
                            'QCa': QCa,
                            'QCb': QCb,
                            'f_plus_Cb': f_plus_Cb,
                            'f_minus_Cb': f_minus_Cb,
                            'f_zero_Cb': f_zero_Cb,
                            'f_plus_Ca': f_plus_Ca,
                            'f_minus_Ca': f_minus_Ca,
                            'f_zero_Ca': f_zero_Ca,
                            'E_HOMO': E_HOMO * (units.mol / units.kcal),
                            'E_LUMO': E_LUMO * (units.mol / units.kcal),
                            'E(initial, kcal/mol)': energy_eV_ini * (units.mol / units.kcal),
                            'E(rattled, kcal/mol)': energy_eV_rattled * (units.mol / units.kcal),
                            'E(opt, kcal/mol)': energy_eV_opt * (units.mol / units.kcal),
                            'time(opt, sec)': end_time - start_time, # (sec)
                            })

            cursor.execute(
                """
                INSERT OR IGNORE INTO optimized (name, rc, smiles, charge, alpha, beta, xyz)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (name, rc, smiles, charge, alpha_idx, beta_idx, string_buffer.getvalue())
                )

            if prev_name and prev_name != name:
                conn.commit()

            prev_name = name