import argparse
import sqlite3
import gzip
import time
import csv
import sqlite3
import argparse
import sys

from enum import Enum
from io import StringIO
from pathlib import Path
from rdkit import Chem
from tqdm import tqdm

from covalent import Geometry, Reaction


def setup_xyz():
    parser = argparse.ArgumentParser(description="Setup Covalent Warhead Workflow",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("sdf", type=str, help="input .sdf.gz file")
    parser.add_argument("db", type=str, help="output sqlite3 .db file")
    parser.add_argument("--name-startswith", type=str, help="filter for name")
    parser.add_argument("--conformers", type=int, default=10, help="maximim number of conformers")
    parser.add_argument("--skip-xtb-opt", action="store_true", help="skip xtb geometry optimization")
    args = parser.parse_args()

    with sqlite3.connect(args.db) as conn:
        cursor = conn.cursor()
        cursor.execute("""CREATE TABLE IF NOT EXISTS reaction (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            rc TEXT,
            smiles TEXT,
            charge INTEGER,
            alpha INTEGER,
            beta INTEGER,
            xyz TEXT )""")
        conn.commit()

        conformers = {}
        batch = []

        with gzip.open(args.sdf, "rb") as f:
            with Chem.ForwardSDMolSupplier(f) as supp:
                for mol in supp:
                    name = mol.GetProp("_Name")
                    if args.name_startswith and (not name.startswith(args.name_startswith)):
                        continue
                    if name in conformers:
                        conformers[name] += 1
                    else:
                        conformers[name] = 1
                    if conformers[name] > args.conformers:
                        continue
                    confid = f"{name}.{conformers[name]}"
                    batch.append((confid, mol))
                    
        with tqdm(batch, total= len(batch)) as pbar: 
            for confid, mol in batch:
                pbar.set_postfix_str(f"processing: {confid}")
                smiles = Chem.MolToSmiles(mol)
                try:
                    rxn = Reaction(reactant=mol, thiol_smiles="SC", verbose=False)
                except:
                    # no implemented pattern detected
                    print(f"Skipping {confid} because applicable reaction pattern is not found")
                    continue

                for (rc, smiles_, charge_, alpha, beta, geom) in [
                    ('reactant', 
                        rxn.reactant_smiles, 
                        rxn.reactant_charge,
                        rxn.alpha_idx,
                        rxn.beta_idx,
                        Geometry(rdmol=rxn.reactant_rdmol, charge=rxn.reactant_charge)),
                    ('intermediate', 
                        rxn.carbanion_smiles, 
                        rxn.carbanion_charge, 
                        rxn.carbanion_alpha_idx,
                        rxn.carbanion_beta_idx,
                        Geometry(rdmol=rxn.carbanion_rdmol, charge=rxn.carbanion_charge)), 
                    ('product', 
                        rxn.product_smiles,
                        rxn.product_charge,
                        rxn.product_alpha_idx,
                        rxn.product_beta_idx,
                        Geometry(rdmol=rxn.product_rdmol, charge=rxn.product_charge)),
                    ]:

                    if not args.skip_xtb_opt:
                        geom.xtb_optimize(water='alpb')

                    cursor.execute("""
                        INSERT OR IGNORE INTO reaction (name, rc, smiles, charge, alpha, beta, xyz)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (confid, rc, smiles_, charge_, alpha, beta, geom.write_xyz())
                        )
                pbar.update(1)
                
        conn.commit()
        
    conn.close()


def run_aimnet2_workflow():
    # docker run -it --rm --gpus all -v .:/home/appuser aimnet2:nse bash

    try:
        from ase.optimize import BFGS
        from ase.io import read, write
        from ase import units
        from aimnet.calculators import AIMNet2Calculator, AIMNet2ASE
    except ImportError as e:
        print(
            f"Missing Docker dependency: {e}\n"
            "docker run -it --rm --gpus all -v .:/home/appuser aimnet2:nse bash",
            file=sys.stderr
        )
        sys.exit(1)


    class Models(Enum):
        """
        isayevlab/aimnet2-wb97m-d3	aimnet2	        General organic chemistry
        isayevlab/aimnet2-nse	    aimnet2-nse	    Open-shell chemistry
        isayevlab/aimnet2-2025	    aimnet2-2025	Improved intermolecular interactions
        """
        _wb97m = 'isayevlab/aimnet2-wb97m-d3'
        _nse   = 'isayevlab/aimnet2-nse'
        _2025  = 'isayevlab/aimnet2-2025'


    parser = argparse.ArgumentParser(description="Run Covalent Warhead Workflow")
    parser.add_argument("db", type=str, help="input sqlite3 .db file")
    parser.add_argument("--model", type=str, choices=[e.value for e in Models], default=Models._wb97m.value)
    args = parser.parse_args()

    model_alias = args.model.split("/")[-1]
    prefix = Path(args.db).stem
    o_dbfile = Path(f"{prefix}_{model_alias}.db")
    o_csvfile = Path(f"{prefix}_{model_alias}.csv")


    device="cuda"
    calculator = AIMNet2Calculator(args.model, compile_model=True, device=device) 
    # compile_model=True is recommended for GPU usage

    with sqlite3.connect(args.db) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, rc, smiles, charge, alpha, beta, xyz FROM reaction")
        workload = cursor.fetchall()
    conn.close()


    with sqlite3.connect(o_dbfile) as conn:
        cursor = conn.cursor()
        cursor.execute("""CREATE TABLE IF NOT EXISTS optimized (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            rc TEXT,
            smiles TEXT,
            charge INTEGER,
            alpha INTEGER,
            beta INTEGER,
            xyz TEXT
            )""")
        conn.commit()

        with open(o_csvfile, "w") as out_csvfile:
            writer = csv.DictWriter(out_csvfile, fieldnames=[
                'Name', 'RC', 'SMILES', 'charge', 'QCa', 'QCb', 'E_HOMO', 'E_LUMO',
                'f_plus_Cb', 'f_minus_Cb', 'f_zero_Cb', 'f_plus_Ca', 'f_minus_Ca', 'f_zero_Ca',
                'E(opt, kcal/mol)', 'time(opt, sec)', 'E(rattled, kcal/mol)', 'E(initial, kcal/mol)'])
            writer.writeheader()

            for row in workload:
                row_id, name, rc, smiles, charge, alpha_idx, beta_idx, xyz = row
                print(f"Processing {name} ...")

                ase_atoms = read(StringIO(xyz), format='xyz') 
                # read with ASE to get the correct format for the calculator
                ase_atoms.calc = AIMNet2ASE(calculator, charge=charge, mult=1)
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

                partial_charges = ase_atoms.calc.results["charges"]
                QCa = partial_charges[alpha_idx]
                QCb = partial_charges[beta_idx]

                # calculate cation energy +1 on the same geometry (vertical IP)
                ase_atoms.calc = AIMNet2ASE(calculator, charge=charge+1, mult=1)
                energy_eV_plus1 = ase_atoms.get_potential_energy() # eV
                partial_charges_plus1 = ase_atoms.calc.results["charges"]

                # calculate cation energy -1 on the same geometry (vertical EA)
                ase_atoms.calc = AIMNet2ASE(calculator, charge=charge-1, mult=1)
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

                end_time = time.perf_counter()

                string_buffer = StringIO()
                write(string_buffer, ase_atoms, format='xyz')
                
                writer.writerow({'Name': name, 
                                'RC': rc, 
                                'charge': charge,
                                'SMILES': smiles,
                                'QCa': QCa,
                                'QCb': QCb,
                                'f_plus_Cb': f_plus[beta_idx],
                                'f_minus_Cb': f_minus[beta_idx],
                                'f_zero_Cb': f_zero[beta_idx],
                                'f_plus_Ca': f_plus[alpha_idx],
                                'f_minus_Ca': f_minus[alpha_idx],
                                'f_zero_Ca': f_zero[alpha_idx],
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