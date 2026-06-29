# docker run -it --rm --gpus all -v .:/home/appuser aimnet2:nse bash

import ase
import json
import time

from pathlib import Path
from collections import defaultdict
from ase.optimize import BFGS
from ase.io import read, iread, write
from ase import units
from ase import Atoms

from aimnet.calculators import AIMNet2Calculator, AIMNet2ASE

# isayevlab/aimnet2-wb97m-d3	aimnet2	        General organic chemistry
# isayevlab/aimnet2-2025	    aimnet2-2025	Improved intermolecular interactions
# isayevlab/aimnet2-nse	        aimnet2-nse	    Open-shell chemistry


import gzip
import re
import numpy as np


# SDF atom symbol → used directly; charge map for V2000 charge field
_CHG_MAP = {0: 0, 1: 3, 2: 2, 3: 1, 4: 0, 5: -1, 6: -2, 7: -3}


def _parse_v2000_block(atom_lines, bond_lines=None):
    """Parse atom block from V2000 molfile. Returns (symbols, positions)."""
    symbols = []
    positions = []
    for line in atom_lines:
        # V2000 atom line: x y z symbol [dd] [charge] ...
        parts = line.split()
        if len(parts) < 4:
            continue
        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
        sym = parts[3]
        symbols.append(sym)
        positions.append([x, y, z])
    return symbols, positions


def _parse_v3000_block(lines):
    """Parse V3000 CTAB atom block. Returns (symbols, positions)."""
    symbols = []
    positions = []
    in_atom_block = False
    for line in lines:
        line = line.strip()
        if line.upper() == "M  V30 BEGIN ATOM":
            in_atom_block = True
            continue
        if line.upper() == "M  V30 END ATOM":
            in_atom_block = False
            continue
        if in_atom_block and line.startswith("M  V30"):
            # M  V30 index type x y z map [options]
            parts = line.split()
            # parts: ['M', 'V30', index, type, x, y, z, ...]
            if len(parts) < 7:
                continue
            sym = parts[3]
            x, y, z = float(parts[4]), float(parts[5]), float(parts[6])
            symbols.append(sym)
            positions.append([x, y, z])
    return symbols, positions


def read_sdf_gz(filepath: Path, sanitize=True):
    """
    Parse a (possibly gzipped) multi-conformer SDF file without RDKit.

    Parameters
    ----------
    filepath : str
        Path to .sdf or .sdf.gz file.
    sanitize : bool
        If True, strip trailing whitespace/nulls from element symbols.

    Returns
    -------
    list of ase.Atoms
        One Atoms object per conformer ($$$$-delimited record).
    """
    opener = gzip.open if filepath.suffix == ".gz" else open
    mode = "rt"  # text mode — works for both gzip.open and open

    conformers = []

    with opener(filepath, mode, encoding="utf-8", errors="replace") as fh:
        record_lines = []
        for raw_line in fh:
            line = raw_line.rstrip("\n")
            if line.strip() == "$$$$":
                atoms = _molblock_to_atoms(record_lines, sanitize=sanitize)
                if atoms is not None:
                    conformers.append(atoms)
                record_lines = []
            else:
                record_lines.append(line)

        # Handle file without trailing $$$$
        if record_lines:
            atoms = _molblock_to_atoms(record_lines, sanitize=sanitize)
            if atoms is not None:
                conformers.append(atoms)

    return conformers


def _molblock_to_atoms(lines, sanitize=True):
    """
    Convert a single molfile block (list of str) to an ase.Atoms object.
    Supports both V2000 and V3000 CTAB formats.
    """
    if len(lines) < 4:
        return None

    # Lines 0-2: molecule name, program/date, comment
    mol_name = lines[0].strip()

    # Line 3: counts line (V2000) or "M  V30 COUNTS ..." (V3000)
    counts_line = lines[3]

    # Detect format
    is_v3000 = "V3000" in counts_line or any("M  V30" in l for l in lines[:10])

    if is_v3000:
        symbols, positions = _parse_v3000_block(lines[4:])
    else:
        # V2000: counts line cols 0-2 = num_atoms, num_bonds, ...
        try:
            n_atoms = int(counts_line[0:3].strip())
            n_bonds = int(counts_line[3:6].strip())
        except ValueError:
            return None

        atom_start = 4
        atom_end = atom_start + n_atoms
        atom_lines = lines[atom_start:atom_end]
        bond_lines = lines[atom_end: atom_end + n_bonds]
        symbols, positions = _parse_v2000_block(atom_lines, bond_lines)

    if not symbols:
        return None

    if sanitize:
        # Strip junk characters sometimes present in element symbols
        symbols = [re.sub(r"[^A-Za-z]", "", s).capitalize() for s in symbols]

    atoms = Atoms(
        symbols=symbols,
        positions=np.array(positions, dtype=float),
    )
    atoms.info["mol_name"] = mol_name
    return atoms



for model_name in [
    "isayevlab/aimnet2-nse",
    "isayevlab/aimnet2-wb97m-d3",
    "isayevlab/aimnet2-2025",
    ]:

    model_name_alias = f"{model_name.split('/')[-1]}" # e.g. "aimnet2-nse"

    device="cuda" # or device="cuda"
    calc_nse = AIMNet2Calculator(model_name, compile_model=True, device=device) 
    # compile_model=True is recommended for GPU usage

    for dataset in [
        '../data/Danilack_et_al_2024/si/S2_conformers', 
        '../data/Danilack_et_al_2024/si/S3_conformers', 
        '../data/Danilack_et_al_2024/si/S4_conformers', 
        '../data/Danilack_et_al_2024/si/S5_conformers']:
        workdir = Path(dataset)
        infile = workdir / "original_rc_fixpka_omega.sdf.gz"
        dataset_name = dataset.split("/")[-1].replace("_conformers", "")
        outdir = workdir.parent.parent / f"{model_name_alias}_opt/{dataset_name}_conformers"
        outdir.mkdir(exist_ok=True, parents=True)
        
        print(f"Processing dataset: {dataset}")
        print(f"Input directory: {workdir}")
        print(f"Output directory: {outdir}\n")

        outdir.mkdir(exist_ok=True, parents=True)
        outfile = outdir / "energies.json"

        data = defaultdict(list)

        for sdf_file in workdir.glob("*.sdf.gz"):
            if sdf_file.name == "original_rc_fixpka_omega.sdf.gz":
                continue
            name, rc = sdf_file.stem.split("_")

            conformers = read_sdf_gz(sdf_file)
            print(f"Processing {sdf_file} (RC: {rc})")
            print(f"Number of conformers: {len(conformers)}\n")

            for idx, ase_atoms in enumerate(conformers): 
                out_xyz_file = outdir / f"{name}_{rc}_{idx:02d}.xyz"
                
                if rc == "intermediate":
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
                dyn = BFGS(ase_atoms, logfile=None, trajectory=None)
                dyn.run(fmax=0.01)
                energy_eV_opt = ase_atoms.get_potential_energy() # eV
                end_time = time.perf_counter()

                print(f"Optimized Energy: {energy_eV_opt:.4f} eV")
                print(f"Optimization Time: {end_time - start_time:.2f} seconds\n")

                # write optimized geometry to xyz file
                # print(f"Writing optimized geometry to {outdir / out_xyz_file}")
                # write(outdir / out_xyz_file, ase_atoms) 

                data[name].append({
                    'E(initial, kcal/mol)': energy_eV_ini * (units.mol / units.kcal),
                    'E(rattled, kcal/mol)': energy_eV_rattled * (units.mol / units.kcal),
                    'E(opt, kcal/mol)': energy_eV_opt * (units.mol / units.kcal),
                    'time(opt, sec)': end_time - start_time, # (sec)
                })

        with open(outfile, "w") as f:
            json.dump(data, f, indent=4)