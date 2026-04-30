"""
Fukui Function Calculator using Psi4
=====================================
Computes Fukui functions f+, f-, and f0 via finite differences of
electron density on a real-space grid, using Mulliken/Hirshfeld
condensed-to-atom fukui indices as well.

Requirements:
    pip install psi4 numpy matplotlib

Usage:
    python fukui_psi4.py

Adjust the CONFIGURATION section below for your molecule.
"""

import numpy as np
import psi4
import os

from covalent import Geometry
from pathlib import Path

m = Geometry(smiles="C=CC(=O)N")  # Acrylamide
if Path("xacrylamide.xyz").exists():
    m.update_coords("acrylamide.xyz")
else:
    m.pre_optimize() # Pre-optimization with GFN2-xTB to get a reasonable starting geometry for Psi4 DFT optimization.
    m.write_xyz("acrylamide_xtb.xyz")  # Save the pre-optimized geometry for reference
    m.optimize() # B3LYP/6-31G* optimization to get a good geometry for the Fukui calculation
    m.write_xyz("acrylamide_dft.xyz")

# m.update_coords("acrylamide_dft.xyz")  # Load the DFT-optimized geometry for the Fukui calculation
MOLECULE_XYZ = f"{m.charge} 1\n{m.xyz_block}\n"
# Format: "charge multiplicity\n<xyz coords>"  
 
# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit this section
# ─────────────────────────────────────────────────────────────────────────────

# MOLECULE_XYZ = """
# 0 1
# O   0.000000   0.000000   0.117176
# H   0.000000   0.757306  -0.468706
# H   0.000000  -0.757306  -0.468706
# """

# ^ Format: "charge multiplicity\n<xyz coords>"
# For cations:  "1 1\n..."
# For radicals: "0 2\n..."
# OPT_DISPERSION = "d3bj"        # Grimme D3(BJ) dispersion correction
# BASIS = "6-31+G*" # Diffuse functions (+) essential for anions
# METHOD    = "b3lyp" 
BASIS = "6-311+G(2d,2p)"
METHOD = "wb97x-d"
MEMORY    = "2 GB"
NTHREADS  = 4

# Grid settings for real-space density cube files (via Psi4 cubeprop)
GRID_SPACING = 0.2          # Voxel spacing in Ångströms (smaller = finer, slower)
GRID_PADDING = 3.0          # Ångströms of padding around molecule bounding box

OUTPUT_DIR = "fukui_output"  # Directory for output files

# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

psi4.set_memory(MEMORY)
psi4.set_num_threads(NTHREADS)
psi4.core.set_output_file(f"{OUTPUT_DIR}/psi4_output.log", False)


def parse_charge_mult(xyz_block):
    """Extract charge and multiplicity from the XYZ block header."""
    lines = xyz_block.strip().splitlines()
    charge, mult = map(int, lines[0].split())
    return charge, mult


# Atomic numbers by symbol (covers H–Xe, sufficient for most organic/inorganic work)
_ATOMIC_NUMBERS = {
    "H":1,"HE":2,"LI":3,"BE":4,"B":5,"C":6,"N":7,"O":8,"F":9,"NE":10,
    "NA":11,"MG":12,"AL":13,"SI":14,"P":15,"S":16,"CL":17,"AR":18,
    "K":19,"CA":20,"SC":21,"TI":22,"V":23,"CR":24,"MN":25,"FE":26,
    "CO":27,"NI":28,"CU":29,"ZN":30,"GA":31,"GE":32,"AS":33,"SE":34,
    "BR":35,"KR":36,"RB":37,"SR":38,"Y":39,"ZR":40,"NB":41,"MO":42,
    "TC":43,"RU":44,"RH":45,"PD":46,"AG":47,"CD":48,"IN":49,"SN":50,
    "SB":51,"TE":52,"I":53,"XE":54,
}


def count_electrons(xyz_block, charge_offset=0):
    """Count total electrons = sum(atomic numbers) - (charge + charge_offset)."""
    lines = xyz_block.strip().splitlines()
    orig_charge = int(lines[0].split()[0])
    total_charge = orig_charge + charge_offset
    n_protons = sum(
        _ATOMIC_NUMBERS[line.split()[0].upper()]
        for line in lines[1:]
        if line.strip()
    )
    return n_protons - total_charge


def lowest_mult(n_electrons):
    """
    Return the lowest valid multiplicity for a given electron count.
    Even electrons → singlet (1); odd electrons → doublet (2).
    """
    return 1 if n_electrons % 2 == 0 else 2


def build_molecule(xyz_block, charge_offset=0):
    """
    Build a Psi4 molecule from an XYZ block with an optional charge offset.
    Multiplicity is determined automatically from the electron count so that
    the correct spin state is always used (singlet for even-e, doublet for odd-e).
    """
    lines = xyz_block.strip().splitlines()
    orig_charge = int(lines[0].split()[0])
    new_charge = orig_charge + charge_offset

    n_elec = count_electrons(xyz_block, charge_offset)
    new_mult = lowest_mult(n_elec)

    coord_block = "\n".join(line for line in lines[1:] if line.strip())
    mol_str = f"{new_charge} {new_mult}\n{coord_block}"

    mol = psi4.geometry(mol_str)
    mol.update_geometry()
    return mol, new_mult


def run_calculation(mol, mult, method, basis):
    """
    Run a single-point energy+wavefunction calculation.
    Automatically selects RHF/RKS (singlet) or UHF/UKS (open-shell) reference.
    """
    psi4.core.clean()
    reference = "rhf" if mult == 1 else "uhf"
    psi4.set_options({
        "basis": basis,
        "reference": reference,
        "scf_type": "df",
        "e_convergence": 1e-8,
        "d_convergence": 1e-8,
        "dft_spherical_points": 590,
        "dft_radial_points": 99,
        # "dft_dispersion_parameters": ["d3bj"],   # activates D3(BJ)
    })
    energy, wfn = psi4.energy(f"{method}/{basis}", molecule=mol, return_wfn=True)
    return energy, wfn


def get_atom_populations(wfn):
    """
    Return Mulliken atomic populations (electrons per atom) from a wavefunction.
    """
    psi4.oeprop(wfn, "MULLIKEN_CHARGES")
    charges = np.array(wfn.atomic_point_charges())
    mol = wfn.molecule()
    nuclear_charges = np.array([mol.Z(i) for i in range(mol.natom())])
    populations = nuclear_charges - charges  # N_e per atom
    return populations


def condensed_fukui(pop_N, pop_Np1, pop_Nm1):
    """
    Compute condensed-to-atom Fukui indices.

    f+_k = q_k(N+1) - q_k(N)   [electrophilic attack]
    f-_k = q_k(N-1) - q_k(N)   [nucleophilic attack]
    f0_k = (f+_k + f-_k) / 2   [radical attack]

    where q_k are Mulliken populations.
    """
    f_plus  = pop_Np1 - pop_N
    f_minus = pop_N - pop_Nm1 
    f_zero  = 0.5 * (f_plus + f_minus)
    return f_plus, f_minus, f_zero


def dump_density_cube(wfn, label, outdir, padding=GRID_PADDING, spacing=GRID_SPACING):
    """
    Use Psi4's built-in cubeprop to write the total electron density as a
    Gaussian CUBE file.  Returns the path to the resulting file.

    cubeprop writes  Dt.cube (total density) into cubeprop_filepath.
    We use a per-call temp subdir so successive calls don't collide.
    """
    import shutil, tempfile

    tmpdir = tempfile.mkdtemp(prefix=f"cubeprop_{label}_", dir=outdir)
    psi4.set_options({
        "cubeprop_tasks":     ["density"],
        "cubeprop_filepath":  tmpdir,
        "cubic_grid_overage": [padding, padding, padding],
        "cubic_grid_spacing": [spacing, spacing, spacing],
    })
    psi4.cubeprop(wfn)

    src = os.path.join(tmpdir, "Dt.cube")
    dst = os.path.join(outdir, f"density_{label}.cube")
    shutil.copy(src, dst)
    shutil.rmtree(tmpdir)
    return dst


def read_cube(filepath):
    """
    Parse a Gaussian CUBE file.
    Returns (origin, axes, natom, atom_info, data_3d).
      origin    : (3,) float array, bohr
      axes      : (3,3) float array; axes[i] = voxel step vector for axis i
      natom     : int
      atom_info : list of (Z, x, y, z) tuples (bohr)
      data_3d   : numpy array shape (nx, ny, nz)
    """
    with open(filepath) as fh:
        lines = fh.readlines()

    natom = abs(int(lines[2].split()[0]))
    origin = np.array(list(map(float, lines[2].split()[1:4])))

    npts = np.zeros(3, dtype=int)
    axes = np.zeros((3, 3))
    for i in range(3):
        parts = lines[3 + i].split()
        npts[i] = int(parts[0])
        axes[i] = list(map(float, parts[1:4]))

    atom_info = []
    for i in range(natom):
        p = lines[6 + i].split()
        atom_info.append((int(p[0]), float(p[2]), float(p[3]), float(p[4])))

    flat = []
    for line in lines[6 + natom:]:
        flat.extend(map(float, line.split()))

    return origin, axes, natom, atom_info, np.array(flat).reshape(npts)


def write_cube(filepath, origin, axes, natom, atom_info, data_3d, label="density"):
    """Write a numpy array as a Gaussian CUBE file."""
    npts = data_3d.shape
    with open(filepath, "w") as fh:
        fh.write(f"Fukui {label}\nGenerated by fukui_psi4.py\n")
        fh.write(f"{natom:5d} {origin[0]:12.6f} {origin[1]:12.6f} {origin[2]:12.6f}\n")
        for i in range(3):
            fh.write(f"{npts[i]:5d} {axes[i,0]:12.6f} {axes[i,1]:12.6f} {axes[i,2]:12.6f}\n")
        for Z, x, y, z in atom_info:
            fh.write(f"{Z:5d} {float(Z):12.6f} {x:12.6f} {y:12.6f} {z:12.6f}\n")
        vals = data_3d.ravel()
        for idx, v in enumerate(vals):
            fh.write(f"{v:13.5E}")
            if (idx + 1) % 6 == 0:
                fh.write("\n")
        fh.write("\n")
    print(f"  Saved: {filepath}")





def print_results_table(wfn_N, f_plus, f_minus, f_zero):
    """Pretty-print condensed Fukui indices per atom."""
    mol = wfn_N.molecule()
    natom = mol.natom()
    symbols = [mol.symbol(i) for i in range(natom)]

    header = f"{'Atom':>6} {'Symbol':>8} {'f+ (elec)':>12} {'f- (nuc)':>12} {'f0 (rad)':>12}"
    sep = "-" * len(header)
    print("\n" + sep)
    print("  Condensed Fukui Indices (Mulliken)")
    print(sep)
    print(header)
    print(sep)
    for i in range(natom):
        print(f"{i+1:>6} {symbols[i]:>8} {f_plus[i]:>12.4f} {f_minus[i]:>12.4f} {f_zero[i]:>12.4f}")
    print(sep + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Fukui Function Calculator — Psi4")
    print("=" * 60)

    charge, mult = parse_charge_mult(MOLECULE_XYZ)
    print(f"\nMolecule: charge={charge}, mult={mult}, method={METHOD}/{BASIS}")

    # ── Step 1: Neutral molecule (N electrons) ──────────────────────────────
    print("\n[1/3] Computing N-electron system...")
    mol_N, mult_N = build_molecule(MOLECULE_XYZ, charge_offset=0)
    print(f"      charge={charge}, mult={mult_N}")
    E_N, wfn_N = run_calculation(mol_N, mult_N, METHOD, BASIS)
    pop_N = get_atom_populations(wfn_N)
    print(f"      E(N)   = {E_N:.8f} Eh")

    # ── Step 2: Anion (N+1 electrons) ───────────────────────────────────────
    print("[2/3] Computing (N+1)-electron system (anion)...")
    mol_Np1, mult_Np1 = build_molecule(MOLECULE_XYZ, charge_offset=-1)
    print(f"      charge={charge-1}, mult={mult_Np1}  (auto-assigned)")
    E_Np1, wfn_Np1 = run_calculation(mol_Np1, mult_Np1, METHOD, BASIS)
    pop_Np1 = get_atom_populations(wfn_Np1)
    print(f"      E(N+1) = {E_Np1:.8f} Eh")

    # ── Step 3: Cation (N-1 electrons) ──────────────────────────────────────
    print("[3/3] Computing (N-1)-electron system (cation)...")
    mol_Nm1, mult_Nm1 = build_molecule(MOLECULE_XYZ, charge_offset=+1)
    print(f"      charge={charge+1}, mult={mult_Nm1}  (auto-assigned)")
    E_Nm1, wfn_Nm1 = run_calculation(mol_Nm1, mult_Nm1, METHOD, BASIS)
    pop_Nm1 = get_atom_populations(wfn_Nm1)
    print(f"      E(N-1) = {E_Nm1:.8f} Eh")

    # ── Condensed Fukui indices ──────────────────────────────────────────────
    f_plus, f_minus, f_zero = condensed_fukui(pop_N, pop_Np1, pop_Nm1)
    print_results_table(wfn_N, f_plus, f_minus, f_zero)

    # Save condensed results to CSV
    mol = wfn_N.molecule()
    natom = mol.natom()
    csv_path = f"{OUTPUT_DIR}/condensed_fukui.csv"
    with open(csv_path, "w") as f:
        f.write("atom_index,symbol,f_plus,f_minus,f_zero\n")
        for i in range(natom):
            f.write(f"{i+1},{mol.symbol(i)},{f_plus[i]:.6f},{f_minus[i]:.6f},{f_zero[i]:.6f}\n")
    print(f"  Condensed Fukui indices saved to: {csv_path}")

    # ── Real-space Fukui functions via cubeprop ──────────────────────────────
    print("\nGenerating real-space electron density CUBE files via cubeprop...")
    print(f"  (grid spacing {GRID_SPACING} Å, padding {GRID_PADDING} Å — edit in CONFIGURATION)")

    cube_N   = dump_density_cube(wfn_N,   "N",   OUTPUT_DIR)
    cube_Np1 = dump_density_cube(wfn_Np1, "Np1", OUTPUT_DIR)
    cube_Nm1 = dump_density_cube(wfn_Nm1, "Nm1", OUTPUT_DIR)

    print("  Reading CUBE files...")
    origin, axes, natom_c, atom_info, rho_N   = read_cube(cube_N)
    _,      _,    _,       _,         rho_Np1 = read_cube(cube_Np1)
    _,      _,    _,       _,         rho_Nm1 = read_cube(cube_Nm1)

    # Verify grids are compatible (same shape)
    if not (rho_N.shape == rho_Np1.shape == rho_Nm1.shape):
        raise RuntimeError(
            f"Cube grid shapes differ: {rho_N.shape} vs {rho_Np1.shape} vs {rho_Nm1.shape}. "
            "This should not happen if all three cubeprop calls use the same molecule geometry."
        )

    # Finite-difference Fukui functions
    fukui_plus  = rho_Np1 - rho_N     # f+(r) for electrophilic attack
    fukui_minus = rho_N - rho_Nm1     # f-(r) for nucleophilic attack
    fukui_zero  = 0.5 * (fukui_plus + fukui_minus)  # f0(r) for radical attack

    print("\nWriting Fukui CUBE files...")
    write_cube(f"{OUTPUT_DIR}/fukui_plus.cube",  origin, axes, natom_c, atom_info, fukui_plus,  "f+")
    write_cube(f"{OUTPUT_DIR}/fukui_minus.cube", origin, axes, natom_c, atom_info, fukui_minus, "f-")
    write_cube(f"{OUTPUT_DIR}/fukui_zero.cube",  origin, axes, natom_c, atom_info, fukui_zero,  "f0")

    # ── Optional: 2D slice plot ──────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        mid = fukui_plus.shape[2] // 2   # central z-slice index
        fig, axes_plt = plt.subplots(1, 3, figsize=(15, 4))
        titles = ["f⁺ (electrophilic)", "f⁻ (nucleophilic)", "f⁰ (radical)"]
        data_slices = [
            fukui_plus[:, :, mid],
            fukui_minus[:, :, mid],
            fukui_zero[:, :, mid],
        ]

        for ax, title, dat in zip(axes_plt, titles, data_slices):
            vmax = np.abs(dat).max() or 1e-6
            im = ax.imshow(dat.T, origin="lower", cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax, interpolation="bilinear")
            ax.set_title(title, fontsize=13)
            ax.set_xlabel("x (grid pts)")
            ax.set_ylabel("y (grid pts)")
            plt.colorbar(im, ax=ax, label="e/bohr³")

        plt.suptitle("Fukui Functions — Central z-slice", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plot_path = f"{OUTPUT_DIR}/fukui_slices.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  2D slice plot saved: {plot_path}")
    except ImportError:
        print("\n  (matplotlib not found — skipping 2D slice plot)")

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  DONE — outputs written to:", OUTPUT_DIR)
    print("    condensed_fukui.csv    — atom-condensed Fukui indices")
    print("    density_N.cube         — rho(N) total density")
    print("    density_Np1.cube       — rho(N+1) total density")
    print("    density_Nm1.cube       — rho(N-1) total density")
    print("    fukui_plus.cube        — f+(r)  electrophilic attack")
    print("    fukui_minus.cube       — f-(r)  nucleophilic attack")
    print("    fukui_zero.cube        — f0(r)  radical attack")
    print("    psi4_output.log        — Psi4 log")
    print("  Visualize .cube files with VMD, VESTA, or Avogadro")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
