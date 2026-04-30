import psi4
import numpy as np
from .geometry import Geometry
from pathlib import Path


class FukuiIndices(Geometry):
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

    def __init__(self, 
                 smiles: str, 
                 charge: int = 0, 
                 multiplicity: int = 1,
                 basis: str = "6-311+G(2d,2p)",
                 functional: str = "wb97x-d",
                 memory: str = "4 GB",
                 num_threads: int = 4,
                 output_dir: str | None = None):
        super().__init__(smiles, charge, multiplicity)
        self.basis = basis
        self.functional = functional
        self.psi4_mol_plus = None # one less electron than parent, thus cation if parent is neutral
        self.psi4_mol_plus_mult = None
        self.psi4_mol_minus = None # one more electron than parent, thus anion if parent is neutral
        self.psi4_mol_minus_mult = None
        self.build()
        
        # Grid settings for real-space density cube files (via Psi4 cubeprop)
        self.grid_spacing = 0.2          # Voxel spacing in Ångströms (smaller = finer, slower)
        self.grid_padding = 3.0          # Ångströms of padding around molecule bounding box

        if output_dir is not None:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            psi4.set_memory(memory)
            psi4.set_num_threads(num_threads)
            psi4.core.set_output_file(f"{output_dir}/psi4_output.log", False)
        else:
            self.output_dir = None


    def build(self) -> None:
        """
        Build a Psi4 molecule from an XYZ block with an optional charge offset.
        Multiplicity is determined automatically from the electron count so that
        the correct spin state is always used (singlet for even-e, doublet for odd-e).
        """
        n_electrons = self.count_electrons() - 1
        plus_mult = 1 if n_electrons % 2 == 0 else 2
        mol_str = f"{self.charge + 1} {plus_mult}\n{self.xyz_block}"
        self.psi4_mol_plus = psi4.geometry(mol_str)
        self.psi4_mol_plus = self.psi4_mol_plus.update_geometry()
        self.psi4_mol_plus_mult = plus_mult

        n_electrons = self.count_electrons() + 1
        minus_mult = 1 if n_electrons % 2 == 0 else 2
        mol_str = f"{self.charge - 1} {minus_mult}\n{self.xyz_block}"
        self.psi4_mol_minus = psi4.geometry(mol_str)
        self.psi4_mol_minus = self.psi4_mol_minus.update_geometry()
        self.psi4_mol_minus_mult = minus_mult
    

    def run(self) -> tuple:
        """
        Run a single-point energy+wavefunction calculation.
        Automatically selects RHF/RKS (singlet) or UHF/UKS (open-shell) reference.
        """
        psi4.core.clean()
        psi4.set_options({
            "basis": self.basis,
            "reference": "rhf" if self.psi4_mol_plus_mult == 1 else "uhf",
            "scf_type": "df",
            "e_convergence": 1e-8,
            "d_convergence": 1e-8,
            "dft_spherical_points": 590,
            "dft_radial_points": 99,
        })
        plus_energy, plus_wfn = psi4.energy(f"{self.functional}/{self.basis}", 
                                  molecule=self.psi4_mol_plus, 
                                  return_wfn=True)
        
        psi4.core.clean()
        psi4.set_options({
            "basis": self.basis,
            "reference": "rhf" if self.psi4_mol_minus_mult == 1 else "uhf",
            "scf_type": "df",
            "e_convergence": 1e-8,
            "d_convergence": 1e-8,
            "dft_spherical_points": 590,
            "dft_radial_points": 99,
        })
        minus_energy, minus_wfn = psi4.energy(f"{self.functional}/{self.basis}", 
                                  molecule=self.psi4_mol_minus, 
                                  return_wfn=True)
        
        return plus_energy, plus_wfn, minus_energy, minus_wfn


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

        
        
    
