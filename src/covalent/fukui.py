import psi4
import numpy as np
from pathlib import Path
from .geometry import Geometry


class FukuiIndex:
    def __init__(self, 
                 geometry: Geometry,
                 functional: str = "wb97x-d",
                 basis: str = "6-311+G(2d,2p)",
                 memory: str = "4 GB",
                 num_threads: int = 4,
                 output_dir: str | None = None):
        self.geometry = geometry
        self.basis = basis
        self.functional = functional
        self.systems = {
            0: {
                'mol': self.geometry.psi4_mol, 
                'mult': self.geometry.mult, 
                'energy': None,
                'wfn': None,
                'pop': None,
                },
            }
        
        self.plus = None
        self.minus = None
        self.zero = None

        psi4.set_memory(memory)
        psi4.set_num_threads(num_threads)
            
        if output_dir is not None:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            psi4.core.set_output_file(f"{output_dir}/psi4_output.log", False)
        else:
            self.output_dir = None

        self.add_system(delta=-1), # one less electrons, positive
        self.add_system(delta=+1), # one more electrons, negative



    def add_system(self, delta: int) -> None:
        """
        Build a Psi4 molecule from an XYZ block with an optional charge offset.
        Multiplicity is determined automatically from the electron count so that
        the correct spin state is always used (singlet for even-e, doublet for odd-e).
        """
        ne = self.geometry.count_electrons()
        charge = self.geometry.charge - delta
        mult = 1 if (ne + delta) % 2 == 0 else 2
        mol_str = f"{charge} {mult}\n{self.geometry.xyz_block}"
        mol = psi4.geometry(mol_str)
        mol.update_geometry()
        self.systems[delta] = {'mol': mol, 'mult': mult, 'energy': None, 'wfn': None, 'pop': None}
        

    @staticmethod
    def get_atom_populations(wfn) -> np.ndarray:
        """
        Return Mulliken atomic populations (electrons per atom) from a wavefunction.
        """
        psi4.oeprop(wfn, "MULLIKEN_CHARGES")
        charges = np.array(wfn.atomic_point_charges())
        mol = wfn.molecule()
        nuclear_charges = np.array([mol.Z(i) for i in range(mol.natom())])
        populations = nuclear_charges - charges  # N_e per atom
        return populations


    def run(self) -> None:
        """
        Run a single-point energy + wavefunction calculation.
        Automatically selects RHF/RKS (singlet) or UHF/UKS (open-shell) reference.
        """

        for delta, datadict in self.systems.items():
            psi4.core.clean()
            psi4.set_options({
                "basis": self.basis,
                "reference": "rhf" if datadict['mult'] == 1 else "uhf",
                "scf_type": "df",
                "e_convergence": 1e-8,
                "d_convergence": 1e-8,
                "dft_spherical_points": 590,
                "dft_radial_points": 99,
            })
            energy, wfn = psi4.energy(f"{self.functional}/{self.basis}", 
                                    molecule=datadict['mol'], 
                                    return_wfn=True)
            datadict['energy'] = energy
            datadict['wfn'] = wfn
            datadict['pop'] = self.get_atom_populations(wfn)
    
        """
        Compute condensed-to-atom Fukui indices.

        f+_k = q_k(N+1) - q_k(N)   [electrophilic attack]
        f-_k = q_k(N-1) - q_k(N)   [nucleophilic attack]
        f0_k = (f+_k + f-_k) / 2   [radical attack]

        where q_k are Mulliken populations.
        """
        self.plus  = self.systems[1]['pop'] - self.systems[0]['pop']
        self.minus = self.systems[0]['pop'] - self.systems[-1]['pop'] 
        self.zero  = 0.5 * (self.plus + self.minus)


    def show(self):
        mol = self.systems[0]['wfn'].molecule()
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
            print(f"{i+1:>6} {symbols[i]:>8} {self.plus[i]:>12.4f} {self.minus[i]:>12.4f} {self.zero[i]:>12.4f}")
        print(sep + "\n")