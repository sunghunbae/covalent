import numpy as np
import psi4

from rdkit import Chem
from rdkit.Chem import AllChem


class Geometry():
    def __init__(self, smiles: str, charge: int = 0, mult: int = 1) -> None:
        """
        Parameters
        ----------
        xyz_block : str
            Atomic coordinates in XYZ format (element  x  y  z, one atom per line).
            Do NOT include the line-count header or comment line — just coordinates.
        charge : int
            Total molecular charge (0 = neutral, -1 = anion, +1 = cation).
        multiplicity : int
            Spin multiplicity (1 = singlet, 2 = doublet, …).

        """
        self.smiles = smiles
        self.charge = charge
        self.mult = mult
        self.rdmol = Chem.MolFromSmiles(smiles)
        self.rdmolH = Chem.AddHs(self.rdmol)
        self.symbols = [a.GetSymbol() for a in self.rdmolH.GetAtoms()]
        
        AllChem.EmbedMolecule(self.rdmolH, AllChem.ETKDGv3())
        AllChem.MMFFOptimizeMolecule(self.rdmolH)

        self.xyz_block = self.smiles_to_xyz_block()
        self.mol_str = f"{self.charge} {self.mult}\n{self.xyz_block}\n  symmetry c1\n  no_reorient\n  no_com"
        self.psi4_mol = psi4.geometry(self.mol_str)


    def smiles_to_xyz_block(self) -> str:
        """Convert SMILES to Psi4-ready XYZ block (MMFF pre-optimized)."""
        conf = self.rdmolH.GetConformer()
        atoms = self.rdmolH.GetAtoms()
        lines = []
        for atom in atoms:
            pos = conf.GetAtomPosition(atom.GetIdx())
            lines.append(f"  {atom.GetSymbol()}  {pos.x:.6f}  {pos.y:.6f}  {pos.z:.6f}")
        return "\n".join(lines)


    def update_geometry(self, coords: np.ndarray) -> None:
        """Update the geometry with new coordinates."""
        self.xyz_block = "\n".join(f"  {s}  {c[0]:.6f}  {c[1]:.6f}  {c[2]:.6f}"
                            for s, c in zip(self.symbols, coords))
        self.mol_str = f"{self.charge} {self.mult}\n{self.xyz_block}\n  symmetry c1\n  no_reorient\n  no_com"


    def single_point_energy(self, method: str = "wb97x-d", basis: str  = "6-311+G(2d,2p)") -> float:
        """
        Compute a single-point energy at a previously optimised geometry.

        Returns
        -------
        float : electronic energy in Hartree
        """
        psi4.set_options({"basis": basis})
        E_sp, _ = psi4.energy(f"{method}/{basis}", molecule=self.psi4_mol, return_wfn=True)
        
        return E_sp


    def optimize(self, method: str = 'b3lyp', basis: str = '6-31G*') -> None:
        """
        Embed with RDKit MMFF, then optimize with Psi4 DFT.
        Returns (coords_Nx3, atom_symbols).
        """
        psi4.set_memory('4 GB')
        psi4.set_num_threads(4)
        psi4.set_options({'basis': basis, 'scf_type': 'df',
                        'geom_maxiter': 300, 'd_convergence': 1e-8})

        mol = psi4.geometry(self.mol_str)
        psi4.optimize(f'{method}/{basis}', molecule=mol)

        # Extract optimized geometry
        opt_coords = np.array(mol.geometry().to_array()) * 0.529177  # bohr→Å
        self.update_geometry(opt_coords)
