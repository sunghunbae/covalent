import numpy as np
import psi4

from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops

# Note: psi4 uses Ångström units by default for geometry input

class Geometry():
    def __init__(self, smiles: str, charge: int = 0, mult: int = 1) -> None:
        """
        Parameters
        ----------
        smiles : str
            SMILES string representing the molecule.
        charge : int
            Total molecular charge (0 = neutral, -1 = anion, +1 = cation).
        multiplicity : int
            Spin multiplicity (1 = singlet, 2 = doublet, …).
        """
        self.smiles : str = smiles
        self.charge : int = charge
        self.mult : int = mult
        self.rdmol : Chem.Mol = Chem.MolFromSmiles(smiles)
        self.rdmolH : Chem.Mol = Chem.AddHs(self.rdmol)
        
        # 3D embedding and MMFF optimization with RDKit
        AllChem.EmbedMolecule(self.rdmolH, AllChem.ETKDGv3())
        AllChem.MMFFOptimizeMolecule(self.rdmolH)

        # self.charge = rdmolops.GetFormalCharge(self.rdmolH)
        self.natoms : int = self.rdmolH.GetNumAtoms()
        self.symbols : list[str] = [atom.GetSymbol() for atom in self.rdmolH.GetAtoms()]
        self.numbers : list[int] = [atom.GetAtomicNum() for atom in self.rdmolH.GetAtoms()]

        self.coords : np.ndarray = self.rdmolH.GetConformer().GetPositions()
        # Atomic coordinates in Angstroms, as a list of (x,y,z) tuples.

        self.xyz_block : str = self._update_xyz_block()
        # Atomic coordinates in XYZ format (element  x  y  z, one atom per line) (Angstroms).
        # Do NOT include the line-count header or comment line — just coordinates.
        
        # The XYZ block is used to create the psi4 molecule string, which also includes charge and multiplicity.
        self.mol_str : str = f"{self.charge} {self.mult}\n{self.xyz_block}\n  symmetry c1\n  no_reorient\n  no_com"
        
        # Create a psi4 molecule object from the XYZ string
        self.psi4_mol : psi4.core.Molecule = psi4.geometry(self.mol_str)


    def optimize(self, 
                 method: str = 'b3lyp', 
                 basis: str = '6-31G*', 
                 memory: str = '4 GB', 
                 num_threads: int = 4) -> None:
        """
        Embed with RDKit MMFF, then optimize with Psi4 DFT.
        Returns (coords_Nx3, atom_symbols).
        """
        psi4.set_memory(memory)
        psi4.set_num_threads(num_threads)
        psi4.set_options({'basis': basis, 'scf_type': 'df', 'geom_maxiter': 300, 'd_convergence': 1e-8})
        psi4.optimize(f'{method}/{basis}', molecule=self.psi4_mol)
        
        # self.psi4_mol is updated in-place by psi4.optimize, so we can directly access the new geometry
        # update xyz_block and mol_str with the optimized geometry
        self.coords = self.psi4_mol.geometry().to_array()  # Get optimized coordinates as a NumPy array
        lines = [f"{e:5}  {x:.6f}  {y:.6f}  {z:.6f}" for e, (x, y, z) in zip(self.symbols, self.coords)]
        self.xyz_block = "\n".join(lines)
        self.mol_str = f"{self.charge} {self.mult}\n{self.xyz_block}\n  symmetry c1\n  no_reorient\n  no_com"


    def write_xyz(self, 
                  filename: str, 
                  overwrite: bool = False) -> None:
        lines : list[str] = [f"{self.natoms}", " "]
        for e, (x, y, z) in zip(self.symbols, self.coords):
            lines.append(f"{e:5} {x:23.14f} {y:23.14f} {z:23.14f}")
        with open(filename, "w" if overwrite else "x") as f:
            # x mode will raise an error if the file already exists, 
            # preventing accidental overwrites
            f.write("\n".join(lines))

    
    def single_point_energy(self, 
                            method: str = "wb97x-d", 
                            basis: str  = "6-311+G(2d,2p)", 
                            memory: str = "4 GB", 
                            num_threads: int = 4) -> float:
        """
        Compute a single-point energy at a previously optimised geometry.

        Returns
        -------
        float : electronic energy in Hartree
        """
        psi4.set_memory(memory)
        psi4.set_num_threads(num_threads)
        psi4.set_options({"basis": basis})
        E_sp, _ = psi4.energy(f"{method}/{basis}", molecule=self.psi4_mol, return_wfn=True)
        
        return E_sp