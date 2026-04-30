import numpy as np
import psi4

from psi4.driver.qcdb import vib as qcdb_vib
from .xtb.wrapper import GFN2xTB

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
        self.rdmol2D : Chem.Mol = Chem.MolFromSmiles(smiles)
        self.rdmolH : Chem.Mol = Chem.AddHs(self.rdmol2D)
        
        # 3D embedding and MMFF optimization with RDKit
        AllChem.EmbedMolecule(self.rdmolH, AllChem.ETKDGv3())
        AllChem.MMFFOptimizeMolecule(self.rdmolH)

        # self.charge = rdmolops.GetFormalCharge(self.rdmolH)
        self.natoms : int = self.rdmolH.GetNumAtoms()
        self.symbols : list[str] = [atom.GetSymbol() for atom in self.rdmolH.GetAtoms()]
        self.numbers : list[int] = [atom.GetAtomicNum() for atom in self.rdmolH.GetAtoms()]
        self.coords : np.ndarray | None = None
        self.xyz_block : str = ""
        self.mol_str : str = ""
        self.update_coords()
        
        
    def count_electrons(self) -> int:
        """Count total electrons = sum(atomic numbers) - charge."""
        return sum(self.numbers) - self.charge


    def update_coords(self, source: np.ndarray | str | None = None) -> None:
        if isinstance(source, str) and source.endswith(".xyz"):
            with open(source, "r") as f:
                lines = f.readlines()
                natoms = int(lines[0].strip())
                coords = []
                assert self.natoms == natoms, f"Number of atoms in XYZ file ({natoms}) does not match the current molecule ({self.natoms})."
                for i in range(2, 2 + natoms):
                    e, x, y, z = lines[i].split()
                    assert e == self.symbols[i-2], f"Element symbol in XYZ file ({e}) does not match the current molecule ({self.symbols[i-2]})."
                    coords.append([float(x), float(y), float(z)])
            self.coords : np.ndarray = np.array(coords)
        elif isinstance(source, np.ndarray):
            self.coords : np.ndarray = source
        elif source is None and self.rdmolH is not None:
            self.coords : np.ndarray = self.rdmolH.GetConformer().GetPositions()

        lines = [f"{e:5}  {x:23.14f}  {y:23.14f}  {z:23.14f}" for e, (x, y, z) in zip(self.symbols, self.coords)]
        self.xyz_block : str = "\n".join(lines)
        # Atomic coordinates in XYZ format (element  x  y  z, one atom per line) (Angstroms).
        # Do NOT include the line-count header or comment line — just coordinates.
        self.mol_str : str = f"{self.charge} {self.mult}\n{self.xyz_block}\n  symmetry c1\n  no_reorient\n  no_com"
        # The XYZ block is used to create the psi4 molecule string, which also includes charge and multiplicity.

        # Create a psi4 molecule object from the XYZ string
        self.psi4_mol : psi4.core.Molecule = psi4.geometry(self.mol_str)


    def pre_optimize(self) -> None:
        """
        Pre-optimization with xtb (GFN2-xTB) to get a reasonable starting geometry for Psi4 DFT optimization.
        This can help avoid convergence issues in the subsequent DFT optimization step.
        """
        xtb = GFN2xTB(self.rdmolH)
        xtb_opt_result = xtb.optimize() # returns a NameSpace with .geometry (RDKit Mol)
        self.rdmolH = xtb_opt_result.geometry
        self.update_coords()


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
        coords = self.psi4_mol.geometry().to_array() * psi4.constants.bohr2angstroms  
        # Convert from Bohr to Angstroms
        self.update_coords(coords)


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
    

    def gibbs_free_energy(self,
                          scale_factor: float,
                          functional: str = 'b3lyp',
                          basis: str = '6-31+G(d)',
                          dispersion: str = 'd3bj',
                          temperature: float = 298.15,
                          pressure: float = 101325.0) -> float:
        """
        Perform a frequency calculation, scale the frequencies by a given factor, and compute thermochemical properties.
        Parameters
        ----------
        mol : psi4.core.Molecule
            The molecule for which to calculate frequencies.
        scale_factor : float
            The factor by which to scale the frequencies.
        functional : str, optional
            The functional to use for the calculation. Default is 'b3lyp' (or 'scf').
        basis : str, optional
            The basis set to use for the calculation. Default is '6-31+G(d)' or 'cc-pVDZ'.
            Diffuse functions (e.g. 6-31+G(d)) are recommended for anions to get more accurate thermochemistry.
        dispersion : str, optional
            The dispersion correction to use (e.g. 'd3bj' for Grimme D3 with Becke-Johnson damping). Default is 'd3bj'.
        temperature : float, optional
            The temperature at which to compute thermochemical properties. Default is 298.15 K.
        pressure : float, optional
            The pressure at which to compute thermochemical properties. Default is 101325.0 Pa.
        Returns
        -------
        float
            The corrected Gibbs free energy.
        """

        psi4.set_options({
            "basis": basis,
            "dft_dispersion_parameters": [dispersion],   # activates D3(BJ)
            "geom_maxiter": 200,
            "g_convergence": "gau_tight",
        })

        # 0. Optimise
        E_opt, wfn = psi4.optimize(f"{functional}/{basis}", molecule=self.psi4_mol, return_wfn=True)

        # 1. Run the frequency calculation
        E_freq, wfn_freq = psi4.frequency(
            f'{functional}/{basis}',
            molecule=self.psi4_mol, 
            return_wfn=True)

        # Step 2: Scale the Hessian by SCALE_FACTOR^2
        H_orig    = np.array(wfn_freq.hessian())          # (3N, 3N) non-mass-weighted, Eh/a0^2
        H_scaled  = H_orig * (scale_factor ** 2)

        # Step 3: Re-run harmonic_analysis with the scaled Hessian
        #         This keeps ALL vibinfo fields (omega, ZPE, force constants) consistent
        mol_psi  = wfn_freq.molecule()
        geom     = np.array(mol_psi.geometry())
        mass     = np.array([mol_psi.mass(i) for i in range(mol_psi.natom())])  # in u
        basisset = wfn_freq.basisset()
        irrep_labels = wfn_freq.molecule().irrep_labels()

        scaled_vibinfo, _ = qcdb_vib.harmonic_analysis(
            hess         = H_scaled,
            geom         = geom,
            mass         = mass,
            basisset     = basisset,
            irrep_labels = irrep_labels,
        )

        # Step 4: Extract molecular properties for thermo()
        molecular_mass = sum(mol_psi.mass(i) for i in range(mol_psi.natom()))
        multiplicity   = mol_psi.multiplicity()
        sigma          = mol_psi.rotational_symmetry_number()
        rot_const       = np.asarray(mol_psi.rotational_constants())

        # Step 5: Call thermo() with the fully consistent scaled vibinfo
        thermo, thermo_text = qcdb_vib.thermo(
            vibinfo        = scaled_vibinfo,
            T              = temperature,
            P              = pressure,
            multiplicity   = multiplicity,
            molecular_mass = molecular_mass,
            E0             = E_freq,
            sigma          = sigma,
            rot_const      = rot_const,
        )

        # thermo.keys() = ['E0', 'B', 'sigma', 'T', 'P',
        #   'Cv_trans', 'Cv_rot', 'Cv_vib', 'Cv_elec', 'Cv_tot', 
        #   'Cp_trans', 'Cp_rot', 'Cp_vib', 'Cp_elec', 'Cp_tot',
        #   'E_trans', 'E_rot', 'E_vib', 'E_elec', 'E_corr', 'E_tot',
        #   'H_trans',  'H_rot', 'H_vib', 'H_elec', 'H_corr', 'H_tot',
        #   'G_elec', 'G_trans', 'G_rot', 'G_vib', 'G_corr', 'G_tot',
        #   'S_elec', 'S_trans', 'S_rot', 'S_vib', 'S_tot',
        #   'ZPE_vib',  'ZPE_elec', 'ZPE_trans', 'ZPE_rot', 'ZPE_corr', 'ZPE_tot'])

        # Step 6: Extract results
        ZPE    = thermo["ZPE_vib"].data
        H_corr = thermo["H_corr"].data
        G_corr = thermo["G_corr"].data

        E_zpe   = E_freq + ZPE
        H_total = E_freq + H_corr
        G_total = E_freq + G_corr
        TS      = H_total - G_total
        S       = (TS / temperature) * 627.509474 # hartree/K to kcal/mol/K
        # print(thermo_text)
        # print(f"E_elec      = {E_freq:.6f}  Hartree")
        # print(f"E_zpe       = {E_zpe:.6f}  Hartree")
        # print(f"H({temperature} K) = {H_total:.6f}  Hartree")
        # print(f"G({temperature} K) = {G_total:.6f}  Hartree")
        # print(f"TS          = {TS * 627.509:.4f}  kcal/mol")

        return G_total