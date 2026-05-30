import numpy as np
import psi4
import json

from pathlib import Path
from psi4.driver.qcdb import vib as qcdb_vib

from .xtb.wrapper import GFN2xTB

from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds
from rdkit.Geometry import Point3D


# Note: psi4 uses Ångström units by default for geometry input


class Geometry():
    def __init__(self, 
                 smiles: str = "", 
                 rdmol: Chem.Mol | None = None,
                 xyz_string: str | None = None,
                 charge: int = 0, 
                 mult: int = 1) -> None:
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
        self.smiles : str = ""
        self.rdmol2D : Chem.Mol | None = None
        self.rdmol : Chem.Mol | None = None
        self.natoms : int = 0
        self.symbols : list[str] = []
        self.numbers : list[int] = []
        self.coords : np.ndarray | None = None
        self.xyz_string : str = ""
        self.xyz_block : str = ""
        self.mol_str : str = ""
        self.psi4_mol : psi4.core.Molecule | None = None
        self.charge : int = charge
        self.mult : int = mult

        if smiles or rdmol:
            if smiles:
                self.smiles = smiles    
                self.rdmol2D = Chem.MolFromSmiles(smiles)
                self.rdmol = Chem.AddHs(self.rdmol2D)
            elif rdmol:
                self.smiles = Chem.MolToSmiles(rdmol)
                self.rdmol2D = Chem.MolFromSmiles(self.smiles)
                self.rdmol = Chem.Mol(rdmol)
            # 3D embedding and MMFF optimization with RDKit
            AllChem.EmbedMolecule(self.rdmol, AllChem.ETKDGv3())
            AllChem.MMFFOptimizeMolecule(self.rdmol)
            # self.charge = rdmolops.GetFormalCharge(self.rdmol)
            self.natoms = self.rdmol.GetNumAtoms()
            self.symbols = [atom.GetSymbol() for atom in self.rdmol.GetAtoms()]
            self.numbers = [atom.GetAtomicNum() for atom in self.rdmol.GetAtoms()]
            self.update_coords()
            
        elif xyz_string:
            self.xyz_string = xyz_string
            lines = xyz_string.split('\n')
            natoms = int(lines[0].strip())
            symbols = []
            coords = []
            for i in range(2, 2 + natoms):
                e, x, y, z = lines[i].split()
                symbols.append(e)
                coords.append([float(x), float(y), float(z)])
            self.natoms = natoms
            self.symbols = symbols
            self.numbers = []
            self.coords = np.array(coords)

            lines = [f"{e:5}  {x:23.14f}  {y:23.14f}  {z:23.14f}" for e, (x, y, z) in zip(self.symbols, self.coords)]
            self.xyz_block : str = "\n".join(lines)
            # Atomic coordinates in XYZ format (element  x  y  z, one atom per line) (Angstroms).
            # Do NOT include the line-count header or comment line — just coordinates.
            self.mol_str : str = f"{self.charge} {self.mult}\n{self.xyz_block}\n  symmetry c1\n  no_reorient\n  no_com"
            # The XYZ block is used to create the psi4 molecule string, which also includes charge and multiplicity.
            # Create a psi4 molecule object from the XYZ string
            self.psi4_mol = psi4.geometry(self.mol_str)
            
            # 3. Load the XYZ block into an RDKit molecule
            rdmol = Chem.MolFromXYZBlock(xyz_string)
            rdDetermineBonds.DetermineBonds(rdmol, charge=self.charge)
            # 4. Infer bonds (required, as XYZ block lacks bond information)
            Chem.SanitizeMol(rdmol)
            rdmol = Chem.AddHs(rdmol)
            AllChem.EmbedMolecule(rdmol, AllChem.ETKDGv3())
            AllChem.MMFFOptimizeMolecule(rdmol)
            self.smiles = Chem.MolToSmiles(rdmol)
            self.rdmol2D = Chem.MolFromSmiles(self.smiles)
            self.rdmol = rdmol

        
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
        
        elif source is None and self.rdmol is not None:
            self.coords : np.ndarray = self.rdmol.GetConformer().GetPositions()

        for i, (e, (x, y, z)) in enumerate(zip(self.symbols, self.coords)):
            atom = self.rdmol.GetAtomWithIdx(i)
            assert e == atom.GetSymbol()
            self.rdmol.GetConformer().SetAtomPosition(atom.GetIdx(), Point3D(x, y, z))
        
        lines = [f"{e:5}  {x:23.14f}  {y:23.14f}  {z:23.14f}" for e, (x, y, z) in zip(self.symbols, self.coords)]
        self.xyz_block = "\n".join(lines)
        self.xyz_string = f"{self.natoms}\n\n{self.xyz_block}\n"
        # Atomic coordinates in XYZ format (element  x  y  z, one atom per line) (Angstroms).
        # Do NOT include the line-count header or comment line — just coordinates.
        self.mol_str = f"{self.charge} {self.mult}\n{self.xyz_block}\n  symmetry c1\n  no_reorient\n  no_com"
        # The XYZ block is used to create the psi4 molecule string, which also includes charge and multiplicity.

        # Create a psi4 molecule object from the XYZ string
        self.psi4_mol = psi4.geometry(self.mol_str)


    def xtb_optimize(self) -> None:
        """
        Pre-optimization with xtb (GFN2-xTB) to get a reasonable starting geometry for Psi4 DFT optimization.
        This can help avoid convergence issues in the subsequent DFT optimization step.
        """
        xtb = GFN2xTB(self.rdmol)
        xtb_opt_result = xtb.optimize() # returns a NameSpace with .geometry (RDKit Mol)
        self.rdmol = xtb_opt_result.geometry
        self.update_coords()


    def optimize(self, 
                 functional: str = 'wb97x-d', 
                 basis: str = '6-311+G(d,p)',
                 solvent: str | None = None,
                 solvation_model: str = 'pcm',
                 max_iter: int = 500) -> None:
        """
        Embed with RDKit MMFF, then optimize with Psi4 DFT.
        Returns (coords_Nx3, atom_symbols).
        """
        if solvent:
            psi4.set_options({
                "basis": basis,
                "scf_type": "df", # density fitting - faster & more stable
                "guess": "sad", # superposition of atomic densities (usually best)
                "diis": True, # DIIS extrapolation (should be on by default)
                "diis_min_vecs": 2,
                "diis_max_vecs": 10,
                "maxiter": 500, # default is 100
                "fail_on_maxiter": False, # return best guess instead of crashing
                "geom_maxiter": max_iter,
                "ddx": True,
                "ddx_model": solvation_model,
                "ddx_solvent": solvent,
                "ddx_radii_set": "uff",
                })
        else:
            psi4.set_options({
                'basis': basis,
                "scf_type": "df", # density fitting - faster & more stable
                "guess": "sad", # superposition of atomic densities (usually best)
                "diis": True, # DIIS extrapolation (should be on by default)
                "diis_min_vecs": 2,
                "diis_max_vecs": 10,
                "maxiter": 500, # default is 100
                "fail_on_maxiter": False, # return best guess instead of crashing
                "geom_maxiter": max_iter,
                })
        # DF (Density Fitting): Approximates 4-center integrals using 3-center integrals, drastically speeding up calculations, especially for large systems.

        theory_level = f'{functional}/{basis}' 
        psi4.optimize(theory_level, molecule=self.psi4_mol)
        
        # self.psi4_mol is updated in-place by psi4.optimize, so we can directly access the new geometry
        # update xyz_block and mol_str with the optimized geometry
        coords = self.psi4_mol.geometry().to_array() * psi4.constants.bohr2angstroms  
        
        # Convert from Bohr to Angstroms
        self.update_coords(coords)


    def write_xyz(self, output_path: Path | str, overwrite: bool = False) -> None:
        lines : list[str] = [f"{self.natoms}", " "]
        for e, (x, y, z) in zip(self.symbols, self.coords):
            lines.append(f"{e:5} {x:23.14f} {y:23.14f} {z:23.14f}")
        if isinstance(output_path, str):
            output_path = Path(output_path)
        with open(output_path, "w" if overwrite else "x") as f:
            # x mode will raise an error if the file already exists, 
            # preventing accidental overwrites
            f.write("\n".join(lines))


    def write_sdf(self, output_path: Path | str, overwrite: bool = False) -> None:
        if self.rdmol is None:
            raise ValueError("No RDKit molecule available to write SDF.")
        if isinstance(output_path, str):
            output_path = Path(output_path)
        with Chem.SDWriter(str(output_path)) as w:
            if not overwrite and output_path.exists():
                raise FileExistsError(f"File {output_path} already exists. Set overwrite=True to overwrite it.")
            w.write(self.rdmol)


    def serialize(self) -> str:
        return json.dumps({
            'smiles': self.smiles,
            'natoms': self.natoms,
            'symbols': self.symbols,
            'numbers': self.numbers,
            'coords': self.coords.tolist(),
            'xyz_string' : self.xyz_string,
            'xyz_block': self.xyz_block,
            'mol_str': self.mol_str,
            'charge': self.charge,
            'mult': self.mult,
        })
    

    def deserialize(self, serialized: str) -> None:
        data = json.loads(serialized)
        self.smiles = data.get('smiles')
        self.natoms = data.get('natoms')
        self.symbols = data.get('symbols')
        self.numbers = data.get('numbers')
        self.coords = np.array(data.get('coords'))
        self.xyz_string = data.get('xyz_string')
        self.xyz_block = data.get('xyz_block')
        self.mol_str = data.get('mol_str')
        self.charge = data.get('charge')
        self.mult = data.get('mult')

        self.psi4_mol = psi4.geometry(self.mol_str)

        # 3. Load the XYZ block into an RDKit molecule
        rdmol = Chem.MolFromXYZBlock(self.xyz_string)

        rdDetermineBonds.DetermineBonds(rdmol, charge=self.charge)
        # 4. Infer bonds (required, as XYZ block lacks bond information)
        Chem.SanitizeMol(rdmol)
        rdmol = Chem.AddHs(rdmol)
        AllChem.EmbedMolecule(rdmol, AllChem.ETKDGv3())
        AllChem.MMFFOptimizeMolecule(rdmol)

        self.rdmol2D = Chem.MolFromSmiles(self.smiles)
        self.rdmol = rdmol
        

    def single_point_energy(self, 
                            functional: str = "wb97x-d", 
                            basis: str  = "6-311+G(d,p)",
                            solvent: str | None = None,
                            solvation_model: str = 'pcm') -> float:
        """
        Compute a single-point energy at a previously optimised geometry.
        functional : str, optional
            The functional to use for the calculation. Default is 'wb97x-d' 
            Other choice is 'b3lyp-d3bj2b'.
        basis : str, optional
            The basis set to use for the calculation. Default is 6-311+G(2d,2p). 
            Other choices are '6-31+G(d)' or 'cc-pVDZ'.
            Diffuse functions ('+') are recommended for anions to get more accurate thermochemistry.
        

        Returns
        -------
        float : electronic energy in Hartree
        """
        if solvent:
            psi4.set_options({
                "basis": basis,
                "scf_type": "df", # density fitting - faster & more stable
                "guess": "sad", # superposition of atomic densities (usually best)
                "diis": True, # DIIS extrapolation (should be on by default)
                "diis_min_vecs": 2,
                "diis_max_vecs": 10,
                "maxiter": 500, # default is 100
                "fail_on_maxiter": False, # return best guess instead of crashing
                "ddx": True,
                "ddx_model": solvation_model, # PCM(default), COSMO, LPB
                "ddx_solvent": solvent,
                "ddx_radii_set": "uff",
                })
        else:
            psi4.set_options({
                "basis": basis,
                "scf_type": "df", # density fitting - faster & more stable
                "guess": "sad", # superposition of atomic densities (usually best)
                "diis": True, # DIIS extrapolation (should be on by default)
                "diis_min_vecs": 2,
                "diis_max_vecs": 10,
                "maxiter": 500,
                "fail_on_maxiter": False, # return best guess instead of crashing
                })
        
        theory_level = f"{functional}/{basis}"
        E_sp, wfn = psi4.energy(theory_level, molecule=self.psi4_mol, return_wfn=True)
        
        return E_sp
    

    def gibbs_free_energy(self,
                          functional: str = 'wb97x-d',
                          basis: str = '6-311+G(d,p)',
                          scale_factor: float = 1.0,
                          temperature: float = 298.15,
                          pressure: float = 101325.0) -> float:
        """
        Perform a frequency calculation, scale the frequencies by a given factor, and compute thermochemical properties.
        Parameters
        ----------
        functional : str, optional
            The functional to use for the calculation. Default is 'wb97x-d' 
            Other choice is 'b3lyp-d3bj2b'.
        basis : str, optional
            The basis set to use for the calculation. Default is 6-311+G(2d,2p). 
            Other choices are '6-31+G(d)' or 'cc-pVDZ'.
            Diffuse functions ('+') are recommended for anions to get more accurate thermochemistry.
        scale_factor : float
            The factor by which to scale the frequencies.
        temperature : float, optional
            The temperature at which to compute thermochemical properties. Default is 298.15 K.
        pressure : float, optional
            The pressure at which to compute thermochemical properties. Default is 101325.0 Pa.
        Returns
        -------
        float
            The corrected Gibbs free energy in hartree.
        """
        psi4.set_options({
            "basis": basis,
            "scf_type": "df", # density fitting - faster & more stable
            "guess": "sad", # superposition of atomic densities (usually best)
            "diis": True, # DIIS extrapolation (should be on by default)
            "diis_min_vecs": 2,
            "diis_max_vecs": 10,
            "maxiter": 500,
            "fail_on_maxiter": False, # return best guess instead of crashing
            })
        # note: 'dft_dispersion_parameters' is delicate.
        # for standard D3BJ, it's safer to append it to the functinoal string. 
        # "dft_dispersion_parameters": [dispersion],   # activates D3(BJ)

        theory_level = f"{functional}/{basis}"

        # 1. Run the frequency calculation
        E_freq, wfn_freq = psi4.frequency(theory_level, molecule=self.psi4_mol, return_wfn=True)

        # Step 2: Scale the Hessian by SCALE_FACTOR^2
        H_orig    = np.array(wfn_freq.hessian()) # (3N, 3N) non-mass-weighted, Eh/a0^2
        H_scaled  = H_orig * (scale_factor ** 2)

        # Step 3: Re-run harmonic_analysis with the scaled Hessian
        #         This keeps ALL vibinfo fields (omega, ZPE, force constants) consistent
        mol_psi  = wfn_freq.molecule()
        scaled_vibinfo, _ = qcdb_vib.harmonic_analysis(
            hess         = H_scaled,
            geom         = np.array(mol_psi.geometry()),
            mass         = np.array([mol_psi.mass(i) for i in range(mol_psi.natom())]),
            basisset     = wfn_freq.basisset(),
            irrep_labels = wfn_freq.molecule().irrep_labels(),
        )

        # Thermochemistry
        # CRITICAL: Psi4 qcdb.vib.thermo expects Pressure in Pascals, 
        # but check if your version of Psi4 prefers atmosphere (101325.0 is correct for Pa).
        # Step 5: Call thermo() with the fully consistent scaled vibinfo
        thermo, thermo_text = qcdb_vib.thermo(
            vibinfo        = scaled_vibinfo,
            T              = temperature,
            P              = pressure,
            multiplicity   = mol_psi.multiplicity(),
            molecular_mass = sum(mol_psi.mass(i) for i in range(mol_psi.natom())),
            E0             = E_freq,
            sigma          = mol_psi.rotational_symmetry_number(),
            rot_const      = np.asarray(mol_psi.rotational_constants()),
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
        # ZPE    = thermo["ZPE_vib"].data
        # H_corr = thermo["H_corr"].data
        # G_corr = thermo["G_corr"].data

        # E_zpe   = E_freq + ZPE
        # H_total = E_freq + H_corr
        # G_total = E_freq + G_corr
        # TS      = H_total - G_total
        # S       = (TS / temperature) * 627.509474 # hartree/K to kcal/mol/K
        # print(thermo_text)
        # print(f"E_elec      = {E_freq:.6f}  Hartree")
        # print(f"E_zpe       = {E_zpe:.6f}  Hartree")
        # print(f"H({temperature} K) = {H_total:.6f}  Hartree")
        # print(f"G({temperature} K) = {G_total:.6f}  Hartree")
        # print(f"TS          = {TS * 627.509:.4f}  kcal/mol")

        return thermo["G_tot"].data
