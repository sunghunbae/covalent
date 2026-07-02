from rdkit import Chem
from rdkit.Chem import AllChem

from typing import Self
import covalent.utils

class Reaction:
    """
    Supported reaction(s):
        - Michael addtion

    Given a Michael acceptor SMILES, constructs the corresponding α-carbanion intermediate SMILES
    formed after thiolate attack. 
    The class auto-detects the α/β carbons and EWG type based on SMARTS patterns.

    Constructs the α-carbanion SMILES formed after thiolate (RS⁻) attacks 
    the β-carbon of an α,β-unsaturated Michael acceptor.

        Reaction:
        EWG-Cα=Cβ(R)(R')  +  RS⁻  →  EWG-C⁻α(H)-Cβ(R)(R')-SR

        Strategy:
        1. Identify Michael acceptor pattern: [EWG]-Cα=Cβ
        2. Break the Cα=Cβ π-bond (make it a single bond)
        3. Attach -SCH3 (model thiolate) to Cβ
        4. Assign formal charge -1 to Cα
        5. Return SMILES with overall charge = -1
    """
    
    # ── EWG SMARTS patterns that define Michael acceptors ────────────────────────
    # Each pattern matches: [alpha_carbon]=[beta_carbon] conjugated with EWG
    # Capture group 0 = alpha carbon, group 1 = beta carbon
    # Ordered from most specific to most general to avoid mis-assignment

    EWG_SMARTS = {
        "vinyl_trifluoromethyl": "[C:1](=[CX3H2:2])C(F)(F)F",   # -Cb=Ca-CF3
        "acrylamide_CF3":        "[C:1](=[CX3H1:2]C(F)(F)F)",   # -Ca=Cb-CF3
        "cyanoacrylamide":       "[C:1](=[C:2])C#N",          # -C=C-CN (dual activation)
        "vinyl_sulfone":         "[C:1](=[C:2])S(=O)(=O)",    # -C=C-SO2-
        "vinyl_phosphonate":     "[C:1](=[C:2])P(=O)",        # -C=C-P(O)-
        "vinyl_ketone":          "[C:1](=[C:2])C(=O)[#6]",    # -C=C-C(=O)-C
        "acrylamide":            "[C:1](=[C:2])C(=O)N",       # -C=C-C(=O)N
        "acrylate":              "[C:1](=[C:2])C(=O)O",       # -C=C-C(=O)O
        "acrylonitrile":         "[C:1](=[C:2])C#N",          # -C=C-CN
        "vinyl_aldehyde":        "[C:1](=[C:2])C=O",          # -C=C-CHO
        "generic_ewg":           "[C:1](=[C:2])[$(C=O),$(S(=O)),$(C#N),$(N(=O)=O)]",
    }
    
    

    def __init__(self, 
                 reactant: str | Chem.Mol | None = None,
                 thiol_smiles: str = "SC", # nucleophile
                 alpha_idx: int | None = None,
                 beta_idx: int | None = None,
                 map_Ca: int = 91,
                 map_Cb: int = 92,
                 map_S: int = 93,
                 verbose: bool = False):
        """
        Args:
            reactant_smiles : SMILES of the neutral Michael acceptor
            thiol_smiles    : SMILES fragment for the thiolate model
                              (default "SC" = methylthio; use "SCC(N)C(=O)O"
                               for cysteine-like surrogate)
            alpha_idx       : Override auto-detected alpha carbon atom index
            beta_idx        : Override auto-detected beta carbon atom index
            verbose         : Print detection details
        """
        # Reactant
        self.reactant_rdmol : Chem.Mol | None = None
        self.reactant_smiles : str = ""
        self.reactant_charge : int = 0

        if isinstance(reactant, str):
            reactant = Chem.MolFromSmiles(reactant)

        if isinstance(reactant, Chem.Mol):
            reactant = Chem.AddHs(reactant)
            self.reactant_smiles = Chem.MolToSmiles(reactant)
            self.reactant_rdmol = Chem.RWMol(reactant)
            self.reactant_charge = sum(a.GetFormalCharge() for a in self.reactant_rdmol.GetAtoms())
        
        self.thiol_smiles : str = thiol_smiles
        self.ewg: str = ""
        self.sites : list[tuple[int, int, str]] = []
        self.alpha_idx : int | None = alpha_idx
        self.beta_idx : int | None = beta_idx
        self.map_Ca : int = map_Ca
        self.map_Cb : int = map_Cb
        self.map_S  : int = map_S
        self.verbose : bool = verbose

        self.carbanion_rdmol : Chem.Mol | None = None
        self.carbanion_smiles : str = ""
        self.carbanion_charge: int = 0
        self.carbanion_alpha_idx : int = -1
        self.carbanion_beta_idx : int = -1
        self.carbanion_S_idx : int = -1

        self.product_rdmol : Chem.Mol | None = None
        self.product_smiles : str = ""
        self.product_charge: int = 0
        self.product_alpha_idx : int = -1
        self.product_beta_idx : int = -1
        self.product_S_idx : int = -1
        
        if self.reactant_smiles and self.reactant_rdmol :
            self._find_michael_acceptor_atoms()
            self._build_intermediate_and_product()


    def serialize(self) -> str:
        serialized = covalent.utils.serialize({
            'reactant': Chem.MolToMolBlock(self.reactant_rdmol),
            'carbanion': Chem.MolToMolBlock(self.carbanion_rdmol),
            'product': Chem.MolToMolBlock(self.product_rdmol),
            'thiol_smiles': self.thiol_smiles,
            'ewg': self.ewg,
            'sites': self.sites,
            'alpha_idx': self.alpha_idx,
            'beta_idx': self.beta_idx,
            'map_Ca': self.map_Ca,
            'map_Cb': self.map_Cb,
            'map_S': self.map_S,
        })
        return serialized


    def deserialize(self, serialized: str) -> Self:
        data = covalent.utils.deserialize(serialized)
        self.thiol_smiles = data['thiol_smiles']
        self.ewg = data['ewg']
        self.sites = data['sites']
        self.alpha_idx = int(data['alpha_idx'])
        self.beta_idx = int(data['beta_idx'])
        self.map_Ca = int(data['map_Ca'])
        self.map_Cb = int(data['map_Cb'])
        self.map_S = int(data['map_S'])
        
        self.reactant_rdmol = Chem.MolFromMolBlock(data['reactant'], sanitize=False, removeHs=False)
        self.reactant_smiles = Chem.MolToSmiles(self.reactant_rdmol)
        self.reactant_charge = sum(a.GetFormalCharge() for a in self.reactant_rdmol.GetAtoms())
        
        self.carbanion_rdmol = Chem.MolFromMolBlock(data['carbanion'], sanitize=False, removeHs=False)
        self.carbanion_smiles = Chem.MolToSmiles(self.carbanion_rdmol)
        self.carbanion_charge = sum(a.GetFormalCharge() for a in self.carbanion_rdmol.GetAtoms())
        self.carbanion_beta_idx = next(a.GetIdx() for a in self.carbanion_rdmol.GetAtoms() if a.GetAtomMapNum() == self.map_Cb)
        self.carbanion_alpha_idx = next(a.GetIdx() for a in self.carbanion_rdmol.GetAtoms() if a.GetAtomMapNum() == self.map_Ca)
        self.carbanion_S_idx = next(a.GetIdx() for a in self.carbanion_rdmol.GetAtoms() if a.GetAtomMapNum() == self.map_S)

        self.product_rdmol = Chem.MolFromMolBlock(data['product'], sanitize=False, removeHs=False)
        self.product_smiles = Chem.MolToSmiles(self.product_rdmol)
        self.product_charge = sum(a.GetFormalCharge() for a in self.product_rdmol.GetAtoms())
        self.product_beta_idx = next(a.GetIdx() for a in self.product_rdmol.GetAtoms() if a.GetAtomMapNum() == self.map_Cb)
        self.product_alpha_idx = next(a.GetIdx() for a in self.product_rdmol.GetAtoms() if a.GetAtomMapNum() == self.map_Ca)
        self.product_S_idx = next(a.GetIdx() for a in self.product_rdmol.GetAtoms() if a.GetAtomMapNum() == self.map_S)

        return self


    def _find_michael_acceptor_atoms(self) -> None:
        """
        Identify all (alpha_idx, beta_idx, ewg) tuples in the molecule.
        
        The alpha carbon is directly bonded to the EWG.
        The beta carbon is the terminal alkene carbon (site of thiolate attack).
        
        Returns:
            List of (alpha_carbon_idx, beta_carbon_idx, ewg_name)
            Empty list if no Michael acceptor pattern found.
        """
        matches = []
        seen_pairs = set()

        if self.alpha_idx is None or self.beta_idx is None:
            for ewg_name, smarts in self.EWG_SMARTS.items():
                pattern = Chem.MolFromSmarts(smarts)
                if pattern is None:
                    continue

                match_indices = self.reactant_rdmol.GetSubstructMatches(pattern)
                # ex. match_indices= ((1, 0, 2, 3, 4),)
                if match_indices:
                    map_to_idx = {}
                    for a in pattern.GetAtoms():
                        map_num = a.GetAtomMapNum()
                        # unmapped atoms have map number of 0
                        if map_num > 0:
                            map_to_idx[map_num] = match_indices[0][a.GetIdx()]
    
                    alpha_idx = map_to_idx.get(1)
                    beta_idx = map_to_idx.get(2)
                    pair = (alpha_idx, beta_idx)
                    if pair in seen_pairs:
                        continue  # already found via a more specific pattern
                    
                    # ── Validate the double bond at Cα=Cβ ─────────────────────
                    bond = self.reactant_rdmol.GetBondBetweenAtoms(alpha_idx, beta_idx)
                    if bond is None or bond.GetBondTypeAsDouble() < 1.9:
                        continue
                        
                    seen_pairs.add(pair)
                    matches.append((alpha_idx, beta_idx, ewg_name))
                    
            self.sites = matches
            if not self.sites:
                raise NotImplementedError(
                    f"Acceptor pattern not detected in: {self.reactant_smiles}\n"
                    f"Supported EWGs: {list(self.EWG_SMARTS.keys())}\n"
                    f"You can manually specify alpha_idx and beta_idx to override."
                )

            if self.verbose:
                print(f"  Detected {len(self.sites)} Michael acceptor site(s):")
                for a, b, ewg in self.sites:
                    print(f"    α-C idx= {a}  β-C idx= {b}  EWG= {ewg}")

            # Use the first (most specifically matched) site
            self.alpha_idx, self.beta_idx, self.ewg = self.sites[0]
        else:
            self.ewg = "user_defined"
            if self.verbose:
                print(f"  Using user-specified α-C idx={self.alpha_idx}, β-C idx={self.beta_idx}")
                

    def _build_intermediate_and_product(self) -> None:
        """
        Build the α-carbanion intermediate SMILES from a Michael acceptor.

        The function:
        1. Parses and sanitizes the input SMILES
        2. Auto-detects the α/β carbons (or uses provided indices)
        3. Converts Cα=Cβ double bond → single bond
        4. Attaches -S-CH3 (model thiolate) to Cβ
        5. Sets formal charge -1 on Cα
        6. Returns the carbanion SMILES and metadata

        Returns:
            dict with keys:
            "carbanion_smiles"  : SMILES string of the carbanion intermediate
            "neutral_smiles"    : original input SMILES (sanitized)
            "alpha_idx"         : alpha carbon index in the original molecule
            "beta_idx"          : beta carbon index in the original molecule
            "ewg"               : detected EWG classification
            "charge"            : total charge of carbanion molecule (-1)
            "n_michael_sites"   : number of Michael acceptor sites detected
        
        Raises:
            ValueError: if no Michael acceptor pattern is found and no indices given
        """        
        # ── Build carbanion ─────────────────────────────
        self.reactant_rdmol.GetAtomWithIdx(self.beta_idx).SetAtomMapNum(self.map_Cb)
        self.reactant_rdmol.GetAtomWithIdx(self.alpha_idx).SetAtomMapNum(self.map_Ca)
        self.reactant_smiles = Chem.MolToSmiles(self.reactant_rdmol)

        edit_mol = Chem.RWMol(self.reactant_rdmol)
        
        # a. Convert Cα=Cβ double bond → single bond
        edit_mol.RemoveBond(self.alpha_idx, self.beta_idx)
        edit_mol.AddBond(self.alpha_idx, self.beta_idx, Chem.BondType.SINGLE)
        
        # b. Assign formal charge -1 to α-carbon
        alpha_atom = edit_mol.GetAtomWithIdx(self.alpha_idx)
        alpha_atom.SetFormalCharge(-1)
        
        # c. Attach thiolate (ex. -SCH3) to β-carbon
        #     Parse thiolate fragment and merge into molecule
        thiol_mol = Chem.MolFromSmiles(self.thiol_smiles)
        if thiol_mol is None:
            raise ValueError(f"Invalid thiolate SMILES: {self.thiol_smiles}")
        thiol_mol = Chem.AddHs(thiol_mol)
        
        # d. Find the sulfur atom (or attachment point) in the thiol_mol
        s_atom_idx_in_thiol = None
        for atom in thiol_mol.GetAtoms():
            if atom.GetAtomicNum() == 16: # sulfur
                s_atom_idx_in_thiol = atom.GetIdx()
                break
        if s_atom_idx_in_thiol is None:
            raise ValueError(f"No sulfur atom found in thiolate: {self.thiol_smiles}")
        
        thiol_mol.GetAtomWithIdx(s_atom_idx_in_thiol).SetAtomMapNum(self.map_S)

        # f. Combine the two molecules
        # Atom indices - Chem.CombineMols simply concatenates the atom lists of the input molecules. 
        # The atoms from your first molecule keep their original indices (starting at 0), 
        # while the atoms from the second molecule are shifted by the size of the first molecule.
        # So, alpha_idx and beta_idx should be the same in the combined molecule.
        combined = Chem.RWMol(Chem.CombineMols(edit_mol, thiol_mol))
        
        # g. Bond β-carbon to sulfur
        # s_idx_combined = edit_mol.GetNumAtoms() + s_atom_idx_in_thiol
        s_idx_combined = next(a.GetIdx() for a in combined.GetAtoms() if a.GetAtomMapNum() == self.map_S)
        # Sulfur atom index in the combined molecule
        combined.AddBond(self.beta_idx, s_idx_combined, Chem.BondType.SINGLE)
        
        # ── Remove one H from sulfur (it was SH, now S-) ──────────────────
        s_atom = combined.GetAtomWithIdx(s_idx_combined)
        sh_to_remove = None
        for neighbor in s_atom.GetNeighbors():
            if neighbor.GetAtomicNum() == 1:
                sh_to_remove = neighbor.GetIdx()
                break
        if sh_to_remove is not None:
            combined.RemoveAtom(sh_to_remove)
        
        # ── Sanitize and generate SMILES ──────────────────────────────────
        try:
            Chem.SanitizeMol(combined)
        except Exception as e:
            raise ValueError(f"Sanitization failed after carbanion construction: {e}")
        
        try:
            AllChem.EmbedMolecule(combined, AllChem.ETKDGv3())
            AllChem.MMFFOptimizeMolecule(combined) # coordinates change here
        except Exception as e:
            raise ValueError(f"MMFF optimization failed after carbanion construction: {e}")
        
        self.carbanion_rdmol = Chem.Mol(combined)
        self.carbanion_smiles = Chem.MolToSmiles(combined)
        self.carbanion_charge = sum(a.GetFormalCharge() for a in combined.GetAtoms())

        # Find alpha and beta indices in the carbanion via map numbers
        self.carbanion_beta_idx = next(a.GetIdx() for a in self.carbanion_rdmol.GetAtoms() if a.GetAtomMapNum() == self.map_Cb)
        self.carbanion_alpha_idx = next(a.GetIdx() for a in self.carbanion_rdmol.GetAtoms() if a.GetAtomMapNum() == self.map_Ca)
        self.carbanion_S_idx = next(a.GetIdx() for a in self.carbanion_rdmol.GetAtoms() if a.GetAtomMapNum() == self.map_S)
        
        # The charge of the carbanion intermediate depends on the charge of the reactant molecule
        if (self.carbanion_charge - self.reactant_charge) != -1:
            raise ValueError(f"Unexpected charge: reactant: {self.reactant_charge} carbanion: {self.carbanion_charge}")

        # ── Product SMILES ────────────────────────────────────────────────────
        alpha_atom = None
        for atom in combined.GetAtoms():
            if atom.GetFormalCharge() == -1:
                alpha_atom = atom
                break
        
        assert alpha_atom is not None, "cannot identify negatively charged alpha carbon"

        alpha_atom.SetFormalCharge(0)
        alpha_atom.UpdatePropertyCache() # Recalculate implicit valency (this "adds" the H)
        combined = Chem.AddHs(combined)
        Chem.SanitizeMol(combined)

        self.product_rdmol = Chem.Mol(combined)
        self.product_smiles = Chem.MolToSmiles(combined)
        self.product_charge = sum(a.GetFormalCharge() for a in combined.GetAtoms())
        # Find alpha and beta indices in the carbanion via map numbers
        self.product_beta_idx = next(a.GetIdx() for a in self.product_rdmol.GetAtoms() if a.GetAtomMapNum() == self.map_Cb)
        self.product_alpha_idx = next(a.GetIdx() for a in self.product_rdmol.GetAtoms() if a.GetAtomMapNum() == self.map_Ca)
        self.product_S_idx = next(a.GetIdx() for a in self.product_rdmol.GetAtoms() if a.GetAtomMapNum() == self.map_S)
        
        if self.verbose:
            print()
            print(f"  Reactant                 : {self.reactant_smiles}")
            print(f"    charge                 : {self.reactant_charge}")
            print(f"    # of reaction sites    : {len(self.sites)}")
            print(f"    EWG                    : {self.ewg}")
            print(f"    Cα, Cβ indices         : {self.alpha_idx},{self.beta_idx}")
            print(f"  Thiol                    : {self.thiol_smiles}")
            print(f"  α-carbanion intermediate : {self.carbanion_smiles}")
            print(f"    charge                 : {self.carbanion_charge}")
            print(f"    Cα, Cβ, S indices      : {self.carbanion_alpha_idx},{self.carbanion_beta_idx},{self.carbanion_S_idx}")
            print(f"  Product                  : {self.product_smiles}")
            print(f"    charge                 : {self.product_charge}")
            print(f"    Cα, Cβ, S indices      : {self.product_alpha_idx},{self.product_beta_idx},{self.product_S_idx}")
