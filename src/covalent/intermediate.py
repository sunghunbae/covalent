from rdkit import Chem


class Intermediate:
    """
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
        "cyanoacrylamide":  "[C:1](=[C:2])C#N",          # -C=C-CN (dual activation)
        "vinyl_sulfone":    "[C:1](=[C:2])S(=O)(=O)",     # -C=C-SO2-
        "vinyl_phosphonate":"[C:1](=[C:2])P(=O)",          # -C=C-P(O)-
        "vinyl_ketone":     "[C:1](=[C:2])C(=O)[#6]",     # -C=C-C(=O)-C
        "acrylamide":       "[C:1](=[C:2])C(=O)N",        # -C=C-C(=O)N
        "acrylate":         "[C:1](=[C:2])C(=O)O",        # -C=C-C(=O)O
        "acrylonitrile":    "[C:1](=[C:2])C#N",            # -C=C-CN
        "vinyl_aldehyde":   "[C:1](=[C:2])C=O",            # -C=C-CHO
        "generic_ewg":      "[C:1](=[C:2])[$(C=O),$(S(=O)),$(C#N),$(N(=O)=O)]",
    }

    def __init__(self, 
                 michael_acceptor_smiles: str,
                 thiolate_smiles: str = "SC",
                 alpha_idx: int | None = None,
                 beta_idx: int | None = None,
                 verbose: bool = False):
        """
        Args:
            michael_acceptor_smiles : SMILES of the neutral Michael acceptor
            thiolate_smiles         : SMILES fragment for the thiolate model
                                        (default "SC" = methylthio; use "SCC(N)C(=O)O"
                                        for cysteine-like surrogate)
            alpha_idx               : Override auto-detected alpha carbon atom index
            beta_idx                : Override auto-detected beta carbon atom index
            verbose                 : Print detection details
        """

        self.michael_acceptor_smiles : str = michael_acceptor_smiles
        self.thiolate_smiles : str = thiolate_smiles
        self.alpha_idx : int | None = alpha_idx
        self.beta_idx : int | None = beta_idx
        self.verbose : bool = verbose
        self.carbanion_smiles : str
        self.ewg_type: str = "user_defined"
        self.carbanion_charge: int | None = None
        self.sites : list[tuple[int, int, str]] = []

        self.build_carbanion_smiles()


    def find_michael_acceptor_atoms(self) -> None:
        """
        Identify all (alpha_idx, beta_idx, ewg_type) tuples in the molecule.
        
        The alpha carbon is directly bonded to the EWG.
        The beta carbon is the terminal alkene carbon (site of thiolate attack).
        
        Returns:
            List of (alpha_carbon_idx, beta_carbon_idx, ewg_name)
            Empty list if no Michael acceptor pattern found.
        """
        matches = []
        seen_pairs = set()
        
        for ewg_name, smarts in self.EWG_SMARTS.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is None:
                continue
            
            for match in self.mol.GetSubstructMatches(pattern):
                # match[0] = alpha carbon (attached to EWG, mapped as :1)
                # match[1] = beta carbon (terminal, mapped as :2)
                alpha_idx, beta_idx = match[0], match[1]
                
                pair = (alpha_idx, beta_idx)
                if pair in seen_pairs:
                    continue  # already found via a more specific pattern
                seen_pairs.add(pair)
                
                # Verify this is truly a C=C double bond between these atoms
                bond = self.mol.GetBondBetweenAtoms(alpha_idx, beta_idx)
                if bond is None or bond.GetBondTypeAsDouble() < 1.9:
                    continue
                
                matches.append((alpha_idx, beta_idx, ewg_name))
        
        self.sites = matches


    def build_carbanion_smiles(self) -> None:
        """
        Build the α-carbanion intermediate SMILES from a Michael acceptor.

        The function:
        1. Parses and sanitizes the input SMILES
        2. Auto-detects the α/β carbons (or uses provided indices)
        3. Converts Cα=Cβ double bond → single bond
        4. Attaches -S-CH3 (model thiolate) to Cβ
        5. Sets formal charge -1 on Cα
        6. Returns the carbanion SMILES and metadata

        Args:
            michael_acceptor_smiles : SMILES of the neutral Michael acceptor
            thiolate_smiles         : SMILES fragment for the thiolate model
                                    (default "SC" = methylthio; use "SCC(N)C(=O)O"
                                    for cysteine-like surrogate)
            alpha_idx               : Override auto-detected alpha carbon atom index
            beta_idx                : Override auto-detected beta carbon atom index
            verbose                 : Print detection details

        Returns:
            dict with keys:
            "carbanion_smiles"  : SMILES string of the carbanion intermediate
            "neutral_smiles"    : original input SMILES (sanitized)
            "alpha_idx"         : alpha carbon index in the original molecule
            "beta_idx"          : beta carbon index in the original molecule
            "ewg_type"          : detected EWG classification
            "charge"            : total charge of carbanion molecule (-1)
            "n_michael_sites"   : number of Michael acceptor sites detected
        
        Raises:
            ValueError: if no Michael acceptor pattern is found and no indices given
        """
        # ── Step 1: Parse input ───────────────────────────────────────────────────
        self.mol = Chem.MolFromSmiles(self.michael_acceptor_smiles)
        if self.mol is None:
            raise ValueError(f"Could not parse SMILES: {self.michael_acceptor_smiles}")
        self.mol = Chem.RWMol(Chem.AddHs(self.mol))   # explicit H needed for carbanion geometry
        
        # ── Step 2: Detect Michael acceptor site(s) ───────────────────────────────
        self.ewg_type = "user_defined"
        
        if self.alpha_idx is None or self.beta_idx is None:
            self.find_michael_acceptor_atoms()
            if not self.sites:
                raise ValueError(
                    f"No Michael acceptor pattern detected in: {self.michael_acceptor_smiles}\n"
                    f"Supported EWGs: {list(self.EWG_SMARTS.keys())}\n"
                    f"You can manually specify alpha_idx and beta_idx to override."
                )

            if self.verbose:
                print(f"  Detected {len(self.sites)} Michael acceptor site(s):")
                for a, b, ewg in self.sites:
                    print(f"    α-C idx={a}  β-C idx={b}  EWG={ewg}")

            # Use the first (most specifically matched) site
            self.alpha_idx, self.beta_idx, self.ewg_type = self.sites[0]
        else:
            if self.verbose:
                print(f"  Using user-specified α-C idx={self.alpha_idx}, β-C idx={self.beta_idx}")
        
        # ── Step 3: Validate the Cα=Cβ bond ──────────────────────────────────────
        bond = self.mol.GetBondBetweenAtoms(self.alpha_idx, self.beta_idx)
        if bond is None:
            raise ValueError(f"No bond between atom {self.alpha_idx} and {self.beta_idx}")
        if bond.GetBondTypeAsDouble() < 1.9:
            raise ValueError(
                f"Bond between {self.alpha_idx} and {self.beta_idx} is not a double bond "
                f"(bond order = {bond.GetBondTypeAsDouble():.1f})"
            )
        
        # ── Step 4: Modify molecule — build carbanion ─────────────────────────────
        edit_mol = Chem.RWMol(self.mol)
        
        # 4a. Convert Cα=Cβ double bond → single bond
        edit_mol.RemoveBond(self.alpha_idx, self.beta_idx)
        edit_mol.AddBond(self.alpha_idx, self.beta_idx, Chem.BondType.SINGLE)
        
        # 4b. Assign formal charge -1 to α-carbon
        alpha_atom = edit_mol.GetAtomWithIdx(self.alpha_idx)
        alpha_atom.SetFormalCharge(-1)
        
        # 4c. Attach thiolate (-SCH3) to β-carbon
        #     Parse thiolate fragment and merge into molecule
        thiol_mol = Chem.MolFromSmiles(self.thiolate_smiles)
        if thiol_mol is None:
            raise ValueError(f"Invalid thiolate SMILES: {self.thiolate_smiles}")
        thiol_mol = Chem.AddHs(thiol_mol)
        
        # Find the sulfur atom (attachment point) in thiolate fragment
        s_atom_idx_in_thiol = None
        for atom in thiol_mol.GetAtoms():
            if atom.GetAtomicNum() == 16:   # sulfur
                s_atom_idx_in_thiol = atom.GetIdx()
                break
        if s_atom_idx_in_thiol is None:
            raise ValueError(f"No sulfur atom found in thiolate: {self.thiolate_smiles}")
        
        # Combine the two molecules
        combined = Chem.RWMol(Chem.CombineMols(edit_mol, thiol_mol))
        
        # Index of S in the combined molecule
        s_idx_combined = edit_mol.GetNumAtoms() + s_atom_idx_in_thiol
        
        # Bond β-carbon to sulfur
        combined.AddBond(self.beta_idx, s_idx_combined, Chem.BondType.SINGLE)
        
        # Remove one H from β-carbon (it now has an extra substituent)
        beta_atom = combined.GetAtomWithIdx(self.beta_idx)
        # Find and remove one explicit H bonded to beta carbon
        h_to_remove = None
        for neighbor in beta_atom.GetNeighbors():
            if neighbor.GetAtomicNum() == 1:  # hydrogen
                h_to_remove = neighbor.GetIdx()
                break
        if h_to_remove is not None:
            combined.RemoveAtom(h_to_remove)
        
        # Remove one H from sulfur (it was -SH, now -S-)
        # Re-fetch s_idx after possible atom removal
        # Recalculate s_idx if h_to_remove was before it in index
        if h_to_remove is not None and h_to_remove < s_idx_combined:
            s_idx_combined -= 1
        
        s_atom = combined.GetAtomWithIdx(s_idx_combined)
        sh_to_remove = None
        for neighbor in s_atom.GetNeighbors():
            if neighbor.GetAtomicNum() == 1:
                sh_to_remove = neighbor.GetIdx()
                break
        if sh_to_remove is not None:
            combined.RemoveAtom(sh_to_remove)
        
        # ── Step 5: Sanitize and generate SMILES ──────────────────────────────────
        try:
            Chem.SanitizeMol(combined)
        except Exception as e:
            raise ValueError(f"Sanitization failed after carbanion construction: {e}")
        
        self.carbanion_smiles = Chem.MolToSmiles(combined)
        self.carbanion_charge = sum(atom.GetFormalCharge() for atom in combined.GetAtoms())
        if self.carbanion_charge != -1:
            raise ValueError(f"Constructed carbanion has unexpected charge: {self.carbanion_charge}")

        if self.verbose:
            print(f"\n  Neutral Michael acceptor : {self.michael_acceptor_smiles}")
            print(f"  Michael acceptor sites   : {len(self.sites)}")
            print(f"  EWG type detected        : {self.ewg_type}")
            print(f"  α-carbanion intermediate : {self.carbanion_smiles}")
            print(f"  α-carbanion intermediate charge : {self.carbanion_charge}")