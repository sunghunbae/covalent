import numpy as np

from collections import deque
from rdkit import Chem
from rdkit.Chem import AllChem, rdDepictor

from .xtb.wrapper import GFN2xTB


def get_torsion_angle_atom_indices(rdmol:Chem.Mol, strict:bool=True) -> list[tuple]:
    """Determine dihedral angle atoms (a-b-c-d) and rotating group for each rotatable bond (b-c).

    Args:
        rdmol (Chem.Mol): molecule
        strict (bool): whether to exclude amide/imide/ester/acid bonds.

    Returns:
        [   (a, b, c, d, rot_atom_indices, fix_atom_indices), 
            (a, b, c, d, rot_atom_indices, fix_atom_indices), 
            ...,
        ]
    """
    # https://github.com/rdkit/rdkit/blob/1bf6ef3d65f5c7b06b56862b3fb9116a3839b229/rdkit/Chem/Lipinski.py#L47%3E
    # https://github.com/rdkit/rdkit/blob/de602c88809ea6ceba1e8ed50fd543b6e406e9c4/Code/GraphMol/Descriptors/Lipinski.cpp#L108
    if strict :
        # excludes amide/imide/ester/acid bonds
        rotatable_bond_pattern = Chem.MolFromSmarts(
            (
            "[!$(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])("
            "[CH3])[CH3])&!$([CD3](=[N,O,S])-!@[#7,O,S!D1])&!$([#7,O,S!D1]-!@[CD3]="
            "[N,O,S])&!$([CD3](=[N+])-!@[#7!D1])&!$([#7!D1]-!@[CD3]=[N+])]-,:;!@[!$"
            "(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])(["
            "CH3])[CH3])]"
            )
        )
    else:
        rotatable_bond_pattern = Chem.MolFromSmarts('[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]')
    
    rotatable_bonds = rdmol.GetSubstructMatches(rotatable_bond_pattern)
    ri = rdmol.GetRingInfo()
    atom_rings = ri.AtomRings()

    torsion_angle_atom_indices = []

    # small rings (n=3 or 4)
    small_rings = [ ring for ring in list(atom_rings) if len(ring) < 5 ]
    # ex. = [(1, 37, 35, 34, 3, 2), (29, 28, 30)]

    forbidden_terminal_nuclei = [1, 9, 17, 35, 53] # H,F,Cl,Br,I

    for (b_idx, c_idx) in rotatable_bonds:
        # determine a atom ``a`` that define a dihedral angle 
        a_candidates = []
        for neighbor in rdmol.GetAtomWithIdx(b_idx).GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if neighbor_idx == c_idx:
                continue
            neighbor_atomic_num = neighbor.GetAtomicNum()
            if neighbor_atomic_num not in forbidden_terminal_nuclei:
                a_candidates.append((neighbor_atomic_num, neighbor_idx))
        
        if not a_candidates:
            continue
        
        (a_atomic_num, a_idx) = sorted(a_candidates, key=lambda x: (x[0], -x[1]), reverse=True)[0]

        # is a-b in a small ring (n=3 or 4)?
        is_in_small_ring = False
        for small_ring in small_rings:
            if (a_idx in small_ring) and (b_idx in small_ring):
                is_in_small_ring = True
                break
        
        if is_in_small_ring:
            continue

        # determine a atom ``d`` that define a dihedral angle
        d_candidates = []
        for neighbor in rdmol.GetAtomWithIdx(c_idx).GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if (neighbor_idx == b_idx):
                continue
            neighbor_atomic_num = neighbor.GetAtomicNum()
            if neighbor_atomic_num not in forbidden_terminal_nuclei:
                d_candidates.append((neighbor_atomic_num, neighbor_idx))
        
        if not d_candidates:
            continue
        
        (d_atomic_num, d_idx) = sorted(d_candidates, key=lambda x: (x[0], -x[1]), reverse=True)[0]

        # is c-d in a small ring?
        is_in_small_ring = False
        for small_ring in small_rings:
            if (c_idx in small_ring) and (d_idx in small_ring):
                is_in_small_ring = True
                break
        
        if is_in_small_ring:
            continue

        # check ring closure between a and d - avoid macrocycles
        if any(a_idx in ring and d_idx in ring for ring in atom_rings):
            continue

        # determine a group of atoms to be rotated
        # https://ctr.fandom.com/wiki/Break_rotatable_bonds_and_report_the_fragments
        em = Chem.EditableMol(rdmol)
        em.RemoveBond(b_idx, c_idx)
        fragmented = em.GetMol()
        (frag1, frag2) = Chem.GetMolFrags(fragmented, asMols=False) # returns tuple of tuple
        hac1 = sum([ 1 for i in frag1 if rdmol.GetAtomWithIdx(i).GetAtomicNum() > 1 ])
        hac2 = sum([ 1 for i in frag2 if rdmol.GetAtomWithIdx(i).GetAtomicNum() > 1 ])
        
        # smaller fragment will be rotated and must contain at least three heavy atoms
        if min(hac1, hac2) >= 3:
            (frag_rot, frag_fix) = sorted([(hac1, frag1), (hac2, frag2)])
            torsion_angle_atom_indices.append((a_idx, b_idx, c_idx, d_idx, frag_rot[1], frag_fix[1]))

    return torsion_angle_atom_indices




def find_atoms_at_bond_distance(rdmol: Chem.Mol,
                                start_atom_idx: int,
                                distance: int) -> list[int]:
    """Finds atoms at a specific bond distance from a starting atom.

    Args:
        mol: An RDKit Mol object.
        start_atom_idx: The index of the starting atom.
        distance: The desired bond distance.

    Returns:
        A list of atom indices at the specified distance
    """
    assert start_atom_idx < rdmol.GetNumAtoms(), "start_atom_idx out of range."

    found_atoms = []
    visited = set()

    def dfs(curr_atom_idx: int, curr_bond_dist: int):
        if curr_bond_dist == distance:
            found_atoms.append(curr_atom_idx)
            return

        visited.add(curr_atom_idx)
        curr_atom = rdmol.GetAtomWithIdx(curr_atom_idx)
        for next_atom in curr_atom.GetNeighbors():
            next_atom_idx = next_atom.GetIdx()
            if next_atom_idx not in visited:
                dfs(next_atom_idx, curr_bond_dist + 1)

        # Backtrack
        visited.remove(curr_atom_idx)

    dfs(start_atom_idx, 0)

    return found_atoms


def get_bond_distance(rdmol: Chem.Mol, start_atom_idx: int) -> dict:
    """Get bonds distance from a given atom.

    Args:
        mol: An RDKit Mol object.
        start_atom_idx: The index of the starting atom.
        distance: The desired bond distance.

    Returns:
        A list of atom indices at the specified distance
    """
    assert start_atom_idx < rdmol.GetNumAtoms(), "start_atom_idx out of range."

    bond_distance = {}
    visited = set()

    def dfs(curr_atom_idx: int, curr_bond_dist: int):
        if curr_bond_dist in bond_distance:
            bond_distance[curr_bond_dist].append(curr_atom_idx)
        else:
            bond_distance[curr_bond_dist] = [curr_atom_idx]

        visited.add(curr_atom_idx)
        curr_atom = rdmol.GetAtomWithIdx(curr_atom_idx)
        for next_atom in curr_atom.GetNeighbors():
            next_atom_idx = next_atom.GetIdx()
            if next_atom_idx not in visited:
                dfs(next_atom_idx, curr_bond_dist + 1)

        # Backtrack
        visited.remove(curr_atom_idx)

    dfs(start_atom_idx, 0)

    return bond_distance


def find_bonds_to_prune(rdmol: Chem.Mol, 
                        jk: tuple,
                        bond_dist_threshold: int = 4,
                        bond_order_threshold: float = 1.75,
                        electronegative: list[int] = [7, 8, 9, 17, 35],
                        ) -> dict[int, list[int]]:
    """Find pruning candidate bonds from a given atom to construct fragment.

    Rules for a candidate bond to break:
        
        For (i-j-k-l) torsion,

        1. NOT (bond distance from j or k < 4)
        2. NOT (bond order > 1.75)
        3. NOT (Pauling electronegativity of any of bond atoms > 2.9)

    Args:
        mol: An RDKit Mol object.
        start_atom_idx: The index of the starting atom.
        distance: The desired bond distance.

    Pauling electronegativity:
        ```py
        from mendeleev import element
        for i in range(1, 119):  # 118 is the highest atomic number known
            el = element(i)
            if isinstance(el.en_pauling, float) and el.en_pauling > 2.9:
                print(f"Element {i}: {el.symbol} {el.atomic_number} {el.en_pauling}")
        ```
        Element 7: N 7 3.04
        Element 8: O 8 3.44
        Element 9: F 9 3.98
        Element 17: Cl 17 3.16
        Element 35: Br 35 2.96

    Returns:
        A list of atom indices at the specified distance
    """

    (j, k) = jk
    
    dist_from_j = get_bond_distance(rdmol, j)
    dist_from_k = get_bond_distance(rdmol, k)
    
    # sum(,[]) flattens a list of list
    forbidden  = sum([v for d, v in dist_from_j.items() if d < bond_dist_threshold], [])
    forbidden += sum([v for d, v in dist_from_k.items() if d < bond_dist_threshold], [])
    forbidden = set(forbidden)

    start_atom_idx = k # either j or k yields the same result

    found_bonds = {}
    visited = set()

    def ordered(p: int, q: int) -> list[int]:
        """Returns a list of atom indices by bond distance.

        Args:
            p (int): atom index
            q (int): atom index

        Returns:
            list[int]: (atom index closer to the torsion angle, the other)
        """
        dist_p = []
        dist_q = []
        for d, indices in dist_from_j.items():
            if p in indices:
                dist_p.append(d)
            if q in indices:
                dist_q.append(d)
        for d, indices in dist_from_k.items():
            if p in indices:
                dist_p.append(d)
            if q in indices:
                dist_q.append(d)
        if sum(dist_p) < sum(dist_q):
            return [p, q]
        else:
            return [q, p]

    def dfs(curr_atom_idx: int, bond_dist: int):
        """Depth-first recursive search of bonded atoms.

        Args:
            curr_atom_idx (int): atom index.
            bond_dist (int): bond distance.
        """
        curr_atom = rdmol.GetAtomWithIdx(curr_atom_idx)
        visited.add(curr_atom_idx)
        for next_atom in curr_atom.GetNeighbors():
            next_atom_idx = next_atom.GetIdx()
            bond = rdmol.GetBondBetweenAtoms(curr_atom_idx, next_atom_idx)
            n1 = curr_atom.GetAtomicNum()
            n2 = next_atom.GetAtomicNum()
            # forbidden (rule 1)
            too_close = (curr_atom_idx in forbidden) and (next_atom_idx in forbidden)
            # bond order (rule 2)
            bond_order = not (bond.GetBondTypeAsDouble() > bond_order_threshold)
            # Pauling electronegativity (rule 3)
            bond_pauling = not ((n1 in electronegative) or (n2 in electronegative))
            if (bond_dist >= bond_dist_threshold) and (not too_close) \
                and (not bond.IsInRing()) and bond_order and bond_pauling:
                # determine which atom has shorter bond distance to the torsion angle (j or k)                
                found_bonds[bond.GetIdx()] = ordered(curr_atom_idx, next_atom_idx)
                return
            if next_atom_idx not in visited:
                dfs(next_atom_idx, bond_dist + 1)
        # Backtrack
        visited.remove(curr_atom_idx)

    dfs(start_atom_idx, 0)

    return found_bonds


def get_fragment_idx(parent: Chem.Mol,  
                     indices: tuple, 
                     fragment: Chem.Mol) -> tuple:
    """Get fragment atom indices corresponding to given parent indices.

    It uses 3D coordinates to find matching atoms between parent and fragment.
    In comparison with the MCS-based method `get_fragment_idx_with_mcs()`,
            0 elapsed=0.0006455129478126764 sec.
            1 elapsed=0.0005964740412309766
            2 elapsed=0.0005442029796540737
            3 elapsed=0.000652436981908977
            4 elapsed=0.0006737819639965892
            5 elapsed=0.0004481689538806677
            6 elapsed=0.00035582599230110645
            7 elapsed=0.0003812289796769619
            8 elapsed=0.000359484925866127
            9 elapsed=0.0002818549983203411
            10 elapsed=0.000247497926466167
            11 elapsed=0.0003651580773293972

    Args:
        parent (Chem.Mol): rdkit Chem.Mol object.
        parent_indices (tuple): parent atom indices to map within the MCS.
        fragment (Chem.Mol): fragment originated from the parent.

    Returns:
        dict[int, int]: { parent_atom_index : fragment_atom_index, ...}
    """
    parent_xyz = parent.GetConformer().GetPositions() # numpy.ndarray
    frag_xyz = fragment.GetConformer().GetPositions() # numpy.ndarray
    qpos = [parent_xyz[i] for i in indices]
    
    return tuple(j for q in qpos for j, f in enumerate(frag_xyz) if np.array_equal(f, q))



def get_fragment_idx_with_mcs(parent: Chem.Mol, 
                              indices: tuple, 
                              fragment: Chem.Mol) -> tuple:
    """Get fragment atom indices corresponding to given parent indices.

    Warning:
        It uses MCS and can be extremely slow sometimes.
        For example, below are the elapsed times for 12 torsion angles of atorvastatin:
            0 elapsed=5.525973221054301 sec. **
            1 elapsed=1.9143556850031018 *
            2 elapsed=3.145250838017091 *
            3 elapsed=9.390580283012241 **
            4 elapsed=89.97735002799891 ***
            5 elapsed=0.19022215204313397
            6 elapsed=0.013428106089122593
            7 elapsed=0.023345661000348628
            8 elapsed=0.023358764010481536
            9 elapsed=0.0007965450640767813
            10 elapsed=0.0008196790004149079
            11 elapsed=0.04075543500948697

    Args:
        parent (Chem.Mol): rdkit Chem.Mol object.
        parent_indices (tuple): parent atom indices to map within the MCS.
        fragment (Chem.Mol): fragment originated from the parent.

    Returns:
        dict[int, int]: { parent_atom_index : fragment_atom_index, ...}
    """
    mcs_result = Chem.rdFMCS.FindMCS([parent, fragment])
    mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
    
    parent_matches = parent.GetSubstructMatches(mcs_mol)
    frag_matches = fragment.GetSubstructMatches(mcs_mol)

    indices_idx = None
    for parent_matched_indices in parent_matches:
        # It is possible to have more than one matches (i.e. methyl rotation).
        # However, even if there are more than one matches, the parent indices
        # should be the same.
        if indices_idx is None:
            indices_idx = {x: parent_matched_indices.index(x) for x in indices}
        else:
            assert all([indices_idx[x] == parent_matched_indices.index(x) for x in indices])

    indices_map = None
    for frag_matched_indices in frag_matches:
        # it is possible to have more than one matches (i.e. methyl rotation)
        if indices_map is None:
            indices_map = {x : frag_matched_indices[indices_idx[x]] for x in indices}
        else:
            assert all([indices_map[x] == frag_matched_indices[indices_idx[x]] for x in indices])
                
    return tuple([indices_map[x] for x in indices])



def create_fragment_on_bonds(rdmol2D: Chem.Mol, bonds: dict, cap: bool = True) -> Chem.Mol | None:
    """Create a fragment that preserves defined atoms.

    Args:
        rdmol (Chem.Mol): input molecule.
        bonds (dict): {bond_index : (preserved_atom_index, removed_atom_index), ...}
        cap (bool): whether to cap the dummy atom(s) with hydrogen(s)

    Returns:
        Chem.Mol: resulting fragment molecule.
    """
    fragments = Chem.FragmentOnBonds(rdmol2D, list(bonds))
    preserved_atoms = { preserved for bond_idx, (preserved, removed) in bonds.items() }
    for fragment_indices, fragment_mol in zip(Chem.GetMolFrags(fragments), 
                                              Chem.GetMolFrags(fragments, asMols=True)):
        if preserved_atoms.issubset(set(fragment_indices)):
            if cap:
                # cap dummy atoms with hydrogens
                for atom in fragment_mol.GetAtoms():
                    if atom.GetAtomicNum() == 0: 
                        atom.SetAtomicNum(1)

            return fragment_mol # 2D
        
    return None


def truncate(rdmolH: Chem.Mol, jk: tuple, wbo_tolerance: float = 0.03) -> tuple[str, Chem.Mol, list[int], bool, bool]:
    """Create a close surrogate fragment that captures the PES of the intended torsion.
    
    Fragmentation aims to preserve the local chemical environment around the targeted torsion
    while increase calculation speed and potential complications. To avoid oversimplification
    and inaccurate approximation, two strategies are combined:
        - fragment candidates are generated by a set of reasonably empirical rules        
        - further filtered by Wiberg bond order (WBO) calculated by semi-empirical QM. It has 
        been shown that the Wiberg bond order (WBO) provides a fast and robust measure of 
        whether a torsion profile has been disrupted by fragmentation. Any fragment that causes
        WBO difference larger than 0.03 will be excluded.

    Args:
        rdmol (Chem.Mol): molecule.
        torsion_indices (tuple): (i, j, k, l, atoms to be rotated, atoms to be fixed)

    Returns:
        (Chem.Mol: fragment molecule, 
        list[int]: fragment indices, 
        bool: True if fragmented, 
        bool: True if WBO filtering is used)

    References:
        https://pubs.acs.org/doi/10.1021/acs.jcim.2c01153
        https://www.biorxiv.org/content/10.1101/2020.08.27.270934v2
    """
    (j, k) = jk

    rdmol2D = Chem.Mol(rdmolH)
    # 2. Compute 2D coordinates and clear existing 3D conformers
    rdDepictor.Compute2DCoords(rdmol2D, clearConfs=True)
    candidates = find_bonds_to_prune(rdmol2D, jk)
    print("prune:")
    for k, v in candidates.items():
        print(f"{k} {v[0]} {rdmol2D.GetAtomWithIdx(v[0]).GetAtomicNum()} {v[1]} "
              f"{rdmol2D.GetAtomWithIdx(v[1]).GetAtomicNum()}")
    
    if not candidates:
        # no fragmentation
        return (rdmolH, jk, False, False)

    WBO_filtered = False
    if GFN2xTB().is_ready():
        # fragmented
        # filter candidate(s) by Wiberg bond order (WBO) if xTB is available
        jk = tuple(sorted([j, k]))
        wbo_passed_candidates = {}
        parent = GFN2xTB(rdmolH).singlepoint()
        assert hasattr(parent, 'wbo'), "create_torsion_fragment() Error: no wbo for parent"
        for bond_idx, (p, q) in candidates.items():
            frag_single_break = create_fragment_on_bonds(rdmol2D, {bond_idx: (p, q)})
            # 2D -> 3D
            Chem.AddHs(frag_single_break)
            AllChem.EmbedMolecule(frag_single_break, AllChem.ETKDG())
            AllChem.MMFFOptimizeMolecule(frag_single_break)

            fragment = GFN2xTB(frag_single_break).singlepoint()

            assert hasattr(fragment, 'wbo'), "create_torsion_fragment() Error: no wbo for fragment"
            # WBO difference at the torsion angle bond
            frag_jk = get_fragment_idx(rdmol2D, jk, frag_single_break)
            frag_jk = tuple(sorted(frag_jk))
            if abs(fragment.wbo[frag_jk] - parent.wbo[jk]) < wbo_tolerance:
                wbo_passed_candidates[bond_idx] = (p, q)
        frag_multi_breaks = create_fragment_on_bonds(rdmol2D, wbo_passed_candidates)
        WBO_filtered = True
    else:
        # skip WBO filtering
        frag_multi_breaks = create_fragment_on_bonds(rdmol2D, candidates)

    frag_indices = get_fragment_idx(rdmol2D, (j, k), frag_multi_breaks)
    
    return (Chem.MolToSmiles(frag_multi_breaks), frag_multi_breaks, frag_indices, True, WBO_filtered)