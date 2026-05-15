
from rdkit import Chem
from rdkit.Chem import rdmolops, AllChem
from covalent.xtb.wrapper import GFN2xTB
from itertools import combinations
import numpy as np


def matched_frag_idx(parent_mol: Chem.Mol,  
                     parent_indices: tuple, 
                     fragment_mol: Chem.Mol) -> tuple:
    """Get fragment atom indices corresponding to given parent indices.

    It uses 3D coordinates to find matching atoms between parent and fragment.
    It is much faster than the MCS-based method. 

    Args:
        parent_mol (Chem.Mol): rdkit Chem.Mol object.
        parent_indices (tuple): parent atom indices to map within the MCS.
        fragment (Chem.Mol): fragment originated from the parent.

    Returns:
        dict[int, int]: { parent_atom_index : fragment_atom_index, ...}
    """
    parent_xyz = parent_mol.GetConformer().GetPositions() # numpy.ndarray
    frag_xyz = fragment_mol.GetConformer().GetPositions() # numpy.ndarray
    indice_pos = [parent_xyz[i] for i in parent_indices]
    
    return tuple(j for i in indice_pos for j, f in enumerate(frag_xyz) if np.array_equal(f, i))

    
def prune(mol: Chem.Mol, 
          center: list[int],
          max_wbo_diff: float = 0.03,
          min_topo_dist: int = 4,
          cap_dummy_atom: int = 1,
          bond_order_threshold: float = 1.75,
          electronegative_atoms: list[int] = [7, 8, 9, 17, 35],
          verbose: bool = False
          ) -> list[tuple[int, Chem.Mol, tuple]]:
    # compute distance matrix and find candidate bonds to break
    dm = rdmolops.GetDistanceMatrix(mol)
    
    # find bonds between ring and non-ring atoms
    bis = mol.GetSubstructMatches(Chem.MolFromSmarts('[!R][R]'))

    # only keep the bonds that are at least min_topo_dist away from the wbo_watch atoms
    center = set(center)

    # bonds to watch WBO changes, e.g. bonds in the reaction center
    wbo_watch = [(i,j) for i,j in list(combinations(center, 2)) if mol.GetBondBetweenAtoms(i,j) is not None]

    # electronegative atoms are less likely to be truncated
    # electronegative: list[int] = [7, 8, 9, 17, 35]

    # bonds with bond order less than this threshold are more likely to be truncated
    # bond_order_threshold: float = 1.75

    d_vs_ij = []
    for i,j in bis:
        d_min = []
        for k in center:
            d_min += [dm[k][i], dm[k][j]]
        d = int(min(d_min))
        d_vs_ij.append((d, (i,j)))

    # 3D-conformer is needed to compute WBOs with GFN2xTB, which are used to evaluate the truncation quality.
    rdmolH = Chem.AddHs(mol)
    AllChem.EmbedMolecule(rdmolH, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(rdmolH)

    parent = GFN2xTB(rdmolH).singlepoint()

    assert hasattr(parent, 'wbo'), "Error: no wbo for parent"
    
    if verbose:
        for ij in wbo_watch:
            print(f"parent WBO={parent.wbo[ij]}")
    
    bonds_to_break = []
    truncated = []
    for d, (i,j) in sorted(d_vs_ij, reverse=True):
        bond = mol.GetBondBetweenAtoms(i,j)
        n1 = mol.GetAtomWithIdx(i).GetAtomicNum()
        n2 = mol.GetAtomWithIdx(j).GetAtomicNum()
        bond_order = (bond.GetBondTypeAsDouble() < bond_order_threshold)
        bond_pauling = not((n1 in electronegative_atoms) or (n2 in electronegative_atoms))
        
        # print(f"d={d} {n1} {n2} bond_order={bond.GetBondTypeAsDouble()} pauling={bond_pauling}")
        if verbose:
            print(f"d={d} {n1} {n2} bond_order={bond.GetBondTypeAsDouble()} pauling={bond_pauling}")

        if d >= min_topo_dist :
            
            bonds_to_break.append(bond.GetIdx())
            if verbose:
                print(f"bonds_to_break={bonds_to_break}")
            
            edit_mol = Chem.Mol(rdmolH)

            fragments = Chem.FragmentOnBonds(edit_mol, bonds_to_break)
            
            for fragment_indices, fragment_mol in zip(Chem.GetMolFrags(fragments), 
                                                    Chem.GetMolFrags(fragments, asMols=True)):
                if center.issubset(set(fragment_indices)):
                    if cap_dummy_atom > 0:
                        # cap_dummy_atom dummy atoms with carbon
                        for atom in fragment_mol.GetAtoms():
                            if atom.GetAtomicNum() == 0: 
                                atom.SetAtomicNum(cap_dummy_atom)
                    break

            child = GFN2xTB(fragment_mol).singlepoint()
            assert hasattr(child, 'wbo'), "Error: no wbo for child"
            frag_na = fragment_mol.GetNumAtoms(onlyExplicit=False)

            wbo_diff = []
            for jk in wbo_watch:
                frag_jk = matched_frag_idx(rdmolH, jk, fragment_mol)
                wbo_diff_ = abs(parent.wbo[jk] - child.wbo[frag_jk])
                wbo_diff.append(wbo_diff_ <= max_wbo_diff)
                if verbose:
                    print(f"Child={Chem.MolToSmiles(fragment_mol)}")
                    print(f"WBO frag_jk={frag_jk} WBO={parent.wbo[jk]} -> {child.wbo[frag_jk]} diff={wbo_diff_} {'PASS' if wbo_diff_ <= max_wbo_diff else 'FAIL'}")

            if all(wbo_diff):
                truncated.append((frag_na, 
                                  fragment_mol, 
                                  matched_frag_idx(rdmolH, center, fragment_mol)))

    return truncated


if __name__ == '__main__':
    mol = Chem.MolFromSmiles("FC1=C(Cl)C=C(NC2=NC=NC3=C2C=C(NC(/C=C/CN(C)C)=O)C(O[C@@H]4COCC4)=C3)C=C1")
    
    pattern = Chem.MolFromSmarts("C=CC")
    center = mol.GetSubstructMatch(pattern)

    parent_na = mol.GetNumAtoms(onlyExplicit=False)
    print(parent_na, Chem.MolToSmiles(Chem.RemoveHs(mol)), center)

    frag = prune(mol, center=center, cap_dummy_atom=6, verbose=True)
    na, frag_mol, frag_center = sorted(frag, key=lambda x: x[0])[0] # smallest fragment
    print(na, Chem.MolToSmiles(Chem.RemoveHs(frag_mol)), frag_center)