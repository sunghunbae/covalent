
from rdkit import Chem
from rdkit.Chem import rdmolops, AllChem
from covalent.xtb.wrapper import GFN2xTB

import polars as pl
import numpy as np


def get_matching_fragment_idx(parent: Chem.Mol,  parent_indices: tuple, fragment: Chem.Mol) -> tuple:
    """Get fragment atom indices corresponding to given parent indices.

    It uses 3D coordinates to find matching atoms between parent and fragment.
    It is much faster than the MCS-based method. 

    Args:
        parent (Chem.Mol): rdkit Chem.Mol object.
        parent_indices (tuple): parent atom indices to map within the MCS.
        fragment (Chem.Mol): fragment originated from the parent.

    Returns:
        dict[int, int]: { parent_atom_index : fragment_atom_index, ...}
    """
    parent_xyz = parent.GetConformer().GetPositions() # numpy.ndarray
    frag_xyz = fragment.GetConformer().GetPositions() # numpy.ndarray
    indice_pos = [parent_xyz[i] for i in parent_indices]
    
    return tuple(j for i in indice_pos for j, f in enumerate(frag_xyz) if np.array_equal(f, i))

    
def truncate(mol: Chem.Mol, 
             wbo_watch: tuple[int, int], 
             keep: list[int], 
             cap: int = 1,
             wbo_tolerance: float = 0.03):
    dm = rdmolops.GetDistanceMatrix(mol)
    bis = mol.GetSubstructMatches(Chem.MolFromSmarts('[!R][R]'))
    keep = set(keep)
    electronegative: list[int] = [7, 8, 9, 17, 35]
    bond_order_threshold: float = 1.75

    d_vs_ij = []
    for i,j in bis:
        d_min = []
        for k in keep:
            d_min += [dm[k][i], dm[k][j]]
        d = int(min(d_min))
        d_vs_ij.append((d, (i,j)))

    
    rdmolH = Chem.AddHs(mol)
    AllChem.EmbedMolecule(rdmolH, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(rdmolH)
    parent = GFN2xTB(rdmolH).singlepoint()
    assert hasattr(parent, 'wbo'), "Error: no wbo for parent"
    print("parent WBO=", parent.wbo[wbo_watch])
    
    bonds_to_break = []
    for d, (i,j) in sorted(d_vs_ij, reverse=True):
        bond = mol.GetBondBetweenAtoms(i,j)
        n1 = mol.GetAtomWithIdx(i).GetAtomicNum()
        n2 = mol.GetAtomWithIdx(j).GetAtomicNum()
        bond_order = (bond.GetBondTypeAsDouble() < bond_order_threshold)
        bond_pauling = not((n1 in electronegative) or (n2 in electronegative))
        
        print(f"d={d} {n1} {n2} bond_order={bond.GetBondTypeAsDouble()} pauling={bond_pauling}")

        if d >= 4 : #and bond_order and bond_pauling:
            
            bonds_to_break.append(bond.GetIdx())
            print("bonds_to_break=", bonds_to_break)
            edit_mol = Chem.Mol(rdmolH)

            fragments = Chem.FragmentOnBonds(edit_mol, bonds_to_break)
            
            for fragment_indices, fragment_mol in zip(Chem.GetMolFrags(fragments), 
                                                    Chem.GetMolFrags(fragments, asMols=True)):
                if keep.issubset(set(fragment_indices)):
                    if cap > 0:
                        # cap dummy atoms with carbon
                        for atom in fragment_mol.GetAtoms():
                            if atom.GetAtomicNum() == 0: 
                                atom.SetAtomicNum(cap)
                    break

            child = GFN2xTB(fragment_mol).singlepoint()
            assert hasattr(child, 'wbo'), "Error: no wbo for child"
    
            frag_jk = get_matching_fragment_idx(rdmolH, wbo_watch, fragment_mol)
            print("child=", Chem.MolToSmiles(fragment_mol))
            print("child frag_jk=", frag_jk, "wbo_watch=", wbo_watch)
            print("child WBO=", child.wbo[frag_jk])

            wbo_diff = abs(child.wbo[frag_jk] - parent.wbo[wbo_watch])
            print("child wbo_diff=", wbo_diff, "wbo_tolerance=", wbo_tolerance)
            print("-"*40)



if __name__ == '__main__':
    mol = Chem.MolFromSmiles("FC1=C(Cl)C=C(NC2=NC=NC3=C2C=C(NC(/C=C/CN(C)C)=O)C(O[C@@H]4COCC4)=C3)C=C1")
    truncate(mol, keep=(17,18), wbo_watch=(17,18))
    