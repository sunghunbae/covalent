from rdkit import Chem
from covalent import prune

if __name__ == '__main__':
    mol = Chem.MolFromSmiles("FC1=C(Cl)C=C(NC2=NC=NC3=C2C=C(NC(/C=C/CN(C)C)=O)C(O[C@@H]4COCC4)=C3)C=C1")
    
    pattern = Chem.MolFromSmarts("C=CC")
    center = mol.GetSubstructMatch(pattern)

    parent_na = mol.GetNumAtoms(onlyExplicit=False)
    print(parent_na, Chem.MolToSmiles(Chem.RemoveHs(mol)), center)

    frag = prune(mol, center=center, cap_dummy_atom=1, verbose=False)
    na, frag_mol, frag_center = sorted(frag, key=lambda x: x[0])[0] # smallest fragment
    print(na, Chem.MolToSmiles(Chem.RemoveHs(frag_mol)), frag_center)