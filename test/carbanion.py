from covalent import Geometry
from covalent.carbanion import compute_carbanion_descriptors
from rdkit import Chem


_ = Chem.MolFromSmiles("C=CC(=O)N")
print(Chem.MolToSmiles(_))  # Should print: [CH2-]CC(=O)N
total_h = sum(atom.GetTotalNumHs() for atom in _.GetAtoms())
print(f"Total number of hydrogens: {total_h}")  # Should print the total number of hydrogens


_ = Chem.MolFromSmiles("C[CH-]C(=O)N")
print(Chem.MolToSmiles(_))  # Should print: [CH2-]CC(=O)N
total_h = sum(atom.GetTotalNumHs() for atom in _.GetAtoms())
print(f"Total number of hydrogens: {total_h}")  # Should print the total number of hydrogens

Chem.AddHs(_)
print(Chem.MolToSmiles(_))  # Should print: [CH2-]CC(=O)N with explicit hydrogens
total_h = sum(atom.GetTotalNumHs() for atom in _.GetAtoms())
print(f"Total number of hydrogens: {total_h}")  # Should print the total number of hydrogens


# ── Warhead 1: Acrylamide ─────────────────────────────────────────────
print("\n" + "="*60)
print("  WARHEAD 1: Acrylamide")
print("="*60)

acrylamide = Geometry(smiles="C=CC(=O)N")
acrylamide.optimize()

acrylamide_carbanion = Geometry(smiles="C[CH-]C(=O)N", charge=-1)
acrylamide_carbanion.optimize()

desc_acrylamide = compute_carbanion_descriptors(
    neutral_xyz = acrylamide.xyz_block,
    anion_xyz   = acrylamide_carbanion.xyz_block,
    name        = "Acrylamide"
)

print(desc_acrylamide)