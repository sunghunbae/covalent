from covalent import Geometry
from covalent.carbanion import compute_carbanion_descriptors

# ── Warhead 1: Acrylamide ─────────────────────────────────────────────
print("\n" + "="*60)
print("  WARHEAD 1: Acrylamide")
print("="*60)

acrylamide = Geometry(smiles="C=CC(=O)N")
acrylamide.optimize()

acrylamide_carbanion = Geometry(smiles="[CH2-]CC(=O)N", charge=-1)
acrylamide_carbanion.optimize()

desc_acrylamide = compute_carbanion_descriptors(
    neutral_xyz = acrylamide.xyz_block,
    anion_xyz   = acrylamide_carbanion.xyz_block,
    name        = "Acrylamide"
)

print(desc_acrylamide)