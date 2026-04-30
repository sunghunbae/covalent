from covalent import Geometry
from covalent.carbanion import compute_carbanion_descriptors
from rdkit import Chem
from pathlib import Path

# ── Warhead 1: Acrylamide ─────────────────────────────────────────────
print("\n" + "="*60)
print("  WARHEAD 1: Acrylamide")
print("="*60)

if Path("acrylamide_prod_neutral.xyz").exists():
    acrylamide_neutral = Geometry(smiles="CSCCC(=O)N")
    acrylamide_neutral.update_coords("acrylamide_prod_neutral.xyz")
else:
    acrylamide_neutral = Geometry(smiles="CSCCC(=O)N")
    acrylamide_neutral.pre_optimize() # Pre-optimization with GFN2-xTB to get a reasonable starting geometry for Psi4 DFT optimization.
    acrylamide_neutral.optimize()
    acrylamide_neutral.write_xyz("acrylamide_prod_neutral.xyz")

# if Path("acrylamide_prod_carbanion.xyz").exists():
#     acrylamide_carbanion = Geometry(smiles="CSC[CH-]C(=O)N", charge=-1)
#     acrylamide_carbanion.update_coords("acrylamide_prod_carbanion.xyz")
# else:
#     acrylamide_carbanion = Geometry(smiles="CSC[CH-]C(=O)N", charge=-1)
#     acrylamide_carbanion.pre_optimize()
#     acrylamide_carbanion.optimize()
#     acrylamide_carbanion.write_xyz("acrylamide_prod_carbanion.xyz")

desc_acrylamide = compute_carbanion_descriptors(
    neutral_xyz = acrylamide_neutral.xyz_block,
    anion_xyz = None,
    # anion_xyz   = acrylamide_carbanion.xyz_block,
    name        = "Acrylamide"
)

print(desc_acrylamide)