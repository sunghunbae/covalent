from covalent import Reaction

test_cases = [
    ("acrylamide",        "C=CC(=O)N",              "SC"),
    ("vinyl_sulfone",     "C=CS(=O)(=O)c1ccccc1",   "SC"),
    ("cyanoacrylamide",   "N/C(=C/C#N)C(=O)N",      "SC"),
    ("methyl_vinyl_ketone","C=CC(=O)C",              "SC"),
    # Cysteine-like thiolate surrogate
    ("acrylamide_CysLike","C=CC(=O)N",              "SCC(N)C(=O)O"),
]

print("=" * 65)
print("  α-Carbanion Intermediate Builder — Test Suite")
print("=" * 65)

for name, smiles, thiol in test_cases:
    print(f"\n▶ {name}")
    try:
        rxn = Reaction(smiles, thiol_smiles=thiol, verbose=True)
        print(f"  Reactant SMILES: {smiles}")
        print(f"  Thiol SMILES: {thiol}")
        print(f"  Alpha carbon index: {rxn.alpha_idx}")
        print(f"  Beta carbon index: {rxn.beta_idx}")
        print(f"  Carbanion SMILES: {rxn.carbanion_smiles}")
        print(f"  Product SMILES: {rxn.product_smiles}")
        print(f"  EWG type: {rxn.ewg_type}")
        print()
    except ValueError as e:
        print(f"  ✗ Error: {e}")