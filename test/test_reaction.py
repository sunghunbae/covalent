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
    except ValueError as e:
        print(f"  ✗ Error: {e}")