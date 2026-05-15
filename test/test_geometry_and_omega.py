from covalent import Geometry, omega

smiles = "C=CC(=O)N"  # N-methyl acrylamide
geometry = Geometry(smiles)
print("Initial geometry:")
print(geometry.mol_str)

geometry.optimize()
print("\nOptimized geometry:")
print(geometry.mol_str)

results = omega(geometry)
print("Global electrophilicity index:")
for key, value in results.items():
    print(f"  {key}: {value}")