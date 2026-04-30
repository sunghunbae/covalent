from covalent import Geometry, electrophilicity_index


smiles = "C=CC(=O)N"  # N-methyl acrylamide
geometry = Geometry(smiles)
print("Initial geometry:")
print(geometry.mol_str)

geometry.optimize()
print("\nOptimized geometry:")
print(geometry.mol_str)

gei_results = electrophilicity_index(geometry)
print("Global electrophilicity index:")
for key, value in gei_results.items():
    print(f"  {key}: {value}")