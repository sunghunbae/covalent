from covalent import Geometry, Gibbs_free_energy

smiles = "O"  # water

geometry = Geometry(smiles)
geometry.optimize()

G = Gibbs_free_energy(geometry)
print(f"Gibbs free energy = {G:.6f} Hartree")