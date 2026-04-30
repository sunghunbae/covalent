from covalent import Geometry, FukuiIndex

smiles = "C=CC(=O)N"  # N-methyl acrylamide

geometry = Geometry(smiles)
geometry.optimize()

f = FukuiIndex(geometry)

f.run()
f.show()