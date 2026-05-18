from covalent import Geometry


g = Geometry(smiles="C=CC(=O)N", charge=0)
serialized = g.serialize()

h = Geometry()
h.deserialize(serialized)

assert h.smiles == g.smiles
assert h.xyz_string == g.xyz_string
assert h.mol_str == g.mol_str