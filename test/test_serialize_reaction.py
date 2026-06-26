from covalent import Reaction
from rdkit import Chem

r = Reaction(reactant="C=CC(=O)N", thiol_smiles="SC", verbose=True)
serialized = r.serialize()

print("Serialized string size=", len(serialized))
print("Serialized string:", serialized)

s = Reaction()
s.deserialize(serialized)

t = Reaction().deserialize(serialized)

# compare with s

assert r.thiol_smiles == s.thiol_smiles
assert r.ewg == s.ewg
assert r.alpha_idx == s.alpha_idx
assert r.beta_idx == s.beta_idx

assert r.reactant_smiles == s.reactant_smiles
assert Chem.MolToSmiles(r.reactant_rdmol) == Chem.MolToSmiles(s.reactant_rdmol)
assert r.reactant_charge == s.reactant_charge

assert r.carbanion_smiles == s.carbanion_smiles
assert r.carbanion_charge == s.carbanion_charge
assert Chem.MolToSmiles(r.carbanion_rdmol) == Chem.MolToSmiles(s.carbanion_rdmol)
assert r.carbanion_alpha_idx == s.carbanion_alpha_idx
assert r.carbanion_beta_idx == s.carbanion_beta_idx
assert r.carbanion_S_idx == s.carbanion_S_idx

assert r.product_charge == s.product_charge
assert r.product_smiles == s.product_smiles
assert Chem.MolToSmiles(r.product_rdmol) == Chem.MolToSmiles(s.product_rdmol)
assert r.product_alpha_idx == s.product_alpha_idx
assert r.product_beta_idx == s.product_beta_idx
assert r.product_S_idx == s.product_S_idx

# compare with t

assert r.thiol_smiles == t.thiol_smiles
assert r.ewg == t.ewg
assert r.alpha_idx == t.alpha_idx
assert r.beta_idx == t.beta_idx

assert r.reactant_smiles == t.reactant_smiles
assert Chem.MolToSmiles(r.reactant_rdmol) == Chem.MolToSmiles(t.reactant_rdmol)
assert r.reactant_charge == t.reactant_charge

assert r.carbanion_smiles == t.carbanion_smiles
assert r.carbanion_charge == t.carbanion_charge
assert Chem.MolToSmiles(r.carbanion_rdmol) == Chem.MolToSmiles(t.carbanion_rdmol)

assert r.product_charge == t.product_charge
assert r.product_smiles == t.product_smiles
assert Chem.MolToSmiles(r.product_rdmol) == Chem.MolToSmiles(t.product_rdmol)