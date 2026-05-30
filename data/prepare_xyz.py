from covalent import Geometry, MichaelAddition
from covalent import prune

from rdkit import Chem
from pathlib import Path


dataset = 'S5'

infile = Path(f"Danilack_et_al_2024/si/{dataset}.csv")
outdir = infile.parent / f"{dataset}"
outdir.mkdir(parents=True, exist_ok=True)
outfile = infile.parent / f"{dataset}_struct.csv"

with open(infile, "r") as f, open(outfile, "w") as g:
    g.write("Name,Original_or_Pruned,RC,SMILES\n")
    for line in f:
        if line.startswith("Name"):
            continue

        name, smiles, GSH_half_life_min = line.strip().split(",")
        print(f"Processing {name} with GSH half-life {GSH_half_life_min} min...")
        
        # we need to prune the original molecule for QM calculations
        # fast check fukui index for truncation candidates using xtb 
        rxn_original = MichaelAddition(reactant=smiles, donor="SC", verbose=False)
        jk = (rxn_original.alpha_idx, rxn_original.beta_idx)
        mol = Chem.MolFromSmiles(smiles)
        
        print(f"Original SMILES: {smiles}")
        print(f"Original number of atoms: {mol.GetNumAtoms()}")
        print(f"Original Alpha index: {jk[0]}, Beta index: {jk[1]}")

        for (rc, smiles_, geom) in [
            ('reactant', rxn_original.reactant_smiles, Geometry(smiles=rxn_original.reactant_smiles)),
            ('intermediate', rxn_original.carbanion_smiles, Geometry(rdmol=rxn_original.carbanion_rdmol, charge=-1)), 
            ('product', rxn_original.product_smiles, Geometry(smiles=rxn_original.product_smiles)),
            ]:
            g.write(f"{name},original,{rc},{smiles_}\n")
            title = f"{name}_original_{rc}"
            if (outdir / f"{title}.xyz").exists():
                continue
            geom.xtb_optimize()
            geom.write_xyz(outdir / f"{title}.xyz")
            print(f"Written optimized geometry for {title} to {outdir / f'{title}.xyz'}")

        # ---------------------- PRUNING ------------------        
        frag = prune(mol, center=jk, cap_dummy_atom=1, verbose=False)

        if frag: # if no pruned fragment candidate is returned, use original structure
            na, frag_mol, frag_center = sorted(frag, key=lambda x: x[0])[0] # smallest fragment
            print(f"Pruned Fragment SMILES: {Chem.MolToSmiles(Chem.RemoveHs(frag_mol))}")
            print(f"Pruned Number of atoms in fragment: {na}")
            print(f"Pruned Fragment center: {frag_center}")
        else:
            frag_mol = mol
            frag_center = jk
            print(f"No pruned fragment found, using original molecule.")

        
        rxn_pruned = MichaelAddition(reactant=frag_mol, donor="SC", verbose=False)
        jk_min = (rxn_pruned.alpha_idx, rxn_pruned.beta_idx)
        print(f"Pruned Alpha index: {jk_min[0]}, Beta index: {jk_min[1]}")

        for (rc, smiles_, geom) in [
            ('reactant', rxn_pruned.reactant_smiles, Geometry(smiles=rxn_pruned.reactant_smiles)),
            ('intermediate', rxn_pruned.carbanion_smiles, Geometry(rdmol=rxn_pruned.carbanion_rdmol, charge=-1)), 
            ('product', rxn_pruned.product_smiles, Geometry(smiles=rxn_pruned.product_smiles)),
            ]:
            g.write(f"{name},pruned,{rc},{smiles_}\n")
            title = f"{name}_pruned_{rc}"
            if (outdir / f"{title}.xyz").exists():
                continue
            geom.xtb_optimize()
            geom.write_xyz(outdir / f"{title}.xyz")
            print(f"Written optimized geometry for {title} to {outdir / f'{title}.xyz'}")
