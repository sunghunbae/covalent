from pathlib import Path


for dataset in ['S2', 'S3', 'S4', 'S5']:
    infile = Path(f"Danilack_et_al_2024/si/{dataset}_struct.csv")
    outdir = infile.parent / f"{dataset}_conformers"
    outdir.mkdir(exist_ok=True, parents=True)
    outfile = outdir / "original_rc.smi"

    with open(infile, "r") as f, open(outfile, "w") as g:
        for line in f:
            if line.startswith("Name"): 
                continue # skip header
            name, original_or_pruned, rc, smiles = line.strip().split(",")
            if original_or_pruned == "original":
                g.write(f"{smiles} {name}@{rc}\n")