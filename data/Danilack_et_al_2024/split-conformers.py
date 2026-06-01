from rdkit import Chem
from pathlib import Path
import gzip


workdir = Path(".")

with gzip.open(workdir / "original_rc_fixpka_omega.sdf.gz", "rb") as gzip_file:
    # Pass the stream to the ForwardSDMolSupplier
    with Chem.ForwardSDMolSupplier(gzip_file, removeHs=False) as supplier:
        conformers = {}
        for mol in supplier:
            name = mol.GetProp("_Name")
            if name not in conformers:
                conformers[name] = []
            conformers[name].append(mol)

for name, confs in conformers.items():
    print(f"{name}: {len(confs)} conformers")
    name, rc = name.split("@")
    outfile = workdir / f"{name}_{rc}.sdf.gz"
    with gzip.open(outfile, "wt") as gz_file:
        with Chem.SDWriter(gz_file) as writer:
            for conf in confs:
                writer.write(conf)