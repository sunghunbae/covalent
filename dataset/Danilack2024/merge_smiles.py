with open("SI.csv", "w") as g:
    for dataset in ['S2', 'S3', 'S4', 'S5']:
        with open(f"{dataset}.csv", "r") as f:
            for line in f:
                smi, name = line.strip().split()
                g.write(f"{smi} {dataset}-{name}\n")
