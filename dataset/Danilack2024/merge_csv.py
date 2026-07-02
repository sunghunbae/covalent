import polars as pl

dfs = []
for dataset in ['S2', 'S3', 'S4', 'S5']:
    _ = pl.read_csv(f"{dataset}.csv", schema_overrides={
        'Name': pl.String,
        'SMILES': pl.String,
        'GSH_half_life_min': pl.String})

    _ = _.with_columns(
            pl.lit(dataset).alias('dataset'),
            pl.concat_str([pl.lit(dataset), pl.lit("-"), pl.col("Name")]).alias("id"),
            )
    print(_)
    dfs.append(_)

df = pl.concat(dfs)
df.write_csv("SI.csv")
