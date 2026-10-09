import pandas as pd
df = pd.read_csv("CY25-1_1-#1.csv")
print(df.columns.tolist()); print(len(df)); print(df.dtypes)
cyc = df["cycle number"].unique()
print(cyc[:10], len(cyc))
print(df.groupby("cycle number")["time/s"].agg(["min", "max", "count"]).head(8))
c = df[df["cycle number"] == cyc[1]]
rest = c[c["<I>/mA"].abs() < 5]
print(len(rest)); print(rest[["time/s", "Ecell/V", "<I>/mA"]].head(25))