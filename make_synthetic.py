"""Generate a tiny AMEX-shaped dataset to smoke-test the full pipeline."""
import numpy as np, pandas as pd
from pathlib import Path

rng = np.random.RandomState(0)
RAW = Path("data/raw"); RAW.mkdir(parents=True, exist_ok=True)

N_CUST = 800
rows = []
ids = []
for c in range(N_CUST):
    cid = "".join(rng.choice(list("0123456789abcdef"), size=32))
    ids.append(cid)
    n_stmt = rng.randint(1, 14)                     # 1..13 statements
    base = rng.randn()
    for t in range(n_stmt):
        row = {"customer_ID": cid,
               "S_2": pd.Timestamp("2018-01-01") + pd.Timedelta(days=30*t)}
        for pre, k in [("P", 3), ("B", 8), ("S", 6), ("R", 6), ("D", 12)]:
            for j in range(k):
                row[f"{pre}_{j}"] = base + rng.randn() + 0.1*t
        # categoricals
        row["D_63"] = rng.choice(["CO", "CR", "CL"])
        row["D_64"] = rng.choice(["O", "R", "U", np.nan])
        # a mostly-missing column that should be dropped (like D_87)
        row["D_87"] = np.nan if rng.rand() > 0.05 else rng.randn()
        rows.append(row)

df = pd.DataFrame(rows)
# inject some missingness
for col in ["B_3", "S_2_dummy" if False else "R_4", "D_5"]:
    if col in df: df.loc[df.sample(frac=0.2, random_state=1).index, col] = np.nan
df.to_csv(RAW / "train_data.csv", index=False)

# labels: default prob rises with mean of B_0
lab = (df.groupby("customer_ID")["B_0"].mean()
       .pipe(lambda s: (s > s.median()).astype(int)).reset_index())
lab.columns = ["customer_ID", "target"]
# add noise
flip = lab.sample(frac=0.25, random_state=2).index
lab.loc[flip, "target"] = 1 - lab.loc[flip, "target"]
lab.to_csv(RAW / "train_labels.csv", index=False)
print(f"wrote {len(df)} statement rows for {N_CUST} customers; "
      f"bad-rate={lab['target'].mean():.2%}")
