"""Generate a small PaySim-shaped CSV so the notebook can be smoke-tested
without the 470 MB Kaggle file. Results on this sample are NOT meaningful.

Usage: python scripts/make_sample_data.py  ->  data/paysim_sample.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 300_000
FRAUD_PAIRS = 250  # each fraud = one TRANSFER + one CASH_OUT

# Volume front-loaded in time like the real data; steps 1..743
step = np.sort(rng.choice(np.arange(1, 744), size=N, p=np.r_[np.full(400, 3.0), np.full(343, 0.5)] / (400 * 3 + 343 * 0.5)))
types = rng.choice(["CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER", "DEBIT"], size=N, p=[0.35, 0.34, 0.22, 0.08, 0.01])
amount = np.round(rng.lognormal(10.5, 1.3, N), 2)
old_org = np.round(rng.lognormal(10, 2, N) * (rng.random(N) > 0.3), 2)
sign = np.where(types == "CASH_IN", 1, -1)
new_org = np.clip(old_org + sign * amount, 0, None)
old_dest = np.round(rng.lognormal(12, 2, N) * (rng.random(N) > 0.35), 2)
new_dest = np.where(types == "PAYMENT", 0.0, old_dest - sign * amount)
new_dest = np.clip(new_dest, 0, None)
# real data has messy balance bookkeeping on legit rows too
noise = rng.random(N) < 0.15
new_dest[noise] = old_dest[noise]

df = pd.DataFrame({
    "step": step, "type": types, "amount": amount,
    "nameOrig": [f"C{x}" for x in rng.integers(1e8, 2e9, N)],
    "oldbalanceOrg": old_org, "newbalanceOrig": new_org,
    "nameDest": [("M" if t == "PAYMENT" else "C") + str(x) for t, x in zip(types, rng.integers(1e8, 2e9, N))],
    "oldbalanceDest": old_dest, "newbalanceDest": new_dest,
    "isFraud": 0, "isFlaggedFraud": 0,
})

# Fraud: empty the victim account via TRANSFER, then CASH_OUT; spread uniformly over time
rows = []
for s in rng.integers(1, 744, FRAUD_PAIRS):
    bal = round(float(rng.lognormal(12.5, 1.2)), 2)
    amt = min(bal, 10_000_000)
    for t in ("TRANSFER", "CASH_OUT"):
        od = 0.0 if t == "TRANSFER" else round(float(rng.lognormal(11, 2)), 2)
        rows.append({
            "step": s, "type": t, "amount": amt,
            "nameOrig": f"C{rng.integers(1e8, 2e9)}", "oldbalanceOrg": bal, "newbalanceOrig": 0.0,
            "nameDest": f"C{rng.integers(1e8, 2e9)}", "oldbalanceDest": od,
            "newbalanceDest": 0.0 if rng.random() < 0.5 else od + amt,
            "isFraud": 1, "isFlaggedFraud": int(t == "TRANSFER" and amt > 1_000_000 and rng.random() < 0.05),
        })
df = pd.concat([df, pd.DataFrame(rows)]).sort_values("step", kind="stable")

out = Path(__file__).resolve().parents[1] / "data" / "paysim_sample.csv"
df.to_csv(out, index=False)
print(f"wrote {len(df):,} rows ({df.isFraud.sum()} fraud) to {out}")
