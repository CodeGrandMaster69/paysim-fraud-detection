# Mobile Payments Fraud Detection (PaySim)

**Business question:** Which mobile-money transactions are likely fraud, and how should a risk team set its alert threshold?

This project builds a fraud model on heavily imbalanced data (0.13% fraud) and turns it into a business decision: **how many transactions should the team review to catch most of the fraud?**

> **Note:** PaySim is a **simulated** dataset generated from real mobile-money logs from an African country. It contains no real customers. Results on real data would be lower.

## Data
[PaySim synthetic financial dataset (Kaggle)](https://www.kaggle.com/datasets/ealaxi/paysim1): 6.36M simulated transactions, about 8,200 fraud. Columns: `step` (1 step = 1 hour, 743 steps ≈ 31 days), `type`, `amount`, origin and destination balances before and after, `isFraud`, `isFlaggedFraud`.

Download `PS_20174392719_1491204439457_log.csv` into `data/`. The file is about 470 MB. The notebook loads it with explicit `dtype`s to keep memory down. On a weak laptop, use Google Colab.

## Approach
| Step | What | Why |
|---|---|---|
| 1. Explore | Fraud count and amount by `type` | Fraud occurs **only in TRANSFER and CASH_OUT**, so we filter to those |
| 2. Leakage check | Drop `isFlaggedFraud`, `nameOrig`, `nameDest` | The flag is the simulator's own rule (an output, not a signal). The names are near-unique IDs, so the model could only memorise them |
| 3. Features | `errorOrig = oldbalanceOrg − amount − newbalanceOrig`, `errorDest`, `hour = step % 24`, zero-balance flags | Balance bookkeeping gaps separate fraud very well |
| 4. Split | **By time**: train on the first ~80% of rows, test on later steps | This mirrors production (train on the past, score the future) and avoids leaking future patterns |
| 5. Baseline | Logistic regression, `class_weight='balanced'` | Simple, interpretable reference |
| 6. Main model | Random forest, `n_estimators=200`, `class_weight='balanced'` | Captures non-linear interactions between balances |
| 7. Evaluate | Precision, recall, F1, PR curve, average precision | Accuracy is meaningless: "never fraud" scores 99.87% |
| 8. Threshold | Share of fraud **value** caught when reviewing the top 0.5 / 1 / 2 / 5% | Turns scores into an analyst-capacity decision |
| 9. Explain | Impurity and permutation feature importance | Which signals drive the alerts? |

## Results
Full Kaggle file. Time-based split: train = steps 1–354 (2.22M TRANSFER/CASH_OUT transactions, 3,955 fraud), test = steps 355–743 (552,504 transactions, 4,258 fraud). Because legitimate volume falls off later in the simulation, the test fraud rate (0.77%) is about 4× the train rate (0.18%).

| Model | Average precision (test) |
|---|---|
| Logistic regression (baseline) | 0.984 |
| Random forest (all features) | **1.000** |
| Random forest, *pre-transaction features only* (sanity check) | 0.908 |

**Review-budget table: random forest, time-based test set**

| Review top | Transactions reviewed | Fraud value caught | Fraud cases caught | Precision of alerts |
|---|---|---|---|---|
| 0.5% | 2,763 | 79.1% | 64.9%* | 100.0% |
| **1%** | **5,526** | **100.0%** | **100.0%** | **77.1%** |
| 2% | 11,051 | 100.0% | 100.0% | 38.5% |
| 5% | 27,626 | 100.0% | 100.0% | 15.4% |

\*At 0.5% the team reviews 2,763 transactions, but the test set has 4,258 fraud cases, so 64.9% is the most any model could catch. Every alert at this budget was fraud.

**Chosen operating point:** review the top 1% (alert when score ≥ 0.011). This catches all 4,258 fraud cases (6.7B in simulated value) with 1,268 false alerts.

**Top signals:** `errorOrig` (dropping it costs 0.71 average precision) and `origEmptied` (0.16). Fraud in PaySim drains the origin account *exactly*, so the sender's balance reconciles perfectly, unlike legitimate transactions.

### Is a perfect score believable? (sanity check)
No, not for real-world use. `errorOrig == 0` is largely a **simulation artifact**: the simulator writes fraud as `amount == oldbalanceOrg` and `newbalanceOrig == 0`. To get an honest estimate, I retrained using only information known *before* the transaction posts (amount, opening balances, hour, type):

| Review top | Fraud value caught | Fraud cases caught | Precision |
|---|---|---|---|
| 0.5% | 94.9% | 63.9% | 98.5% |
| 1% | 99.0% | 88.1% | 67.9% |
| 2% | 99.6% | 94.6% | 36.5% |
| 5% | 100.0% | 99.0% | 15.3% |

Even without post-transaction balances, reviewing 1% of transactions catches 99% of fraud value. Large, account-draining transfers are what this simulation flags. Real-world performance would be lower.

**Resume bullet:**
> Built a fraud-detection model on 6.36M simulated mobile-money transactions (0.13% fraud) with a time-based train/test split; a random forest with engineered balance-error features lets a risk team **catch 100% of fraud value by reviewing just the top 1% of transactions** (PR-AUC 1.00 vs 0.98 logistic baseline). A leakage-aware sanity check using only pre-transaction data still caught 99% of fraud value at the same review budget.

### Charts
| | |
|---|---|
| ![](figures/01_fraud_by_type.png) | ![](figures/03_precision_recall.png) |
| ![](figures/04_fraud_value_vs_review.png) | ![](figures/05_confusion_matrix.png) |
| ![](figures/06_feature_importance.png) | ![](figures/02_feature_separation.png) |

## How to run
```bash
pip install -r requirements.txt
# put the Kaggle CSV in data/, then:
jupyter notebook notebooks/paysim_fraud_detection.ipynb
```
If the Kaggle file is missing, the notebook falls back to `data/paysim_sample.csv`, a small synthetic file made by `python scripts/make_sample_data.py`. The sample is only for checking that the code runs. Its numbers are meaningless.

Outputs: charts go to `figures/` and tables to `results/` (`review_budget_table.csv`, `feature_importance.csv`).

## Limitations
- Simulated data: the fraud logic is cleaner than real life. The perfect score depends on `errorOrig`, which is largely a simulation artifact (see the sanity check).
- A single time-based split. A production setup would use rolling (walk-forward) validation and monitor drift.
- No cost model: the threshold is set by review capacity. With a cost per review and an average loss per missed fraud, you could optimise expected cost directly.

See [INTERVIEW_PREP.md](INTERVIEW_PREP.md) for discussion questions.
