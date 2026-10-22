# Adaptive Enterprise Intelligence System for SME Resource Management

A working prototype of a decision-support system for a small business. It forecasts demand, plans inventory and
staffing from those forecasts, flags unusual data, answers questions in plain English, and **adapts**: when new data
arrives it checks whether each product's model has gone stale and retrains only the ones that need it. A built-in
experiment measures, in rupees, what that adaptation is worth.

## Run it

```bash
pip install -r requirements.txt
python app.py
# open http://127.0.0.1:5000   (first start trains models, ~15 seconds)
```

Works offline. No database, no external services, no API keys.
Tests: `python -m unittest discover -s tests -v` (24 tests, about 2 minutes).
Re-run the experiments (results are shipped precomputed): `python -m engine.experiment --scenario surge` and
`--scenario normal` (about 3 minutes each).

## The tabs

| Tab | What it shows |
|---|---|
| Overview | KPIs, 14-day demand forecast, auto-generated alerts |
| Demand Forecast | Per-product forecast with 80% interval, flagged anomalies, model comparison |
| Inventory | Safety stock, reorder point, EOQ, days of cover, stock-out risk, and a **Why?** explanation per product |
| Workforce | Staff needed per day from forecast volume, cost versus a fixed roster |
| Anomalies | Unusual sales, each cross-checked by two independent detection methods |
| Adaptive Engine | Feed new data (simulate, upload CSV), see which models were kept or retrained and why, model registry, learning log |
| Proof: Adaptive vs Static | 180-day replay of two identical shops, one retraining and one frozen, compared in rupees |
| What-if | Sliders for demand change and supplier delay, with instant effect on stock and staffing |
| Assistant | Ask questions in plain English; every number comes from the live engines |

## 5-minute demo script

1. **Overview**: KPIs and alerts. Two products already need reordering.
2. **Inventory**: click **Why?** on a REORDER NOW row to show the explanation. Change service level to 99%.
3. **Adaptive Engine**: click **Simulate 30 days: demand surge (+45%)**. Every model retrains, some switch
   algorithm, and the learning log records each decision. Overview and Inventory update by themselves.
4. **Proof: Adaptive vs Static**: the main result. Show the surge scenario, then **Normal demand** to show the honest
   picture (smaller and uneven gains).
5. **What-if**: click **Supplier strike (+4 days)** and watch products flip to REORDER NOW.
6. **Assistant**: click the suggested questions.
7. Optional: upload `data/sample_upload.csv` (one bad row, one brand-new product). **Reset demo** to start over.

## How it works

| Layer | File | What it does |
|---|---|---|
| Data | `generate_data.py` | Synthetic retail data: weekly and yearly seasonality, trend, promotions, injected anomalies |
| Forecasting | `engine/forecast.py` | Four competing models per product, walk-forward backtesting, model registry, drift detection, retraining |
| Decisions | `engine/operations.py` | Inventory with explanations and what-if parameters, workforce sizing, two-method anomaly detection |
| Assistant | `engine/assistant.py` | Rule-based question answering over live engine outputs |
| Experiment | `engine/experiment.py` | Replays 180 days with a frozen versus an adaptive forecaster and prices the difference |
| Interface | `app.py`, `static/index.html` | Flask API and a single-page dashboard with hand-drawn charts |

**Candidate models** (written from scratch except gradient boosting): seasonal naive, moving average with a weekday
profile, additive Holt-Winters with weekly seasonality, and gradient boosting on calendar and lag features. Each is
scored by walk-forward validation (3 folds of 14 days) using WAPE, and the best is chosen per product.

**Adaptation rule.** New data is tested against the model as it was trained. A product is retrained if:
- *Error rise*: live WAPE above max(0.24, 1.6 x training WAPE)
- *Bias*: systematic over- or under-forecast beyond max(15%, the model's own training error), which signals a level shift
- *Staleness*: more than 42 days of unseen data

**Inventory.** Safety stock = z x sigma x sqrt(lead time), with sigma from backtest residuals. Reorder point = demand
over lead time + safety stock. EOQ uses a 22% holding rate and a flat order cost. Stock-out risk is the probability that
lead-time demand exceeds stock on hand.

**Anomalies.** A rolling median/MAD z-score above 4 is the primary detector. An Isolation Forest provides an
independent second opinion; points flagged by both are marked "Both methods". Flagged points are also winsorised before
training.

**Assistant.** Transparent keyword and product-name matching. It can only restate numbers the engines produced.
A language model could later be added to phrase answers, but is not part of this prototype.

## Results (measured, not estimated)

**Forecast accuracy**: about 87% on held-out backtests; every chosen model beats seasonal naive.

**Drift detector**, 6 random seeds x 6 products = 36 cases per scenario:

| Scenario | Models retrained |
|---|---|
| Normal demand, 20 new days | 1 of 36 (2.8% false alarms) |
| Normal demand, 30 new days | 0 of 36 |
| Demand surge +30%, 30 days | 33 of 36 (92% detected) |
| Demand surge +50%, 30 days | 36 of 36 (100% detected) |

**Adaptive versus static replay**, 180 days, 6 products, identical demand and ordering policy (one run each):

| | Adaptive | Static (never retrained) |
|---|---|---|
| **Demand surge** (+45% from day 60): total cost | ₹1,45,637 | ₹4,61,829 |
| Fill rate | 96.3% | 84.6% |
| Stock-out days | 66 | 220 |
| **Normal demand**: total cost | ₹84,177 | ₹1,13,321 |
| Fill rate | 98.6% | 97.8% |

Read these carefully. Under the surge, adaptation lowered cost on all 6 products, a 68% reduction. Under normal demand
the gain is smaller and uneven: adaptive won on 2 products (Basmati Rice, LED Bulb) and cost ₹700 to ₹1,300 more on the
other 4. Of 21 retrains in the normal run, 17 were scheduled 6-week refreshes and only 4 were drift triggers.

**Tests**: 24 automated tests covering forecasting, adaptation, inventory, what-if maths, explanations, both anomaly
detectors, the assistant, the experiment, and a regression for products with misaligned data dates.

## Limitations (say these before your examiner does)

- The data is **synthetic**, and the experiment is a **single simulated run per scenario**. The rupee figures show the
  mechanism works under stated assumptions; they are not a prediction for a real shop. Real data would be the strongest
  upgrade.
- "Static" means models frozen after day 0. A real business might refit simple models occasionally, which would narrow
  the gap in the normal-demand case.
- Drift thresholds were tuned on this synthetic data and need recalibrating on real data. Weekly checks on only 7 new
  days are noisier than the 20 to 30 day windows used for the detector table above.
- The assistant is rule-based, not a language model. It understands the example question styles, not arbitrary text.
- Not included: pretrained or deep-learning forecasters, reinforcement learning, shift-scheduling optimisation.
- Single-user prototype on Flask's development server with in-memory state, not a production deployment.

## Likely viva questions

- **What makes it adaptive?** A closed loop: monitor live error and bias against the training baseline, then retrain and
  re-select models automatically. The learning log records every decision.
- **How do you know adaptation helps?** The Proof tab: identical shops, same demand, same policy, only forecasting differs.
  It helps a lot under a demand shock and modestly, unevenly under stable demand.
- **Why not one good model?** Products behave differently, and no single model wins everywhere in the backtests.
- **Why WAPE instead of MAPE?** Daily sales for small items approach zero, which makes MAPE explode.
- **How is overfitting avoided?** Selection uses walk-forward validation, never in-sample error.
- **Is the assistant AI?** It is rule-based over trusted engine outputs, deliberately, so it cannot invent numbers.
  Adding a language model for phrasing is a clear next step.
- **What would you do next?** Real client data, a pretrained forecasting model in the model competition, shift
  optimisation, a database, and authentication.

## Using your own data

CSV columns: `date, product, units_sold` (optional: `price, unit_cost, lead_time_days, promo`). Starting stock lives
in `data/stock.csv` or the Inventory tab. For a fresh synthetic dataset, delete `data/sales.csv` and run
`python generate_data.py`.
