"""Baseline experiment: does adapting actually pay off?

Replays 180 days of business with two identical shops that differ only in forecasting:
  * STATIC   - models trained once on the first 600 days and never touched again
  * ADAPTIVE - this system: drift monitoring, retraining and model re-selection
Both see exactly the same real demand and follow the same ordering policy (periodic review,
order up to forecast demand over lead time + review period + safety stock). Money outcomes are
compared: lost margin from stock-outs, holding cost and ordering cost.

Run:  python -m engine.experiment --scenario surge --out data/experiment_surge.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import generate_data as gd
from engine.forecast import AdaptiveForecaster, MODEL_POOL

HOLD_RATE = 0.22
ORDER_COST = 250.0
SERVICE = 0.95


def run_experiment(scenario="surge", future_days=180, review_every=7, seed=500, products=None,
                   surge_day=60, history_days=600, progress=print):
    full = gd.generate(start="2024-01-01", days=640, seed=42)
    t_start = full["date"].min()
    hist = full[full["date"] < t_start + pd.Timedelta(days=history_days)].copy()
    if products:
        hist = hist[hist["product"].isin(products)]
    prods = sorted(hist["product"].unique())
    start = hist["date"].max() + pd.Timedelta(days=1)
    fut = gd.generate(start=str(start.date()), days=future_days, seed=seed, t0=history_days,
                      shift_after=surge_day if scenario == "surge" else None)  # shift_after applies a built-in +45% surge
    fut = fut[fut["product"].isin(prods)]
    meta = hist.groupby("product").agg(price=("price", "last"), cost=("unit_cost", "last"), lead=("lead_time_days", "last"))
    demand = {p: fut[fut["product"] == p].sort_values("date")["units_sold"].values.astype(float) for p in prods}

    # ---- models at day 0
    af = AdaptiveForecaster()
    af.train_all(hist, reason="experiment: initial training")
    static_pred, static_sd, adapt_sd = {}, {}, {}
    for p in prods:
        rec = af.registry.current[p]
        cls = MODEL_POOL[rec["model"]]
        s = af.clean(af.series(hist, p))
        static_pred[p] = cls().fit(s).predict(future_days + 14)
        res = af._residuals(s, cls)
        static_sd[p] = float(np.std(res)) if res else float(s.tail(28).std())
        adapt_sd[p] = static_sd[p]
    z = norm.ppf(SERVICE)
    horizon_needed = lambda p: int(meta.loc[p, "lead"]) + review_every

    def new_state(p):
        base = hist[hist["product"] == p].sort_values("date")["units_sold"].tail(28).mean()
        return dict(on_hand=base * 9, pipeline=[], lost_units=0.0, lost_margin=0.0, hold=0.0, order=0.0,
                    orders=0, stockout_days=0, demand=0.0)

    pol = {"static": {p: new_state(p) for p in prods}, "adaptive": {p: new_state(p) for p in prods}}
    series = {k: [] for k in ("date", "adaptive_cost", "static_cost", "adaptive_lost", "static_lost",
                              "adaptive_fill", "static_fill")}
    wk = {"static": dict(lost=0.0, dem=0.0), "adaptive": dict(lost=0.0, dem=0.0)}
    retrain_events = []
    df_known = hist.copy()

    def totals(name):
        st = pol[name]
        lost_m = sum(v["lost_margin"] for v in st.values())
        hold = sum(v["hold"] for v in st.values())
        order = sum(v["order"] for v in st.values())
        return lost_m, hold, order

    t0 = time.time()
    for d in range(future_days):
        day = start + pd.Timedelta(days=d)
        if d % review_every == 0:
            if d > 0:
                df_known = pd.concat([hist, fut[fut["date"] < day]])
                rep = af.update(df_known)
                for r in rep.itertuples():
                    if r.action == "retrained":
                        retrain_events.append(dict(day=d, date=str(day.date()), product=r.product, reason=r.reason, model=r.model))
                        s = af.clean(af.series(df_known, r.product))
                        res = af._residuals(s, MODEL_POOL[r.model])
                        adapt_sd[r.product] = float(np.std(res)) if res else float(s.tail(28).std())
            # place orders for both policies
            for p in prods:
                L = int(meta.loc[p, "lead"])
                H = L + review_every
                s = af.clean(af.series(df_known, p))
                rec = af.registry.current[p]
                a_fc = MODEL_POOL[rec["model"]]().fit(s).predict(14)
                s_fc = static_pred[p][d:d + 14]
                for name, fc, sd in (("adaptive", a_fc, adapt_sd[p]), ("static", s_fc, static_sd[p])):
                    st = pol[name][p]
                    target = float(np.sum(fc[:H])) + z * sd * math.sqrt(H)
                    position = st["on_hand"] + sum(q for _, q in st["pipeline"])
                    if position < target:
                        qty = math.ceil(target - position)
                        st["pipeline"].append((d + L, qty))
                        st["order"] += ORDER_COST
                        st["orders"] += 1
        # day mechanics
        for name in ("adaptive", "static"):
            for p in prods:
                st = pol[name][p]
                arrived = [q for (a, q) in st["pipeline"] if a <= d]
                st["pipeline"] = [(a, q) for (a, q) in st["pipeline"] if a > d]
                st["on_hand"] += sum(arrived)
                dem = demand[p][d]
                sold = min(st["on_hand"], dem)
                lost = dem - sold
                st["on_hand"] -= sold
                st["demand"] += dem
                if lost > 0:
                    st["lost_units"] += lost
                    st["lost_margin"] += lost * float(meta.loc[p, "price"] - meta.loc[p, "cost"])
                    st["stockout_days"] += 1
                st["hold"] += st["on_hand"] * float(meta.loc[p, "cost"]) * HOLD_RATE / 365
                wk[name]["lost"] += lost
                wk[name]["dem"] += dem
        if d % 7 == 6:
            series["date"].append(str(day.date()))
            for name in ("adaptive", "static"):
                lm, hold, order = totals(name)
                series[f"{name}_cost"].append(round(lm + hold + order))
                series[f"{name}_lost"].append(round(lm))
                series[f"{name}_fill"].append(round(1 - wk[name]["lost"] / max(wk[name]["dem"], 1), 4))
                wk[name] = dict(lost=0.0, dem=0.0)
        if progress and d % 28 == 27:
            progress(f"  day {d + 1}/{future_days}  ({time.time() - t0:.0f}s)")

    summary, per_product = {}, {}
    for name in ("adaptive", "static"):
        st = pol[name]
        lm, hold, order = totals(name)
        dem = sum(v["demand"] for v in st.values())
        lost_u = sum(v["lost_units"] for v in st.values())
        summary[name] = dict(fill_rate=1 - lost_u / dem, lost_units=lost_u, lost_margin=lm, holding_cost=hold,
                             ordering_cost=order, total_cost=lm + hold + order,
                             stockout_days=sum(v["stockout_days"] for v in st.values()),
                             orders=sum(v["orders"] for v in st.values()))
        per_product[name] = {p: dict(fill_rate=1 - v["lost_units"] / max(v["demand"], 1), lost_margin=v["lost_margin"],
                                     total_cost=v["lost_margin"] + v["hold"] + v["order"]) for p, v in st.items()}
    return dict(scenario=scenario, future_days=future_days, review_every=review_every, surge_day=surge_day if scenario == "surge" else None,
                surge_mult=1.45 if scenario == "surge" else None, products=prods, summary=summary,
                per_product=per_product, series=series, retrain_events=retrain_events,
                generated=pd.Timestamp.now().isoformat(timespec="seconds"),
                assumptions=dict(service_level=SERVICE, holding_rate=HOLD_RATE, order_cost=ORDER_COST,
                                 policy="periodic review every %d days, order up to forecast demand over lead time + review period + safety stock" % review_every,
                                 lost_sales="unmet demand is lost (valued at price minus unit cost)"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["surge", "normal"], default="surge")
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--out", default=None)
    ap.add_argument("--products", nargs="*", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join("data", f"experiment_{a.scenario}.json")
    print(f"Running '{a.scenario}' experiment for {a.days} days...")
    res = run_experiment(a.scenario, future_days=a.days, products=a.products)
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    s = res["summary"]
    print(f"adaptive: fill {s['adaptive']['fill_rate']:.3f}  cost Rs{s['adaptive']['total_cost']:,.0f}")
    print(f"static:   fill {s['static']['fill_rate']:.3f}  cost Rs{s['static']['total_cost']:,.0f}")
    print("saved", out)
