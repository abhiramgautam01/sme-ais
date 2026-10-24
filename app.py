"""Adaptive Enterprise Intelligence System for SME Resource Management - web app.

Run:  python app.py      then open http://127.0.0.1:5000
"""
from __future__ import annotations

import io
import json
import math
import os
import threading

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

from engine.forecast import AdaptiveForecaster, ModelRegistry, HORIZON
from engine.operations import inventory_plan, workforce_plan, roster_cost, detect_anomalies
from engine import assistant as asst
import generate_data as gd

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
INITIAL_DAYS = 600

app = Flask(__name__, static_folder=os.path.join(ROOT, "static"))
LOCK = threading.Lock()
S: dict = {}          # application state


# ------------------------------------------------------------------ helpers
def clean_json(o):
    if isinstance(o, dict):
        return {k: clean_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean_json(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if (math.isnan(o) or math.isinf(o)) else float(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o.date())
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def reply(o, code=200):
    return jsonify(clean_json(o)), code


def load_initial():
    if not os.path.exists(os.path.join(DATA, "sales.csv")):
        df = gd.generate()
        df.to_csv(os.path.join(DATA, "sales.csv"), index=False)
        gd.initial_stock(df).to_csv(os.path.join(DATA, "stock.csv"), index=False)
    full = pd.read_csv(os.path.join(DATA, "sales.csv"), parse_dates=["date"])
    cutoff = full["date"].min() + pd.Timedelta(days=INITIAL_DAYS)
    df = full[full["date"] < cutoff].reset_index(drop=True)
    stock = pd.read_csv(os.path.join(DATA, "stock.csv"))
    return df, stock


def refresh_forecasts():
    af: AdaptiveForecaster = S["af"]
    fc, sd = {}, {}
    for p in sorted(S["df"]["product"].unique()):
        fc[p], sd[p] = af.forecast(S["df"], p, until=S["df"]["date"].max())
    S["fc"], S["sd"] = fc, sd
    S["anoms"] = detect_anomalies(S["df"][S["df"]["date"] >= S["df"]["date"].max() - pd.Timedelta(days=150)].copy())


def reset_state():
    df, stock = load_initial()
    S.clear()
    S.update(df=df, stock=stock, af=AdaptiveForecaster(), last_report=None, sim_count=0)
    S["af"].train_all(df, reason="initial training")
    refresh_forecasts()
    save_registry()


def save_registry():
    with open(os.path.join(DATA, "registry.json"), "w") as f:
        f.write(S["af"].registry.to_json())


def apply_new_data(new: pd.DataFrame, source: str):
    """Merge new rows (dedupe on date+product), run the adaptive update, refresh outputs."""
    new = new.copy()
    new["date"] = pd.to_datetime(new["date"])
    meta = S["df"].groupby("product").last()[["price", "unit_cost", "lead_time_days"]]
    for col, default in (("price", None), ("unit_cost", None), ("lead_time_days", None), ("promo", 0)):
        if col not in new.columns:
            new[col] = new["product"].map(meta[col]) if default is None and col in meta else default
    new = new.fillna({"price": 100, "unit_cost": 70, "lead_time_days": 5, "promo": 0})
    cols = ["date", "product", "units_sold", "price", "unit_cost", "lead_time_days", "promo"]
    merged = pd.concat([S["df"], new[cols]]).drop_duplicates(["date", "product"], keep="last")
    S["df"] = merged.sort_values(["product", "date"]).reset_index(drop=True)
    for p in new["product"].unique():
        if p not in set(S["stock"]["product"]):
            S["stock"] = pd.concat([S["stock"], pd.DataFrame({"product": [p], "on_hand": [int(new[new["product"] == p]["units_sold"].mean() * 7)]})])
    report = S["af"].update(S["df"])
    S["last_report"] = dict(source=source, rows=len(new), report=report.to_dict("records"),
                            through=str(S["df"]["date"].max().date()))
    refresh_forecasts()
    save_registry()
    return S["last_report"]


# ------------------------------------------------------------------ pages
@app.route("/favicon.ico")
def favicon():
    return ("", 204)


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


# ------------------------------------------------------------------ API
@app.route("/api/overview")
def api_overview():
    with LOCK:
        meta = S["df"].groupby("product").agg(price=("price", "last"))
        inv = inventory_plan(S["df"], S["fc"], S["sd"], S["stock"])
        wf = workforce_plan(S["fc"], S["df"])
        rc = roster_cost(wf)
        rev = sum(float(f["forecast"].sum() * meta.loc[p, "price"]) for p, f in S["fc"].items())
        units = sum(float(f["forecast"].sum()) for f in S["fc"].values())
        accs = [1 - r["backtest_wape"] for r in S["af"].registry.current.values() if r["backtest_wape"] is not None and not math.isnan(r["backtest_wape"])]
        acc = float(np.mean(accs)) if accs else None
        total_hist = S["df"].groupby("date")["units_sold"].sum().tail(60)
        total_fc = None
        for f in S["fc"].values():
            x = f.set_index("date")["forecast"]
            total_fc = x if total_fc is None else total_fc.add(x, fill_value=0)
        alerts = []
        for _, r in inv.iterrows():
            if r["status"] == "REORDER NOW":
                alerts.append(dict(level="high", text=f"{r['product']}: {r['days_of_cover']} days of cover, below reorder point ({r['reorder_point']}). {r['action']}."))
            elif r["status"] == "WATCH":
                alerts.append(dict(level="mid", text=f"{r['product']}: stock-out risk {r['stockout_risk']*100:.0f}% within lead time."))
        an = S["anoms"]
        if len(an):
            recent = an[pd.to_datetime(an["date"]) >= S["df"]["date"].max() - pd.Timedelta(days=14)]
            for _, r in recent.head(3).iterrows():
                alerts.append(dict(level="mid", text=f"{r['product']}: unusual {r['type'].lower()} on {str(r['date'])[:10]} ({int(r['units_sold'])} vs ~{r['expected']:.0f} expected)."))
        last = S["last_report"]
        if last:
            n = sum(1 for r in last["report"] if r["action"] == "retrained")
            alerts.append(dict(level="info", text=f"Adaptive engine: last update ({last['source']}) retrained {n} of {len(last['report'])} product models."))
        return reply(dict(
            kpis=dict(units=units, revenue=rev, reorder=int((inv["status"] == "REORDER NOW").sum()),
                      watch=int((inv["status"] == "WATCH").sum()), capital=int(inv["capital_tied"].sum()),
                      accuracy=acc, saving=rc["saving"], data_through=str(S["df"]["date"].max().date()),
                      rows=len(S["df"]), products=int(S["df"]["product"].nunique())),
            history=[dict(date=str(d.date()), v=float(v)) for d, v in total_hist.items()],
            forecast=[dict(date=str(d.date()), v=float(v)) for d, v in total_fc.items()],
            alerts=alerts))


@app.route("/api/products")
def api_products():
    return reply(sorted(S["df"]["product"].unique()))


@app.route("/api/forecast")
def api_forecast():
    p = request.args.get("product")
    with LOCK:
        if p not in S["fc"]:
            return reply({"error": "unknown product"}, 404)
        s = AdaptiveForecaster.series(S["df"], p).tail(90)
        raw = S["df"][S["df"]["product"] == p].set_index("date")["units_sold"]
        an = S["anoms"]
        an_dates = set(str(d)[:10] for d in an[an["product"] == p]["date"]) if len(an) else set()
        rec = S["af"].registry.current[p]
        return reply(dict(
            product=p, model=rec["model"], wape=rec["backtest_wape"], scores=rec["scores"],
            trained_through=rec["trained_through"], sd=S["sd"][p],
            history=[dict(date=str(d.date()), v=float(v), anomaly=str(d.date()) in an_dates) for d, v in s.items()],
            forecast=S["fc"][p].assign(date=lambda x: x["date"].dt.strftime("%Y-%m-%d")).to_dict("records")))


@app.route("/api/inventory")
def api_inventory():
    sl = float(request.args.get("sl", 0.95))
    with LOCK:
        inv = inventory_plan(S["df"], S["fc"], S["sd"], S["stock"], service_level=sl)
        return reply(inv.to_dict("records"))


@app.route("/api/stock", methods=["POST"])
def api_stock():
    j = request.get_json(force=True)
    with LOCK:
        m = S["stock"]["product"] == j["product"]
        if not m.any():
            return reply({"error": "unknown product"}, 404)
        S["stock"].loc[m, "on_hand"] = max(0, int(j["on_hand"]))
    return reply({"ok": True})


@app.route("/api/workforce")
def api_workforce():
    cap = float(request.args.get("cap", 75))
    staff = int(request.args.get("staff", 5))
    wage = float(request.args.get("wage", 700))
    with LOCK:
        wf = workforce_plan(S["fc"], S["df"], units_per_staff_day=cap)
        rc = roster_cost(wf, daily_wage=wage, current_staff=staff)
        wf["date"] = wf["date"].dt.strftime("%Y-%m-%d")
        return reply(dict(plan=wf.to_dict("records"), cost=rc))


@app.route("/api/anomalies")
def api_anomalies():
    with LOCK:
        a = S["anoms"].copy()
        if len(a):
            a["date"] = pd.to_datetime(a["date"]).dt.strftime("%Y-%m-%d")
        return reply(a.head(60).to_dict("records") if len(a) else [])


@app.route("/api/adaptive")
def api_adaptive():
    with LOCK:
        reg = S["af"].registry
        return reply(dict(models=list(reg.current.values()), history=reg.history[-60:][::-1],
                          last_report=S["last_report"], data_through=str(S["df"]["date"].max().date())))


@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    """Append N days of new data. mode='normal' (business as usual) or 'shift' (+45% demand shock)."""
    j = request.get_json(force=True) or {}
    mode = j.get("mode", "normal")
    days = int(j.get("days", 30))
    with LOCK:
        start = S["df"]["date"].max() + pd.Timedelta(days=1)
        t0 = (start - pd.Timestamp("2024-01-01")).days
        S["sim_count"] += 1
        new = gd.generate(start=str(start.date()), days=days, seed=1000 + S["sim_count"], t0=t0,
                          level_mult=1.45 if mode == "shift" else 1.0)
        rep = apply_new_data(new, f"simulated {days} days, {'demand shift +45%' if mode == 'shift' else 'normal demand'}")
    return reply(rep)


@app.route("/api/upload", methods=["POST"])
def api_upload():
    f = request.files.get("file")
    if not f:
        return reply({"error": "No file uploaded"}, 400)
    try:
        new = pd.read_csv(io.BytesIO(f.read()))
    except Exception as e:  # noqa: BLE001
        return reply({"error": f"Could not read CSV: {e}"}, 400)
    need = {"date", "product", "units_sold"}
    if not need.issubset(new.columns):
        return reply({"error": f"CSV must contain columns: {sorted(need)}. Found: {list(new.columns)}"}, 400)
    new["date"] = pd.to_datetime(new["date"], errors="coerce")
    new["units_sold"] = pd.to_numeric(new["units_sold"], errors="coerce")
    bad = int(new[["date", "units_sold"]].isna().any(axis=1).sum())
    new = new.dropna(subset=["date", "units_sold"])
    new = new[new["units_sold"] >= 0]
    if new.empty:
        return reply({"error": "No valid rows after cleaning"}, 400)
    with LOCK:
        rep = apply_new_data(new, f"uploaded file '{f.filename}'")
    rep["skipped_rows"] = bad
    return reply(rep)


@app.route("/api/retrain", methods=["POST"])
def api_retrain():
    with LOCK:
        report = S["af"].update(S["df"], force=True)
        S["last_report"] = dict(source="manual full retrain", rows=0, report=report.to_dict("records"),
                                through=str(S["df"]["date"].max().date()))
        refresh_forecasts()
        save_registry()
    return reply(S["last_report"])


@app.route("/api/reset", methods=["POST"])
def api_reset():
    with LOCK:
        reset_state()
    return reply({"ok": True})



def assistant_ctx():
    inv = inventory_plan(S["df"], S["fc"], S["sd"], S["stock"])
    wf = workforce_plan(S["fc"], S["df"])
    meta = S["df"].groupby("product").agg(price=("price", "last"))
    revenue = sum(float(f["forecast"].sum() * meta.loc[p, "price"]) for p, f in S["fc"].items())
    units = sum(float(f["forecast"].sum()) for f in S["fc"].values())
    accs = [1 - r["backtest_wape"] for r in S["af"].registry.current.values()
            if r["backtest_wape"] is not None and not math.isnan(r["backtest_wape"])]
    return dict(inv=inv, fc=S["fc"], af=S["af"], df=S["df"], wf=wf, cost=roster_cost(wf), anoms=S["anoms"],
                last_report=S["last_report"], revenue=revenue, units=units,
                accuracy=float(np.mean(accs)) if accs else 0.0, through=str(S["df"]["date"].max().date()))


@app.route("/api/ask", methods=["POST"])
def api_ask():
    q = (request.get_json(force=True) or {}).get("question", "")
    with LOCK:
        return reply(dict(answer=asst.answer(q, assistant_ctx()), examples=asst.EXAMPLES))


@app.route("/api/whatif")
def api_whatif():
    """Compare today's plan with a scenario: demand change (%) and extra supplier lead time (days)."""
    dm = 1 + float(request.args.get("demand", 0)) / 100.0
    lead = int(request.args.get("lead", 0))
    sl = float(request.args.get("sl", 0.95))
    cap = float(request.args.get("cap", 75))
    staff = int(request.args.get("staff", 5))
    wage = float(request.args.get("wage", 700))
    with LOCK:
        base = inventory_plan(S["df"], S["fc"], S["sd"], S["stock"], service_level=sl)
        scen = inventory_plan(S["df"], S["fc"], S["sd"], S["stock"], service_level=sl, demand_mult=dm, lead_add=lead)
        rows = []
        for b, s in zip(base.itertuples(), scen.itertuples()):
            rows.append(dict(product=b.product, on_hand=b.on_hand, base_status=b.status, scen_status=s.status,
                             base_rop=b.reorder_point, scen_rop=s.reorder_point, base_cover=b.days_of_cover,
                             scen_cover=s.days_of_cover, scen_risk=s.stockout_risk, scen_action=s.action,
                             extra_units=max(0, int(s.reorder_point - b.reorder_point)),
                             changed=b.status != s.status))
        wf0 = workforce_plan(S["fc"], S["df"], units_per_staff_day=cap)
        wf1 = workforce_plan(S["fc"], S["df"], units_per_staff_day=cap, demand_mult=dm)
        c0, c1 = roster_cost(wf0, daily_wage=wage, current_staff=staff), roster_cost(wf1, daily_wage=wage, current_staff=staff)
        return reply(dict(rows=rows,
                          summary=dict(reorder_base=int((base["status"] == "REORDER NOW").sum()),
                                       reorder_scen=int((scen["status"] == "REORDER NOW").sum()),
                                       capital_base=int(base["capital_tied"].sum()),
                                       extra_safety_units=int((scen["safety_stock"] - base["safety_stock"]).sum()),
                                       staff_days_base=int(wf0["staff_needed"].sum()), staff_days_scen=int(wf1["staff_needed"].sum()),
                                       staff_cost_base=c0["adaptive_cost"], staff_cost_scen=c1["adaptive_cost"],
                                       short_base=c0["understaffed_days_units"], short_scen=c1["understaffed_days_units"])))


@app.route("/api/experiment")
def api_experiment():
    sc = request.args.get("scenario", "surge")
    path = os.path.join(DATA, f"experiment_{sc}.json")
    if sc not in ("surge", "normal") or not os.path.exists(path):
        return reply({"error": f"Experiment results not found. Run: python -m engine.experiment --scenario {sc}"}, 404)
    with open(path) as f:
        return jsonify(json.load(f))



# Initialize the application when the module is loaded.
# This is required when Render starts the app using Gunicorn.
with LOCK:
    reset_state()


if __name__ == "__main__":
    print("Ready -> http://127.0.0.1:5000")
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
