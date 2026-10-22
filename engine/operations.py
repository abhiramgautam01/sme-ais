"""Decision-support modules: inventory, workforce and anomaly detection."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import IsolationForest


# ------------------------------------------------------------- inventory
def inventory_plan(df_sales, forecasts: dict, sd: dict, stock: pd.DataFrame, service_level=0.95,
                   order_cost=250.0, holding_rate=0.22, demand_mult=1.0, lead_add=0):
    """Reorder point, safety stock, EOQ, days of cover and stock-out risk per product.

    forecasts:   product -> DataFrame(date, forecast)
    sd:          product -> daily forecast error std-dev
    stock:       DataFrame(product, on_hand)
    demand_mult: what-if multiplier applied to forecast demand (1.0 = as forecast)
    lead_add:    what-if extra days added to every supplier lead time
    Each row carries a plain-English `why` explaining the recommendation.
    """
    z = norm.ppf(service_level)
    rows = []
    meta = df_sales.groupby("product").agg(unit_cost=("unit_cost", "last"), price=("price", "last"),
                                           lead=("lead_time_days", "last"))
    for p, f in forecasts.items():
        lead = int(meta.loc[p, "lead"]) + int(lead_add)
        cost = float(meta.loc[p, "unit_cost"])
        fv = f["forecast"].values * demand_mult
        daily = float(fv.mean())
        sigma = sd.get(p, daily * 0.2) * demand_mult
        safety = z * sigma * np.sqrt(lead)
        rop = daily * lead + safety
        annual = daily * 365
        eoq = np.sqrt(2 * annual * order_cost / max(cost * holding_rate, 1e-9))
        on_hand = float(stock.loc[stock["product"] == p, "on_hand"].iloc[0])
        cum = fv.cumsum()
        if cum[-1] > on_hand:
            i = int(np.searchsorted(cum, on_hand))
            prior = cum[i - 1] if i > 0 else 0.0
            cover = i + (on_hand - prior) / max(fv[i], 1e-9)
        else:
            cover = len(cum) + (on_hand - cum[-1]) / max(daily, 1e-9)
        # stock-out risk within lead time: P(demand over lead > on_hand)
        mu_l = daily * lead
        sd_l = max(sigma * np.sqrt(lead), 1e-9)
        risk = float(1 - norm.cdf((on_hand - mu_l) / sd_l))
        recent = float(df_sales[df_sales["product"] == p].sort_values("date")["units_sold"].tail(28).mean())
        trend = daily / recent - 1 if recent > 0 else 0.0
        trend_txt = f" Forecast demand is {trend * 100:+.0f}% versus the last 4 weeks." if abs(trend) >= 0.10 else ""
        if on_hand <= rop:
            status = "REORDER NOW"
            qty = int(np.ceil(max(eoq, rop - on_hand + daily * 7)))
            action = f"Order {qty} units"
            why = (f"Stock of {int(on_hand)} is below the reorder point of {int(np.ceil(rop))}. Expected demand over the "
                   f"{lead}-day lead time is about {int(round(daily * lead))} units, plus {int(np.ceil(safety))} safety stock. "
                   f"At {daily:.0f} units/day, stock lasts {cover:.1f} days.{trend_txt}")
        elif risk > 0.10:
            status, action = "WATCH", "Stock-out risk rising - prepare order"
            why = (f"Stock is above the reorder point, but there is a {risk * 100:.0f}% chance of running out before an order "
                   f"could arrive ({lead}-day lead time).{trend_txt}")
        else:
            status, action = "OK", "No action"
            why = (f"Stock of {int(on_hand)} covers {cover:.1f} days, comfortably above the {lead}-day lead time and the "
                   f"reorder point of {int(np.ceil(rop))}.{trend_txt}")
        rows.append(dict(product=p, on_hand=int(on_hand), avg_daily_forecast=round(daily, 1), lead_days=lead,
                         safety_stock=int(np.ceil(safety)), reorder_point=int(np.ceil(rop)), eoq=int(np.ceil(eoq)),
                         days_of_cover=round(cover, 1), stockout_risk=round(risk, 3), status=status, action=action,
                         capital_tied=int(on_hand * cost), why=why))
    return pd.DataFrame(rows)


# ------------------------------------------------------------- workforce
def workforce_plan(forecasts: dict, df_sales, units_per_staff_day=75, min_staff=2, max_staff=9, demand_mult=1.0):
    """Translate forecasted unit volume into daily staffing requirements."""
    total = None
    for p, f in forecasts.items():
        x = f.set_index("date")["forecast"] * demand_mult
        total = x if total is None else total.add(x, fill_value=0)
    plan = pd.DataFrame({"date": total.index, "forecast_units": total.values})
    need = np.ceil(plan["forecast_units"] / units_per_staff_day)
    plan["staff_needed"] = np.clip(need, min_staff, max_staff).astype(int)
    plan["day"] = plan["date"].dt.day_name().str[:3]
    plan["load_pct"] = (plan["forecast_units"] / (plan["staff_needed"] * units_per_staff_day) * 100).round(0)
    return plan


def roster_cost(plan, daily_wage=700, current_staff=5):
    fixed = current_staff * daily_wage * len(plan)
    adaptive = plan["staff_needed"].sum() * daily_wage
    short = int((plan["staff_needed"] - current_staff).clip(lower=0).sum())
    idle = int((current_staff - plan["staff_needed"]).clip(lower=0).sum())
    return dict(fixed_cost=fixed, adaptive_cost=int(adaptive), saving=int(fixed - adaptive),
                understaffed_days_units=short, overstaffed_days_units=idle)


# ------------------------------------------------------------- anomalies
def detect_anomalies(df_sales, window=21, thresh=4.0):
    """Two independent detectors per product.

    1. Robust (median/MAD) rolling z-score on daily units: the primary detector.
    2. Isolation Forest on (units / local median, z-score): a second opinion.
    `iso_agrees` is True when both methods flag the point, which gives higher confidence.
    """
    out = []
    for p, g in df_sales.groupby("product"):
        g = g.sort_values("date").copy()
        y = g["units_sold"].astype(float)
        med = y.rolling(window, center=True, min_periods=7).median()
        mad = (y - med).abs().rolling(window, center=True, min_periods=7).median().replace(0, 1)
        g["z"] = (y - med) / (1.4826 * mad)
        g["expected"] = med
        ratio = (y / med.replace(0, np.nan)).fillna(1.0).clip(0, 10)
        if len(g) >= 40:
            feats = np.column_stack([ratio.values, g["z"].fillna(0).clip(-15, 15).values])
            iso = IsolationForest(n_estimators=150, contamination=0.04, random_state=0)
            g["iso_agrees"] = iso.fit_predict(feats) == -1
        else:
            g["iso_agrees"] = False
        out.append(g[g["z"].abs() > thresh][["date", "product", "units_sold", "expected", "z", "iso_agrees"]])
    res = pd.concat(out) if out else pd.DataFrame()
    if len(res):
        res["type"] = np.where(res["z"] > 0, "Spike", "Drop")
        res["expected"] = res["expected"].round(1)
        res["z"] = res["z"].round(1)
        res = res.sort_values("date", ascending=False)
    return res
