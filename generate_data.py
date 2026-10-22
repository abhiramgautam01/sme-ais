"""Generate a realistic synthetic dataset for a small retail business.

Produces data/sales.csv with daily sales per product, including weekly and
yearly seasonality, trend, promotions, occasional anomalies, and (optionally)
a demand shift partway through, which is used to demo adaptive retraining.
"""
import argparse
import numpy as np
import pandas as pd

PRODUCTS = {
    # name: (base daily demand, weekly amplitude, yearly amplitude, unit cost, price, lead time days)
    "Basmati Rice 5kg": (38, 0.25, 0.10, 310, 395, 4),
    "Cooking Oil 1L": (52, 0.20, 0.08, 118, 149, 3),
    "Tea Powder 500g": (30, 0.30, 0.18, 165, 215, 5),
    "Notebook A4": (24, 0.15, 0.45, 28, 45, 7),
    "LED Bulb 9W": (18, 0.10, 0.12, 62, 99, 6),
    "Detergent 1kg": (34, 0.22, 0.10, 78, 110, 4),
}


def generate(start="2024-01-01", days=640, seed=42, shift_after=None, t0=0, level_mult=1.0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days, freq="D")
    rows = []
    for name, (base, wk, yr, cost, price, lead) in PRODUCTS.items():
        t = np.arange(days) + t0
        weekly = 1 + wk * np.where(dates.dayofweek >= 5, 1.0, -0.3 * (dates.dayofweek / 4))
        yearly = 1 + yr * np.sin(2 * np.pi * (dates.dayofyear - 60) / 365.25)
        trend = 1 + 0.0004 * t
        promo = (rng.random(days) < 0.06).astype(int)
        mu = base * weekly * yearly * trend * (1 + 0.35 * promo)
        if shift_after is not None:
            mu = np.where(np.arange(days) >= shift_after, mu * 1.45, mu)  # sudden demand shift
        mu = mu * level_mult
        demand = rng.poisson(np.clip(mu, 1, None)).astype(float)
        # sprinkle anomalies (data errors / bulk orders)
        idx = rng.choice(days, size=max(1, days // 120), replace=False)
        demand[idx] *= rng.choice([0.2, 3.2], size=len(idx))
        demand = np.round(demand)
        for d, q, p in zip(dates, demand, promo):
            rows.append((d, name, int(q), price, cost, lead, int(p)))
    df = pd.DataFrame(rows, columns=["date", "product", "units_sold", "price", "unit_cost", "lead_time_days", "promo"])
    return df


def initial_stock(df):
    """Starting stock snapshot used by the inventory module (units on hand today)."""
    out = {}
    rng = np.random.default_rng(7)
    for p, g in df.groupby("product"):
        avg = g.units_sold.tail(28).mean()
        out[p] = int(avg * rng.uniform(2.5, 9))
    return pd.DataFrame({"product": list(out), "on_hand": list(out.values())})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sales.csv")
    ap.add_argument("--days", type=int, default=640)
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--shift-after", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    df = generate(a.start, a.days, a.seed, a.shift_after)
    df.to_csv(a.out, index=False)
    initial_stock(df).to_csv("data/stock.csv", index=False)
    print(f"Wrote {len(df)} rows to {a.out}")
