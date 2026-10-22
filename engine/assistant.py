"""Business assistant: answers plain-English questions from the live system state.

This is a transparent rule-based assistant (keyword intent matching + product-name matching).
Every number it states is read from the forecasting, inventory and workforce engines, so it can
never invent figures. A language model could later be plugged in to *phrase* answers, while the
numbers would still come from these same engine outputs.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

EXAMPLES = [
    "Which products will run out soon?",
    "What is the forecast for rice?",
    "How many staff do I need on the busiest day?",
    "Were there any unusual sales recently?",
    "Which model is most accurate?",
    "What did the system learn from the last update?",
    "Give me a summary",
]


def _match_products(q: str, products):
    found = []
    ql = q.lower()
    for p in products:
        toks = [w.lower() for w in re.findall(r"[A-Za-z]{3,}", p) if w.lower() not in {"the", "kg"}]
        if any(re.search(rf"\b{re.escape(t)}s?\b", ql) for t in toks):
            found.append(p)
    return found


def _has(q, *words):
    return any(re.search(rf"\b{w}", q) for w in words)


def answer(question: str, ctx: dict) -> str:
    q = question.lower().strip()
    if not q:
        return "Ask me something about demand, stock, staffing, anomalies or the models."
    inv: pd.DataFrame = ctx["inv"]
    products = list(ctx["fc"].keys())
    mentioned = _match_products(q, products)

    if _has(q, "help", "what can you", "hello", "hi$", "hey"):
        return "I can answer questions about the live system. Try:\n" + "\n".join("• " + e for e in EXAMPLES)

    # --- stock / reorder
    if _has(q, "run out", "stock", "reorder", "order", "shortage", "low", "inventory", "supply", "buy", "restock"):
        if mentioned:
            rows = inv[inv["product"].isin(mentioned)]
            return "\n".join(f"**{r.product}**: {r.status}. {r.why} Suggested: {r.action}." for r in rows.itertuples())
        urgent = inv[inv["status"] == "REORDER NOW"]
        watch = inv[inv["status"] == "WATCH"]
        if urgent.empty and watch.empty:
            return "No product needs reordering right now. All stock levels are above their reorder points."
        lines = []
        for r in urgent.itertuples():
            lines.append(f"• **{r.product}** needs reordering now: {r.on_hand} on hand, {r.days_of_cover} days of cover, "
                         f"lead time {r.lead_days} days. {r.action}.")
        for r in watch.itertuples():
            lines.append(f"• **{r.product}** is on watch: {r.stockout_risk * 100:.0f}% risk of running out within the lead time.")
        return f"{len(urgent)} product(s) need reordering now and {len(watch)} are on watch:\n" + "\n".join(lines)

    # --- forecast / demand for a product
    if _has(q, "forecast", "demand", "sell", "sales", "predict", "expect", "next week", "next month") \
            and not _has(q, "unusual", "anomal", "spike", "drop", "outlier", "strange", "weird"):
        if mentioned:
            out = []
            for p in mentioned:
                f = ctx["fc"][p]
                nxt7 = f["forecast"].head(7).sum()
                tot = f["forecast"].sum()
                rec = ctx["af"].registry.current[p]
                recent = ctx["df"][ctx["df"]["product"] == p].sort_values("date")["units_sold"].tail(14).sum()
                chg = (f["forecast"].head(14).sum() / recent - 1) * 100 if recent else 0
                out.append(f"**{p}**: about {nxt7:.0f} units over the next 7 days and {tot:.0f} over 14 days "
                           f"({chg:+.0f}% versus the last 14 days). Model: {rec['model'].replace('_', ' ')}, "
                           f"accuracy about {(1 - rec['backtest_wape']) * 100:.0f}%.")
            return "\n".join(out)
        tot = sum(float(f["forecast"].sum()) for f in ctx["fc"].values())
        top = sorted(ctx["fc"].items(), key=lambda kv: -kv[1]["forecast"].sum())[:3]
        return (f"Total forecast demand for the next 14 days is about {tot:.0f} units (revenue about "
                f"₹{ctx['revenue']:,.0f}). Highest-volume products: " +
                ", ".join(f"{p} ({f['forecast'].sum():.0f})" for p, f in top) + ".")

    # --- staffing
    if _has(q, "staff", "roster", "worker", "employee", "shift", "people", "hire", "labour", "labor"):
        wf = ctx["wf"]
        peak = wf.loc[wf["staff_needed"].idxmax()]
        c = ctx["cost"]
        lines = [f"The busiest day is {peak['day']} {str(peak['date'])[:10]}, needing {int(peak['staff_needed'])} staff "
                 f"for about {peak['forecast_units']:.0f} units.",
                 f"Over 14 days the demand-based plan needs {int(wf['staff_needed'].sum())} staff-days versus "
                 f"{c['fixed_cost'] / 700:.0f} on the fixed roster."]
        if c["saving"] >= 0:
            lines.append(f"That is a saving of about ₹{c['saving']:,.0f}.")
        else:
            lines.append(f"Demand exceeds the current roster by about {c['understaffed_days_units']} staff-days "
                         f"(extra cost about ₹{-c['saving']:,.0f}).")
        return " ".join(lines)

    # --- anomalies
    if _has(q, "anomal", "unusual", "spike", "drop", "strange", "outlier", "weird"):
        a = ctx["anoms"]
        if mentioned and len(a):
            a = a[a["product"].isin(mentioned)]
        if a is None or len(a) == 0:
            return "No unusual sales were detected."
        top = a.head(5)
        both = int(a["iso_agrees"].sum()) if "iso_agrees" in a else 0
        return (f"{len(a)} unusual days detected, {both} confirmed by both detection methods. Most recent:\n" +
                "\n".join(f"• {str(r.date)[:10]}: {r.product} sold {int(r.units_sold)} (expected about {r.expected:.0f}), "
                          f"a {r.type.lower()}." for r in top.itertuples()))

    # --- accuracy / models
    if _has(q, "model", "accura", "algorithm", "best", "error", "reliable", "trust"):
        rows = []
        for p, r in ctx["af"].registry.current.items():
            if mentioned and p not in mentioned:
                continue
            acc = "n/a (too little history)" if r["backtest_wape"] is None or np.isnan(r["backtest_wape"]) \
                else f"{(1 - r['backtest_wape']) * 100:.0f}%"
            rows.append((p, r["model"].replace("_", " "), acc))
        return "Current production models:\n" + "\n".join(f"• {p}: {m}, accuracy {a}" for p, m, a in rows)

    # --- adaptation / learning
    if _has(q, "learn", "retrain", "adapt", "drift", "update", "changed", "last"):
        lr = ctx["last_report"]
        if not lr:
            return "No new data has been added since the initial training, so nothing has been retrained yet."
        n = sum(1 for r in lr["report"] if r["action"] == "retrained")
        det = "; ".join(f"{r['product']} ({r['reason']})" for r in lr["report"] if r["action"] == "retrained")
        return (f"The last update ({lr['source']}) checked {len(lr['report'])} products and retrained {n}." +
                (f" Retrained: {det}." if det else " All models were kept because no drift was detected."))

    # --- revenue
    if _has(q, "revenue", "money", "income", "turnover", "profit"):
        return f"Forecast revenue for the next 14 days is about ₹{ctx['revenue']:,.0f} at current prices."

    # --- summary
    if _has(q, "summary", "overview", "status", "how are we", "report", "everything"):
        urgent = inv[inv["status"] == "REORDER NOW"]["product"].tolist()
        return (f"Next 14 days: about {ctx['units']:.0f} units and ₹{ctx['revenue']:,.0f} in revenue. "
                f"{len(urgent)} product(s) need reordering" + (f" ({', '.join(urgent)})" if urgent else "") +
                f". Average forecast accuracy is about {ctx['accuracy'] * 100:.0f}%. Data runs through {ctx['through']}.")

    if mentioned:
        p = mentioned[0]
        r = inv[inv["product"] == p].iloc[0]
        return f"**{p}**: {r.status}. {r.why}"

    return "I did not understand that. Try one of these:\n" + "\n".join("• " + e for e in EXAMPLES)
