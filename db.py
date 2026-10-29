"""SQLite persistence for the SME Adaptive Intelligence System.

Why SQLite: it is part of Python (nothing to install), it is one file (data/sme.db), it is transactional, and it
is plenty for one business.  Set SME_DB to use a different file.  For many users or many shops, the same schema
moves to PostgreSQL with only this file changing.

Tables
  sales          one row per product per day (the history the models learn from)
  stock          current units on hand per product
  model_registry the model currently chosen for each product (name, scores, trained-through date)
  learning_log   every retrain / drift check the adaptive engine has ever made, with the reason
  upload_log     every batch of data added (source, rows, how many models retrained)
  meta           small settings (simulation counter, last update report)

Command line
  python db.py status
  python db.py import --sales my_sales.csv [--stock my_stock.csv]    load a real business, replacing the demo data
  python db.py export sales_out.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from contextlib import closing
from datetime import datetime

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
SALES_COLS = ["date", "product", "units_sold", "price", "unit_cost", "lead_time_days", "promo"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS sales (
    date TEXT NOT NULL, product TEXT NOT NULL, units_sold REAL NOT NULL,
    price REAL, unit_cost REAL, lead_time_days INTEGER, promo INTEGER DEFAULT 0,
    PRIMARY KEY (date, product)
);
CREATE INDEX IF NOT EXISTS idx_sales_product ON sales(product, date);
CREATE TABLE IF NOT EXISTS stock (
    product TEXT PRIMARY KEY, on_hand INTEGER NOT NULL, updated_at TEXT
);
CREATE TABLE IF NOT EXISTS model_registry (
    product TEXT PRIMARY KEY, record TEXT NOT NULL, saved_at TEXT
);
CREATE TABLE IF NOT EXISTS learning_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, time TEXT, product TEXT, event TEXT, reason TEXT, extra TEXT
);
CREATE TABLE IF NOT EXISTS upload_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, time TEXT, source TEXT, rows_added INTEGER,
    skipped_rows INTEGER, retrained INTEGER, kept INTEGER, data_through TEXT
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


def path() -> str:
    return os.environ.get("SME_DB") or os.path.join(ROOT, "data", "sme.db")


def connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(path()), exist_ok=True)
    con = sqlite3.connect(path(), timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.executescript(SCHEMA)
    return con


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ------------------------------------------------------------------ state checks
def has_data() -> bool:
    with closing(connect()) as c:
        return c.execute("SELECT COUNT(*) FROM sales").fetchone()[0] > 0


def wipe() -> None:
    """Delete everything (used by Reset demo and by import)."""
    with closing(connect()) as c, c:
        for t in ("sales", "stock", "model_registry", "learning_log", "upload_log", "meta"):
            c.execute(f"DELETE FROM {t}")


# ------------------------------------------------------------------ sales and stock
def load_sales() -> pd.DataFrame:
    with closing(connect()) as c:
        df = pd.read_sql_query("SELECT * FROM sales ORDER BY product, date", c, parse_dates=["date"])
    df["promo"] = df["promo"].fillna(0).astype(int)
    return df[SALES_COLS]


def upsert_sales(df: pd.DataFrame) -> None:
    d = df[SALES_COLS].copy()
    d["date"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")
    d = d.astype(object).where(d.notna(), None)
    with closing(connect()) as c, c:
        c.executemany(
            "INSERT OR REPLACE INTO sales (date, product, units_sold, price, unit_cost, lead_time_days, promo) "
            "VALUES (?,?,?,?,?,?,?)",
            [(r[0], r[1], float(r[2]), r[3], r[4], None if r[5] is None else int(r[5]), int(r[6] or 0))
             for r in d.itertuples(index=False, name=None)])


def load_stock() -> pd.DataFrame:
    with closing(connect()) as c:
        return pd.read_sql_query("SELECT product, on_hand FROM stock ORDER BY product", c)


def save_stock(stock: pd.DataFrame) -> None:
    with closing(connect()) as c, c:
        c.executemany(
            "INSERT OR REPLACE INTO stock (product, on_hand, updated_at) VALUES (?,?,?)",
            [(str(r.product), int(r.on_hand), now()) for r in stock.itertuples()])


# ------------------------------------------------------------------ model registry + learning log
def save_registry(registry, already_logged: int) -> int:
    """Store the current models and append any new log events.  Returns how many events are now stored."""
    with closing(connect()) as c, c:
        c.execute("DELETE FROM model_registry")
        c.executemany("INSERT INTO model_registry (product, record, saved_at) VALUES (?,?,?)",
                      [(p, json.dumps(rec, default=str), now()) for p, rec in registry.current.items()])
        for ev in registry.history[already_logged:]:
            extra = {k: v for k, v in ev.items() if k not in ("time", "product", "event", "reason")}
            c.execute("INSERT INTO learning_log (time, product, event, reason, extra) VALUES (?,?,?,?,?)",
                      (ev.get("time"), ev.get("product"), ev.get("event"), ev.get("reason"),
                       json.dumps(extra, default=str)))
    return len(registry.history)


def load_registry():
    """Return a ModelRegistry rebuilt from the database, or None if nothing is stored."""
    from engine.forecast import ModelRegistry
    with closing(connect()) as c:
        cur = c.execute("SELECT product, record FROM model_registry").fetchall()
        logs = c.execute("SELECT time, product, event, reason, extra FROM learning_log ORDER BY id").fetchall()
    if not cur:
        return None
    hist = []
    for t, p, e, r, x in logs:
        ev = {"time": t, "product": p, "event": e, "reason": r}
        ev.update(json.loads(x) if x else {})
        hist.append(ev)
    return ModelRegistry(current={p: json.loads(rec) for p, rec in cur}, history=hist)


# ------------------------------------------------------------------ uploads, meta
def log_upload(report: dict, skipped: int = 0) -> None:
    rows = report.get("report", [])
    with closing(connect()) as c, c:
        c.execute("INSERT INTO upload_log (time, source, rows_added, skipped_rows, retrained, kept, data_through) "
                  "VALUES (?,?,?,?,?,?,?)",
                  (now(), report.get("source"), report.get("rows", 0), skipped,
                   sum(1 for r in rows if r.get("action") == "retrained"),
                   sum(1 for r in rows if r.get("action") == "kept"), report.get("through")))


def set_meta(key: str, value) -> None:
    with closing(connect()) as c, c:
        c.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)", (key, json.dumps(value, default=str)))


def get_meta(key: str, default=None):
    with closing(connect()) as c:
        r = c.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return json.loads(r[0]) if r else default


# ------------------------------------------------------------------ reporting
def stats() -> dict:
    with closing(connect()) as c:
        counts = {t: c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                  for t in ("sales", "stock", "model_registry", "learning_log", "upload_log")}
        span = c.execute("SELECT MIN(date), MAX(date), COUNT(DISTINCT product) FROM sales").fetchone()
        uploads = c.execute("SELECT time, source, rows_added, skipped_rows, retrained, kept, data_through "
                            "FROM upload_log ORDER BY id DESC LIMIT 8").fetchall()
    p = path()
    return dict(engine="SQLite " + sqlite3.sqlite_version, file=os.path.basename(p),
                size_kb=round(os.path.getsize(p) / 1024, 1) if os.path.exists(p) else 0,
                counts=counts, first_date=span[0], last_date=span[1], products=span[2],
                recent_uploads=[dict(time=u[0], source=u[1], rows=u[2], skipped=u[3], retrained=u[4], kept=u[5],
                                     through=u[6]) for u in uploads])


# ------------------------------------------------------------------ importing a real business
def clean_sales_csv(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    need = {"date", "product", "units_sold"}
    if not need.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {sorted(need)}; found {list(df.columns)}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["units_sold"] = pd.to_numeric(df["units_sold"], errors="coerce")
    bad = int(df[["date", "units_sold"]].isna().any(axis=1).sum())
    df = df.dropna(subset=["date", "units_sold"])
    df = df[df["units_sold"] >= 0]
    df["product"] = df["product"].astype(str).str.strip()
    for col, default in (("price", 100.0), ("unit_cost", 70.0), ("lead_time_days", 5), ("promo", 0)):
        if col not in df.columns:
            df[col] = default
        df[col] = df[col].fillna(default)
    df = df.drop_duplicates(["date", "product"], keep="last")
    return df[SALES_COLS], bad


def import_business(sales_csv: str, stock_csv: str | None = None) -> dict:
    df, bad = clean_sales_csv(pd.read_csv(sales_csv))
    if df.empty:
        raise ValueError("No valid rows after cleaning")
    wipe()
    upsert_sales(df)
    if stock_csv:
        st = pd.read_csv(stock_csv)[["product", "on_hand"]]
    else:  # no stock file: assume one week of average sales on the shelf, and say so
        last = df[df["date"] >= df["date"].max() - pd.Timedelta(days=28)]
        st = (last.groupby("product")["units_sold"].mean() * 7).round().astype(int).rename("on_hand").reset_index()
    save_stock(st)
    set_meta("source", f"imported from {os.path.basename(sales_csv)}")
    return dict(rows=len(df), products=int(df["product"].nunique()), skipped=bad,
                first=str(df["date"].min().date()), last=str(df["date"].max().date()),
                stock_source="file" if stock_csv else "estimated: 7 days of average sales (edit in the Inventory tab)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    im = sub.add_parser("import")
    im.add_argument("--sales", required=True)
    im.add_argument("--stock")
    ex = sub.add_parser("export")
    ex.add_argument("out")
    a = ap.parse_args(argv)
    if a.cmd == "status":
        print(json.dumps(stats(), indent=2))
    elif a.cmd == "import":
        print(json.dumps(import_business(a.sales, a.stock), indent=2))
        print("Done. Start the app (python app.py); it trains models on this data at first start.")
    elif a.cmd == "export":
        load_sales().to_csv(a.out, index=False)
        print("Wrote", a.out)


if __name__ == "__main__":
    sys.exit(main())
