"""Adaptive demand forecasting.

For each product the system keeps a small pool of candidate models:
  * seasonal_naive   - repeat the value from 7 days ago
  * moving_average   - mean of the last 28 days with a weekday profile
  * holt_winters     - additive trend + weekly seasonality (hand-written, no deps)
  * gbm              - gradient boosting on lag / calendar features
Models are scored with rolling-origin backtests (walk-forward validation) and the
best one is selected *per product*. The ModelRegistry stores the choice, the
error, and a history so the dashboard can show how the system adapts. When fresh
data arrives, `AdaptiveForecaster.update()` re-scores everything and flags
concept drift when the live error rises well above the error recorded at
training time.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

HORIZON = 14


# ----------------------------------------------------------------- metrics
def wape(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.abs(y - yhat).sum() / max(y.sum(), 1e-9))


# ----------------------------------------------------------------- models
class SeasonalNaive:
    name = "seasonal_naive"

    def fit(self, s: pd.Series):
        self.s = s
        return self

    def predict(self, h):
        last = self.s.values[-7:]
        return np.array([last[i % 7] for i in range(h)], float)


class MovingAverage:
    name = "moving_average"

    def fit(self, s: pd.Series):
        self.s = s
        return self

    def predict(self, h):
        recent = self.s.tail(56)
        level = self.s.tail(28).mean()
        prof = recent.groupby(recent.index.dayofweek).mean()
        prof = prof / prof.mean()
        start = self.s.index[-1] + pd.Timedelta(days=1)
        dows = [(start + pd.Timedelta(days=i)).dayofweek for i in range(h)]
        return np.array([level * prof.get(d, 1.0) for d in dows], float)


class HoltWinters:
    """Additive Holt-Winters with weekly seasonality; parameters chosen by small grid search."""
    name = "holt_winters"
    m = 7

    def _run(self, y, a, b, g):
        m = self.m
        level = y[:m].mean()
        trend = (y[m:2 * m].mean() - y[:m].mean()) / m
        season = list(y[:m] - level)
        sse = 0.0
        for t in range(m, len(y)):
            s = season[t - m]
            f = level + trend + s
            sse += (y[t] - f) ** 2
            new_level = a * (y[t] - s) + (1 - a) * (level + trend)
            trend = b * (new_level - level) + (1 - b) * trend
            season.append(g * (y[t] - new_level) + (1 - g) * s)
            level = new_level
        return sse, level, trend, season

    def fit(self, s: pd.Series):
        y = s.values.astype(float)[-120:]
        best = None
        for a in (0.1, 0.25, 0.4):
            for b in (0.0, 0.02):
                for g in (0.1, 0.3):
                    sse, level, trend, season = self._run(y, a, b, g)
                    if best is None or sse < best[0]:
                        best = (sse, level, trend, season)
        _, self.level, self.trend, self.season = best
        self.n = len(y)
        return self

    def predict(self, h):
        out = []
        for i in range(1, h + 1):
            s = self.season[self.n - self.m + ((i - 1) % self.m)]
            out.append(self.level + i * self.trend + s)
        return np.clip(out, 0, None)


def _features(idx: pd.DatetimeIndex):
    return np.column_stack([
        idx.dayofweek, idx.dayofyear, idx.month,
        np.sin(2 * np.pi * idx.dayofyear / 365.25), np.cos(2 * np.pi * idx.dayofyear / 365.25),
        (idx.dayofweek >= 5).astype(int),
    ])


class GBM:
    """Recursive gradient boosting using calendar features + recent level / lags."""
    name = "gbm"

    def _row(self, hist: np.ndarray, d: pd.Timestamp):
        cal = _features(pd.DatetimeIndex([d]))[0]
        lag7 = hist[-7] if len(hist) >= 7 else hist[-1]
        lag14 = hist[-14] if len(hist) >= 14 else hist[-1]
        return np.concatenate([cal, [lag7, lag14, hist[-28:].mean(), hist[-7:].mean()]])

    def fit(self, s: pd.Series):
        y = s.values.astype(float)
        X, t = [], []
        for i in range(28, len(y)):
            X.append(self._row(y[:i], s.index[i]))
            t.append(y[i])
        self.model = GradientBoostingRegressor(n_estimators=120, max_depth=3, learning_rate=0.06,
                                               subsample=0.8, random_state=0)
        self.model.fit(np.array(X), np.array(t))
        self.hist = y.copy()
        self.last = s.index[-1]
        return self

    def predict(self, h):
        hist = list(self.hist)
        out = []
        for i in range(1, h + 1):
            d = self.last + pd.Timedelta(days=i)
            p = float(self.model.predict(self._row(np.array(hist), d).reshape(1, -1))[0])
            p = max(p, 0.0)
            out.append(p)
            hist.append(p)
        return np.array(out)


MODEL_POOL = {m.name: m for m in (SeasonalNaive, MovingAverage, HoltWinters, GBM)}


# ----------------------------------------------------------------- backtesting
def backtest(model_cls, s: pd.Series, folds=3, h=HORIZON, step=14):
    """Walk-forward validation. Returns mean WAPE across folds."""
    errs = []
    n = len(s)
    for k in range(folds, 0, -1):
        cut = n - k * step
        if cut < 70:
            continue
        train, test = s.iloc[:cut], s.iloc[cut:cut + h]
        if len(test) < h:
            continue
        pred = model_cls().fit(train).predict(h)
        errs.append(wape(test.values, pred))
    return float(np.mean(errs)) if errs else float("nan")


# ----------------------------------------------------------------- registry
@dataclass
class ModelRecord:
    product: str
    model: str
    backtest_wape: float
    scores: dict
    trained_at: str
    trained_through: str
    n_obs: int


@dataclass
class ModelRegistry:
    current: dict = field(default_factory=dict)       # product -> ModelRecord(dict)
    history: list = field(default_factory=list)       # events

    def log(self, **kw):
        kw["time"] = datetime.now().isoformat(timespec="seconds")
        self.history.append(kw)

    def to_json(self):
        return json.dumps(asdict(self), indent=2, default=str)

    @staticmethod
    def from_json(txt):
        d = json.loads(txt)
        return ModelRegistry(current=d["current"], history=d["history"])


# ----------------------------------------------------------------- adaptive controller
class AdaptiveForecaster:
    DRIFT_RATIO = 1.6    # live error this much above training error => drift
    DRIFT_FLOOR = 0.24   # ...and above this absolute WAPE

    def __init__(self, registry: ModelRegistry | None = None):
        self.registry = registry or ModelRegistry()

    @staticmethod
    def series(df: pd.DataFrame, product: str) -> pd.Series:
        g = df[df["product"] == product].copy()
        g["date"] = pd.to_datetime(g["date"])
        s = g.groupby("date")["units_sold"].sum().asfreq("D")
        return s.interpolate().clip(lower=0)

    @staticmethod
    def clean(s: pd.Series) -> pd.Series:
        """Winsorise outliers using a rolling median/MAD so bad data does not poison training."""
        med = s.rolling(15, center=True, min_periods=5).median()
        mad = (s - med).abs().rolling(15, center=True, min_periods=5).median().replace(0, 1)
        z = (s - med) / (1.4826 * mad)
        return s.where(z.abs() < 4, med)

    def train_product(self, df, product, reason="initial training"):
        s = self.clean(self.series(df, product))
        scores = {name: backtest(cls, s) for name, cls in MODEL_POOL.items()}
        if all(np.isnan(v) for v in scores.values()):   # too little history to backtest (< ~70 days)
            best = "moving_average"
        else:
            best = min(scores, key=lambda k: (np.isnan(scores[k]), scores[k]))
        prev = self.registry.current.get(product)
        rec = ModelRecord(product, best, scores[best], scores, datetime.now().isoformat(timespec="seconds"),
                          str(s.index[-1].date()), len(s))
        self.registry.current[product] = asdict(rec)
        self.registry.log(event="retrain", product=product, reason=reason, model=best,
                          wape=round(scores[best], 4),
                          switched_from=(prev["model"] if prev and prev["model"] != best else None))
        return rec

    def train_all(self, df, reason="initial training"):
        for p in df["product"].unique():
            self.train_product(df, p, reason)
        return self.registry

    BIAS_LIMIT = 0.15    # systematic over/under-forecast beyond max(this, the model's own backtest error) => drift
    STALE_DAYS = 42      # retrain anyway after this many days of unseen data

    def check_drift(self, df, product, max_eval=42):
        """Judge the *frozen* model (as trained) against every new observation since training.

        Three triggers: error rise (WAPE well above training WAPE), systematic bias
        (consistent over/under-forecasting = level shift) and staleness.
        """
        rec = self.registry.current.get(product)
        if not rec:
            return {"drift": True, "reason": "new product", "live_wape": None, "bias": None,
                    "train_wape": None, "new_days": 0}
        s = self.clean(self.series(df, product))
        cutoff = pd.Timestamp(rec["trained_through"])
        new = s[s.index > cutoff]
        base = dict(live_wape=None, bias=None, train_wape=rec["backtest_wape"], new_days=len(new))
        if len(new) < 7:
            return {"drift": False, "reason": "not enough new data", **base}
        h = min(len(new), max_eval)
        pred = MODEL_POOL[rec["model"]]().fit(s[s.index <= cutoff]).predict(h)
        act = new.values[:h]
        live = wape(act, pred)
        bias = float((act.sum() - pred.sum()) / max(act.sum(), 1e-9))
        tr = rec["backtest_wape"]
        reason = None
        if live > max(self.DRIFT_FLOOR, tr * self.DRIFT_RATIO):
            reason = "error rise"
        elif abs(bias) > max(self.BIAS_LIMIT, tr):
            reason = "demand level shift (" + ("under" if bias > 0 else "over") + "-forecasting)"
        elif len(new) > self.STALE_DAYS:
            reason = "model stale"
        return {"drift": reason is not None, "reason": reason or "no drift", **base | {"live_wape": live, "bias": bias}}

    def update(self, df, force=False):
        """Called when new data is uploaded. Retrains products whose error has drifted."""
        report = []
        for p in df["product"].unique():
            d = self.check_drift(df, p)
            retrain = force or d["drift"] or p not in self.registry.current
            if retrain:
                reason = "forced" if force else ("new product" if p not in self.registry.current else d["reason"])
                rec = self.train_product(df, p, reason)
                report.append({"product": p, "action": "retrained", "reason": reason, "model": rec.model,
                               "live_wape": d["live_wape"], "bias": d["bias"], "new_wape": rec.backtest_wape})
            else:
                self.registry.log(event="checked", product=p, reason=d["reason"], model=self.registry.current[p]["model"],
                                  wape=round(d["live_wape"], 4) if d["live_wape"] is not None else None)
                report.append({"product": p, "action": "kept", "reason": d["reason"],
                               "model": self.registry.current[p]["model"], "live_wape": d["live_wape"],
                               "bias": d["bias"], "new_wape": None})
        return pd.DataFrame(report)

    def forecast(self, df, product, h=HORIZON, until=None):
        """Forecast `h` days *after* `until` (default: the last date in df for any product).

        Products whose own data ends earlier are forecast further ahead and the gap is dropped,
        so every product's forecast covers exactly the same calendar days.
        """
        rec = self.registry.current[product]
        s = self.clean(self.series(df, product))
        until = pd.Timestamp(until) if until is not None else pd.to_datetime(df["date"]).max()
        gap = max(0, (until - s.index[-1]).days)
        cls = MODEL_POOL[rec["model"]]
        model = cls().fit(s)
        mean = model.predict(h + gap)[gap:]
        resid = self._residuals(s, cls)
        sd = float(np.std(resid)) if len(resid) else float(s.tail(28).std())
        idx = pd.date_range(until + pd.Timedelta(days=1), periods=h)
        z = 1.28  # ~80% interval
        widen = np.sqrt(np.arange(1, h + 1)) ** 0.5
        return pd.DataFrame({"date": idx, "forecast": mean, "lower": np.clip(mean - z * sd * widen, 0, None),
                             "upper": mean + z * sd * widen}), sd

    @staticmethod
    def _residuals(s, cls, folds=3, h=HORIZON, step=14):
        res = []
        n = len(s)
        for k in range(folds, 0, -1):
            cut = n - k * step
            if cut < 70 or len(s.iloc[cut:cut + h]) < h:
                continue
            pred = cls().fit(s.iloc[:cut]).predict(h)
            res.extend(s.iloc[cut:cut + h].values - pred)
        return res
