"""Automated checks for the core claims of the system.  Run:  python -m unittest discover -s tests -v"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import generate_data as gd
from engine.forecast import AdaptiveForecaster, MODEL_POOL, backtest, wape
from engine.operations import inventory_plan, workforce_plan, detect_anomalies, roster_cost


def make_df(days=420, **kw):
    return gd.generate(start="2024-01-01", days=days, seed=3, **kw)


class TestForecasting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = make_df()
        cls.af = AdaptiveForecaster()
        cls.af.train_all(cls.df)

    def test_every_product_has_a_model(self):
        self.assertEqual(set(self.af.registry.current), set(self.df["product"].unique()))

    def test_best_model_beats_seasonal_naive(self):
        better = 0
        for rec in self.af.registry.current.values():
            if rec["backtest_wape"] <= rec["scores"]["seasonal_naive"]:
                better += 1
        self.assertEqual(better, len(self.af.registry.current))

    def test_accuracy_is_reasonable(self):
        for rec in self.af.registry.current.values():
            self.assertLess(rec["backtest_wape"], 0.30)

    def test_forecast_shape_and_interval(self):
        for p in self.df["product"].unique():
            f, sd = self.af.forecast(self.df, p)
            self.assertEqual(len(f), 14)
            self.assertTrue((f["lower"] <= f["forecast"]).all() and (f["forecast"] <= f["upper"]).all())
            self.assertTrue((f["forecast"] >= 0).all())

    def test_all_models_run_and_return_horizon(self):
        s = AdaptiveForecaster.series(self.df, self.df["product"].iloc[0])
        for name, cls in MODEL_POOL.items():
            out = cls().fit(s).predict(10)
            self.assertEqual(len(out), 10, name)
            self.assertFalse(np.isnan(out).any(), name)


class TestAdaptation(unittest.TestCase):
    """New-data batches of realistic size.  (The system also retrains any model that has gone
    more than 42 days without an update, so batches here stay under that.)"""

    def setUp(self):
        self.full = make_df(days=470)

    def split(self, new_days):
        cut = self.full["date"].min() + pd.Timedelta(days=470 - new_days)
        return self.full[self.full["date"] < cut], cut

    def test_no_false_alarm_on_normal_data(self):
        old, _ = self.split(20)
        af = AdaptiveForecaster()
        af.train_all(old)
        rep = af.update(self.full)
        self.assertLessEqual((rep["action"] == "retrained").sum(), 1)

    def test_detects_demand_shift_and_retrains(self):
        old, cut = self.split(30)
        af = AdaptiveForecaster()
        af.train_all(old)
        shifted = self.full.copy()
        m = shifted["date"] >= cut
        shifted.loc[m, "units_sold"] = (shifted.loc[m, "units_sold"] * 1.5).round()
        rep = af.update(shifted)
        self.assertEqual((rep["action"] == "retrained").sum(), 6)

    def test_retraining_reduces_error_after_shift(self):
        old, cut = self.split(30)
        af = AdaptiveForecaster()
        af.train_all(old)
        shifted = self.full.copy()
        m = shifted["date"] >= cut
        shifted.loc[m, "units_sold"] = (shifted.loc[m, "units_sold"] * 1.5).round()
        p = shifted["product"].iloc[0]
        before = af.check_drift(shifted, p)["live_wape"]
        af.train_product(shifted, p, "test")
        after = af.registry.current[p]["backtest_wape"]
        self.assertLess(after, before)

    def test_stale_models_are_refreshed(self):
        old, _ = self.split(50)
        af = AdaptiveForecaster()
        af.train_all(old)
        rep = af.update(self.full)
        self.assertEqual((rep["action"] == "retrained").sum(), 6)

    def test_registry_logs_events(self):
        old, _ = self.split(30)
        af = AdaptiveForecaster()
        af.train_all(old)
        self.assertGreaterEqual(len(af.registry.history), 6)
        self.assertIn("reason", af.registry.history[0])


class TestPartialData(unittest.TestCase):
    """Regression: products whose data ends on different dates must still forecast the same 14 days."""

    def test_forecasts_share_calendar_and_short_history_works(self):
        df = make_df(days=300)
        prods = list(df["product"].unique())
        # first product's data ends 9 days earlier than the rest
        df = df[~((df["product"] == prods[0]) & (df["date"] > df["date"].max() - pd.Timedelta(days=9)))]
        # a brand-new product with only 10 days of history
        extra = df[df["product"] == prods[1]].tail(10).assign(product="New Item")
        df = pd.concat([df, extra])
        af = AdaptiveForecaster()
        af.train_all(df)
        until = df["date"].max()
        dates = None
        fc = {}
        for p in df["product"].unique():
            f, sd = af.forecast(df, p, until=until)
            fc[p] = f
            d = list(f["date"])
            dates = dates or d
            self.assertEqual(d, dates, p)
            self.assertFalse(f["forecast"].isna().any(), p)
        self.assertEqual(dates[0], until + pd.Timedelta(days=1))
        self.assertEqual(len(workforce_plan(fc, df)), 14)


class TestOperations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = make_df()
        cls.af = AdaptiveForecaster()
        cls.af.train_all(cls.df)
        cls.fc, cls.sd = {}, {}
        for p in cls.df["product"].unique():
            cls.fc[p], cls.sd[p] = cls.af.forecast(cls.df, p)

    def _stock(self, mult):
        return pd.DataFrame({"product": list(self.fc), "on_hand": [int(self.fc[p]["forecast"].mean() * mult) for p in self.fc]})

    def test_low_stock_triggers_reorder(self):
        inv = inventory_plan(self.df, self.fc, self.sd, self._stock(1))
        self.assertTrue((inv["status"] == "REORDER NOW").all())

    def test_high_stock_is_ok(self):
        inv = inventory_plan(self.df, self.fc, self.sd, self._stock(30))
        self.assertTrue((inv["status"] == "OK").all())

    def test_higher_service_level_means_more_safety_stock(self):
        a = inventory_plan(self.df, self.fc, self.sd, self._stock(8), service_level=0.90)
        b = inventory_plan(self.df, self.fc, self.sd, self._stock(8), service_level=0.99)
        self.assertTrue((b["safety_stock"] >= a["safety_stock"]).all())
        self.assertTrue((b["safety_stock"] > a["safety_stock"]).any())

    def test_workforce_bounds_and_weekend_peak(self):
        wf = workforce_plan(self.fc, self.df)
        self.assertTrue(wf["staff_needed"].between(2, 9).all())
        self.assertEqual(len(wf), 14)
        c = roster_cost(wf)
        self.assertEqual(c["fixed_cost"] - c["adaptive_cost"], c["saving"])

    def test_anomaly_detector_finds_injected_spike(self):
        df = self.df.copy()
        i = df.index[(df["product"] == df["product"].iloc[0])][200]
        df.loc[i, "units_sold"] = 900
        found = detect_anomalies(df)
        self.assertTrue(((found["units_sold"] == 900)).any())


class TestNewFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = make_df()
        cls.af = AdaptiveForecaster()
        cls.af.train_all(cls.df)
        cls.fc, cls.sd = {}, {}
        for p in cls.df["product"].unique():
            cls.fc[p], cls.sd[p] = cls.af.forecast(cls.df, p)
        cls.stock = pd.DataFrame({"product": list(cls.fc), "on_hand": [int(cls.fc[p]["forecast"].mean() * 8) for p in cls.fc]})

    def test_what_if_demand_raises_reorder_point(self):
        a = inventory_plan(self.df, self.fc, self.sd, self.stock)
        b = inventory_plan(self.df, self.fc, self.sd, self.stock, demand_mult=1.3)
        self.assertTrue((b["reorder_point"] > a["reorder_point"]).all())

    def test_what_if_supplier_delay_raises_reorder_point_and_risk(self):
        a = inventory_plan(self.df, self.fc, self.sd, self.stock)
        b = inventory_plan(self.df, self.fc, self.sd, self.stock, lead_add=4)
        self.assertTrue((b["reorder_point"] > a["reorder_point"]).all())
        self.assertTrue((b["stockout_risk"] >= a["stockout_risk"]).all())
        self.assertTrue((b["lead_days"] == a["lead_days"] + 4).all())

    def test_every_row_has_an_explanation_that_matches_its_status(self):
        stock = self.stock.copy()
        stock.loc[0, "on_hand"] = 1
        inv = inventory_plan(self.df, self.fc, self.sd, stock)
        self.assertTrue(inv["why"].str.len().gt(30).all())
        self.assertIn("below the reorder point", inv.loc[0, "why"])

    def test_workforce_demand_mult(self):
        a = workforce_plan(self.fc, self.df)
        b = workforce_plan(self.fc, self.df, demand_mult=1.5)
        self.assertGreaterEqual(b["staff_needed"].sum(), a["staff_needed"].sum())

    def test_second_detector_confirms_a_large_spike(self):
        df = self.df.copy()
        i = df.index[(df["product"] == df["product"].iloc[0])][200]
        df.loc[i, "units_sold"] = 900
        found = detect_anomalies(df)
        row = found[found["units_sold"] == 900]
        self.assertEqual(len(row), 1)
        self.assertTrue(bool(row["iso_agrees"].iloc[0]))

    def _ctx(self):
        wf = workforce_plan(self.fc, self.df)
        return dict(inv=inventory_plan(self.df, self.fc, self.sd, self.stock), fc=self.fc, af=self.af, df=self.df,
                    wf=wf, cost=roster_cost(wf), anoms=detect_anomalies(self.df), last_report=None,
                    revenue=1.0, units=1.0, accuracy=0.87, through="2025-01-01")

    def test_assistant_answers_core_questions_with_real_numbers(self):
        from engine import assistant as asst
        ctx = self._ctx()
        p0 = list(self.fc)[0]
        word = [w for w in p0.split() if len(w) > 3][0].lower()
        self.assertIn(p0, asst.answer(f"what is the forecast for {word}?", ctx))
        self.assertIn("staff", asst.answer("how many staff do I need", ctx).lower())
        self.assertIn("unusual", asst.answer("any unusual sales?", ctx).lower())
        self.assertIn("accuracy", asst.answer("which model is most accurate", ctx).lower())
        self.assertIn("nothing has been retrained", asst.answer("what did the system learn", ctx))
        self.assertIn("did not understand", asst.answer("xyzzy plugh", ctx))

    def test_assistant_stock_answer_matches_engine_status(self):
        from engine import assistant as asst
        ctx = self._ctx()
        ctx["inv"].loc[0, "status"] = "REORDER NOW"
        ctx["inv"].loc[0, "action"] = "Order 100 units"
        self.assertIn(ctx["inv"].loc[0, "product"], asst.answer("which products will run out soon?", ctx))


class TestExperiment(unittest.TestCase):
    def test_small_experiment_runs_and_is_consistent(self):
        from engine.experiment import run_experiment
        r = run_experiment("surge", future_days=35, products=["Cooking Oil 1L", "Notebook A4"], surge_day=7, progress=None)
        for k in ("adaptive", "static"):
            s = r["summary"][k]
            self.assertTrue(0 <= s["fill_rate"] <= 1)
            self.assertAlmostEqual(s["total_cost"], s["lost_margin"] + s["holding_cost"] + s["ordering_cost"], places=3)
        # identical demand for both shops => identical number of orders under the same review schedule is not
        # guaranteed, but both must have placed orders
        self.assertGreater(r["summary"]["adaptive"]["orders"], 0)
        self.assertGreater(r["summary"]["static"]["orders"], 0)
        self.assertEqual(len(r["series"]["date"]), 5)



if __name__ == "__main__":
    unittest.main(verbosity=2)
