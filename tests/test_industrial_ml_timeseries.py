#!/usr/bin/env python3
import unittest
from src.industrial_ml_timeseries_pipeline import run_pipeline, BASE_DIR


class TestIndustrialMLTimeseriesPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = run_pipeline()

    def test_dataset_scale_and_quality_gates(self):
        self.assertEqual(self.summary["machine_telemetry_rows"], 10000)
        self.assertEqual(self.summary["timeseries_sensor_rows"], 19735)
        self.assertEqual(self.summary["verified_datasets_total_rows"], 29735)
        self.assertEqual(self.summary["data_quality_gates_passed"], "12/12 (100.0%)")

    def test_predictive_model_evaluation(self):
        pm = self.summary["predictive_maintenance_evaluation"]
        self.assertGreaterEqual(pm["ensemble_roc_auc"], 0.99)
        self.assertGreaterEqual(pm["deep_mlp_roc_auc"], 0.99)
        self.assertLessEqual(pm["brier_score"], 0.03)

    def test_walk_forward_timeseries_forecasting(self):
        ts = self.summary["walk_forward_timeseries_evaluation"]
        self.assertEqual(ts["walk_forward_folds"], 5)
        self.assertLess(ts["model_mape_pct"], 4.0)
        self.assertGreater(ts["forecast_error_reduction_pct"], 40.0)

    def test_sql_variant_kpi_breakdown(self):
        variants = self.summary["variant_kpi_breakdown"]
        self.assertEqual(len(variants), 3)
        self.assertEqual(sum(v["total_units"] for v in variants), 10000)
        self.assertEqual(sum(v["total_failures"] for v in variants), 339)

    def test_powerbi_and_llm_handoff_exports(self):
        out_dir = BASE_DIR / "outputs"
        self.assertTrue((out_dir / "powerbi_dim_product_variants.csv").exists())
        self.assertTrue((out_dir / "powerbi_fct_machine_predictions.csv").exists())
        self.assertTrue((out_dir / "powerbi_fct_timeseries_forecasts.csv").exists())
        self.assertEqual(self.summary["llm_grounding_verification"]["schema_compliance_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
