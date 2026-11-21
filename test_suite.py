"""
Module 7: test_suite.py
COS10009 Introduction to Programming — Automated Verification & Unit Test Suite (ULO 1)
Author: Anh Vu Pham (Student ID: 106217835)
"""

import os
import unittest
from domain_models import AppState, CrashRecord
from benchmark_runner import (
    ingest_and_prepare_repository,
    run_policy_preset_simulations,
    run_rq3_scaling_benchmarks,
    train_and_benchmark_models,
)
from io_persistence_lib import filter_invalid_records_in_place, query_sqlite_record_counts
from metrics_stats_lib import compute_gini_impurity, compute_survival_rate, floats_are_equal
from ml_policy_engine import (
    binary_search_speed_range,
    linear_search_speed_range,
)


class TestSafeRoadWorkbench(unittest.TestCase):
    """Automated unit test suite for COS10009 ULO 1 verification."""

    def test_01_float_safety_and_survival_rate(self) -> None:
        self.assertTrue(floats_are_equal(0.1 + 0.2, 0.3))
        r1 = CrashRecord(1, 25, "Female", 50, True, True, 1, "test")
        r2 = CrashRecord(2, 30, "Male", 95, False, False, 0, "test")
        rate = compute_survival_rate([r1, r2])
        gini = compute_gini_impurity([r1, r2])
        self.assertTrue(floats_are_equal(rate, 0.5))
        self.assertTrue(floats_are_equal(gini, 0.5))

    def test_02_in_place_while_loop_deletion(self) -> None:
        records = [
            CrashRecord(1, 25, "Female", 60, True, True, 1, "t"),
            CrashRecord(2, 12, "Male", 60, True, True, 0, "t"),    # Invalid age < 18
            CrashRecord(3, 30, "Male", 150, False, False, 0, "t"),  # Invalid speed > 119
            CrashRecord(4, 40, "Female", 70, True, False, 1, "t"),
        ]
        removed = filter_invalid_records_in_place(records)
        self.assertEqual(removed, 2)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].record_id, 1)
        self.assertEqual(records[1].record_id, 4)

    def test_03_full_pipeline_dedup_sort_search_and_sqlite(self) -> None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        state = AppState()
        ingest_and_prepare_repository(state, base_dir)

        self.assertEqual(len(state.repository.raw_records), 1200)
        self.assertEqual(len(state.repository.dedup_records), 599)

        lin_matches, _ = linear_search_speed_range(state.repository.dedup_records, 75, 86)
        bin_matches, bin_comps = binary_search_speed_range(state.repository.sorted_by_speed, 75, 86)
        self.assertEqual(len(lin_matches), len(bin_matches))
        self.assertLessEqual(bin_comps, 24)

        train_and_benchmark_models(state)
        run_policy_preset_simulations(state, base_dir)
        run_rq3_scaling_benchmarks(state)

        self.assertEqual(len(state.metrics_list), 6)
        self.assertGreaterEqual(len(state.policy_history), 4)
        db_counts = query_sqlite_record_counts(state.sqlite_db_path)
        self.assertEqual(db_counts["crash_records"], 599)
        self.assertGreaterEqual(db_counts["model_benchmarks"], 6)
        self.assertGreaterEqual(db_counts["policy_simulations"], 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
