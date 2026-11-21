"""
Module 2: metrics_stats_lib.py
COS10009 Introduction to Programming — Custom Reusable Evaluation & Summary Library (77D/85HD)
Author: Anh Vu Pham (Student ID: 106217835)

Provides straightforward, first-year-friendly helper functions for:
1. Safe floating-point comparisons using math.isclose() (avoiding == precision bugs).
2. Subgroup survival rate calculations using explicit indexed while loops.
3. Confusion matrix evaluation (Accuracy, Precision, Recall, Specificity, and F1-Score).
4. Profiling the 86 km/h speed cliff and restraint combinations.
"""

import math
from typing import Dict, List
from domain_models import CrashRecord, EvaluationMetrics


def floats_are_equal(val_a: float, val_b: float) -> bool:
    """Pure Function: Compares two floats safely using math.isclose instead of ==."""
    return math.isclose(val_a, val_b, rel_tol=1e-9, abs_tol=1e-9)


def compute_survival_rate(records: List[CrashRecord]) -> float:
    """Pure Function: Calculates the proportion of survivors (0.0 to 1.0) using a while loop."""
    n_total: int = len(records)
    if n_total == 0:
        return 0.0

    survived_count: int = 0
    i: int = 0
    while i < n_total:
        if records[i].survived == 1:
            survived_count += 1
        i += 1

    return survived_count / n_total


def compute_gini_impurity(records: List[CrashRecord]) -> float:
    """Pure Function: Calculates simple Gini impurity 2 * p * (1 - p) to measure group mix.
    A pure group (all survived or all fatal) has impurity 0.0; a 50/50 mix has 0.50."""
    n_total: int = len(records)
    if n_total == 0:
        return 0.0
    p_surv: float = compute_survival_rate(records)
    p_fatal: float = 1.0 - p_surv
    return 2.0 * p_surv * p_fatal


def compute_split_improvement(
    parent_records: List[CrashRecord],
    left_records: List[CrashRecord],
    right_records: List[CrashRecord],
) -> float:
    """Pure Function: Measures how much a threshold split reduces group impurity."""
    n_parent: int = len(parent_records)
    n_left: int = len(left_records)
    n_right: int = len(right_records)
    if n_parent == 0 or n_left == 0 or n_right == 0:
        return 0.0

    parent_impurity: float = compute_gini_impurity(parent_records)
    left_weight: float = n_left / n_parent
    right_weight: float = n_right / n_parent
    weighted_child_impurity: float = (
        left_weight * compute_gini_impurity(left_records)
        + right_weight * compute_gini_impurity(right_records)
    )
    improvement: float = parent_impurity - weighted_child_impurity
    if improvement <= 0.0 or floats_are_equal(improvement, 0.0):
        return 0.0
    return improvement


def evaluate_predictions(
    model_name: str,
    evaluation_mode: str,
    test_records: List[CrashRecord],
    predicted_probs: List[float],
    threshold: float = 0.5,
    train_time_ms: float = 0.0,
) -> EvaluationMetrics:
    """Pure Function: Counts True Positives (TP), False Positives (FP), True Negatives (TN),
    and False Negatives (FN) using an indexed while loop and returns EvaluationMetrics."""
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    n_test: int = len(test_records)
    i: int = 0
    while i < n_test:
        actual: int = test_records[i].survived
        prob: float = predicted_probs[i]
        pred: int = 1 if (prob > threshold or floats_are_equal(prob, threshold)) else 0

        if actual == 1 and pred == 1:
            tp += 1
        elif actual == 0 and pred == 1:
            fp += 1
        elif actual == 0 and pred == 0:
            tn += 1
        else:
            fn += 1
        i += 1

    return EvaluationMetrics(
        model_name=model_name,
        evaluation_mode=evaluation_mode,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        train_time_ms=train_time_ms,
    )


def compute_kinetic_cliff_summary(records: List[CrashRecord]) -> Dict[str, float]:
    """Pure Function: Counts survival percentages above 86 km/h vs. at or below 86 km/h."""
    high_speed_total: int = 0
    high_speed_surv: int = 0
    low_speed_total: int = 0
    low_speed_surv: int = 0

    # Subgroup counters within the survivable speed zone (<= 86 km/h)
    nn_tot: int = 0
    nn_surv: int = 0
    yn_tot: int = 0
    yn_surv: int = 0
    ny_tot: int = 0
    ny_surv: int = 0
    yy_tot: int = 0
    yy_surv: int = 0

    i: int = 0
    n_rec: int = len(records)
    while i < n_rec:
        rec: CrashRecord = records[i]
        if rec.speed_of_impact > 86:
            high_speed_total += 1
            if rec.survived == 1:
                high_speed_surv += 1
        else:
            low_speed_total += 1
            if rec.survived == 1:
                low_speed_surv += 1

            if (not rec.helmet_used) and (not rec.seatbelt_used):
                nn_tot += 1
                nn_surv += rec.survived
            elif rec.helmet_used and (not rec.seatbelt_used):
                yn_tot += 1
                yn_surv += rec.survived
            elif (not rec.helmet_used) and rec.seatbelt_used:
                ny_tot += 1
                ny_surv += rec.survived
            else:
                yy_tot += 1
                yy_surv += rec.survived
        i += 1

    return {
        "total_records": float(n_rec),
        "high_speed_total": float(high_speed_total),
        "high_speed_surv_rate": (high_speed_surv / high_speed_total * 100.0) if high_speed_total > 0 else 0.0,
        "low_speed_total": float(low_speed_total),
        "low_speed_surv_rate": (low_speed_surv / low_speed_total * 100.0) if low_speed_total > 0 else 0.0,
        "unrestrained_surv_rate": (nn_surv / nn_tot * 100.0) if nn_tot > 0 else 0.0,
        "helmet_only_surv_rate": (yn_surv / yn_tot * 100.0) if yn_tot > 0 else 0.0,
        "seatbelt_only_surv_rate": (ny_surv / ny_tot * 100.0) if ny_tot > 0 else 0.0,
        "both_restraints_surv_rate": (yy_surv / yy_tot * 100.0) if yy_tot > 0 else 0.0,
    }
