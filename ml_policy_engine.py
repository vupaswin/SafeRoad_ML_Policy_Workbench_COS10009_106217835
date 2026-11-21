"""
Module 4: ml_policy_engine.py
COS10009 Introduction to Programming — Pure-Python Decision Rules, Sorting, Binary Search & Policy Simulator
Author: Anh Vu Pham (Student ID: 106217835)

Implements transparent, first-year-accessible algorithms using COS10009 structured programming:
1. O(N log N) MergeSort ordering CrashRecord lists by speed_of_impact.
2. O(N) Indexed while-loop Linear Search vs. O(log N) Binary Search vs. O(1) Dictionary Lookup.
3. Method 1: Rule-Based Decision Tree (DecisionNode pointer tree splitting on the 86 km/h speed cliff
   and restraint compliance subgroups).
4. Method 2: Weighted Linear Point-Scoring Model (LinearScoreModel adding/subtracting points per feature).
5. Method 3: Single-Factor Subgroup Rate Model (SubgroupRateModel averaging single-attribute survival rates).
6. Traffic Policy Simulator (evaluating speed limit caps and helmet/seatbelt compliance on N=599 unique crashes).
"""

import random
from typing import Dict, List, Optional, Tuple
from domain_models import (
    CrashRecord,
    DecisionNode,
    LinearScoreModel,
    PolicyScenario,
    SubgroupRateModel,
)
from metrics_stats_lib import (
    compute_split_improvement,
    compute_survival_rate,
    floats_are_equal,
)


# ============================================================================
# 1. CUSTOM O(N log N) MERGESORT & O(N) / O(log N) / O(1) SEARCH ALGORITHMS
# ============================================================================
def _merge_by_speed(left_list: List[CrashRecord], right_list: List[CrashRecord]) -> List[CrashRecord]:
    """Pure Helper Function: Merges two sorted CrashRecord lists in O(N) using indexed while loops."""
    merged: List[CrashRecord] = []
    i: int = 0
    j: int = 0
    while i < len(left_list) and j < len(right_list):
        if left_list[i].speed_of_impact <= right_list[j].speed_of_impact:
            merged.append(left_list[i])
            i += 1
        else:
            merged.append(right_list[j])
            j += 1

    while i < len(left_list):
        merged.append(left_list[i])
        i += 1

    while j < len(right_list):
        merged.append(right_list[j])
        j += 1

    return merged


def merge_sort_records_by_speed(records: List[CrashRecord]) -> List[CrashRecord]:
    """Pure Function: Recursively sorts CrashRecord objects by speed_of_impact ascending in O(N log N)."""
    n_len: int = len(records)
    if n_len <= 1:
        return list(records)
    mid: int = n_len // 2
    left_half: List[CrashRecord] = merge_sort_records_by_speed(records[:mid])
    right_half: List[CrashRecord] = merge_sort_records_by_speed(records[mid:])
    return _merge_by_speed(left_half, right_half)


def linear_search_speed_range(
    records: List[CrashRecord],
    min_speed: int,
    max_speed: int,
) -> Tuple[List[CrashRecord], int]:
    """Pure Function: Scans unsorted records one-by-one in O(N) using an indexed while loop."""
    matches: List[CrashRecord] = []
    comparisons: int = 0
    i: int = 0
    n_len: int = len(records)
    while i < n_len:
        comparisons += 1
        spd: int = records[i].speed_of_impact
        if min_speed <= spd <= max_speed:
            matches.append(records[i])
        i += 1
    return (matches, comparisons)


def binary_search_first_speed(
    sorted_records: List[CrashRecord],
    target_speed: int,
) -> Tuple[int, int]:
    """Pure Function: Iterative O(log N) binary search finding the first index with speed >= target_speed."""
    low: int = 0
    high: int = len(sorted_records) - 1
    first_idx: int = len(sorted_records)
    comparisons: int = 0

    while low <= high:
        mid: int = (low + high) // 2
        comparisons += 1
        if sorted_records[mid].speed_of_impact >= target_speed:
            first_idx = mid
            high = mid - 1
        else:
            low = mid + 1

    return (first_idx, comparisons)


def binary_search_speed_range(
    sorted_records: List[CrashRecord],
    min_speed: int,
    max_speed: int,
) -> Tuple[List[CrashRecord], int]:
    """Pure Function: Uses O(log N) binary search to slice records within [min_speed, max_speed]."""
    start_idx, comp_low = binary_search_first_speed(sorted_records, min_speed)
    end_idx, comp_high = binary_search_first_speed(sorted_records, max_speed + 1)
    matches: List[CrashRecord] = sorted_records[start_idx:end_idx]
    return (matches, comp_low + comp_high)


def hash_index_lookup(
    hash_index: Dict[str, CrashRecord],
    tuple_hash_key: str,
) -> Tuple[Optional[CrashRecord], int]:
    """Pure Function: O(1) dictionary key lookup."""
    return (hash_index.get(tuple_hash_key, None), 1)


# ============================================================================
# 2. METHOD 1: RULE-BASED DECISION TREE (DecisionNode Pointer Tree)
# ============================================================================
def _extract_feature_value(record: CrashRecord, feature_name: str) -> float:
    """Pure Helper Function: Reads the numerical value of a feature from a CrashRecord."""
    if feature_name == "speed_of_impact":
        return float(record.speed_of_impact)
    elif feature_name == "seatbelt_used":
        return 1.0 if record.seatbelt_used else 0.0
    elif feature_name == "helmet_used":
        return 1.0 if record.helmet_used else 0.0
    elif feature_name == "age":
        return float(record.age)
    else:
        return 1.0 if record.gender == "Female" else 0.0


def _partition_by_threshold(
    records: List[CrashRecord],
    feature_name: str,
    threshold: float,
) -> Tuple[List[CrashRecord], List[CrashRecord]]:
    """Pure Helper Function: Splits a list of CrashRecord objects into left (<=) and right (>) lists."""
    left_list: List[CrashRecord] = []
    right_list: List[CrashRecord] = []
    i: int = 0
    while i < len(records):
        val: float = _extract_feature_value(records[i], feature_name)
        if val <= threshold:
            left_list.append(records[i])
        else:
            right_list.append(records[i])
        i += 1
    return (left_list, right_list)


def build_decision_tree(
    records: List[CrashRecord],
    depth: int = 0,
    max_depth: int = 5,
    min_samples_split: int = 6,
) -> DecisionNode:
    """Pure Function: Builds a binary DecisionNode tree by testing candidate thresholds with while loops."""
    n_samples: int = len(records)
    surv_prob: float = compute_survival_rate(records)

    if (
        n_samples < min_samples_split
        or depth >= max_depth
        or floats_are_equal(surv_prob, 0.0)
        or floats_are_equal(surv_prob, 1.0)
    ):
        return DecisionNode(
            is_leaf=True,
            prob_survived=surv_prob,
            sample_count=n_samples,
        )

    candidate_splits: List[Tuple[str, float]] = [
        ("speed_of_impact", 86.5),
        ("speed_of_impact", 75.5),
        ("speed_of_impact", 62.5),
        ("speed_of_impact", 48.5),
        ("speed_of_impact", 34.5),
        ("seatbelt_used", 0.5),
        ("helmet_used", 0.5),
        ("age", 30.5),
        ("age", 45.5),
        ("age", 55.5),
        ("gender", 0.5),
    ]

    best_score: float = 0.0
    best_feature: str = ""
    best_threshold: float = 0.0
    best_left: List[CrashRecord] = []
    best_right: List[CrashRecord] = []

    idx: int = 0
    while idx < len(candidate_splits):
        feat_name, thresh = candidate_splits[idx]
        left_part, right_part = _partition_by_threshold(records, feat_name, thresh)
        if len(left_part) >= 2 and len(right_part) >= 2:
            score: float = compute_split_improvement(records, left_part, right_part)
            # Prioritize the empirical 86.5 km/h speed cliff at the root node
            if depth == 0 and feat_name == "speed_of_impact" and floats_are_equal(thresh, 86.5):
                score += 0.25
            if score > best_score:
                best_score = score
                best_feature = feat_name
                best_threshold = thresh
                best_left = left_part
                best_right = right_part
        idx += 1

    if best_feature == "" or floats_are_equal(best_score, 0.0):
        return DecisionNode(
            is_leaf=True,
            prob_survived=surv_prob,
            sample_count=n_samples,
        )

    node = DecisionNode(
        feature_name=best_feature,
        threshold=best_threshold,
        split_improvement=best_score,
        is_leaf=False,
        prob_survived=surv_prob,
        sample_count=n_samples,
    )
    node.left_child = build_decision_tree(best_left, depth + 1, max_depth, min_samples_split)
    node.right_child = build_decision_tree(best_right, depth + 1, max_depth, min_samples_split)
    return node


def predict_tree_prob(root_node: Optional[DecisionNode], record: CrashRecord) -> float:
    """Pure Function: Walks down the DecisionNode tree pointers to return the predicted survival rate."""
    curr: Optional[DecisionNode] = root_node
    while curr is not None and (not curr.is_leaf):
        val: float = _extract_feature_value(record, curr.feature_name)
        if val <= curr.threshold:
            curr = curr.left_child
        else:
            curr = curr.right_child
    return curr.prob_survived if curr is not None else 0.0


# ============================================================================
# 3. METHOD 2: WEIGHTED LINEAR POINT-SCORING MODEL (Straight-Line Baseline)
# ============================================================================
def _extract_linear_features(record: CrashRecord) -> List[float]:
    """Pure Helper Function: Scales record attributes into simple numerical inputs."""
    speed_scaled: float = (record.speed_of_impact - 65.0) / 50.0
    seatbelt_val: float = 1.0 if record.seatbelt_used else 0.0
    helmet_val: float = 1.0 if record.helmet_used else 0.0
    age_scaled: float = (record.age - 43.0) / 50.0
    gender_val: float = 1.0 if record.gender == "Female" else 0.0
    return [speed_scaled, seatbelt_val, helmet_val, age_scaled, gender_val]


def predict_linear_score_prob(model: LinearScoreModel, record: CrashRecord) -> float:
    """Pure Function: Computes a clamped linear survival score between 0.0 and 1.0.
    Demonstrates why a straight-line formula makes errors across the hard 86 km/h cliff."""
    feats: List[float] = _extract_linear_features(record)
    score: float = model.base_score
    j: int = 0
    while j < len(model.weights):
        score += model.weights[j] * feats[j]
        j += 1
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


def fit_linear_score_model(
    train_records: List[CrashRecord],
    learning_rate: float = 0.05,
    max_epochs: int = 300,
) -> LinearScoreModel:
    """Pure Function: Fits linear point weights using a simple error-correction while loop."""
    model = LinearScoreModel(learning_rate=learning_rate, max_epochs=max_epochs)
    n_train: int = len(train_records)
    if n_train == 0:
        return model

    model.base_score = 0.32
    epoch: int = 0
    while epoch < max_epochs:
        grad_b0: float = 0.0
        grad_w: List[float] = [0.0, 0.0, 0.0, 0.0, 0.0]
        total_abs_err: float = 0.0

        i: int = 0
        while i < n_train:
            rec: CrashRecord = train_records[i]
            feats: List[float] = _extract_linear_features(rec)
            pred_p: float = predict_linear_score_prob(model, rec)
            error: float = float(rec.survived) - pred_p
            total_abs_err += abs(error)

            grad_b0 += error
            k: int = 0
            while k < 5:
                grad_w[k] += error * feats[k]
                k += 1
            i += 1

        model.base_score += learning_rate * (grad_b0 / n_train)
        k = 0
        while k < 5:
            model.weights[k] += learning_rate * (grad_w[k] / n_train)
            k += 1

        if epoch % 50 == 0 or epoch == max_epochs - 1:
            model.error_history.append(total_abs_err / n_train)
        epoch += 1

    model.weight_summary = {
        "speed_penalty": round(model.weights[0], 3),
        "seatbelt_bonus": round(model.weights[1], 3),
        "helmet_bonus": round(model.weights[2], 3),
    }
    return model


# ============================================================================
# 4. METHOD 3: SINGLE-FACTOR SUBGROUP RATE BASELINE
# ============================================================================
def fit_subgroup_rate_model(train_records: List[CrashRecord]) -> SubgroupRateModel:
    """Pure Function: Computes simple historical survival rates for each individual attribute."""
    model = SubgroupRateModel()
    n_total: int = len(train_records)
    if n_total == 0:
        return model

    surv_total: int = 0
    low_spd_tot: int = 0
    low_spd_surv: int = 0
    high_spd_tot: int = 0
    high_spd_surv: int = 0

    helm_tot: int = 0
    helm_surv: int = 0
    no_helm_tot: int = 0
    no_helm_surv: int = 0

    seat_tot: int = 0
    seat_surv: int = 0
    no_seat_tot: int = 0
    no_seat_surv: int = 0

    i: int = 0
    while i < n_total:
        r: CrashRecord = train_records[i]
        surv_total += r.survived
        if r.speed_of_impact <= 75:
            low_spd_tot += 1
            low_spd_surv += r.survived
        else:
            high_spd_tot += 1
            high_spd_surv += r.survived

        if r.helmet_used:
            helm_tot += 1
            helm_surv += r.survived
        else:
            no_helm_tot += 1
            no_helm_surv += r.survived

        if r.seatbelt_used:
            seat_tot += 1
            seat_surv += r.survived
        else:
            no_seat_tot += 1
            no_seat_surv += r.survived
        i += 1

    model.overall_surv_rate = surv_total / n_total
    model.factor_rates["low_speed_rate"] = low_spd_surv / low_spd_tot if low_spd_tot > 0 else 0.5
    model.factor_rates["high_speed_rate"] = high_spd_surv / high_spd_tot if high_spd_tot > 0 else 0.1
    model.factor_rates["helmet_yes_rate"] = helm_surv / helm_tot if helm_tot > 0 else 0.5
    model.factor_rates["helmet_no_rate"] = no_helm_surv / no_helm_tot if no_helm_tot > 0 else 0.3
    model.factor_rates["seatbelt_yes_rate"] = seat_surv / seat_tot if seat_tot > 0 else 0.5
    model.factor_rates["seatbelt_no_rate"] = no_seat_surv / no_seat_tot if no_seat_tot > 0 else 0.3
    return model


def predict_subgroup_rate_prob(model: SubgroupRateModel, record: CrashRecord) -> float:
    """Pure Function: Averages the single-factor survival rates (speed, helmet, seatbelt).
    Shows why averaging factors independently fails when speed > 86 km/h overrides restraints."""
    r_spd: float = (
        model.factor_rates.get("low_speed_rate", 0.5)
        if record.speed_of_impact <= 75
        else model.factor_rates.get("high_speed_rate", 0.1)
    )
    r_helm: float = (
        model.factor_rates.get("helmet_yes_rate", 0.5)
        if record.helmet_used
        else model.factor_rates.get("helmet_no_rate", 0.3)
    )
    r_seat: float = (
        model.factor_rates.get("seatbelt_yes_rate", 0.5)
        if record.seatbelt_used
        else model.factor_rates.get("seatbelt_no_rate", 0.3)
    )
    return (r_spd + r_helm + r_seat) / 3.0


# ============================================================================
# 5. TRAFFIC POLICY INTERVENTION SIMULATOR (Evaluated on N=599 Unique Crashes)
# ============================================================================
def simulate_policy_intervention(
    cohort: List[CrashRecord],
    scenario_name: str,
    max_speed_cap: int,
    target_helmet_rate: float,
    target_seatbelt_rate: float,
    tree_root: Optional[DecisionNode],
    linear_model: Optional[LinearScoreModel],
    active_classifier: str = "DecisionTree",
) -> PolicyScenario:
    """Pure Function: Simulates a speed-limit cap and mandatory restraint policy across a crash cohort."""
    scenario = PolicyScenario(
        scenario_name=scenario_name,
        max_speed_cap=max_speed_cap,
        target_helmet_rate=target_helmet_rate,
        target_seatbelt_rate=target_seatbelt_rate,
        classifier_used=active_classifier,
    )
    n_cohort: int = len(cohort)
    scenario.cohort_size = n_cohort
    if n_cohort == 0:
        return scenario

    rng = random.Random(106217835)
    base_surv_sum: float = 0.0
    proj_surv_sum: float = 0.0

    i: int = 0
    while i < n_cohort:
        orig: CrashRecord = cohort[i]
        if active_classifier == "LinearScore" and linear_model is not None:
            p_base: float = predict_linear_score_prob(linear_model, orig)
        else:
            p_base = predict_tree_prob(tree_root, orig)
        base_surv_sum += p_base

        # Construct modified crash record under the proposed speed cap and restraint law
        capped_speed: int = (
            orig.speed_of_impact if orig.speed_of_impact <= max_speed_cap else max_speed_cap
        )
        new_helmet: bool = orig.helmet_used or (rng.random() < target_helmet_rate)
        new_seatbelt: bool = orig.seatbelt_used or (rng.random() < target_seatbelt_rate)

        modified_rec = CrashRecord(
            record_id=orig.record_id,
            age=orig.age,
            gender=orig.gender,
            speed_of_impact=capped_speed,
            helmet_used=new_helmet,
            seatbelt_used=new_seatbelt,
            survived=orig.survived,
            source_file=orig.source_file,
        )

        if active_classifier == "LinearScore" and linear_model is not None:
            p_proj: float = predict_linear_score_prob(linear_model, modified_rec)
        else:
            p_proj = predict_tree_prob(tree_root, modified_rec)
        proj_surv_sum += p_proj
        i += 1

    scenario.baseline_survivors = base_surv_sum
    scenario.baseline_fatalities = float(n_cohort) - base_surv_sum
    scenario.projected_survivors = proj_surv_sum
    scenario.projected_fatalities = float(n_cohort) - proj_surv_sum
    scenario.lives_saved = max(0.0, proj_surv_sum - base_surv_sum)
    if scenario.baseline_fatalities > 0.0:
        scenario.fatality_reduction_pct = (
            scenario.lives_saved / scenario.baseline_fatalities
        ) * 100.0
    return scenario
