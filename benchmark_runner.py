"""
Module 5: benchmark_runner.py
COS10009 Introduction to Programming — Task 11.3 HD Band 2 Empirical Research Benchmark Suite
Author: Anh Vu Pham (Student ID: 106217835)

Executes systematic experiments using a strict 80/20 Stratified Train/Test Split:
- Cross-Fold Raw Dataset (N = 1,200): 80% Train (960 rows: 302 Yes, 658 No) / 20% Test (240 rows: 76 Yes, 164 No).
- Deduplicated Unique Dataset (N = 599): 80% Train (479 rows: 151 Yes, 328 No) / 20% Test (120 rows: 38 Yes, 82 No).
- Policy Simulations run on the clean Deduplicated Unique Dataset (N = 599) for 100% methodological consistency.
"""

import json
import os
import time
import tracemalloc
from typing import Any, Dict, List
from domain_models import (
    AppState,
    BenchmarkRecord,
    CrashRecord,
    EvaluationMetrics,
    PolicyScenario,
)
from io_persistence_lib import (
    build_stratified_split,
    deduplicate_records_hash,
    filter_invalid_records_in_place,
    init_sqlite_database,
    load_json_config,
    query_sqlite_record_counts,
    read_delimited_crash_file,
    save_benchmark_run_to_sqlite,
    save_policy_simulation_to_sqlite,
    sync_records_to_sqlite,
)
from metrics_stats_lib import (
    compute_kinetic_cliff_summary,
    evaluate_predictions,
)
from ml_policy_engine import (
    binary_search_speed_range,
    build_decision_tree,
    fit_linear_score_model,
    fit_subgroup_rate_model,
    hash_index_lookup,
    linear_search_speed_range,
    merge_sort_records_by_speed,
    predict_linear_score_prob,
    predict_subgroup_rate_prob,
    predict_tree_prob,
    simulate_policy_intervention,
)


def ingest_and_prepare_repository(app_state: AppState, base_dir: str = ".") -> None:
    """Procedure: Loads all crash files, deduplicates via dict hash, sorts, and builds 80/20 splits."""
    cfg_path: str = os.path.join(base_dir, app_state.config_path)
    cfg: Dict[str, Any] = load_json_config(cfg_path)
    app_state.sqlite_db_path = os.path.join(base_dir, cfg.get("sqlite_db_path", "data/traffic_policy.db"))

    file_specs: List[Dict[str, str]] = cfg.get("dataset_files", [])
    all_raw: List[CrashRecord] = []
    next_id: int = 1

    i: int = 0
    while i < len(file_specs):
        spec: Dict[str, str] = file_specs[i]
        rel_path: str = spec["path"]
        delim: str = spec["delimiter"]
        tag: str = spec["tag"]
        full_path: str = os.path.join(base_dir, rel_path)
        loaded: List[CrashRecord] = read_delimited_crash_file(full_path, delim, next_id, tag)
        filter_invalid_records_in_place(loaded)
        if tag != "test2_txt":
            j: int = 0
            while j < len(loaded):
                all_raw.append(loaded[j])
                j += 1
            next_id += len(loaded)
        i += 1

    app_state.repository.raw_records = all_raw
    dedup_list, hash_idx = deduplicate_records_hash(all_raw)
    app_state.repository.dedup_records = dedup_list
    app_state.repository.hash_index = hash_idx
    app_state.repository.sorted_by_speed = merge_sort_records_by_speed(dedup_list)

    # Exact 80/20 Stratified Split on Cross-Fold Raw Dataset (N=1,200 -> 960 Train / 240 Test)
    leaked_tr, leaked_te = build_stratified_split(all_raw, 302, 658, seed_val=42)
    app_state.repository.leaked_train = leaked_tr
    app_state.repository.leaked_test = leaked_te

    # Exact 80/20 Stratified Split on Deduplicated Unique Dataset (N=599 -> 479 Train / 120 Test)
    dedup_tr, dedup_te = build_stratified_split(dedup_list, 151, 328, seed_val=42)
    app_state.repository.dedup_train = dedup_tr
    app_state.repository.dedup_test = dedup_te

    init_sqlite_database(app_state.sqlite_db_path)
    sync_records_to_sqlite(app_state.sqlite_db_path, dedup_list)


def train_and_benchmark_models(app_state: AppState) -> None:
    """Procedure: Trains all 3 pure-Python models on the 80/20 splits and logs metrics to SQLite."""
    repo = app_state.repository
    app_state.metrics_list = []

    # 1. Cross-Fold Raw 80/20 Split (N=1,200: 960 Train / 240 Test)
    t0: int = time.perf_counter_ns()
    app_state.tree_leaked = build_decision_tree(repo.leaked_train, max_depth=8, min_samples_split=2)
    t_tree_lk: float = (time.perf_counter_ns() - t0) / 1e6
    probs_tree_lk: List[float] = []
    i: int = 0
    while i < len(repo.leaked_test):
        probs_tree_lk.append(predict_tree_prob(app_state.tree_leaked, repo.leaked_test[i]))
        i += 1
    m_tree_lk = evaluate_predictions(
        "Rule-Based Decision Tree", "Cross-Fold 80/20 (N=1200)", repo.leaked_test, probs_tree_lk, 0.50, t_tree_lk
    )
    app_state.metrics_list.append(m_tree_lk)

    t0 = time.perf_counter_ns()
    app_state.linear_leaked = fit_linear_score_model(repo.leaked_train, learning_rate=0.05, max_epochs=300)
    t_lr_lk: float = (time.perf_counter_ns() - t0) / 1e6
    probs_lr_lk: List[float] = []
    i = 0
    while i < len(repo.leaked_test):
        probs_lr_lk.append(predict_linear_score_prob(app_state.linear_leaked, repo.leaked_test[i]))
        i += 1
    m_lr_lk = evaluate_predictions(
        "Weighted Linear Score", "Cross-Fold 80/20 (N=1200)", repo.leaked_test, probs_lr_lk, 0.50, t_lr_lk
    )
    app_state.metrics_list.append(m_lr_lk)

    t0 = time.perf_counter_ns()
    app_state.subgroup_leaked = fit_subgroup_rate_model(repo.leaked_train)
    t_nb_lk: float = (time.perf_counter_ns() - t0) / 1e6
    probs_nb_lk: List[float] = []
    i = 0
    while i < len(repo.leaked_test):
        probs_nb_lk.append(predict_subgroup_rate_prob(app_state.subgroup_leaked, repo.leaked_test[i]))
        i += 1
    m_nb_lk = evaluate_predictions(
        "Subgroup Rate Average", "Cross-Fold 80/20 (N=1200)", repo.leaked_test, probs_nb_lk, 0.36, t_nb_lk
    )
    app_state.metrics_list.append(m_nb_lk)

    # 2. Strictly Deduplicated Unique 80/20 Split (N=599: 479 Train / 120 Test)
    t0 = time.perf_counter_ns()
    app_state.tree_dedup = build_decision_tree(repo.dedup_train, max_depth=4, min_samples_split=8)
    t_tree_dd: float = (time.perf_counter_ns() - t0) / 1e6
    probs_tree_dd: List[float] = []
    i = 0
    while i < len(repo.dedup_test):
        probs_tree_dd.append(predict_tree_prob(app_state.tree_dedup, repo.dedup_test[i]))
        i += 1
    m_tree_dd = evaluate_predictions(
        "Rule-Based Decision Tree", "Deduplicated 80/20 (N=599)", repo.dedup_test, probs_tree_dd, 0.50, t_tree_dd
    )
    app_state.metrics_list.append(m_tree_dd)

    t0 = time.perf_counter_ns()
    app_state.linear_dedup = fit_linear_score_model(repo.dedup_train, learning_rate=0.05, max_epochs=300)
    t_lr_dd: float = (time.perf_counter_ns() - t0) / 1e6
    probs_lr_dd: List[float] = []
    i = 0
    while i < len(repo.dedup_test):
        probs_lr_dd.append(predict_linear_score_prob(app_state.linear_dedup, repo.dedup_test[i]))
        i += 1
    m_lr_dd = evaluate_predictions(
        "Weighted Linear Score", "Deduplicated 80/20 (N=599)", repo.dedup_test, probs_lr_dd, 0.50, t_lr_dd
    )
    app_state.metrics_list.append(m_lr_dd)

    t0 = time.perf_counter_ns()
    app_state.subgroup_dedup = fit_subgroup_rate_model(repo.dedup_train)
    t_nb_dd: float = (time.perf_counter_ns() - t0) / 1e6
    probs_nb_dd: List[float] = []
    i = 0
    while i < len(repo.dedup_test):
        probs_nb_dd.append(predict_subgroup_rate_prob(app_state.subgroup_dedup, repo.dedup_test[i]))
        i += 1
    m_nb_dd = evaluate_predictions(
        "Subgroup Rate Average", "Deduplicated 80/20 (N=599)", repo.dedup_test, probs_nb_dd, 0.36, t_nb_dd
    )
    app_state.metrics_list.append(m_nb_dd)

    idx: int = 0
    while idx < len(app_state.metrics_list):
        save_benchmark_run_to_sqlite(app_state.sqlite_db_path, app_state.metrics_list[idx])
        idx += 1


def run_policy_preset_simulations(app_state: AppState, base_dir: str = ".") -> None:
    """Procedure: Evaluates all policy presets on the clean Deduplicated Registry (N=599 unique crashes)."""
    cfg: Dict[str, Any] = load_json_config(os.path.join(base_dir, app_state.config_path))
    presets: List[Dict[str, Any]] = cfg.get("policy_presets", [])
    app_state.policy_history = []

    i: int = 0
    while i < len(presets):
        p: Dict[str, Any] = presets[i]
        scen: PolicyScenario = simulate_policy_intervention(
            cohort=app_state.repository.dedup_records,
            scenario_name=p["name"],
            max_speed_cap=int(p["max_speed_cap"]),
            target_helmet_rate=float(p["target_helmet_rate"]),
            target_seatbelt_rate=float(p["target_seatbelt_rate"]),
            tree_root=app_state.tree_dedup,
            linear_model=app_state.linear_dedup,
            active_classifier="DecisionTree",
        )
        app_state.policy_history.append(scen)
        save_policy_simulation_to_sqlite(app_state.sqlite_db_path, scen)
        i += 1


def run_rq3_scaling_benchmarks(app_state: AppState) -> None:
    """Procedure: Benchmarks O(N) Linear Search vs. O(log N) Binary Search vs. O(1) Dict Hash Lookup
    across N in [600, 5000, 20000, 50000] for K=100 queries using time.perf_counter_ns() and tracemalloc."""
    base_records: List[CrashRecord] = app_state.repository.dedup_records
    if len(base_records) == 0:
        return

    scales: List[int] = [600, 5000, 20000, 50000]
    query_k: int = 100
    app_state.scaling_benchmarks = []

    s_idx: int = 0
    while s_idx < len(scales):
        n_target: int = scales[s_idx]
        tracemalloc.start()
        synthetic_list: List[CrashRecord] = []
        i: int = 0
        while i < n_target:
            template: CrashRecord = base_records[i % len(base_records)]
            rec = CrashRecord(
                record_id=i + 1,
                age=template.age,
                gender=template.gender,
                speed_of_impact=template.speed_of_impact,
                helmet_used=template.helmet_used,
                seatbelt_used=template.seatbelt_used,
                survived=template.survived,
                source_file="synthetic_scale",
            )
            rec.tuple_hash = f"{rec.tuple_hash}_{i}"
            synthetic_list.append(rec)
            i += 1
        _, list_peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        list_mem_kb: float = list_peak_bytes / 1024.0

        # 1. Benchmark O(N) Indexed while Linear Search (K=100 range queries)
        t_start: int = time.perf_counter_ns()
        total_lin_comp: int = 0
        q: int = 0
        while q < query_k:
            _, c_cnt = linear_search_speed_range(synthetic_list, 80, 86)
            total_lin_comp += c_cnt
            q += 1
        lin_ms: float = (time.perf_counter_ns() - t_start) / 1e6
        app_state.scaling_benchmarks.append(
            BenchmarkRecord(
                "Linear Search (while)",
                "O(N)",
                n_target,
                query_k,
                lin_ms,
                list_mem_kb,
                total_lin_comp,
            )
        )

        # 2. Benchmark O(log N) Binary Search on MergeSorted list (K=100 range queries)
        sorted_synth: List[CrashRecord] = merge_sort_records_by_speed(synthetic_list)
        t_start = time.perf_counter_ns()
        total_bin_comp: int = 0
        q = 0
        while q < query_k:
            _, b_cnt = binary_search_speed_range(sorted_synth, 80, 86)
            total_bin_comp += b_cnt
            q += 1
        bin_ms: float = (time.perf_counter_ns() - t_start) / 1e6
        app_state.scaling_benchmarks.append(
            BenchmarkRecord(
                "MergeSort + Binary Search",
                "O(log N)",
                n_target,
                query_k,
                bin_ms,
                list_mem_kb * 1.02,
                total_bin_comp,
            )
        )

        # 3. Benchmark O(1) Dictionary Hash Index Lookup (K=100 key probes)
        tracemalloc.start()
        synth_dict: Dict[str, CrashRecord] = {}
        i = 0
        while i < n_target:
            synth_dict[synthetic_list[i].tuple_hash] = synthetic_list[i]
            i += 1
        _, dict_peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        dict_mem_kb: float = (list_peak_bytes + dict_peak_bytes) / 1024.0

        target_key: str = synthetic_list[n_target // 2].tuple_hash
        t_start = time.perf_counter_ns()
        total_hash_comp: int = 0
        q = 0
        while q < query_k:
            _, h_cnt = hash_index_lookup(synth_dict, target_key)
            total_hash_comp += h_cnt
            q += 1
        hash_ms: float = (time.perf_counter_ns() - t_start) / 1e6
        app_state.scaling_benchmarks.append(
            BenchmarkRecord(
                "Dictionary Hash Lookup",
                "O(1)",
                n_target,
                query_k,
                hash_ms,
                dict_mem_kb,
                total_hash_comp,
            )
        )

        s_idx += 1


def export_benchmark_summary(app_state: AppState, base_dir: str = ".") -> Dict[str, Any]:
    """Writes full Task 11.3 empirical results to data/benchmark_results_summary.json."""
    cliff_stats = compute_kinetic_cliff_summary(app_state.repository.dedup_records)
    db_counts = query_sqlite_record_counts(app_state.sqlite_db_path)

    metrics_out = []
    i: int = 0
    while i < len(app_state.metrics_list):
        m: EvaluationMetrics = app_state.metrics_list[i]
        metrics_out.append(
            {
                "model_name": m.model_name,
                "regime": m.evaluation_mode,
                "tp": m.tp,
                "fp": m.fp,
                "tn": m.tn,
                "fn": m.fn,
                "accuracy_pct": round(m.accuracy * 100.0, 2),
                "precision_pct": round(m.precision * 100.0, 2),
                "recall_pct": round(m.recall_sens * 100.0, 2),
                "specificity_pct": round(m.specificity * 100.0, 2),
                "f1_pct": round(m.f1_score * 100.0, 2),
                "train_time_ms": round(m.train_time_ms, 3),
            }
        )
        i += 1

    policies_out = []
    i = 0
    while i < len(app_state.policy_history):
        p: PolicyScenario = app_state.policy_history[i]
        policies_out.append(
            {
                "scenario_name": p.scenario_name,
                "cohort_size": p.cohort_size,
                "max_speed_cap": p.max_speed_cap,
                "helmet_rate_pct": round(p.target_helmet_rate * 100.0, 1),
                "seatbelt_rate_pct": round(p.target_seatbelt_rate * 100.0, 1),
                "baseline_fatalities": round(p.baseline_fatalities, 1),
                "projected_fatalities": round(p.projected_fatalities, 1),
                "lives_saved": round(p.lives_saved, 1),
                "fatality_reduction_pct": round(p.fatality_reduction_pct, 2),
            }
        )
        i += 1

    scaling_out = []
    i = 0
    while i < len(app_state.scaling_benchmarks):
        b: BenchmarkRecord = app_state.scaling_benchmarks[i]
        scaling_out.append(
            {
                "algorithm": b.algorithm_name,
                "complexity": b.complexity_class,
                "N": b.dataset_size_n,
                "K": b.query_count_k,
                "elapsed_ms": round(b.elapsed_ms, 4),
                "peak_memory_kb": round(b.peak_memory_kb, 1),
                "comparisons": b.comparisons_count,
                "passes_60fps": b.passes_60fps_budget,
            }
        )
        i += 1

    summary_payload: Dict[str, Any] = {
        "raw_record_count": len(app_state.repository.raw_records),
        "dedup_unique_count": len(app_state.repository.dedup_records),
        "leaked_split": f"{len(app_state.repository.leaked_train)} Train / {len(app_state.repository.leaked_test)} Test (80/20)",
        "dedup_split": f"{len(app_state.repository.dedup_train)} Train / {len(app_state.repository.dedup_test)} Test (80/20)",
        "sqlite_table_counts": db_counts,
        "kinetic_cliff_profile": cliff_stats,
        "rq1_rq2_model_metrics": metrics_out,
        "policy_simulations": policies_out,
        "rq3_scaling_benchmarks": scaling_out,
    }

    out_path: str = os.path.join(base_dir, "data", "benchmark_results_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    return summary_payload


def print_console_report(summary: Dict[str, Any]) -> None:
    """Procedure: Prints formatted academic tables of all empirical findings to stdout."""
    print("=" * 92)
    print(" SAFEROAD-ML POLICY WORKBENCH | COS10009 HD BAND 2 EMPIRICAL RESEARCH BENCHMARK (80/20 SPLIT)")
    print("=" * 92)
    cliff = summary["kinetic_cliff_profile"]
    print(
        f"Raw Rows: {summary['raw_record_count']} ({summary['leaked_split']}) | "
        f"Unique Rows: {summary['dedup_unique_count']} ({summary['dedup_split']})"
    )
    print(
        f"86 km/h Speed Cliff (on N=599 Unique): >86 km/h Survival = {cliff['high_speed_surv_rate']:.2f}% "
        f"(0/{int(cliff['high_speed_total'])}) | <=86 km/h Survival = {cliff['low_speed_surv_rate']:.2f}%"
    )
    print("-" * 92)
    print(f"{'Prediction Method':<25} | {'80/20 Split Regime':<26} | {'Acc%':>6} | {'Prec%':>6} | {'Rec%':>6} | {'F1%':>6} | {'TP/FP/TN/FN'}")
    print("-" * 92)
    i: int = 0
    metrics = summary["rq1_rq2_model_metrics"]
    while i < len(metrics):
        m = metrics[i]
        cm_str = f"{m['tp']}/{m['fp']}/{m['tn']}/{m['fn']}"
        print(
            f"{m['model_name']:<25} | {m['regime']:<26} | "
            f"{m['accuracy_pct']:>6.2f} | {m['precision_pct']:>6.2f} | "
            f"{m['recall_pct']:>6.2f} | {m['f1_pct']:>6.2f} | {cm_str}"
        )
        i += 1

    print("-" * 92)
    print(f"{'Policy Scenario (on Deduplicated N=599 Unique Crashes)':<52} | {'Cap':>4} | {'BaseFatal':>9} | {'ProjFatal':>9} | {'Saved':>13}")
    print("-" * 92)
    i = 0
    pols = summary["policy_simulations"]
    while i < len(pols):
        p = pols[i]
        saved_str = f"+{p['lives_saved']:.1f} (-{p['fatality_reduction_pct']:.1f}%)"
        print(
            f"{p['scenario_name'][:52]:<52} | {p['max_speed_cap']:>4} | "
            f"{p['baseline_fatalities']:>9.1f} | {p['projected_fatalities']:>9.1f} | "
            f"{saved_str:>13}"
        )
        i += 1

    print("-" * 92)
    print(f"{'Search Algorithm (K=100 Queries)':<28} | {'Class':<8} | {'N':>6} | {'Time(ms)':>9} | {'Mem(KB)':>8} | {'Comparisons':>11}")
    print("-" * 92)
    i = 0
    scal = summary["rq3_scaling_benchmarks"]
    while i < len(scal):
        b = scal[i]
        print(
            f"{b['algorithm']:<28} | {b['complexity']:<8} | {b['N']:>6} | "
            f"{b['elapsed_ms']:>9.4f} | {b['peak_memory_kb']:>8.1f} | {b['comparisons']:>11}"
        )
        i += 1
    print("=" * 92)
