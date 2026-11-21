"""
Module 3: io_persistence_lib.py
COS10009 Introduction to Programming — Custom Multi-Format File I/O & SQLite Persistence Library (73D/77D/83HD)
Author: Anh Vu Pham (Student ID: 106217835)

Handles sequential file parsing (.readline().strip()) for comma-delimited (.csv) and tab-delimited (.txt)
police crash logs, in-place indexed while-loop validation, O(1) dictionary hash deduplication,
Fisher-Yates deterministic in-place shuffling, and relational SQLite3 database storage.
"""

import json
import os
import random
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Tuple
from domain_models import (
    CrashRecord,
    CrashRepository,
    EvaluationMetrics,
    PolicyScenario,
)


def parse_single_crash_line(
    line_str: str,
    delimiter: str,
    record_id: int,
    source_file: str,
) -> CrashRecord:
    """Pure Function: Parses a single sanitized line into a CrashRecord object."""
    tokens: List[str] = line_str.split(delimiter)
    age_val: int = int(tokens[0].strip())
    gender_val: str = tokens[1].strip()
    speed_val: int = int(tokens[2].strip())
    helmet_bool: bool = tokens[3].strip().lower() == "yes"
    seatbelt_bool: bool = tokens[4].strip().lower() == "yes"
    survived_int: int = 1 if tokens[5].strip().lower() == "yes" else 0

    return CrashRecord(
        record_id=record_id,
        age=age_val,
        gender=gender_val,
        speed_of_impact=speed_val,
        helmet_used=helmet_bool,
        seatbelt_used=seatbelt_bool,
        survived=survived_int,
        source_file=source_file,
    )


def read_delimited_crash_file(
    filepath: str,
    delimiter: str,
    starting_id: int,
    source_tag: str,
) -> List[CrashRecord]:
    """Pure Function: Reads a delimited crash file sequentially using .readline().strip() and a while loop."""
    records: List[CrashRecord] = []
    if not os.path.exists(filepath):
        return records

    with open(filepath, "r", encoding="utf-8") as file_handle:
        # Read header line first
        header_line: str = file_handle.readline().strip()
        if header_line == "":
            return records

        current_id: int = starting_id
        raw_line: str = file_handle.readline()
        while raw_line != "":
            clean_line: str = raw_line.strip()
            if clean_line != "":
                rec: CrashRecord = parse_single_crash_line(
                    line_str=clean_line,
                    delimiter=delimiter,
                    record_id=current_id,
                    source_file=source_tag,
                )
                records.append(rec)
                current_id += 1
            raw_line = file_handle.readline()

    return records


def filter_invalid_records_in_place(records: List[CrashRecord]) -> int:
    """Procedure-like Filter (returns count removed): Safely removes out-of-bound records in-place.
    Demonstrates the COS10009 safe while-loop deletion pattern (incrementing i only in else)."""
    removed_count: int = 0
    i: int = 0
    while i < len(records):
        rec: CrashRecord = records[i]
        valid_age: bool = 18 <= rec.age <= 69
        valid_speed: bool = 20 <= rec.speed_of_impact <= 119
        valid_gender: bool = rec.gender in ("Female", "Male")
        if (not valid_age) or (not valid_speed) or (not valid_gender):
            records.pop(i)
            removed_count += 1
        else:
            i += 1
    return removed_count


def deduplicate_records_hash(
    raw_records: List[CrashRecord],
) -> Tuple[List[CrashRecord], Dict[str, CrashRecord]]:
    """Pure Function: Deduplicates crash records in O(N) total time (O(1) per lookup) using a dict hash index."""
    unique_list: List[CrashRecord] = []
    hash_index: Dict[str, CrashRecord] = {}

    i: int = 0
    n_raw: int = len(raw_records)
    new_id: int = 1
    while i < n_raw:
        rec: CrashRecord = raw_records[i]
        if rec.tuple_hash not in hash_index:
            dedup_rec = CrashRecord(
                record_id=new_id,
                age=rec.age,
                gender=rec.gender,
                speed_of_impact=rec.speed_of_impact,
                helmet_used=rec.helmet_used,
                seatbelt_used=rec.seatbelt_used,
                survived=rec.survived,
                source_file=rec.source_file,
            )
            hash_index[dedup_rec.tuple_hash] = dedup_rec
            unique_list.append(dedup_rec)
            new_id += 1
        i += 1

    return (unique_list, hash_index)


def fisher_yates_shuffle_in_place(records: List[CrashRecord], seed_val: int = 42) -> None:
    """Procedure: Deterministic Fisher-Yates in-place shuffle using an indexed while loop."""
    rng = random.Random(seed_val)
    i: int = len(records) - 1
    while i > 0:
        j: int = rng.randint(0, i)
        temp: CrashRecord = records[i]
        records[i] = records[j]
        records[j] = temp
        i -= 1


def build_stratified_split(
    records: List[CrashRecord],
    train_surv_target: int,
    train_fatal_target: int,
    seed_val: int = 42,
) -> Tuple[List[CrashRecord], List[CrashRecord]]:
    """Pure Function: Partitions records into stratified balanced training and holdout test sets."""
    surv_pool: List[CrashRecord] = []
    fatal_pool: List[CrashRecord] = []

    i: int = 0
    while i < len(records):
        if records[i].survived == 1:
            surv_pool.append(records[i])
        else:
            fatal_pool.append(records[i])
        i += 1

    fisher_yates_shuffle_in_place(surv_pool, seed_val)
    fisher_yates_shuffle_in_place(fatal_pool, seed_val + 1)

    n_surv_train: int = min(train_surv_target, len(surv_pool))
    n_fatal_train: int = min(train_fatal_target, len(fatal_pool))

    train_set: List[CrashRecord] = []
    test_set: List[CrashRecord] = []

    idx: int = 0
    while idx < len(surv_pool):
        if idx < n_surv_train:
            train_set.append(surv_pool[idx])
        else:
            test_set.append(surv_pool[idx])
        idx += 1

    idx = 0
    while idx < len(fatal_pool):
        if idx < n_fatal_train:
            train_set.append(fatal_pool[idx])
        else:
            test_set.append(fatal_pool[idx])
        idx += 1

    fisher_yates_shuffle_in_place(train_set, seed_val + 2)
    fisher_yates_shuffle_in_place(test_set, seed_val + 3)
    return (train_set, test_set)


def load_json_config(config_path: str) -> Dict[str, Any]:
    """Pure Function: Reads external JSON configuration file for data-driven design."""
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def init_sqlite_database(db_path: str) -> None:
    """Procedure: Initializes the local SQLite3 relational database schema (traffic_policy.db)."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS crash_records (
            record_id INTEGER PRIMARY KEY,
            tuple_hash TEXT UNIQUE,
            age INTEGER,
            gender TEXT,
            speed_of_impact INTEGER,
            helmet_used INTEGER,
            seatbelt_used INTEGER,
            survived INTEGER,
            source_file TEXT
        )
        """
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_crash_speed ON crash_records(speed_of_impact)"
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS model_benchmarks (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            model_name TEXT,
            dataset_mode TEXT,
            accuracy REAL,
            precision_val REAL,
            recall_val REAL,
            f1_val REAL,
            train_time_ms REAL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS policy_simulations (
            sim_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            scenario_name TEXT,
            classifier_used TEXT,
            speed_cap INTEGER,
            helmet_rate REAL,
            seatbelt_rate REAL,
            baseline_deaths REAL,
            projected_deaths REAL,
            lives_saved REAL
        )
        """
    )
    conn.commit()
    conn.close()


def sync_records_to_sqlite(db_path: str, dedup_records: List[CrashRecord]) -> None:
    """Procedure: Persists deduplicated crash records into the SQLite3 crash_records table."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM crash_records")

    i: int = 0
    while i < len(dedup_records):
        r: CrashRecord = dedup_records[i]
        cursor.execute(
            """
            INSERT OR REPLACE INTO crash_records
            (record_id, tuple_hash, age, gender, speed_of_impact, helmet_used, seatbelt_used, survived, source_file)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                r.record_id,
                r.tuple_hash,
                r.age,
                r.gender,
                r.speed_of_impact,
                1 if r.helmet_used else 0,
                1 if r.seatbelt_used else 0,
                r.survived,
                r.source_file,
            ),
        )
        i += 1

    conn.commit()
    conn.close()


def save_benchmark_run_to_sqlite(db_path: str, metrics: EvaluationMetrics) -> None:
    """Procedure: Logs a model evaluation run into the SQLite3 model_benchmarks table."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    now_str: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        """
        INSERT INTO model_benchmarks
        (timestamp, model_name, dataset_mode, accuracy, precision_val, recall_val, f1_val, train_time_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            now_str,
            metrics.model_name,
            metrics.evaluation_mode,
            round(metrics.accuracy, 4),
            round(metrics.precision, 4),
            round(metrics.recall_sens, 4),
            round(metrics.f1_score, 4),
            round(metrics.train_time_ms, 3),
        ),
    )
    conn.commit()
    conn.close()


def save_policy_simulation_to_sqlite(db_path: str, scenario: PolicyScenario) -> None:
    """Procedure: Saves a counterfactual policy simulation scenario into SQLite3."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    now_str: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        """
        INSERT INTO policy_simulations
        (timestamp, scenario_name, classifier_used, speed_cap, helmet_rate, seatbelt_rate,
         baseline_deaths, projected_deaths, lives_saved)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            now_str,
            scenario.scenario_name,
            scenario.classifier_used,
            scenario.max_speed_cap,
            round(scenario.target_helmet_rate, 3),
            round(scenario.target_seatbelt_rate, 3),
            round(scenario.baseline_fatalities, 2),
            round(scenario.projected_fatalities, 2),
            round(scenario.lives_saved, 2),
        ),
    )
    conn.commit()
    conn.close()


def query_sqlite_record_counts(db_path: str) -> Dict[str, int]:
    """Pure Function: Queries row counts across the three SQLite tables for verification."""
    if not os.path.exists(db_path):
        return {"crash_records": 0, "model_benchmarks": 0, "policy_simulations": 0}
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    counts: Dict[str, int] = {}
    tables: List[str] = ["crash_records", "model_benchmarks", "policy_simulations"]
    i: int = 0
    while i < len(tables):
        t_name: str = tables[i]
        cursor.execute(f"SELECT COUNT(*) FROM {t_name}")
        row = cursor.fetchone()
        counts[t_name] = int(row[0]) if row is not None else 0
        i += 1
    conn.close()
    return counts
