"""
Module 1: domain_models.py
COS10009 Introduction to Programming — Custom Program & HD Band 2 Project
Author: Anh Vu Pham (Student ID: 106217835)
Campus: Swinburne University of Technology (Can Tho Campus)

Defines custom classes for crash records, decision tree nodes, baseline scoring models,
policy simulation scenarios, and the central AppState container (zero global variables).
"""

from typing import Dict, List, Optional


class CrashRecord:
    """Represents a single road accident record with demographic, speed, and restraint fields."""

    def __init__(
        self,
        record_id: int,
        age: int,
        gender: str,
        speed_of_impact: int,
        helmet_used: bool,
        seatbelt_used: bool,
        survived: int,
        source_file: str,
    ) -> None:
        self.record_id: int = record_id
        self.age: int = age
        self.gender: str = gender
        self.speed_of_impact: int = speed_of_impact
        self.helmet_used: bool = helmet_used
        self.seatbelt_used: bool = seatbelt_used
        self.survived: int = survived
        self.source_file: str = source_file
        helm_bit: int = 1 if helmet_used else 0
        seat_bit: int = 1 if seatbelt_used else 0
        # Combined string key used for O(1) dictionary duplicate filtering
        self.tuple_hash: str = (
            f"{age}_{gender}_{speed_of_impact}_{helm_bit}_{seat_bit}_{survived}"
        )


class DecisionNode:
    """Binary decision tree node using left_child and right_child object references."""

    def __init__(
        self,
        feature_name: str = "",
        threshold: float = 0.0,
        split_improvement: float = 0.0,
        is_leaf: bool = True,
        prob_survived: float = 0.0,
        sample_count: int = 0,
    ) -> None:
        self.feature_name: str = feature_name
        self.threshold: float = threshold
        self.split_improvement: float = split_improvement
        self.is_leaf: bool = is_leaf
        self.prob_survived: float = prob_survived
        self.sample_count: int = sample_count
        self.left_child: Optional["DecisionNode"] = None
        self.right_child: Optional["DecisionNode"] = None


class LinearScoreModel:
    """Stores simple additive point weights for the Weighted Linear Point-Scoring Baseline."""

    def __init__(
        self,
        learning_rate: float = 0.05,
        max_epochs: int = 300,
    ) -> None:
        self.base_score: float = 0.35
        # Feature order: [speed_scaled, seatbelt_bit, helmet_bit, age_scaled, gender_bit]
        self.weights: List[float] = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.learning_rate: float = learning_rate
        self.max_epochs: int = max_epochs
        self.error_history: List[float] = []
        self.weight_summary: Dict[str, float] = {}


class SubgroupRateModel:
    """Stores single-factor historical survival rates for the Subgroup Rate Averaging Baseline."""

    def __init__(self) -> None:
        self.overall_surv_rate: float = 0.315
        self.factor_rates: Dict[str, float] = {}


class EvaluationMetrics:
    """Stores confusion matrix counts (TP, FP, TN, FN) and accuracy/precision/recall/F1."""

    def __init__(
        self,
        model_name: str,
        evaluation_mode: str,
        tp: int,
        fp: int,
        tn: int,
        fn: int,
        train_time_ms: float = 0.0,
    ) -> None:
        self.model_name: str = model_name
        self.evaluation_mode: str = evaluation_mode
        self.tp: int = tp
        self.fp: int = fp
        self.tn: int = tn
        self.fn: int = fn
        self.train_time_ms: float = train_time_ms

        total: int = tp + fp + tn + fn
        self.accuracy: float = (tp + tn) / total if total > 0 else 0.0
        self.precision: float = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        self.recall_sens: float = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        self.specificity: float = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        denom_f1: float = self.precision + self.recall_sens
        self.f1_score: float = (
            (2.0 * self.precision * self.recall_sens) / denom_f1 if denom_f1 > 0.0 else 0.0
        )


class PolicyScenario:
    """Represents a what-if traffic policy simulation on the deduplicated crash dataset (N=599)."""

    def __init__(
        self,
        scenario_name: str,
        max_speed_cap: int,
        target_helmet_rate: float,
        target_seatbelt_rate: float,
        classifier_used: str = "Rule-Based Decision Tree",
    ) -> None:
        self.scenario_name: str = scenario_name
        self.max_speed_cap: int = max_speed_cap
        self.target_helmet_rate: float = target_helmet_rate
        self.target_seatbelt_rate: float = target_seatbelt_rate
        self.classifier_used: str = classifier_used
        self.cohort_size: int = 0
        self.baseline_survivors: float = 0.0
        self.baseline_fatalities: float = 0.0
        self.projected_survivors: float = 0.0
        self.projected_fatalities: float = 0.0
        self.lives_saved: float = 0.0
        self.fatality_reduction_pct: float = 0.0


class BenchmarkRecord:
    """Stores a single search timing and memory measurement for Task 11.3 RQ3."""

    def __init__(
        self,
        algorithm_name: str,
        complexity_class: str,
        dataset_size_n: int,
        query_count_k: int,
        elapsed_ms: float,
        peak_memory_kb: float,
        comparisons_count: int,
    ) -> None:
        self.algorithm_name: str = algorithm_name
        self.complexity_class: str = complexity_class
        self.dataset_size_n: int = dataset_size_n
        self.query_count_k: int = query_count_k
        self.elapsed_ms: float = elapsed_ms
        self.peak_memory_kb: float = peak_memory_kb
        self.comparisons_count: int = comparisons_count
        self.passes_60fps_budget: bool = elapsed_ms <= 16.67


class CrashRepository:
    """Holds raw, deduplicated, sorted, and 80/20 train/test split crash collections."""

    def __init__(self) -> None:
        self.raw_records: List[CrashRecord] = []
        self.dedup_records: List[CrashRecord] = []
        self.sorted_by_speed: List[CrashRecord] = []
        self.hash_index: Dict[str, CrashRecord] = {}
        # 80/20 Stratified Split on Cross-Fold Raw Data (N=1,200 -> 960 Train / 240 Test)
        self.leaked_train: List[CrashRecord] = []
        self.leaked_test: List[CrashRecord] = []
        # 80/20 Stratified Split on Deduplicated Unique Data (N=599 -> 479 Train / 120 Test)
        self.dedup_train: List[CrashRecord] = []
        self.dedup_test: List[CrashRecord] = []


class AppState:
    """Central state object passed by reference across all functions and procedures (Zero Globals)."""

    def __init__(self, config_path: str = "config_presets.json") -> None:
        self.config_path: str = config_path
        self.active_screen: str = "SCREEN_POLICY_SIM"
        self.active_dedup_mode: bool = True
        self.active_classifier: str = "DecisionTree"
        self.sqlite_db_path: str = "data/traffic_policy.db"
        self.repository: CrashRepository = CrashRepository()

        # Trained models for both regimes
        self.tree_leaked: Optional[DecisionNode] = None
        self.tree_dedup: Optional[DecisionNode] = None
        self.linear_leaked: Optional[LinearScoreModel] = None
        self.linear_dedup: Optional[LinearScoreModel] = None
        self.subgroup_leaked: Optional[SubgroupRateModel] = None
        self.subgroup_dedup: Optional[SubgroupRateModel] = None

        # Evaluation results & policy logs
        self.metrics_list: List[EvaluationMetrics] = []
        self.policy_history: List[PolicyScenario] = []
        self.scaling_benchmarks: List[BenchmarkRecord] = []
        self.status_message: str = "System initialized. Ready."
