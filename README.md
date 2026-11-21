# SafeRoad-ML Policy Workbench v1.0 (COS10009 Custom Program & HD Band 2 Project)

- **Student Author**: Anh Vu Pham (`106217835`)
- **Unit**: COS10009 — Introduction to Programming (Semester September 2026, 12.5 Credit Points)
- **Campus**: Swinburne University of Technology — Swinburne Vietnam (Can Tho Campus, Room Sydney 308)
- **Unit Coordinator & Tutor**: Mr. Quach Huu Nghia (`hquach@swin.edu.au`)
- **Target Grade Tier**: **High Distinction Band 2 (95–100 Marks)** (Tasks 6.3D, 9.2D/HD, 9.3HD, and 11.3HD)

---

## 1. Project Overview
**SafeRoad-ML Policy Workbench** is a standalone, event-driven Python desktop application and empirical research testbed designed for traffic police analysts and transport policy makers. It ingests multi-format collision registries (`.csv` comma-delimited and `.txt` tab-delimited logs) alongside a local relational database (`sqlite3`), trains three pure-Python white-box Machine Learning classifiers from scratch, and simulates counterfactual traffic safety policies (speed-limit enforcement caps and mandatory helmet/seatbelt compliance rates).

---

## 2. Package Architecture (4-Tier Layered Design)
1. `domain_models.py` — Composite Plain-Old-Data (POD) classes (`CrashRecord`, `DecisionNode`, `LogisticModel`, `NaiveBayesModel`, `EvaluationMetrics`, `PolicyScenario`, `BenchmarkRecord`, `CrashRepository`, `AppState`) allocated on the Dynamic Heap and passed via **Pass-by-Object-Reference** (zero global variables).
2. `io_persistence_lib.py` — Custom reusable persistence library providing sequential `.readline().strip()` multi-delimiter parsing, safe in-place indexed `while`-loop filtering, $O(1)$ dictionary hash deduplication, Fisher-Yates deterministic shuffling, and `sqlite3` database transactions (`data/traffic_policy.db`).
3. `metrics_stats_lib.py` — Custom reusable statistical library providing floating-point safe comparisons (`math.isclose`), Shannon Entropy, Information Gain Ratio, Gaussian PDF, and confusion matrix evaluation.
4. `ml_policy_engine.py` — Custom $O(N\log N)$ MergeSort, $O(N)$ Linear Search, $O(\log N)$ Binary Search, $O(1)$ Dictionary Hash Lookup, three pure-Python Machine Learning classifiers (**Recursive Decision Tree**, **Iterative Logistic Regression**, and **Gaussian/Categorical Naïve Bayes**), and the **Traffic Policy Counterfactual Simulator**.
5. `gui_policy_workbench.py` — 4-Tab Tkinter Desktop GUI Workbench implementing the **Model-View-Controller (MVC)**, **State**, and **Strategy** design patterns.
6. `benchmark_runner.py` — Task 11.3 HD Band 2 empirical research runner testing RQ1 (cross-fold duplicate leakage vs. hash deduplication), RQ2 ($86\text{ km/h}$ kinetic survival cliff partitioning), and RQ3 ($O(N)$ vs. $O(\log N)$ vs. $O(1)$ query scaling against a $16.67\text{ ms}$ 60 FPS GUI frame budget).
7. `test_suite.py` — Automated `unittest` verification suite (ULO 1).
8. `main.py` — Application entry point supporting interactive GUI mode and headless CLI benchmark mode.

---

## 3. Quick Start Commands
```bash
# 1. Run the Automated Unit Test Suite (ULO 1 Verification)
python3 test_suite.py

# 2. Run the Full Empirical Benchmark Suite & Populate SQLite Database (Headless Mode)
python3 main.py --headless

# 3. Launch the Interactive 4-Tab Desktop GUI Workbench
python3 main.py
```
