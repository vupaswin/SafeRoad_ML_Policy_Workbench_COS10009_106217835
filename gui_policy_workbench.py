"""
Module 6: gui_policy_workbench.py
COS10009 Introduction to Programming — Event-Driven Desktop GUI Workbench (80-100 HD)
Author: Anh Vu Pham (Student ID: 106217835)
"""

import tkinter as tk
from tkinter import ttk
from domain_models import AppState, CrashRecord, EvaluationMetrics, PolicyScenario
from io_persistence_lib import save_policy_simulation_to_sqlite
from ml_policy_engine import (
    binary_search_speed_range,
    simulate_policy_intervention,
)


class SafeRoadWorkbenchGUI:
    """Controller & View class managing the 4-Tab desktop GUI without global variables."""

    def __init__(self, root_window: tk.Tk, app_state: AppState) -> None:
        self.root: tk.Tk = root_window
        self.state: AppState = app_state
        self.root.title(
            "SafeRoad-ML Policy Workbench v1.0 | COS10009 Custom Program & HD Band 2 (ID: 106217835)"
        )
        self.root.geometry("1260x760")
        self.root.configure(bg="#F8FAFC")

        self.speed_cap_var: tk.IntVar = tk.IntVar(value=80)
        self.seatbelt_rate_var: tk.DoubleVar = tk.DoubleVar(value=95.0)
        self.helmet_rate_var: tk.DoubleVar = tk.DoubleVar(value=90.0)
        self.classifier_var: tk.StringVar = tk.StringVar(value="DecisionTree")
        self.search_min_var: tk.IntVar = tk.IntVar(value=80)
        self.search_max_var: tk.IntVar = tk.IntVar(value=86)

        self._build_header_bar()
        self.content_frame: tk.Frame = tk.Frame(self.root, bg="#F8FAFC")
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)
        self._build_status_bar()
        self.switch_screen("SCREEN_POLICY_SIM")

    def _build_header_bar(self) -> None:
        """Procedure: Builds the top navigation bar with 4 screen tab buttons."""
        header = tk.Frame(self.root, bg="#1E293B", height=54)
        header.pack(fill=tk.X)

        title_lbl = tk.Label(
            header,
            text="SafeRoad-ML Policy Workbench | COS10009 Custom Program (Anh Vu Pham - 106217835)",
            fg="white",
            bg="#1E293B",
            font=("Helvetica", 12, "bold"),
        )
        title_lbl.pack(side=tk.LEFT, padx=14, pady=12)

        tabs = [
            ("1. Data Audit & Search", "SCREEN_DATA_AUDIT"),
            ("2. Pure-Python Models (80/20)", "SCREEN_ML_TRAINING"),
            ("3. Policy Simulator (N=599)", "SCREEN_POLICY_SIM"),
            ("4. Task 11.3 O(N) Lab", "SCREEN_RESEARCH_LAB"),
        ]
        i: int = len(tabs) - 1
        while i >= 0:
            label_txt, scr_code = tabs[i]
            btn = tk.Button(
                header,
                text=label_txt,
                bg="#0F4C81",
                fg="white",
                activebackground="#C8102E",
                activeforeground="white",
                font=("Helvetica", 9, "bold"),
                relief=tk.FLAT,
                padx=10,
                pady=6,
                command=lambda code=scr_code: self.switch_screen(code),
            )
            btn.pack(side=tk.RIGHT, padx=5, pady=10)
            i -= 1

    def _build_status_bar(self) -> None:
        """Procedure: Builds the bottom status bar displaying SQLite & memory state."""
        self.status_lbl = tk.Label(
            self.root,
            text=self.state.status_message,
            bg="#E2E8F0",
            fg="#1E293B",
            anchor="w",
            font=("Helvetica", 9),
            padx=12,
            pady=4,
        )
        self.status_lbl.pack(fill=tk.X, side=tk.BOTTOM)

    def switch_screen(self, screen_code: str) -> None:
        """Procedure: Switches AppState.active_screen and redraws the active view."""
        self.state.active_screen = screen_code
        children = self.content_frame.winfo_children()
        i: int = 0
        while i < len(children):
            children[i].destroy()
            i += 1

        if screen_code == "SCREEN_DATA_AUDIT":
            self._render_data_audit_screen()
        elif screen_code == "SCREEN_ML_TRAINING":
            self._render_ml_training_screen()
        elif screen_code == "SCREEN_RESEARCH_LAB":
            self._render_research_lab_screen()
        else:
            self._render_policy_simulator_screen()

    def _render_policy_simulator_screen(self) -> None:
        """Procedure: Renders Tab 3 (Interactive Policy Simulator on Deduplicated N=599 Crashes)."""
        left_panel = tk.LabelFrame(
            self.content_frame,
            text=" Policy Intervention Controls (N=599 Unique) ",
            bg="white",
            fg="#0F4C81",
            font=("Helvetica", 10, "bold"),
            padx=12,
            pady=10,
        )
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        tk.Label(
            left_panel, text="Max Speed Cap Enforcement (km/h):", bg="white", font=("Helvetica", 9, "bold")
        ).pack(anchor="w", pady=(4, 0))
        tk.Scale(
            left_panel,
            from_=40,
            to=119,
            orient=tk.HORIZONTAL,
            variable=self.speed_cap_var,
            bg="white",
            length=240,
        ).pack(anchor="w", pady=(0, 8))

        tk.Label(
            left_panel, text="Mandatory Seatbelt Target (%):", bg="white", font=("Helvetica", 9, "bold")
        ).pack(anchor="w", pady=(4, 0))
        tk.Scale(
            left_panel,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.seatbelt_rate_var,
            bg="white",
            length=240,
        ).pack(anchor="w", pady=(0, 8))

        tk.Label(
            left_panel, text="Mandatory Helmet Target (%):", bg="white", font=("Helvetica", 9, "bold")
        ).pack(anchor="w", pady=(4, 0))
        tk.Scale(
            left_panel,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.helmet_rate_var,
            bg="white",
            length=240,
        ).pack(anchor="w", pady=(0, 8))

        tk.Label(
            left_panel, text="Active Prediction Model:", bg="white", font=("Helvetica", 9, "bold")
        ).pack(anchor="w", pady=(6, 2))
        ttk.Combobox(
            left_panel,
            textvariable=self.classifier_var,
            values=["DecisionTree", "LinearScore"],
            state="readonly",
            width=26,
        ).pack(anchor="w", pady=(0, 12))

        run_btn = tk.Button(
            left_panel,
            text="RUN POLICY SIMULATION & SAVE TO SQLITE",
            bg="#C8102E",
            fg="white",
            font=("Helvetica", 9, "bold"),
            padx=8,
            pady=8,
            command=self._on_run_policy_clicked,
        )
        run_btn.pack(fill=tk.X, pady=6)

        self.kpi_label = tk.Label(
            left_panel,
            text="",
            bg="#F1F5F9",
            fg="#1E293B",
            justify=tk.LEFT,
            font=("Courier", 9),
            padx=8,
            pady=8,
        )
        self.kpi_label.pack(fill=tk.X, pady=8)

        right_panel = tk.LabelFrame(
            self.content_frame,
            text=" 86 km/h Speed Cliff & Policy Simulation Output (Deduplicated Registry N=599) ",
            bg="white",
            fg="#0F4C81",
            font=("Helvetica", 10, "bold"),
            padx=10,
            pady=10,
        )
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.sim_canvas = tk.Canvas(right_panel, bg="white", height=420, highlightthickness=0)
        self.sim_canvas.pack(fill=tk.BOTH, expand=True)
        self._on_run_policy_clicked()

    def _on_run_policy_clicked(self) -> None:
        """Procedure: Executes policy simulation on deduplicated N=599 records and saves to SQLite."""
        cap: int = int(self.speed_cap_var.get())
        seat_r: float = float(self.seatbelt_rate_var.get()) / 100.0
        helm_r: float = float(self.helmet_rate_var.get()) / 100.0
        clf_name: str = self.classifier_var.get()

        scen: PolicyScenario = simulate_policy_intervention(
            cohort=self.state.repository.dedup_records,
            scenario_name=f"GUI Custom ({cap} km/h Cap, Seat={int(seat_r*100)}%, Helm={int(helm_r*100)}%)",
            max_speed_cap=cap,
            target_helmet_rate=helm_r,
            target_seatbelt_rate=seat_r,
            tree_root=self.state.tree_dedup,
            linear_model=self.state.linear_dedup,
            active_classifier=clf_name,
        )
        save_policy_simulation_to_sqlite(self.state.sqlite_db_path, scen)
        self.state.policy_history.append(scen)

        self.kpi_label.configure(
            text=(
                f"Unique Cohort   : {scen.cohort_size} crashes\n"
                f"Baseline Deaths : {scen.baseline_fatalities:.1f}\n"
                f"Policy Deaths   : {scen.projected_fatalities:.1f}\n"
                f"Lives Saved     : +{scen.lives_saved:.1f} survivors\n"
                f"Fatality Drop   : -{scen.fatality_reduction_pct:.1f}%"
            )
        )
        self.status_lbl.configure(
            text=f"Saved scenario '{scen.scenario_name}' to SQLite ({self.state.sqlite_db_path})."
        )
        self._draw_kinetic_cliff_canvas(scen)

    def _draw_kinetic_cliff_canvas(self, scen: PolicyScenario) -> None:
        """Procedure: Draws the 86 km/h speed cliff on the Tkinter Canvas."""
        c = self.sim_canvas
        c.delete("all")
        c.create_rectangle(60, 40, 420, 280, fill="#ECFDF5", outline="#CBD5E1")
        c.create_rectangle(420, 40, 620, 280, fill="#FEF2F2", outline="#CBD5E1")
        c.create_line(420, 35, 420, 285, fill="#C8102E", width=2, dash=(4, 3))
        c.create_text(
            420,
            22,
            text="86 km/h Speed Cliff (0.0% Survival > 86 km/h)",
            fill="#C8102E",
            font=("Helvetica", 9, "bold"),
        )
        c.create_text(
            240, 58, text="Survivable Zone (20-86 km/h): 48.84% Survival", fill="#0D9488", font=("Helvetica", 9, "bold")
        )
        c.create_text(
            520, 58, text="Lethal Zone (87-119 km/h)\n0 / 212 Unique Survivors (0.0%)", fill="#C8102E", font=("Helvetica", 9, "bold")
        )
        c.create_line(70, 110, 260, 125, 418, 155, 422, 275, 610, 275, fill="#0F4C81", width=3)
        c.create_text(
            340,
            325,
            text=(
                f"Active Policy (N=599): Speed Cap = {scen.max_speed_cap} km/h | "
                f"Projected Lives Saved = +{scen.lives_saved:.1f} "
                f"(-{scen.fatality_reduction_pct:.1f}% Fatalities)"
            ),
            fill="#1E293B",
            font=("Helvetica", 11, "bold"),
        )

    def _render_data_audit_screen(self) -> None:
        """Procedure: Renders Tab 1 (Data Audit & O(log N) Binary Search Inspector)."""
        top_box = tk.LabelFrame(
            self.content_frame,
            text=" O(log N) Binary Search Speed-Range Crash Inspector ",
            bg="white",
            fg="#0F4C81",
            font=("Helvetica", 10, "bold"),
            padx=10,
            pady=8,
        )
        top_box.pack(fill=tk.X, pady=(0, 8))

        tk.Label(top_box, text="Min Speed (km/h):", bg="white").pack(side=tk.LEFT, padx=4)
        tk.Entry(top_box, textvariable=self.search_min_var, width=6).pack(side=tk.LEFT, padx=4)
        tk.Label(top_box, text="Max Speed (km/h):", bg="white").pack(side=tk.LEFT, padx=4)
        tk.Entry(top_box, textvariable=self.search_max_var, width=6).pack(side=tk.LEFT, padx=4)

        self.search_info_lbl = tk.Label(top_box, text="", bg="white", fg="#0D9488", font=("Helvetica", 9, "bold"))
        self.search_info_lbl.pack(side=tk.LEFT, padx=16)

        tk.Button(
            top_box,
            text="Run O(log N) Binary Search",
            bg="#0F4C81",
            fg="white",
            command=self._on_binary_search_clicked,
        ).pack(side=tk.RIGHT, padx=6)

        self.audit_list = tk.Listbox(self.content_frame, font=("Courier", 9), height=24)
        self.audit_list.pack(fill=tk.BOTH, expand=True)
        self._on_binary_search_clicked()

    def _on_binary_search_clicked(self) -> None:
        """Procedure: Runs binary_search_speed_range and populates the listbox."""
        min_v: int = int(self.search_min_var.get())
        max_v: int = int(self.search_max_var.get())
        matches, comps = binary_search_speed_range(
            self.state.repository.sorted_by_speed, min_v, max_v
        )
        self.search_info_lbl.configure(
            text=f"Found {len(matches)} unique records in [{min_v}..{max_v} km/h] using {comps} binary comparisons."
        )
        self.audit_list.delete(0, tk.END)
        i: int = 0
        while i < min(len(matches), 120):
            r: CrashRecord = matches[i]
            self.audit_list.insert(
                tk.END,
                f"ID={r.record_id:03d} | Age={r.age:02d} | {r.gender:<6} | "
                f"Speed={r.speed_of_impact:03d} km/h | Helmet={str(r.helmet_used):<5} | "
                f"Seatbelt={str(r.seatbelt_used):<5} | Survived={r.survived}",
            )
            i += 1

    def _render_ml_training_screen(self) -> None:
        """Procedure: Renders Tab 2 (80/20 Split Model Comparison)."""
        box = tk.LabelFrame(
            self.content_frame,
            text=" RQ1 & RQ2: 80/20 Train/Test Split Evaluation (Cross-Fold N=1200 vs. Deduplicated N=599) ",
            bg="white",
            fg="#0F4C81",
            font=("Helvetica", 10, "bold"),
            padx=12,
            pady=12,
        )
        box.pack(fill=tk.BOTH, expand=True)

        txt = tk.Text(box, font=("Courier", 10), bg="#F8FAFC", height=24)
        txt.pack(fill=tk.BOTH, expand=True)
        txt.insert(
            tk.END,
            f"{'Prediction Method':<26} | {'80/20 Split Regime':<26} | {'Acc%':>6} | {'Prec%':>6} | {'Recall%':>7} | {'F1%':>6}\n",
        )
        txt.insert(tk.END, "-" * 92 + "\n")
        i: int = 0
        while i < len(self.state.metrics_list):
            m: EvaluationMetrics = self.state.metrics_list[i]
            txt.insert(
                tk.END,
                f"{m.model_name:<26} | {m.evaluation_mode:<26} | "
                f"{m.accuracy*100:>6.2f} | {m.precision*100:>6.2f} | "
                f"{m.recall_sens*100:>7.2f} | {m.f1_score*100:>6.2f}\n",
            )
            i += 1

    def _render_research_lab_screen(self) -> None:
        """Procedure: Renders Tab 4 (Task 11.3 Algorithmic Lookup Scaling Lab)."""
        box = tk.LabelFrame(
            self.content_frame,
            text=" RQ3: Search Speed Scaling (O(N) vs. O(log N) vs. O(1)) against 16.67 ms 60 FPS Ceiling ",
            bg="white",
            fg="#0F4C81",
            font=("Helvetica", 10, "bold"),
            padx=12,
            pady=12,
        )
        box.pack(fill=tk.BOTH, expand=True)

        txt = tk.Text(box, font=("Courier", 10), bg="#F8FAFC", height=24)
        txt.pack(fill=tk.BOTH, expand=True)
        txt.insert(
            tk.END,
            f"{'Search Algorithm (K=100 Queries)':<28} | {'Class':<8} | {'N':>6} | {'Time(ms)':>9} | {'Mem(KB)':>8} | {'60 FPS Check'}\n",
        )
        txt.insert(tk.END, "-" * 90 + "\n")
        i: int = 0
        while i < len(self.state.scaling_benchmarks):
            b = self.state.scaling_benchmarks[i]
            status = "PASS (<16.67ms)" if b.passes_60fps_budget else "LAG (>16.67ms)"
            txt.insert(
                tk.END,
                f"{b.algorithm_name:<28} | {b.complexity_class:<8} | {b.dataset_size_n:>6} | "
                f"{b.elapsed_ms:>9.4f} | {b.peak_memory_kb:>8.1f} | {status}\n",
            )
            i += 1
