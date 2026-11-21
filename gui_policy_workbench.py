"""
Module 6: gui_policy_workbench.py
COS10009 Introduction to Programming — Event-Driven Desktop GUI Workbench (80-100 HD)
Author: Anh Vu Pham (Student ID: 106217835)
"""

import time
import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Optional
from domain_models import AppState, CrashRecord, EvaluationMetrics, PolicyScenario
from io_persistence_lib import save_policy_simulation_to_sqlite
from metrics_stats_lib import compute_kinetic_cliff_summary
from ml_policy_engine import (
    binary_search_speed_range,
    predict_linear_score_prob,
    predict_subgroup_rate_prob,
    predict_tree_prob,
    simulate_policy_intervention,
)


class SafeRoadWorkbenchGUI:
    """Controller & View class managing the 4-Tab desktop GUI without global variables."""

    def __init__(self, root_window: tk.Tk, app_state: AppState) -> None:
        self.root: tk.Tk = root_window
        self.state: AppState = app_state
        self.root.title(
            "SafeRoad-ML Policy Workbench v1.0 — COS10009 Custom Program (Anh Vu Pham - 106217835)"
        )
        self.root.geometry("1280x780")
        self.root.minsize(1180, 720)
        self.root.configure(bg="#F1F5F9")

        self.speed_cap_var: tk.IntVar = tk.IntVar(value=80)
        self.seatbelt_rate_var: tk.DoubleVar = tk.DoubleVar(value=95.0)
        self.helmet_rate_var: tk.DoubleVar = tk.DoubleVar(value=90.0)
        self.classifier_var: tk.StringVar = tk.StringVar(value="DecisionTree")
        self.dedup_mode_var: tk.BooleanVar = tk.BooleanVar(value=True)
        self.search_min_var: tk.IntVar = tk.IntVar(value=80)
        self.search_max_var: tk.IntVar = tk.IntVar(value=86)

        self.tab_buttons: Dict[str, tk.Button] = {}
        self._build_header_bar()
        self.content_frame: tk.Frame = tk.Frame(self.root, bg="#F1F5F9")
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=14, pady=10)
        self._build_status_bar()
        self.switch_screen("SCREEN_POLICY_SIM")

    def _build_header_bar(self) -> None:
        """Procedure: Builds the top navigation bar with 4 screen tab buttons."""
        header = tk.Frame(self.root, bg="#1E293B", height=54)
        header.pack(fill=tk.X)

        title_lbl = tk.Label(
            header,
            text="SafeRoad-ML Policy Workbench | COS10009 (ID: 106217835)",
            fg="white",
            bg="#1E293B",
            font=("Helvetica", 13, "bold"),
        )
        title_lbl.pack(side=tk.LEFT, padx=16, pady=12)

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
                relief=tk.SOLID,
                bd=1,
                padx=12,
                pady=6,
                cursor="hand2",
                command=lambda code=scr_code: self.switch_screen(code),
            )
            btn.pack(side=tk.RIGHT, padx=5, pady=10)
            self.tab_buttons[scr_code] = btn
            i -= 1

    def _update_header_tab_styles(self, active_code: str) -> None:
        """Procedure: Highlights the active tab in crimson (#C8102E) with an asterisk."""
        tabs = [
            ("1. Data Audit & Search", "SCREEN_DATA_AUDIT"),
            ("2. Pure-Python Models (80/20)", "SCREEN_ML_TRAINING"),
            ("3. Policy Simulator (N=599)", "SCREEN_POLICY_SIM"),
            ("4. Task 11.3 O(N) Lab", "SCREEN_RESEARCH_LAB"),
        ]
        i: int = 0
        while i < len(tabs):
            base_txt, code = tabs[i]
            if code in self.tab_buttons:
                btn = self.tab_buttons[code]
                if code == active_code:
                    btn.configure(bg="#C8102E", text=f"{base_txt} *")
                else:
                    btn.configure(bg="#0F4C81", text=base_txt)
            i += 1

    def _build_status_bar(self) -> None:
        """Procedure: Builds the bottom status bar displaying SQLite & memory state."""
        n_dedup: int = len(self.state.repository.dedup_records)
        n_train: int = len(self.state.repository.dedup_train)
        n_test: int = len(self.state.repository.dedup_test)
        default_status: str = (
            f"SQLite3 Connected: {self.state.sqlite_db_path} "
            f"({n_dedup} unique crash_records | 80/20 Split: {n_train} Train / {n_test} Test) | Zero Global Variables"
        )
        self.state.status_message = default_status
        self.status_lbl = tk.Label(
            self.root,
            text=self.state.status_message,
            bg="#E2E8F0",
            fg="#1E293B",
            anchor="w",
            font=("Helvetica", 9, "bold"),
            padx=14,
            pady=5,
        )
        self.status_lbl.pack(fill=tk.X, side=tk.BOTTOM)

    def switch_screen(self, screen_code: str) -> None:
        """Procedure: Switches AppState.active_screen and redraws the active view."""
        self.state.active_screen = screen_code
        self._update_header_tab_styles(screen_code)
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

    def _find_metric(self, model_sub: str, regime_sub: str) -> Optional[EvaluationMetrics]:
        """Pure Helper: Finds an EvaluationMetrics entry matching model and split regime."""
        i: int = 0
        while i < len(self.state.metrics_list):
            m = self.state.metrics_list[i]
            if model_sub in m.model_name and regime_sub in m.evaluation_mode:
                return m
            i += 1
        return None

    def _predict_record_prob(self, rec: CrashRecord) -> float:
        """Pure Helper: Predicts survival probability using the active classifier and hygiene mode."""
        use_dedup: bool = bool(self.dedup_mode_var.get())
        clf: str = self.classifier_var.get()
        if clf == "LinearScore":
            lin_m = self.state.linear_dedup if use_dedup else self.state.linear_leaked
            return predict_linear_score_prob(lin_m, rec) if lin_m is not None else 0.0
        elif clf == "SubgroupRate":
            sub_m = self.state.subgroup_dedup if use_dedup else self.state.subgroup_leaked
            return predict_subgroup_rate_prob(sub_m, rec) if sub_m is not None else 0.0
        else:
            tree_m = self.state.tree_dedup if use_dedup else self.state.tree_leaked
            return predict_tree_prob(tree_m, rec)

    # =========================================================================
    # TAB 3: INTERACTIVE POLICY SIMULATOR (3-Column Dashboard + Bottom Preview)
    # =========================================================================
    def _render_policy_simulator_screen(self) -> None:
        """Procedure: Renders Tab 3 matching the 3-column KPI dashboard + bottom binary search preview."""
        top_row = tk.Frame(self.content_frame, bg="#F1F5F9")
        top_row.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        # ---------------------------------------------------------------------
        # LEFT COLUMN: POLICY INTERVENTION CONTROLS
        # ---------------------------------------------------------------------
        left_card = tk.Frame(top_row, bg="white", highlightbackground="#CBD5E1", highlightthickness=1, width=325)
        left_card.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_card.pack_propagate(False)

        left_inner = tk.Frame(left_card, bg="white", padx=14, pady=10)
        left_inner.pack(fill=tk.BOTH, expand=True)

        self.left_title_lbl = tk.Label(
            left_inner,
            text="POLICY INTERVENTION CONTROLS (N=599)",
            bg="white",
            fg="#0F4C81",
            font=("Helvetica", 10, "bold"),
        )
        self.left_title_lbl.pack(anchor="w")
        tk.Frame(left_inner, bg="#E2E8F0", height=1).pack(fill=tk.X, pady=(4, 8))

        # Slider 1: Max Speed Cap
        self.speed_cap_lbl = tk.Label(
            left_inner,
            text=f"Max Speed Cap Enforcement: {self.speed_cap_var.get()} km/h",
            bg="white",
            fg="#1E293B",
            font=("Helvetica", 9, "bold"),
        )
        self.speed_cap_lbl.pack(anchor="w")
        tk.Scale(
            left_inner,
            from_=40,
            to=119,
            orient=tk.HORIZONTAL,
            variable=self.speed_cap_var,
            showvalue=False,
            bg="white",
            fg="#1E293B",
            troughcolor="#C8102E",
            activebackground="#C8102E",
            highlightthickness=0,
            bd=0,
            sliderrelief=tk.FLAT,
            command=lambda _v: self._on_slider_adjusted(),
        ).pack(fill=tk.X, pady=(2, 0))
        s1_range = tk.Frame(left_inner, bg="white")
        s1_range.pack(fill=tk.X, pady=(0, 6))
        tk.Label(s1_range, text="40 km/h", bg="white", fg="#64748B", font=("Helvetica", 8)).pack(side=tk.LEFT)
        tk.Label(s1_range, text="119 km/h", bg="white", fg="#64748B", font=("Helvetica", 8)).pack(side=tk.RIGHT)

        # Slider 2: Mandatory Seatbelt Target
        self.seatbelt_lbl = tk.Label(
            left_inner,
            text=f"Mandatory Seatbelt Target: {self.seatbelt_rate_var.get():.1f}%",
            bg="white",
            fg="#1E293B",
            font=("Helvetica", 9, "bold"),
        )
        self.seatbelt_lbl.pack(anchor="w")
        tk.Scale(
            left_inner,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.seatbelt_rate_var,
            showvalue=False,
            bg="white",
            fg="#1E293B",
            troughcolor="#0D9488",
            activebackground="#0D9488",
            highlightthickness=0,
            bd=0,
            sliderrelief=tk.FLAT,
            command=lambda _v: self._on_slider_adjusted(),
        ).pack(fill=tk.X, pady=(2, 0))
        s2_range = tk.Frame(left_inner, bg="white")
        s2_range.pack(fill=tk.X, pady=(0, 6))
        tk.Label(s2_range, text="0%", bg="white", fg="#64748B", font=("Helvetica", 8)).pack(side=tk.LEFT)
        tk.Label(s2_range, text="100%", bg="white", fg="#64748B", font=("Helvetica", 8)).pack(side=tk.RIGHT)

        # Slider 3: Mandatory Helmet Target
        self.helmet_lbl = tk.Label(
            left_inner,
            text=f"Mandatory Helmet Target: {self.helmet_rate_var.get():.1f}%",
            bg="white",
            fg="#1E293B",
            font=("Helvetica", 9, "bold"),
        )
        self.helmet_lbl.pack(anchor="w")
        tk.Scale(
            left_inner,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.helmet_rate_var,
            showvalue=False,
            bg="white",
            fg="#1E293B",
            troughcolor="#0D9488",
            activebackground="#0D9488",
            highlightthickness=0,
            bd=0,
            sliderrelief=tk.FLAT,
            command=lambda _v: self._on_slider_adjusted(),
        ).pack(fill=tk.X, pady=(2, 0))
        s3_range = tk.Frame(left_inner, bg="white")
        s3_range.pack(fill=tk.X, pady=(0, 6))
        tk.Label(s3_range, text="0%", bg="white", fg="#64748B", font=("Helvetica", 8)).pack(side=tk.LEFT)
        tk.Label(s3_range, text="100%", bg="white", fg="#64748B", font=("Helvetica", 8)).pack(side=tk.RIGHT)

        # Active Prediction Model (Strategy Toggle Button)
        tk.Label(
            left_inner,
            text="Active Prediction Model (Strategy):",
            bg="white",
            fg="#1E293B",
            font=("Helvetica", 9, "bold"),
        ).pack(anchor="w", pady=(2, 2))

        self.model_toggle_btn = tk.Button(
            left_inner,
            text="[x] Rule-Based Decision Tree (80/20 Split)",
            bg="#EFF6FF",
            fg="#0F4C81",
            activebackground="#DBEAFE",
            activeforeground="#0F4C81",
            font=("Courier", 8, "bold"),
            relief=tk.SOLID,
            bd=1,
            anchor="w",
            padx=8,
            pady=5,
            cursor="hand2",
            command=self._on_toggle_classifier,
        )
        self.model_toggle_btn.pack(fill=tk.X, pady=(0, 6))

        # Dataset Hygiene Mode (RQ1 Toggle Button)
        tk.Label(
            left_inner,
            text="Dataset Hygiene Mode (RQ1):",
            bg="white",
            fg="#1E293B",
            font=("Helvetica", 9, "bold"),
        ).pack(anchor="w", pady=(2, 2))

        self.hygiene_toggle_btn = tk.Button(
            left_inner,
            text="[x] Deduplicated Unique Registry (N = 599)",
            bg="#ECFDF5",
            fg="#0D9488",
            activebackground="#D1FAE5",
            activeforeground="#0D9488",
            font=("Courier", 8, "bold"),
            relief=tk.SOLID,
            bd=1,
            anchor="w",
            padx=8,
            pady=5,
            cursor="hand2",
            command=self._on_toggle_hygiene_mode,
        )
        self.hygiene_toggle_btn.pack(fill=tk.X, pady=(0, 10))

        run_btn = tk.Button(
            left_inner,
            text="RUN SIMULATION & SAVE TO SQLITE DB",
            bg="#C8102E",
            fg="white",
            activebackground="#991B1B",
            activeforeground="white",
            font=("Helvetica", 9, "bold"),
            relief=tk.FLAT,
            padx=10,
            pady=8,
            cursor="hand2",
            command=lambda: self._execute_policy_simulation(save_to_db=True),
        )
        run_btn.pack(fill=tk.X, pady=(2, 0))

        # ---------------------------------------------------------------------
        # CENTER COLUMN: 86 km/h KINETIC SURVIVAL CLIFF & SUBGROUP RATES
        # ---------------------------------------------------------------------
        center_card = tk.Frame(top_row, bg="white", highlightbackground="#CBD5E1", highlightthickness=1)
        center_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        center_inner = tk.Frame(center_card, bg="white", padx=14, pady=10)
        center_inner.pack(fill=tk.BOTH, expand=True)

        self.center_title_lbl = tk.Label(
            center_inner,
            text="86 km/h KINETIC SURVIVAL CLIFF & SUBGROUP RATES (N = 599)",
            bg="white",
            fg="#0F4C81",
            font=("Helvetica", 10, "bold"),
        )
        self.center_title_lbl.pack(anchor="w")
        tk.Frame(center_inner, bg="#E2E8F0", height=1).pack(fill=tk.X, pady=(4, 6))

        self.sim_canvas = tk.Canvas(center_inner, bg="white", height=295, highlightthickness=0)
        self.sim_canvas.pack(fill=tk.BOTH, expand=True)
        self.sim_canvas.bind("<Configure>", lambda _e: self._redraw_current_canvas())

        insight_box = tk.Frame(
            center_inner,
            bg="#F8FAFC",
            highlightbackground="#CBD5E1",
            highlightthickness=1,
            padx=10,
            pady=6,
        )
        insight_box.pack(fill=tk.X, pady=(4, 0))

        self.insight_line1_lbl = tk.Label(
            insight_box,
            text="",
            bg="#F8FAFC",
            fg="#1E293B",
            anchor="w",
            font=("Helvetica", 8),
        )
        self.insight_line1_lbl.pack(fill=tk.X)

        self.insight_line2_lbl = tk.Label(
            insight_box,
            text="",
            bg="#F8FAFC",
            fg="#0D9488",
            anchor="w",
            font=("Helvetica", 9, "bold"),
        )
        self.insight_line2_lbl.pack(fill=tk.X)

        # ---------------------------------------------------------------------
        # RIGHT COLUMN: POLICY SIMULATION KPI (N=599) & 80/20 MODEL COMPARISON
        # ---------------------------------------------------------------------
        right_card = tk.Frame(top_row, bg="white", highlightbackground="#CBD5E1", highlightthickness=1, width=325)
        right_card.pack(side=tk.LEFT, fill=tk.Y)
        right_card.pack_propagate(False)

        right_inner = tk.Frame(right_card, bg="white", padx=14, pady=10)
        right_inner.pack(fill=tk.BOTH, expand=True)

        self.right_title_lbl = tk.Label(
            right_inner,
            text="POLICY SIMULATION KPI (N=599)",
            bg="white",
            fg="#0F4C81",
            font=("Helvetica", 10, "bold"),
        )
        self.right_title_lbl.pack(anchor="w")
        tk.Frame(right_inner, bg="#E2E8F0", height=1).pack(fill=tk.X, pady=(4, 8))

        # Green KPI Card: Projected Lives Saved
        green_card = tk.Frame(
            right_inner,
            bg="#ECFDF5",
            highlightbackground="#0D9488",
            highlightthickness=2,
            padx=10,
            pady=8,
        )
        green_card.pack(fill=tk.X, pady=(0, 10))
        self.kpi_saved_header_lbl = tk.Label(
            green_card,
            text="Projected Lives Saved (Unique N=599)",
            bg="#ECFDF5",
            fg="#0D9488",
            anchor="w",
            font=("Helvetica", 9, "bold"),
        )
        self.kpi_saved_header_lbl.pack(fill=tk.X)
        self.kpi_saved_val_lbl = tk.Label(
            green_card,
            text="+249.9 Survivors (-60.9%)",
            bg="#ECFDF5",
            fg="#0D9488",
            anchor="w",
            font=("Helvetica", 15, "bold"),
        )
        self.kpi_saved_val_lbl.pack(fill=tk.X, pady=(2, 0))

        # Red KPI Card: Expected Fatalities
        red_card = tk.Frame(
            right_inner,
            bg="#FEF2F2",
            highlightbackground="#C8102E",
            highlightthickness=2,
            padx=10,
            pady=8,
        )
        red_card.pack(fill=tk.X, pady=(0, 10))
        self.kpi_fatal_header_lbl = tk.Label(
            red_card,
            text="Expected Fatalities (Baseline -> Policy)",
            bg="#FEF2F2",
            fg="#C8102E",
            anchor="w",
            font=("Helvetica", 9, "bold"),
        )
        self.kpi_fatal_header_lbl.pack(fill=tk.X)
        self.kpi_fatal_val_lbl = tk.Label(
            red_card,
            text="410.5 -> 160.6 Deaths",
            bg="#FEF2F2",
            fg="#C8102E",
            anchor="w",
            font=("Helvetica", 15, "bold"),
        )
        self.kpi_fatal_val_lbl.pack(fill=tk.X, pady=(2, 0))

        # 80/20 Split Model Comparison Card
        comp_card = tk.Frame(
            right_inner,
            bg="#F8FAFC",
            highlightbackground="#CBD5E1",
            highlightthickness=1,
            padx=10,
            pady=8,
        )
        comp_card.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            comp_card,
            text="80/20 SPLIT MODEL COMPARISON",
            bg="#F8FAFC",
            fg="#0F4C81",
            anchor="w",
            font=("Helvetica", 9, "bold"),
        ).pack(fill=tk.X)

        self.comp_subtitle_lbl = tk.Label(
            comp_card,
            text="Deduplicated Test Set (n = 120):",
            bg="#F8FAFC",
            fg="#1E293B",
            anchor="w",
            font=("Helvetica", 9, "bold"),
        )
        self.comp_subtitle_lbl.pack(fill=tk.X, pady=(2, 4))

        self.comp_m1_title = tk.Label(
            comp_card,
            text="1. Rule-Based Decision Tree:",
            bg="#F8FAFC",
            fg="#0D9488",
            anchor="w",
            font=("Helvetica", 9, "bold"),
        )
        self.comp_m1_title.pack(fill=tk.X)
        self.comp_m1_body = tk.Label(
            comp_card,
            text="",
            bg="#F8FAFC",
            fg="#334155",
            anchor="w",
            justify=tk.LEFT,
            font=("Courier", 8),
        )
        self.comp_m1_body.pack(fill=tk.X, padx=(8, 0), pady=(0, 4))

        self.comp_m2_title = tk.Label(
            comp_card,
            text="2. Weighted Linear Score:",
            bg="#F8FAFC",
            fg="#0F4C81",
            anchor="w",
            font=("Helvetica", 9, "bold"),
        )
        self.comp_m2_title.pack(fill=tk.X)
        self.comp_m2_body = tk.Label(
            comp_card,
            text="",
            bg="#F8FAFC",
            fg="#334155",
            anchor="w",
            justify=tk.LEFT,
            font=("Courier", 8),
        )
        self.comp_m2_body.pack(fill=tk.X, padx=(8, 0), pady=(0, 4))

        self.comp_m3_title = tk.Label(
            comp_card,
            text="3. Subgroup Rate Average:",
            bg="#F8FAFC",
            fg="#64748B",
            anchor="w",
            font=("Helvetica", 9, "bold"),
        )
        self.comp_m3_title.pack(fill=tk.X)
        self.comp_m3_body = tk.Label(
            comp_card,
            text="",
            bg="#F8FAFC",
            fg="#334155",
            anchor="w",
            justify=tk.LEFT,
            font=("Courier", 8),
        )
        self.comp_m3_body.pack(fill=tk.X, padx=(8, 0))

        # ---------------------------------------------------------------------
        # BOTTOM ROW: TAB 1 PREVIEW: O(log N) BINARY SEARCH CRASH INSPECTOR
        # ---------------------------------------------------------------------
        bottom_card = tk.Frame(
            self.content_frame,
            bg="white",
            highlightbackground="#CBD5E1",
            highlightthickness=1,
            padx=14,
            pady=8,
        )
        bottom_card.pack(fill=tk.X)

        tk.Label(
            bottom_card,
            text=f"TAB 1 PREVIEW: O(log N) BINARY SEARCH CRASH INSPECTOR & SQLITE3 AUDIT LOG ({self.state.sqlite_db_path})",
            bg="white",
            fg="#0F4C81",
            anchor="w",
            font=("Helvetica", 9, "bold"),
        ).pack(fill=tk.X)

        self.preview_summary_lbl = tk.Label(
            bottom_card,
            text="",
            bg="white",
            fg="#0D9488",
            anchor="w",
            font=("Courier", 9, "bold"),
        )
        self.preview_summary_lbl.pack(fill=tk.X, pady=(2, 4))

        self.preview_rows_frame = tk.Frame(bottom_card, bg="white")
        self.preview_rows_frame.pack(fill=tk.X)

        self._sync_toggle_button_labels()
        self._execute_policy_simulation(save_to_db=False)

    def _sync_toggle_button_labels(self) -> None:
        """Procedure: Synchronizes Strategy and RQ1 Hygiene button text and colors."""
        clf: str = self.classifier_var.get()
        if clf == "LinearScore":
            self.model_toggle_btn.configure(
                text="[x] Weighted Linear Score (80/20 Split)",
                bg="#EFF6FF",
                fg="#0F4C81",
            )
        else:
            self.model_toggle_btn.configure(
                text="[x] Rule-Based Decision Tree (80/20 Split)",
                bg="#EFF6FF",
                fg="#0F4C81",
            )

        use_dedup: bool = bool(self.dedup_mode_var.get())
        if use_dedup:
            self.hygiene_toggle_btn.configure(
                text=f"[x] Deduplicated Unique Registry (N = {len(self.state.repository.dedup_records)})",
                bg="#ECFDF5",
                fg="#0D9488",
            )
        else:
            self.hygiene_toggle_btn.configure(
                text=f"[!] Cross-Fold Leaked Raw Registry (N = {len(self.state.repository.raw_records)})",
                bg="#FEF2F2",
                fg="#C8102E",
            )

    def _on_slider_adjusted(self) -> None:
        """Procedure: Live-updates slider headers and KPI calculations when any slider moves."""
        self.speed_cap_lbl.configure(text=f"Max Speed Cap Enforcement: {int(self.speed_cap_var.get())} km/h")
        self.seatbelt_lbl.configure(text=f"Mandatory Seatbelt Target: {float(self.seatbelt_rate_var.get()):.1f}%")
        self.helmet_lbl.configure(text=f"Mandatory Helmet Target: {float(self.helmet_rate_var.get()):.1f}%")
        self._execute_policy_simulation(save_to_db=False)

    def _on_toggle_classifier(self) -> None:
        """Procedure: Switches between Rule-Based Decision Tree and Weighted Linear Score."""
        curr: str = self.classifier_var.get()
        next_clf: str = "LinearScore" if curr == "DecisionTree" else "DecisionTree"
        self.classifier_var.set(next_clf)
        self.state.active_classifier = next_clf
        self._sync_toggle_button_labels()
        self._execute_policy_simulation(save_to_db=False)

    def _on_toggle_hygiene_mode(self) -> None:
        """Procedure: Switches between Deduplicated Unique Registry (N=599) and Raw Registry (N=1200)."""
        next_mode: bool = not bool(self.dedup_mode_var.get())
        self.dedup_mode_var.set(next_mode)
        self.state.active_dedup_mode = next_mode
        self._sync_toggle_button_labels()
        self._execute_policy_simulation(save_to_db=False)

    def _execute_policy_simulation(self, save_to_db: bool = True) -> None:
        """Procedure: Executes policy simulation, updates KPI cards, model comparison, canvas, and preview."""
        cap: int = int(self.speed_cap_var.get())
        seat_r: float = float(self.seatbelt_rate_var.get()) / 100.0
        helm_r: float = float(self.helmet_rate_var.get()) / 100.0
        clf_name: str = self.classifier_var.get()
        use_dedup: bool = bool(self.dedup_mode_var.get())

        cohort: List[CrashRecord] = (
            self.state.repository.dedup_records if use_dedup else self.state.repository.raw_records
        )
        tree_model = self.state.tree_dedup if use_dedup else self.state.tree_leaked
        lin_model = self.state.linear_dedup if use_dedup else self.state.linear_leaked
        n_cohort: int = len(cohort)

        scen: PolicyScenario = simulate_policy_intervention(
            cohort=cohort,
            scenario_name=f"GUI Custom ({cap} km/h Cap, Seat={int(seat_r*100)}%, Helm={int(helm_r*100)}%)",
            max_speed_cap=cap,
            target_helmet_rate=helm_r,
            target_seatbelt_rate=seat_r,
            tree_root=tree_model,
            linear_model=lin_model,
            active_classifier=clf_name,
        )
        self.last_scenario: PolicyScenario = scen

        if save_to_db:
            save_policy_simulation_to_sqlite(self.state.sqlite_db_path, scen)
            self.state.policy_history.append(scen)
            self.status_lbl.configure(
                text=(
                    f"Saved scenario '{scen.scenario_name}' to SQLite ({self.state.sqlite_db_path}) | "
                    f"Lives Saved: +{scen.lives_saved:.1f} (-{scen.fatality_reduction_pct:.1f}%)"
                )
            )

        # Update Section Titles & KPI Cards
        tag_label: str = f"Unique N={n_cohort}" if use_dedup else f"Raw N={n_cohort}"
        self.left_title_lbl.configure(text=f"POLICY INTERVENTION CONTROLS (N={n_cohort})")
        self.center_title_lbl.configure(text=f"86 km/h KINETIC SURVIVAL CLIFF & SUBGROUP RATES (N = {n_cohort})")
        self.right_title_lbl.configure(text=f"POLICY SIMULATION KPI (N={n_cohort})")

        self.kpi_saved_header_lbl.configure(text=f"Projected Lives Saved ({tag_label})")
        self.kpi_saved_val_lbl.configure(
            text=f"+{scen.lives_saved:.1f} Survivors (-{scen.fatality_reduction_pct:.1f}%)"
        )
        self.kpi_fatal_val_lbl.configure(
            text=f"{scen.baseline_fatalities:.1f} -> {scen.projected_fatalities:.1f} Deaths"
        )

        # Update 80/20 Split Model Comparison Card
        regime_key: str = "Deduplicated" if use_dedup else "Cross-Fold"
        test_n: int = len(self.state.repository.dedup_test) if use_dedup else len(self.state.repository.leaked_test)
        self.comp_subtitle_lbl.configure(
            text=f"{'Deduplicated' if use_dedup else 'Cross-Fold Raw'} Test Set (n = {test_n}):"
        )

        m_tree = self._find_metric("Decision Tree", regime_key)
        m_lin = self._find_metric("Linear Score", regime_key)
        m_sub = self._find_metric("Subgroup Rate", regime_key)

        if m_tree is not None:
            self.comp_m1_body.configure(
                text=(
                    f"Acc: {m_tree.accuracy*100:.1f}% | Prec: {m_tree.precision*100:.1f}%\n"
                    f"Recall: {m_tree.recall_sens*100:.1f}% | F1: {m_tree.f1_score*100:.1f}%"
                )
            )
        if m_lin is not None:
            self.comp_m2_body.configure(
                text=(
                    f"Acc: {m_lin.accuracy*100:.1f}% | Prec: {m_lin.precision*100:.1f}%\n"
                    f"Recall: {m_lin.recall_sens*100:.1f}% | F1: {m_lin.f1_score*100:.1f}%"
                )
            )
        if m_sub is not None:
            self.comp_m3_body.configure(
                text=f"Acc: {m_sub.accuracy*100:.1f}% | Recall: {m_sub.recall_sens*100:.1f}%"
            )

        # Update Policy Insight Footer & Canvas
        cliff = compute_kinetic_cliff_summary(cohort)
        high_n: int = int(cliff["high_speed_total"])
        both_rate: float = cliff["both_restraints_surv_rate"]
        self.insight_line1_lbl.configure(
            text=f"Policy Insight: Capping speed at <= {cap} km/h moves all {high_n} high-speed crashes below"
        )
        self.insight_line2_lbl.configure(
            text=(
                f"the 86 km/h cliff so {int(seat_r*100)}% seatbelt + {int(helm_r*100)}% helmet compliance "
                f"raises survival to {both_rate:.1f}% (+{scen.lives_saved:.1f} lives saved)."
            )
        )

        self._redraw_current_canvas()
        self._update_bottom_binary_search_preview()

    def _redraw_current_canvas(self) -> None:
        """Procedure: Redraws the 86 km/h Kinetic Survival Cliff & Subgroup Rates chart."""
        if not hasattr(self, "sim_canvas") or not self.sim_canvas.winfo_exists():
            return
        use_dedup: bool = bool(self.dedup_mode_var.get())
        cohort: List[CrashRecord] = (
            self.state.repository.dedup_records if use_dedup else self.state.repository.raw_records
        )
        cliff = compute_kinetic_cliff_summary(cohort)

        c = self.sim_canvas
        c.delete("all")
        w: int = max(c.winfo_width(), 520)
        h: int = max(c.winfo_height(), 275)

        x_left: int = 48
        x_right: int = w - 22
        y_top: int = 26
        y_bot: int = h - 38

        # Map speed [20..119] to x coordinate
        def speed_to_x(spd: float) -> float:
            return x_left + ((spd - 20.0) / (119.0 - 20.0)) * (x_right - x_left)

        # Map percentage [0..100] to y coordinate
        def pct_to_y(pct: float) -> float:
            return y_bot - (pct / 100.0) * (y_bot - y_top)

        x_cliff: float = speed_to_x(86.0)

        # Background zones (Green survivable <= 86 km/h, Red lethal > 86 km/h)
        c.create_rectangle(x_left, y_top, x_cliff, y_bot, fill="#ECFDF5", outline="#CBD5E1")
        c.create_rectangle(x_cliff, y_top, x_right, y_bot, fill="#FEF2F2", outline="#CBD5E1")

        # Horizontal grid lines & Y-axis labels (0%, 25%, 50%, 75%, 100%)
        y_ticks = [0, 25, 50, 75, 100]
        idx: int = 0
        while idx < len(y_ticks):
            pct_val = y_ticks[idx]
            yv = pct_to_y(float(pct_val))
            c.create_line(x_left, yv, x_right, yv, fill="#E2E8F0", width=1)
            c.create_text(
                x_left - 8,
                yv,
                text=f"{pct_val}%",
                anchor="e",
                fill="#475569",
                font=("Helvetica", 8),
            )
            idx += 1

        # X-axis ticks (20, 40, 60, 80, 86, 100, 119)
        x_ticks = [20, 40, 60, 80, 86, 100, 119]
        idx = 0
        while idx < len(x_ticks):
            spd_val = x_ticks[idx]
            xv = speed_to_x(float(spd_val))
            c.create_line(xv, y_bot, xv, y_bot + 4, fill="#475569", width=1)
            font_spec = ("Helvetica", 8, "bold") if spd_val == 86 else ("Helvetica", 8)
            c.create_text(
                xv,
                y_bot + 13,
                text=str(spd_val),
                fill="#1E293B",
                font=font_spec,
            )
            idx += 1

        c.create_text(
            (x_left + x_right) / 2,
            y_bot + 28,
            text="Impact Speed (km/h)",
            fill="#1E293B",
            font=("Helvetica", 9, "bold"),
        )

        # Subgroup survival horizontal lines in <= 86 km/h zone
        both_pct: float = cliff["both_restraints_surv_rate"]
        seat_pct: float = cliff["seatbelt_only_surv_rate"]
        helm_pct: float = cliff["helmet_only_surv_rate"]
        none_pct: float = cliff["unrestrained_surv_rate"]

        y_both: float = pct_to_y(both_pct)
        y_seat: float = pct_to_y(seat_pct)
        y_none: float = pct_to_y(none_pct)

        # 1. Neither Restraint line (Orange)
        c.create_line(x_left, y_none, x_cliff, y_none, fill="#D97706", width=2)
        c.create_text(
            x_left + 10,
            y_none - 9,
            text=f"Neither Restraint: {none_pct:.1f}% Survival",
            anchor="w",
            fill="#B45309",
            font=("Helvetica", 8, "bold"),
        )

        # 2. Seatbelt Only / Helmet Only line (Navy Blue)
        c.create_line(x_left, y_seat, x_cliff, y_seat, fill="#0F4C81", width=2)
        c.create_text(
            x_left + 10,
            y_seat + 11,
            text=f"Seatbelt Only: {seat_pct:.1f}% | Helmet Only: {helm_pct:.1f}%",
            anchor="w",
            fill="#0F4C81",
            font=("Helvetica", 8),
        )

        # 3. Helmet + Seatbelt line (Teal Green) dropping to 0% after 86 km/h
        c.create_line(
            x_left,
            y_both,
            x_cliff,
            y_both,
            x_cliff,
            y_bot,
            x_right,
            y_bot,
            fill="#0D9488",
            width=3,
        )
        c.create_text(
            x_left + 10,
            y_both - 10,
            text=f"Helmet + Seatbelt: {both_pct:.1f}% Survival",
            anchor="w",
            fill="#0D9488",
            font=("Helvetica", 9, "bold"),
        )

        # Vertical 86 km/h Speed Cliff Line (Crimson)
        c.create_line(x_cliff, y_top - 6, x_cliff, y_bot, fill="#C8102E", width=3)
        c.create_text(
            x_cliff + 6,
            y_top - 10,
            text="86 km/h Cliff (0% Survival > 86)",
            anchor="center",
            fill="#C8102E",
            font=("Helvetica", 9, "bold"),
        )

        # Lethal Zone Annotation
        high_tot: int = int(cliff["high_speed_total"])
        c.create_text(
            x_cliff + 12,
            y_top + 68,
            text=(
                f"Lethal Zone\n"
                f"(87-119 km/h)\n"
                f"0 / {high_tot} Survived\n"
                f"(0.00%)"
            ),
            anchor="w",
            fill="#991B1B",
            font=("Helvetica", 9, "bold"),
        )

    def _update_bottom_binary_search_preview(self) -> None:
        """Procedure: Runs O(log N) binary search [80..86 km/h] and renders preview rows."""
        if not hasattr(self, "preview_rows_frame") or not self.preview_rows_frame.winfo_exists():
            return
        t0: int = time.perf_counter_ns()
        matches, comps = binary_search_speed_range(self.state.repository.sorted_by_speed, 80, 86)
        elapsed_ms: float = (time.perf_counter_ns() - t0) / 1e6

        n_sorted: int = len(self.state.repository.sorted_by_speed)
        self.preview_summary_lbl.configure(
            text=(
                f"Binary Search Query [80..86 km/h] on MergeSorted List (N={n_sorted}): "
                f"Found {len(matches)} unique records in {elapsed_ms:.4f} ms ({comps} comparisons)"
            )
        )

        children = self.preview_rows_frame.winfo_children()
        i: int = 0
        while i < len(children):
            children[i].destroy()
            i += 1

        # Select 4 representative records from [80..86 km/h] + 1 lethal cliff record (>86 km/h)
        preview_list: List[CrashRecord] = []
        target_speeds: List[int] = [80, 82, 85, 86]
        t_idx: int = 0
        while t_idx < len(target_speeds):
            t_spd = target_speeds[t_idx]
            k: int = 0
            found_rec: Optional[CrashRecord] = None
            while k < len(matches):
                if matches[k].speed_of_impact == t_spd:
                    if t_spd < 86 and matches[k].survived == 1:
                        found_rec = matches[k]
                        break
                    elif t_spd == 86 and matches[k].survived == 0:
                        found_rec = matches[k]
                        break
                    elif found_rec is None:
                        found_rec = matches[k]
                k += 1
            if found_rec is not None:
                preview_list.append(found_rec)
            t_idx += 1

        # Add 1 record just above the 86 km/h cliff (e.g., 88 km/h) with restraints
        high_matches, _ = binary_search_speed_range(self.state.repository.sorted_by_speed, 87, 92)
        k = 0
        cliff_rec: Optional[CrashRecord] = None
        while k < len(high_matches):
            if high_matches[k].helmet_used and high_matches[k].seatbelt_used:
                cliff_rec = high_matches[k]
                break
            k += 1
        if cliff_rec is None and len(high_matches) > 0:
            cliff_rec = high_matches[0]
        if cliff_rec is not None:
            preview_list.append(cliff_rec)

        r_idx: int = 0
        while r_idx < len(preview_list):
            rec = preview_list[r_idx]
            prob: float = self._predict_record_prob(rec)
            helm_str: str = "Yes" if rec.helmet_used else "No "
            seat_str: str = "Yes" if rec.seatbelt_used else "No "
            if rec.speed_of_impact > 86:
                status_str = ">86 CLIFF"
            elif rec.survived == 1:
                status_str = "SURVIVED"
            else:
                status_str = "HIGH RISK"

            row_bg = "#F8FAFC" if (r_idx % 2 == 0) else "white"
            line_txt = (
                f" ID={rec.record_id:03d} | Age={rec.age:02d} | {rec.gender:<6} | "
                f"Speed={rec.speed_of_impact:02d} km/h | Helmet={helm_str} | "
                f"Seatbelt={seat_str} | Survived={rec.survived} | "
                f"Tree P(Surv)={prob:.3f} | Status: {status_str}"
            )
            tk.Label(
                self.preview_rows_frame,
                text=line_txt,
                bg=row_bg,
                fg="#1E293B",
                anchor="w",
                font=("Courier", 8),
                padx=4,
                pady=1,
            ).pack(fill=tk.X)
            r_idx += 1

    # =========================================================================
    # TAB 1: DATA AUDIT & O(log N) BINARY SEARCH INSPECTOR
    # =========================================================================
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
            font=("Helvetica", 9, "bold"),
            padx=10,
            pady=4,
            cursor="hand2",
            command=self._on_binary_search_clicked,
        ).pack(side=tk.RIGHT, padx=6)

        self.audit_list = tk.Listbox(self.content_frame, font=("Courier", 9), height=24)
        self.audit_list.pack(fill=tk.BOTH, expand=True)
        self._on_binary_search_clicked()

    def _on_binary_search_clicked(self) -> None:
        """Procedure: Runs binary_search_speed_range and populates the listbox."""
        min_v: int = int(self.search_min_var.get())
        max_v: int = int(self.search_max_var.get())
        t0: int = time.perf_counter_ns()
        matches, comps = binary_search_speed_range(
            self.state.repository.sorted_by_speed, min_v, max_v
        )
        elapsed_ms: float = (time.perf_counter_ns() - t0) / 1e6
        self.search_info_lbl.configure(
            text=(
                f"Found {len(matches)} unique records in [{min_v}..{max_v} km/h] "
                f"in {elapsed_ms:.4f} ms ({comps} binary comparisons)."
            )
        )
        self.audit_list.delete(0, tk.END)
        i: int = 0
        while i < min(len(matches), 200):
            r: CrashRecord = matches[i]
            prob: float = predict_tree_prob(self.state.tree_dedup, r)
            helm_str: str = "Yes" if r.helmet_used else "No "
            seat_str: str = "Yes" if r.seatbelt_used else "No "
            status_str: str = ">86 CLIFF" if r.speed_of_impact > 86 else ("SURVIVED" if r.survived == 1 else "HIGH RISK")
            self.audit_list.insert(
                tk.END,
                f"ID={r.record_id:03d} | Age={r.age:02d} | {r.gender:<6} | "
                f"Speed={r.speed_of_impact:03d} km/h | Helmet={helm_str} | "
                f"Seatbelt={seat_str} | Survived={r.survived} | "
                f"Tree P(Surv)={prob:.3f} | Status: {status_str}",
            )
            i += 1

    # =========================================================================
    # TAB 2: PURE-PYTHON MODELS (80/20 SPLIT EVALUATION)
    # =========================================================================
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
            f"{'Prediction Method':<26} | {'80/20 Split Regime':<26} | {'Acc%':>6} | {'Prec%':>6} | {'Recall%':>7} | {'F1%':>6} | {'TP/FP/TN/FN'}\n",
        )
        txt.insert(tk.END, "-" * 106 + "\n")
        i: int = 0
        while i < len(self.state.metrics_list):
            m: EvaluationMetrics = self.state.metrics_list[i]
            cm_str = f"{m.tp}/{m.fp}/{m.tn}/{m.fn}"
            txt.insert(
                tk.END,
                f"{m.model_name:<26} | {m.evaluation_mode:<26} | "
                f"{m.accuracy*100:>6.2f} | {m.precision*100:>6.2f} | "
                f"{m.recall_sens*100:>7.2f} | {m.f1_score*100:>6.2f} | {cm_str}\n",
            )
            i += 1

    # =========================================================================
    # TAB 4: TASK 11.3 ALGORITHMIC LOOKUP SCALING LAB
    # =========================================================================
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
            f"{'Search Algorithm (K=100 Queries)':<28} | {'Class':<8} | {'N':>6} | {'Time(ms)':>9} | {'Mem(KB)':>8} | {'Comparisons':>11} | {'60 FPS Check'}\n",
        )
        txt.insert(tk.END, "-" * 106 + "\n")
        i: int = 0
        while i < len(self.state.scaling_benchmarks):
            b = self.state.scaling_benchmarks[i]
            status = "PASS (<16.67ms)" if b.passes_60fps_budget else "LAG (>16.67ms)"
            txt.insert(
                tk.END,
                f"{b.algorithm_name:<28} | {b.complexity_class:<8} | {b.dataset_size_n:>6} | "
                f"{b.elapsed_ms:>9.4f} | {b.peak_memory_kb:>8.1f} | {b.comparisons_count:>11} | {status}\n",
            )
            i += 1
