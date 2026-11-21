"""
Module 8: main.py
COS10009 Introduction to Programming — Root Application Entry Point
Author: Anh Vu Pham (Student ID: 106217835)
Campus: Swinburne University of Technology (Can Tho Campus)
Unit Coordinator: Mr. Quach Huu Nghia (hquach@swin.edu.au)

Usage:
  1. Launch Interactive Desktop GUI Workbench (Windows / macOS / Desktop Linux):
       python3 main.py
  2. Run Headless Empirical Benchmark & Verification Suite (for CLI / CI / Task 11.3):
       python3 main.py --headless
"""

import os
import sys
from domain_models import AppState
from benchmark_runner import (
    export_benchmark_summary,
    ingest_and_prepare_repository,
    print_console_report,
    run_policy_preset_simulations,
    run_rq3_scaling_benchmarks,
    train_and_benchmark_models,
)


def main() -> None:
    """Root Entry Procedure: Initializes AppState on the heap and runs headless or GUI mode."""
    base_dir: str = os.path.dirname(os.path.abspath(__file__))
    app_state = AppState()

    ingest_and_prepare_repository(app_state, base_dir)
    train_and_benchmark_models(app_state)
    run_policy_preset_simulations(app_state, base_dir)
    run_rq3_scaling_benchmarks(app_state)
    summary = export_benchmark_summary(app_state, base_dir)

    headless_flag: bool = ("--headless" in sys.argv) or (
        sys.platform != "win32" and sys.platform != "darwin" and os.environ.get("DISPLAY", "") == ""
    )
    if headless_flag:
        print_console_report(summary)
    else:
        try:
            import tkinter as tk
            from gui_policy_workbench import SafeRoadWorkbenchGUI

            root_win = tk.Tk()
            _gui = SafeRoadWorkbenchGUI(root_win, app_state)
            root_win.mainloop()
        except ImportError:
            print("Tkinter GUI library not found in this Python environment; falling back to console report:")
            print_console_report(summary)


if __name__ == "__main__":
    main()
