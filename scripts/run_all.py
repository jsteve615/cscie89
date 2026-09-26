"""Run every script in order in one shared namespace (like notebook cells).

Usage (from the repository root):  python scripts/run_all.py
"""
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
namespace = {"__name__": "__main__"}
for script in sorted(SCRIPTS_DIR.glob("[0-9][0-9]_*.py")):
    print(f"\n===== {script.name} =====")
    exec(compile(script.read_text(), str(script), "exec"), namespace)
