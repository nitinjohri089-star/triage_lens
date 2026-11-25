"""Train the emergency triage models and save project outputs."""

from pathlib import Path
from runpy import run_module
import sys


PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

if __name__ == "__main__":
    run_module("triage_app.train", run_name="__main__")
