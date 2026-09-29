"""The evaluation script must fail loudly when trained models are missing."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RUN_WITH_MODELS_DIR = """
import sys
from pathlib import Path
sys.path.insert(0, {root!r})
import config
config.MODELS_FIXED_DIR = Path({models_dir!r})
import runpy
runpy.run_path({script!r}, run_name="__main__")
"""


def test_evaluation_exits_nonzero_when_models_missing(tmp_path):
    code = RUN_WITH_MODELS_DIR.format(
        root=str(ROOT),
        models_dir=str(tmp_path),
        script=str(ROOT / "scripts" / "test_models_refactored.py"),
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=300
    )
    assert result.returncode != 0
    assert "Missing" in result.stdout + result.stderr
