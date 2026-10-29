import subprocess
import sys
from pathlib import Path


def test_api_import_does_not_load_training_framework():
    project_root = Path(__file__).resolve().parents[1]
    script = """
import builtins

original_import = builtins.__import__

def reject_torch(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise AssertionError("API import must not load the training framework")
    return original_import(name, *args, **kwargs)

builtins.__import__ = reject_torch
import api.main
"""

    subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
