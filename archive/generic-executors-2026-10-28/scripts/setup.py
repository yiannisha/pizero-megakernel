"""Run with the benchmark venv's Python after installing requirements.txt."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for name, spec in json.loads((ROOT / "upstreams.json").read_text()).items():
    dest = ROOT / "third_party" / name
    if not dest.exists():
        subprocess.run(["git", "clone", spec["url"], str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", spec["rev"]], check=True)
    if (dest / ".gitmodules").exists():
        subprocess.run(
            ["git", "-C", str(dest), "submodule", "update", "--init", "--recursive"],
            check=True,
        )
    if name == "Megakernels":
        for filename in ("megakernels-serial-barrier-init.patch", "megakernels-dynamic-instructions.patch"):
            patch = ROOT / "native_pi0/patches" / filename
            applied = subprocess.run(
                ["git", "-C", str(dest), "apply", "--reverse", "--check", str(patch)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            ).returncode == 0
            if not applied:
                subprocess.run(["git", "-C", str(dest), "apply", str(patch)], check=True)
import transformers

assert transformers.__version__ == "4.53.2"
assert sys.prefix != sys.base_prefix, (
    "Use an isolated virtual environment for the transformers overlay"
)
shutil.copytree(
    ROOT / "third_party/openpi/src/openpi/models_pytorch/transformers_replace",
    Path(transformers.__file__).parent,
    dirs_exist_ok=True,
)
print("Pinned upstreams and installed the required OpenPI transformers overlay.")
