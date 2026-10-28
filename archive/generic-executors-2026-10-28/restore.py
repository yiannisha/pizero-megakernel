"""Restore the archived experiments to a new directory without overwriting work."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

source=Path(__file__).resolve().parent
if len(sys.argv)!=2:raise SystemExit('Usage: restore.py NEW_DIRECTORY')
destination=Path(sys.argv[1]).expanduser().resolve()
if destination.exists():raise SystemExit('Destination already exists; nothing changed')
manifest=json.loads((source/'MANIFEST.json').read_text())
for name,expected in manifest['snapshot_sha256'].items():
    if hashlib.sha256((source/name).read_bytes()).hexdigest()!=expected:raise SystemExit('Archive hash mismatch: '+name)
destination.mkdir(parents=True)
for name in ('native_pi0','scripts','benchmarks','results'):shutil.copytree(source/name,destination/name)
for name in ('README.md','PLAN.md','upstreams.json'):shutil.copy2(source/name,destination/name)
(destination/'third_party').symlink_to((source/'third_party').resolve(),target_is_directory=True)
print('Verified historical snapshot restored to',destination)
