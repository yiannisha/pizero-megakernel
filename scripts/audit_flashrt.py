"""Reproducible source inventory for the native port (not a semantic audit).

Records every source in the required implementation families and all calls in
the four primary model pipelines. Run again after changing the pinned revision.
"""

import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = (
    "csrc/gemm/fp4/**/*", "csrc/gemm/mega/**/*", "csrc/gemm/fp8_smallM_*",
    "csrc/kernels/pi05/**/*", "csrc/fused_fp4/**/*", "csrc/attention/**/*",
    "csrc/kernels/fused_qkv_attention.*", "csrc/kernels/fusion.*",
    "csrc/kernels/attention_seqused_fused.*",
)


def local_dependencies(source, seeds):
    """Follow repository-local imports/includes without importing GPU modules.

    Dynamic imports and preprocessor-selected implementations still need human
    review. Unresolved C++ includes are recorded rather than silently dropped.
    """
    pending = list(seeds)
    visited, edges, unresolved = set(), [], []
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        content = path.read_text(errors="replace")
        dependencies = []
        if path.suffix == ".py":
            tree = ast.parse(content)
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    if node.level:
                        package = list(path.relative_to(source).parts[:-1])
                        package = package[:len(package) - node.level + 1]
                        base = ".".join(package + ([node.module] if node.module else []))
                    else:
                        base = node.module or ""
                    names = [base] + [base + "." + alias.name for alias in node.names]
                for name in names:
                    if not name.startswith("flash_rt"):
                        continue
                    module = source.joinpath(*name.split("."))
                    for candidate in (module.with_suffix(".py"), module / "__init__.py"):
                        if candidate.is_file():
                            dependencies.append((candidate, node.lineno))
        elif path.suffix in (".cu", ".cuh", ".h", ".hpp", ".cpp", ".cc"):
            for line, text in enumerate(content.splitlines(), 1):
                match = re.match(r'\s*#\s*include\s*"([^"]+)"', text)
                if not match:
                    continue
                name = match.group(1)
                candidates = (path.parent / name, source / "csrc" / name, source / name)
                candidate = next((p.resolve() for p in candidates if p.is_file()), None)
                if candidate is not None and candidate.is_relative_to(source):
                    dependencies.append((candidate, line))
                else:
                    unresolved.append({"source": str(path.relative_to(source)), "line": line, "include": name})
        for target, line in dependencies:
            edges.append({"source": str(path.relative_to(source)), "line": line,
                          "target": str(target.relative_to(source))})
            pending.append(target)
    return visited, sorted(edges, key=lambda r: (r["source"], r["line"], r["target"])), unresolved
PIPELINES = (
    "flash_rt/models/pi0/pipeline_rtx.py",
    "flash_rt/models/pi0/pipeline_thor.py",
    "flash_rt/models/pi05/pipeline_rtx.py",
    "flash_rt/models/pi05/pipeline_thor_fp4.py",
)


def inventory():
    source = (ROOT / "third_party/FlashRT").resolve()
    expected = json.loads((ROOT / "upstreams.json").read_text())["FlashRT"]["rev"]
    actual = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != expected:
        raise RuntimeError(f"FlashRT revision mismatch: {actual} != {expected}")
    seeds = {p for pattern in FAMILIES for p in source.glob(pattern) if p.is_file()}
    seeds.update(source / p for p in PIPELINES)
    files, dependencies, unresolved = local_dependencies(source, seeds)
    calls = {}
    for name in PIPELINES:
        tree = ast.parse((source / name).read_text())
        calls[name] = sorted(
            ({"symbol": ast.unparse(n.func), "line": n.lineno}
             for n in ast.walk(tree) if isinstance(n, ast.Call)),
            key=lambda row: (row["line"], row["symbol"]),
        )
    return {
        "revision": actual,
        "scope": "Required source families; includes vendored attention templates. Semantic review remains separate.",
        "sources": [
            {"path": str(p.relative_to(source)),
             "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
             "lines": len(p.read_bytes().splitlines())}
            for p in sorted(files)
        ],
        "pipeline_calls": calls,
        "local_dependencies": dependencies,
        "unresolved_quoted_includes": sorted(unresolved, key=lambda r: (r["source"], r["line"])),
    }


if __name__ == "__main__":
    result = inventory()
    dest = ROOT / "native_pi0/flashrt_inventory.json"
    dest.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Inventoried {len(result['sources'])} sources and {sum(map(len, result['pipeline_calls'].values()))} pipeline calls")
