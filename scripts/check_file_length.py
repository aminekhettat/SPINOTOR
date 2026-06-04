"""File-length quality gate.

Enforces per-folder maximum line counts to prevent unbounded file growth.
Current thresholds are set to the next clean ceiling above existing maximums,
giving every file a small headroom while blocking runaway accumulation.

Roadmap targets (to be tightened in future releases):
  src/ui/        →  3 000 lines  (requires main_window.py refactor into panels)
  src/control/   →  1 000 lines  (requires splitting calibrators)
  src/core/      →    800 lines
  src/           →    600 lines  (default)
  tests/         →  1 000 lines
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
# Each entry maps a path glob (relative to the project root) to the maximum
# number of lines allowed.  The first matching pattern wins.
_THRESHOLDS: list[tuple[str, int]] = [
    # UI module: main_window.py grown with auto-calibration Phase 1 additions
    ("src/ui/**/*.py", 7_300),
    # FOC controller grown with STSMO/ActiveFlux/MRAS additions
    ("src/control/foc_controller.py", 2_800),
    # Control algorithms: calibrators are large but bounded
    ("src/control/**/*.py", 1_600),
    # Core simulation engine and models
    ("src/core/**/*.py", 1_100),
    # Utilities and visualization
    ("src/utils/**/*.py", 700),
    ("src/visualization/**/*.py", 850),
    # Hardware abstraction (thin wrappers expected)
    ("src/hardware/**/*.py", 400),
    # Test files tolerate some length for parametrize / fixture tables
    ("tests/**/*.py", 1_500),
    # Examples are standalone scripts — generous limit
    ("examples/**/*.py", 2_000),
    # Default catch-all
    ("**/*.py", 600),
]


def _threshold_for(path: Path, root: Path) -> int:
    # Use forward-slash relative path for reliable cross-platform prefix matching.
    rel = path.relative_to(root).as_posix()
    for pattern, limit in _THRESHOLDS:
        # Strip leading **/ for simple prefix tests; exact suffix match otherwise.
        bare = pattern.lstrip("*/")
        if rel.startswith(bare.split("*")[0].rstrip("/")):
            return limit
    return 600  # fallback


def main() -> int:
    root = Path(__file__).resolve().parent.parent

    # Directories to scan (relative to project root)
    scan_dirs = ["src", "tests", "examples"]
    exclude_dirs = {"__pycache__", ".venv", ".git", "docs/_build", "data"}

    failures: list[tuple[Path, int, int]] = []

    for scan in scan_dirs:
        scan_path = root / scan
        if not scan_path.exists():
            continue
        for py_file in sorted(scan_path.rglob("*.py")):
            # Skip excluded sub-trees
            parts = set(py_file.relative_to(root).parts)
            if parts & exclude_dirs:
                continue

            lines = py_file.read_text(encoding="utf-8", errors="replace").count("\n") + 1
            limit = _threshold_for(py_file, root)
            if lines > limit:
                failures.append((py_file.relative_to(root), lines, limit))

    if failures:
        print("File-length gate FAILED — the following files exceed their limits:\n")
        print(f"  {'File':<60} {'Lines':>6}  {'Limit':>6}  {'Over':>6}")
        print(f"  {'-'*60} {'------':>6}  {'------':>6}  {'------':>6}")
        for rel, count, limit in failures:
            over = count - limit
            print(f"  {str(rel):<60} {count:>6}  {limit:>6}  +{over:>5}")
        print(
            f"\n{len(failures)} file(s) exceeded limits.  "
            "Refactor or split before merging."
        )
        return 1

    checked = sum(
        1
        for d in scan_dirs
        for _ in (root / d).rglob("*.py")
        if (root / d).exists()
    )
    print(f"File-length gate PASSED ({checked} files checked).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
