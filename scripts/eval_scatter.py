#!/usr/bin/env python3
"""Phase 8 — joint HF-ratio × re-id scatter plot for RESULTS.md.

Reads ``hf_ratio_summary.csv`` and ``reid_summary.csv`` produced by
``scripts/eval_freq.py`` and ``scripts/eval_reid.py``, and writes a
scatter PNG where each point is one baseline plotted in
``(HF ratio, re-id rate)`` space. The cel-shade story per ``spec.md``
§7.5 lives in the lower-left quadrant: low HF ratio (texture
suppressed → privacy) while still keeping enough head shape for the
re-id rate to drop below the blur baseline.

Run::

    .venv/bin/python scripts/eval_scatter.py
    .venv/bin/python scripts/eval_scatter.py --clip-stem clip
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_freq import _try_import_matplotlib  # noqa: E402
from scripts.eval_render_baselines import BASELINE_LABELS, BASELINE_ORDER  # noqa: E402


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="") as fh:
        return list(csv.DictReader(fh))


def _to_float(value: str | None) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eval-dir", type=Path, default=None,
                        help="dir with hf_ratio_summary.csv + reid_summary.csv")
    parser.add_argument("--clip-stem", default="clip")
    parser.add_argument("--output", type=Path, default=None,
                        help="output PNG (default: <eval-dir>/joint_scatter.png)")
    args = parser.parse_args(argv)

    eval_dir = (
        args.eval_dir
        if args.eval_dir is not None
        else ROOT / "data" / "output" / "phase8" / args.clip_stem / "eval"
    )

    hf_rows = {r["baseline"]: r for r in _read_csv(eval_dir / "hf_ratio_summary.csv")}
    reid_rows = {r["baseline"]: r for r in _read_csv(eval_dir / "reid_summary.csv")}

    if not hf_rows:
        print(f"missing hf_ratio_summary.csv under {eval_dir}", file=sys.stderr)
        return 2

    plt = _try_import_matplotlib()
    if plt is None:
        return 1

    fig, ax = plt.subplots(figsize=(7, 5))
    palette = {
        "control": "#888888",
        "blur": "#4477aa",
        "mosaic": "#ee7733",
        "ivp_per_frame": "#cc3311",
        "ivp_full": "#009988",
    }

    plotted = 0
    for name in BASELINE_ORDER:
        if name not in hf_rows:
            continue
        hf = _to_float(hf_rows[name].get("mean"))
        # control has no re-id pair (it IS the anchor); plot it at y=1.0
        # as a visual reference for "fully identifiable".
        if name == "control":
            reid = 1.0
        else:
            reid = _to_float(reid_rows.get(name, {}).get("reid_rate"))
        if math.isnan(hf) or math.isnan(reid):
            continue
        ax.scatter(hf, reid, color=palette.get(name, "black"), s=120,
                   edgecolor="white", linewidth=1.5, zorder=3)
        ax.annotate(
            BASELINE_LABELS.get(name, name),
            (hf, reid),
            xytext=(8, 6), textcoords="offset points", fontsize=9,
        )
        plotted += 1

    if plotted == 0:
        print("no plottable points", file=sys.stderr)
        return 1

    ax.set_xlabel("HF energy ratio (lower = more texture suppressed)")
    ax.set_ylabel("Re-id rate (lower = harder to match to control)")
    ax.set_xlim(left=0)
    ax.set_ylim(-0.02, 1.05)
    ax.axhline(1.0, color="#999", linestyle=":", linewidth=0.8)
    ax.grid(linestyle=":", alpha=0.5)
    ax.set_title("Phase 8 - joint HF ratio x re-id rate per baseline")
    fig.tight_layout()

    output = args.output if args.output is not None else eval_dir / "joint_scatter.png"
    fig.savefig(output, dpi=140)
    plt.close(fig)
    print(f"[scatter] wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
