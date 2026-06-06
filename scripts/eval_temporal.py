#!/usr/bin/env python3
"""Phase 8 — temporal-stability quantification for the IVP renders.

The two IVP baselines (``ivp_per_frame.mp4`` vs ``ivp_full.mp4``) differ
only in two knobs:

* ``BboxSmoother`` (One Euro filter on ``(cx, cy, w, h)``) — ON in full,
  OFF in per-frame (a fresh smoother is constructed per frame so it
  passes the raw bbox straight through).
* ``TemporalState.ema`` (K-means cluster EMA across frames in
  ``cel_shade``) — 0.85 in full, 0.0 in per-frame.

Both are "temporal stability" interventions per ``spec.md`` §5; the
re-id / HF numbers reported in RESULTS.md §2–3 are per-sampled-frame
so they don't capture the difference. This script adds two cheap,
post-hoc metrics that do.

Metric 1 — **bbox jitter** (px/frame).
    Re-run the project's Haar detector on every frame of the output
    video and measure the L2 distance between consecutive bbox
    centres. Without One Euro the raw Haar bbox dances frame-to-frame
    even when the head is still; with One Euro it stays put.

Metric 2 — **head-region frame-to-frame MAD** (intensity units).
    Mean absolute difference between consecutive frames inside the
    head ellipse ROI. Both clips share the same source motion, so the
    *excess* MAD over the control clip is the cel-shade flicker
    (K-means cluster centres re-permuting / shifting between frames).

Both metrics are computed on the **rendered output videos** (no
re-render needed) so the script is fast (~5 s per clip).

Outputs (under ``<eval-dir>/``):

* ``temporal_summary.csv`` — one row per baseline with both metrics.
* ``temporal_summary.png`` — grouped bar chart (bbox jitter | head MAD).

Run::

    .venv/bin/python scripts/eval_temporal.py
    .venv/bin/python scripts/eval_temporal.py --clip-stem dancing
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from avatarshield.detect import Bbox, HaarFaceDetector, head_ellipse_mask  # noqa: E402
from avatarshield.video_io import VideoReader  # noqa: E402
from scripts.eval_render_baselines import BASELINE_LABELS, BASELINE_ORDER  # noqa: E402
from scripts.eval_freq import _try_import_matplotlib  # noqa: E402


# ---------------------------------------------------------------------------
# Per-clip statistics — single-pass through the video.
# ---------------------------------------------------------------------------


@dataclass
class TemporalStats:
    name: str
    n_frames: int
    bbox_jitter_mean_px: float
    bbox_jitter_std_px: float
    head_mad_mean: float
    head_mad_std: float
    n_jitter: int  # frames that contributed to bbox jitter (need a bbox in both t and t-1)
    n_mad: int     # frames that contributed to head MAD (need a head mask in t)


def _bbox_center(b: Bbox) -> tuple[float, float]:
    return (b.cx, b.cy)


def analyse_video(
    video_path: Path, detector: HaarFaceDetector
) -> TemporalStats:
    """Walk the clip once, recording bbox jitter + head-region MAD."""
    jitters: list[float] = []
    mads: list[float] = []
    prev_bbox: Bbox | None = None
    prev_frame: np.ndarray | None = None
    n_frames = 0

    reader = VideoReader(video_path)
    try:
        for frame in reader:
            n_frames += 1
            bbox = detector.detect(frame)

            if bbox is not None and prev_bbox is not None:
                ax, ay = _bbox_center(bbox)
                bx, by = _bbox_center(prev_bbox)
                jitters.append(float(math.hypot(ax - bx, ay - by)))

            if bbox is not None and prev_frame is not None:
                h, w = frame.shape[:2]
                mask = head_ellipse_mask(bbox, (w, h)) > 0
                if mask.any():
                    diff = cv2.absdiff(frame, prev_frame)
                    # ``mean`` on a 2-D mask wants single-channel; collapse first.
                    diff_gray = diff.mean(axis=2).astype(np.float32)
                    mads.append(float(diff_gray[mask].mean()))

            prev_bbox = bbox if bbox is not None else prev_bbox
            prev_frame = frame
    finally:
        reader.close()

    def _safe_mean(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else math.nan

    def _safe_std(xs: list[float]) -> float:
        return float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0

    return TemporalStats(
        name=video_path.stem,
        n_frames=n_frames,
        bbox_jitter_mean_px=_safe_mean(jitters),
        bbox_jitter_std_px=_safe_std(jitters),
        head_mad_mean=_safe_mean(mads),
        head_mad_std=_safe_std(mads),
        n_jitter=len(jitters),
        n_mad=len(mads),
    )


# ---------------------------------------------------------------------------
# Writers.
# ---------------------------------------------------------------------------


def write_csv(path: Path, stats: list[TemporalStats]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "baseline", "n_frames",
            "bbox_jitter_mean_px", "bbox_jitter_std_px", "n_jitter",
            "head_mad_mean", "head_mad_std", "n_mad",
        ])
        for s in stats:
            writer.writerow([
                s.name, s.n_frames,
                f"{s.bbox_jitter_mean_px:.4f}", f"{s.bbox_jitter_std_px:.4f}",
                s.n_jitter,
                f"{s.head_mad_mean:.4f}", f"{s.head_mad_std:.4f}",
                s.n_mad,
            ])


def plot_grouped_bars(path: Path, stats: list[TemporalStats]) -> None:
    plt = _try_import_matplotlib()
    if plt is None:
        return
    labels = [BASELINE_LABELS.get(s.name, s.name) for s in stats]
    jitter = [s.bbox_jitter_mean_px for s in stats]
    mad = [s.head_mad_mean for s in stats]
    x = np.arange(len(stats))
    width = 0.4
    fig, ax_left = plt.subplots(figsize=(9, 4.5))
    ax_right = ax_left.twinx()

    bars_left = ax_left.bar(x - width / 2, jitter, width=width,
                            color="#4477aa", label="bbox jitter (px/frame, mean)")
    bars_right = ax_right.bar(x + width / 2, mad, width=width,
                              color="#ee7733", label="head MAD (intensity, mean)")

    ax_left.set_xticks(x)
    ax_left.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax_left.set_ylabel("bbox jitter (px/frame)")
    ax_right.set_ylabel("head-region MAD (0-255)")
    ax_left.set_title("Phase 8 - temporal stability (lower = stabler)")
    ax_left.grid(axis="y", linestyle=":", alpha=0.5)

    for rect, v in zip(bars_left, jitter):
        ax_left.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                     f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    for rect, v in zip(bars_right, mad):
        ax_right.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                      f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    # Single legend for both axes
    handles = [bars_left, bars_right]
    ax_left.legend(handles, ["bbox jitter (px/frame)", "head MAD (intensity)"],
                   loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--baseline-dir", type=Path, default=None,
        help="dir containing <baseline>.mp4 (default: data/output/phase8/<clip-stem>/)",
    )
    parser.add_argument("--clip-stem", default="clip")
    parser.add_argument(
        "--baselines", default="control,ivp_per_frame,ivp_full",
        help=(
            "comma-separated subset of " + ",".join(BASELINE_ORDER)
            + " (default: control,ivp_per_frame,ivp_full)"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="dir to write CSV + PNG into (default: <baseline-dir>/eval/)",
    )
    args = parser.parse_args(argv)

    baseline_dir = (
        args.baseline_dir
        if args.baseline_dir is not None
        else ROOT / "data" / "output" / "phase8" / args.clip_stem
    )
    if not baseline_dir.exists():
        print(f"baseline dir not found: {baseline_dir}", file=sys.stderr)
        print("run scripts/eval_render_baselines.py first.", file=sys.stderr)
        return 2

    output_dir = (
        args.output_dir if args.output_dir is not None else baseline_dir / "eval"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    names = [n.strip() for n in args.baselines.split(",") if n.strip()]
    detector = HaarFaceDetector()
    print(f"[eval-temporal] baseline dir: {baseline_dir}")
    print(f"[eval-temporal] baselines: {', '.join(names)}")

    stats: list[TemporalStats] = []
    for name in names:
        path = baseline_dir / f"{name}.mp4"
        if not path.exists():
            print(f"  [skip] {name}: missing {path}")
            continue
        print(f"  [{name}] scanning {path.name} ...", flush=True)
        s = analyse_video(path, detector)
        s.name = name  # use the baseline name, not the file stem
        stats.append(s)
        print(
            f"  [{name}] n_frames={s.n_frames} "
            f"bbox_jitter_px={s.bbox_jitter_mean_px:.3f}+-{s.bbox_jitter_std_px:.3f} "
            f"head_MAD={s.head_mad_mean:.3f}+-{s.head_mad_std:.3f}"
        )

    write_csv(output_dir / "temporal_summary.csv", stats)
    plot_grouped_bars(output_dir / "temporal_summary.png", stats)
    print(f"[eval-temporal] wrote {output_dir / 'temporal_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
