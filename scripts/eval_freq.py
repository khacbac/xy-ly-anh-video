#!/usr/bin/env python3
"""Phase 8 — FFT high-frequency energy ratio (privacy proxy, S05.02).

For every baseline rendered by ``scripts/eval_render_baselines.py`` we
compute the *high-frequency energy ratio* per sampled frame:

    F(u, v)   = fftshift(fft2(L(f)))                  # luminance only
    HF energy = sum |F(u, v)|^2  over { (u, v) : r > r0 }
    Total     = sum |F(u, v)|^2
    HF ratio  = HF energy / Total                     in [0, 1]

with the cutoff radius ``r0 = cutoff_frac * min(H, W) / 2`` (default
``cutoff_frac = 0.20`` — Nyquist fraction per ``spec.md`` §7.2).

The expected ordering (low → high HF retained) is::

    blur  <  mosaic*  <  ivp-full  ~  ivp-per-frame  <  control

(*mosaic introduces sharp 8 px block edges and so sometimes lands above
ivp; that's the "blur kills utility too" story the report uses to defend
cel-shade as the better trade-off.)

Outputs (under ``data/output/phase8/<clip-stem>/eval/``):

* ``hf_ratio_per_frame.csv`` — one row per sampled frame, columns =
  ``frame_index, control, blur, mosaic, ivp_per_frame, ivp_full``.
* ``hf_ratio_summary.csv``   — per-baseline mean / std / median.
* ``hf_ratio_summary.png``   — bar chart with error bars.
* ``hf_ratio_timeline.png``  — line chart over sampled frames.

Run::

    .venv/bin/python scripts/eval_freq.py
    .venv/bin/python scripts/eval_freq.py --clip-stem clip --sample-stride 15
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from avatarshield.detect import HaarFaceDetector, head_ellipse_mask  # noqa: E402
from avatarshield.video_io import VideoReader  # noqa: E402
from scripts.eval_render_baselines import (  # noqa: E402
    BASELINE_LABELS,
    BASELINE_ORDER,
)


# ---------------------------------------------------------------------------
# Core metric.
# ---------------------------------------------------------------------------


def hf_energy_ratio(luma: np.ndarray, cutoff_frac: float = 0.20) -> float:
    """High-frequency energy fraction of a single-channel image.

    The luminance plane is FFT'd (centred via ``fftshift``); we sum
    ``|F|^2`` outside a disc of radius ``cutoff_frac * min(H, W) / 2``
    and divide by the total energy. Following ``spec.md`` §7.2 the
    cutoff is expressed as a fraction of Nyquist so the metric is
    resolution-independent.
    """
    if luma.ndim != 2:
        raise ValueError(f"expected single-channel image, got shape {luma.shape}")
    f = np.fft.fftshift(np.fft.fft2(luma.astype(np.float32)))
    power = (f.real * f.real) + (f.imag * f.imag)

    h, w = luma.shape
    cy, cx = h / 2.0, w / 2.0
    yy, xx = np.indices(luma.shape)
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r0 = cutoff_frac * 0.5 * float(min(h, w))
    hf_mask = rr > r0

    total = float(power.sum())
    if total <= 0.0:
        return 0.0
    return float(power[hf_mask].sum()) / total


def luminance_from_bgr(frame_bgr: np.ndarray) -> np.ndarray:
    """ITU-R BT.601 luminance (single channel, float32 in [0, 255])."""
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)


# ---------------------------------------------------------------------------
# Frame sampling across a video.
# ---------------------------------------------------------------------------


def sample_frame_indices(frame_count: int, stride: int) -> list[int]:
    if frame_count <= 0:
        return []
    stride = max(1, int(stride))
    return list(range(0, frame_count, stride))


def _head_crop(
    frame_bgr: np.ndarray, detector: HaarFaceDetector
) -> np.ndarray | None:
    """Return a square head-ellipse crop, or ``None`` if no face detected.

    Matches the head ROI the IVP pipeline stylizes (``head_ellipse_mask``
    with the spec.md §5.1 extension factors). The crop is the bounding
    box of the ellipse, padded square so the FFT sees a balanced
    frequency response.
    """
    bbox = detector.detect(frame_bgr)
    if bbox is None:
        return None
    h, w = frame_bgr.shape[:2]
    mask = head_ellipse_mask(bbox, (w, h))
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    side = max(y1 - y0, x1 - x0)
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    half = side // 2
    x0p = max(0, cx - half)
    y0p = max(0, cy - half)
    x1p = min(w, cx + half)
    y1p = min(h, cy + half)
    if x1p <= x0p or y1p <= y0p:
        return None
    return frame_bgr[y0p:y1p, x0p:x1p]


def hf_ratios_for_video(
    video_path: Path,
    sample_indices: Iterable[int],
    cutoff_frac: float,
    *,
    head_only: bool = False,
    detector: HaarFaceDetector | None = None,
) -> list[float]:
    """Run :func:`hf_energy_ratio` on every ``sample_indices`` frame.

    Uses sequential decode (skipping with ``read_frame`` would seek on
    every step and is slower than reading all frames). Frames outside
    the sample set are decoded but discarded.

    When ``head_only=True`` the metric is computed only on the head
    ellipse crop — a fair per-region measurement for the cel-shade
    pipeline, which only stylizes the head and leaves the body /
    background untouched. Frames where the detector misses the face
    are skipped (no NaN row).
    """
    sample_set = set(sample_indices)
    ratios: list[float] = []
    reader = VideoReader(video_path)
    try:
        for idx, frame in enumerate(reader):
            if idx not in sample_set:
                continue
            if head_only:
                if detector is None:
                    raise RuntimeError("head_only=True requires a detector")
                crop = _head_crop(frame, detector)
                if crop is None:
                    continue
                ratios.append(hf_energy_ratio(luminance_from_bgr(crop), cutoff_frac))
            else:
                ratios.append(hf_energy_ratio(luminance_from_bgr(frame), cutoff_frac))
    finally:
        reader.close()
    return ratios


# ---------------------------------------------------------------------------
# Summary + plotting.
# ---------------------------------------------------------------------------


@dataclass
class BaselineSummary:
    name: str
    mean: float
    std: float
    median: float
    n: int


def summarise(name: str, values: list[float]) -> BaselineSummary:
    if not values:
        return BaselineSummary(name, math.nan, math.nan, math.nan, 0)
    arr = np.asarray(values, dtype=np.float64)
    return BaselineSummary(
        name=name,
        mean=float(arr.mean()),
        std=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        median=float(np.median(arr)),
        n=int(arr.size),
    )


def write_per_frame_csv(
    path: Path, sample_indices: list[int], by_baseline: dict[str, list[float]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [n for n in BASELINE_ORDER if n in by_baseline]
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame_index", *names])
        for i, idx in enumerate(sample_indices):
            row = [idx]
            for n in names:
                values = by_baseline[n]
                row.append(f"{values[i]:.6f}" if i < len(values) else "")
            writer.writerow(row)


def write_summary_csv(path: Path, summaries: list[BaselineSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["baseline", "mean", "std", "median", "n"])
        for s in summaries:
            writer.writerow(
                [s.name, f"{s.mean:.6f}", f"{s.std:.6f}", f"{s.median:.6f}", s.n]
            )


def _try_import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as exc:
        print(f"[warn] matplotlib unavailable, skipping plots ({exc})")
        return None


def plot_summary_bar(path: Path, summaries: list[BaselineSummary]) -> None:
    plt = _try_import_matplotlib()
    if plt is None:
        return
    labels = [BASELINE_LABELS.get(s.name, s.name) for s in summaries]
    means = [s.mean for s in summaries]
    stds = [s.std for s in summaries]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(range(len(summaries)), means, yerr=stds, capsize=4,
                  color=["#888", "#4477aa", "#ee7733", "#cc3311", "#009988"])
    ax.set_xticks(range(len(summaries)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("HF energy ratio (mean +- std)")
    ax.set_title("Phase 8 - HF energy ratio per baseline (lower = more anonymisation)")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    for rect, s in zip(bars, summaries):
        ax.text(rect.get_x() + rect.get_width() / 2,
                rect.get_height(),
                f"{s.mean:.3f}",
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_timeline(path: Path, sample_indices: list[int],
                  by_baseline: dict[str, list[float]]) -> None:
    plt = _try_import_matplotlib()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name in BASELINE_ORDER:
        if name not in by_baseline:
            continue
        ax.plot(sample_indices[: len(by_baseline[name])], by_baseline[name],
                label=BASELINE_LABELS.get(name, name), linewidth=1.4)
    ax.set_xlabel("frame index")
    ax.set_ylabel("HF energy ratio")
    ax.set_title("Phase 8 - HF energy ratio over time")
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=None,
        help="dir containing <baseline>.mp4 (default: data/output/phase8/<clip-stem>/)",
    )
    parser.add_argument(
        "--clip-stem",
        default="clip",
        help="clip stem used by eval_render_baselines (default: clip)",
    )
    parser.add_argument(
        "--baselines",
        default=None,
        help=(
            "comma-separated subset of " + ",".join(BASELINE_ORDER)
            + " (default: all baselines present on disk)"
        ),
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=15,
        help="sample every Nth frame (default: 15)",
    )
    parser.add_argument(
        "--cutoff-frac",
        type=float,
        default=0.20,
        help="HF cutoff radius as fraction of Nyquist (default: 0.20)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="dir to write CSV + PNG into (default: <baseline-dir>/eval/)",
    )
    parser.add_argument(
        "--head-only",
        action="store_true",
        help=(
            "compute HF ratio on the head-ellipse crop only (per-region "
            "measurement — fair for the IVP pipeline which leaves body / "
            "background untouched). Skips frames with no face detected."
        ),
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

    if args.baselines:
        names = [n.strip() for n in args.baselines.split(",") if n.strip()]
    else:
        names = [n for n in BASELINE_ORDER if (baseline_dir / f"{n}.mp4").exists()]
    if not names:
        print(f"no baseline .mp4 found under {baseline_dir}", file=sys.stderr)
        return 2

    # Anchor sampling to the control video's frame count so every baseline
    # uses the *same* frame indices — the per-baseline CSV needs aligned rows.
    anchor = baseline_dir / f"{names[0]}.mp4"
    with VideoReader(anchor) as r:
        frame_count = r.meta.frame_count
    indices = sample_frame_indices(frame_count, args.sample_stride)

    print(f"[eval-freq] baseline dir: {baseline_dir}")
    print(f"[eval-freq] frames sampled: {len(indices)} / {frame_count}")
    print(f"[eval-freq] cutoff frac: {args.cutoff_frac}")
    print(f"[eval-freq] head-only mode: {args.head_only}")

    detector = HaarFaceDetector() if args.head_only else None

    by_baseline: dict[str, list[float]] = {}
    for name in names:
        path = baseline_dir / f"{name}.mp4"
        if not path.exists():
            print(f"  [skip] {name}: missing {path}")
            continue
        print(f"  [{name}] FFT-scanning {path.name} ...", flush=True)
        ratios = hf_ratios_for_video(
            path, indices, args.cutoff_frac,
            head_only=args.head_only, detector=detector,
        )
        by_baseline[name] = ratios
        summary = summarise(name, ratios)
        print(
            f"  [{name}] mean={summary.mean:.4f} std={summary.std:.4f} "
            f"median={summary.median:.4f} n={summary.n}"
        )

    output_dir = (
        args.output_dir if args.output_dir is not None else baseline_dir / "eval"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = "_head" if args.head_only else ""
    if not args.head_only:
        # Per-frame CSV only makes sense with aligned indices (whole-frame mode).
        write_per_frame_csv(
            output_dir / f"hf_ratio_per_frame{suffix}.csv", indices, by_baseline
        )
        plot_timeline(
            output_dir / f"hf_ratio_timeline{suffix}.png", indices, by_baseline
        )
    summaries = [summarise(name, by_baseline[name]) for name in by_baseline]
    write_summary_csv(output_dir / f"hf_ratio_summary{suffix}.csv", summaries)
    plot_summary_bar(output_dir / f"hf_ratio_summary{suffix}.png", summaries)

    print(
        f"[eval-freq] wrote summary to "
        f"{output_dir / f'hf_ratio_summary{suffix}.csv'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
