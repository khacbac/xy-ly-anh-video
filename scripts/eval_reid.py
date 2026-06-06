#!/usr/bin/env python3
"""Phase 8 — face re-identification rate (privacy proxy, S08).

For each baseline rendered by ``scripts/eval_render_baselines.py`` we
test whether a face-recognition embedder can still match the stylized
frame back to the original. The protocol per ``spec.md`` §7.1:

* Sample every ``--sample-stride`` frame (default 15) from the
  ``control.mp4`` (original) and the baseline under test.
* Crop the largest detected face with the Haar detector already used by
  the render pipeline. Skip frames with no detection.
* Embed each crop with ``facenet-pytorch`` (InceptionResnetV1, ``vggface2``
  weights — lightweight, CPU-friendly).
* Pair each baseline embedding with its corresponding control embedding;
  report:
    - ``embed_distance``  — cosine distance per pair (lower = more
      identifiable).
    - ``reid_rate``       — fraction with distance below the FaceNet
      verification threshold (default 0.7 cosine; spec §7.1).

The whole thing is optional: when ``facenet-pytorch`` is not installed
the script prints a banner and writes ``reid_summary.csv`` rows with NaN
values so ``RESULTS.md`` can still link to it.

Outputs (under ``data/output/phase8/<clip-stem>/eval/``):

* ``reid_per_frame.csv``  — frame_index, baseline, distance, match
* ``reid_summary.csv``    — baseline, mean_dist, median_dist, reid_rate, n
* ``reid_summary.png``    — bar chart (re-id rate per baseline)

Run::

    .venv/bin/python scripts/eval_reid.py
    .venv/bin/python scripts/eval_reid.py --sample-stride 30 --threshold 0.8
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

from avatarshield.detect import HaarFaceDetector  # noqa: E402
from avatarshield.video_io import VideoReader  # noqa: E402
from scripts.eval_render_baselines import (  # noqa: E402
    BASELINE_LABELS,
    BASELINE_ORDER,
)
from scripts.eval_freq import _try_import_matplotlib  # noqa: E402


# ---------------------------------------------------------------------------
# Embedder — optional facenet-pytorch dependency.
# ---------------------------------------------------------------------------


def _try_load_embedder():
    """Return a ``(embed_fn, label)`` pair or ``(None, reason_str)``.

    ``embed_fn`` takes a list of HxWx3 BGR uint8 face crops and returns
    an ``(N, D)`` float32 array of L2-normalised embeddings. We use
    cosine distance downstream, so the L2 normalisation makes
    ``distance = 1 - dot(a, b)`` cheap to compute.
    """
    try:
        import torch
        from facenet_pytorch import InceptionResnetV1
    except Exception as exc:  # noqa: BLE001
        return None, f"facenet-pytorch / torch not importable ({exc!r})"

    device = "cpu"  # spec calls out CPU-portable
    model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    @torch.no_grad()
    def embed(crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, 512), dtype=np.float32)
        batch = []
        for img in crops:
            # facenet-pytorch expects 160x160 RGB float, normalised [-1, 1].
            resized = cv2.resize(img, (160, 160), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
            rgb = (rgb - 127.5) / 128.0
            batch.append(rgb.transpose(2, 0, 1))  # CHW
        tens = torch.from_numpy(np.stack(batch)).to(device)
        emb = model(tens).cpu().numpy()
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
        return (emb / norms).astype(np.float32)

    return embed, "facenet-pytorch (InceptionResnetV1 vggface2)"


# ---------------------------------------------------------------------------
# Frame sampling + face crop.
# ---------------------------------------------------------------------------


@dataclass
class FaceCrop:
    frame_index: int
    crop: np.ndarray  # BGR uint8, square


def sample_face_crops(
    video_path: Path,
    sample_indices: set[int],
    detector: HaarFaceDetector,
    *,
    crop_margin: float = 0.20,
) -> list[FaceCrop]:
    """Decode the video and grab a square face crop for each sample index.

    ``crop_margin`` enlarges the Haar bbox by N % on each side before
    cropping — FaceNet expects a tight head, not a tight face. Frames
    with no detection are simply skipped.
    """
    crops: list[FaceCrop] = []
    reader = VideoReader(video_path)
    try:
        for idx, frame in enumerate(reader):
            if idx not in sample_indices:
                continue
            bbox = detector.detect(frame)
            if bbox is None:
                continue
            h, w = frame.shape[:2]
            mx = int(round(bbox.w * crop_margin))
            my = int(round(bbox.h * crop_margin))
            x0 = max(0, bbox.x - mx)
            y0 = max(0, bbox.y - my)
            x1 = min(w, bbox.x + bbox.w + mx)
            y1 = min(h, bbox.y + bbox.h + my)
            if x1 <= x0 or y1 <= y0:
                continue
            crop = frame[y0:y1, x0:x1].copy()
            crops.append(FaceCrop(frame_index=idx, crop=crop))
    finally:
        reader.close()
    return crops


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


@dataclass
class PairResult:
    baseline: str
    frame_index: int
    distance: float
    match: bool


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """``1 - dot(a, b)``; both vectors assumed L2-normalised."""
    return float(1.0 - float(np.dot(a, b)))


def evaluate_baseline(
    baseline_name: str,
    baseline_video: Path,
    control_index_to_emb: dict[int, np.ndarray],
    detector: HaarFaceDetector,
    embed_fn,
    threshold: float,
) -> list[PairResult]:
    """Embed faces in ``baseline_video`` and pair them with control by index."""
    sample_indices = set(control_index_to_emb.keys())
    crops = sample_face_crops(baseline_video, sample_indices, detector)
    if not crops:
        return []
    embeds = embed_fn([c.crop for c in crops])
    out: list[PairResult] = []
    for crop, emb in zip(crops, embeds):
        anchor = control_index_to_emb.get(crop.frame_index)
        if anchor is None:
            continue
        dist = cosine_distance(anchor, emb)
        out.append(PairResult(
            baseline=baseline_name,
            frame_index=crop.frame_index,
            distance=dist,
            match=dist < threshold,
        ))
    return out


def write_per_frame_csv(path: Path, rows: list[PairResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["baseline", "frame_index", "distance", "match"])
        for r in rows:
            writer.writerow([
                r.baseline, r.frame_index, f"{r.distance:.6f}",
                "1" if r.match else "0",
            ])


@dataclass
class ReidSummary:
    baseline: str
    mean_dist: float
    median_dist: float
    reid_rate: float
    n: int


def summarise_baseline(name: str, rows: list[PairResult]) -> ReidSummary:
    if not rows:
        return ReidSummary(name, math.nan, math.nan, math.nan, 0)
    dists = np.asarray([r.distance for r in rows])
    matches = np.asarray([r.match for r in rows])
    return ReidSummary(
        baseline=name,
        mean_dist=float(dists.mean()),
        median_dist=float(np.median(dists)),
        reid_rate=float(matches.mean()),
        n=int(matches.size),
    )


def write_summary_csv(path: Path, summaries: list[ReidSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["baseline", "mean_dist", "median_dist", "reid_rate", "n"])
        for s in summaries:
            writer.writerow([
                s.baseline,
                f"{s.mean_dist:.6f}" if not math.isnan(s.mean_dist) else "",
                f"{s.median_dist:.6f}" if not math.isnan(s.median_dist) else "",
                f"{s.reid_rate:.6f}" if not math.isnan(s.reid_rate) else "",
                s.n,
            ])


def plot_summary(path: Path, summaries: list[ReidSummary]) -> None:
    plt = _try_import_matplotlib()
    if plt is None:
        return
    labels = [BASELINE_LABELS.get(s.baseline, s.baseline) for s in summaries]
    rates = [0.0 if math.isnan(s.reid_rate) else s.reid_rate for s in summaries]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(range(len(summaries)), rates, color=["#4477aa", "#ee7733", "#cc3311", "#009988"])
    ax.set_xticks(range(len(summaries)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Re-id rate (lower = more anonymisation)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Phase 8 - FaceNet re-id rate per baseline (vs control)")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    for rect, s in zip(bars, summaries):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                f"{s.reid_rate:.2f}" if not math.isnan(s.reid_rate) else "n/a",
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI.
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
        "--clip-stem", default="clip",
        help="clip stem used by eval_render_baselines (default: clip)",
    )
    parser.add_argument(
        "--baselines", default=None,
        help=(
            "comma-separated subset of " + ",".join(BASELINE_ORDER)
            + " (default: all baselines present on disk; control is the anchor)"
        ),
    )
    parser.add_argument("--sample-stride", type=int, default=15)
    parser.add_argument(
        "--threshold", type=float, default=0.7,
        help="cosine-distance threshold for 'matched' (default: 0.7 per spec §7.1)",
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

    embed_fn, embed_label = _try_load_embedder()
    if embed_fn is None:
        print("=" * 72)
        print("[eval-reid] SKIP — facenet-pytorch is not installed.")
        print(f"           reason: {embed_label}")
        print("           install with `pip install facenet-pytorch torch` "
              "to compute re-id metric.")
        print("=" * 72)
        # Write stub CSVs so RESULTS.md can still link to them.
        names = [n for n in BASELINE_ORDER
                 if (baseline_dir / f"{n}.mp4").exists() and n != "control"]
        summaries = [ReidSummary(n, math.nan, math.nan, math.nan, 0) for n in names]
        write_summary_csv(output_dir / "reid_summary.csv", summaries)
        return 0

    print(f"[eval-reid] embedder: {embed_label}")
    print(f"[eval-reid] threshold: cosine distance < {args.threshold}")

    control_path = baseline_dir / "control.mp4"
    if not control_path.exists():
        print(f"control video missing: {control_path}", file=sys.stderr)
        return 2

    with VideoReader(control_path) as r:
        frame_count = r.meta.frame_count
    stride = max(1, args.sample_stride)
    sample_indices = set(range(0, frame_count, stride))
    print(f"[eval-reid] frames sampled: {len(sample_indices)} / {frame_count}")

    detector = HaarFaceDetector()
    print(f"[eval-reid] [control] extracting + embedding faces ...", flush=True)
    control_crops = sample_face_crops(control_path, sample_indices, detector)
    control_embeds = embed_fn([c.crop for c in control_crops])
    control_index_to_emb: dict[int, np.ndarray] = {
        c.frame_index: e for c, e in zip(control_crops, control_embeds)
    }
    print(f"[eval-reid] [control] usable anchors: {len(control_index_to_emb)}")

    if args.baselines:
        names = [n.strip() for n in args.baselines.split(",") if n.strip()]
    else:
        names = [n for n in BASELINE_ORDER
                 if (baseline_dir / f"{n}.mp4").exists() and n != "control"]

    all_rows: list[PairResult] = []
    summaries: list[ReidSummary] = []
    for name in names:
        path = baseline_dir / f"{name}.mp4"
        if not path.exists():
            print(f"  [skip] {name}: missing {path}")
            continue
        print(f"  [{name}] embedding + pairing ...", flush=True)
        rows = evaluate_baseline(
            name, path, control_index_to_emb, detector, embed_fn, args.threshold
        )
        all_rows.extend(rows)
        s = summarise_baseline(name, rows)
        summaries.append(s)
        print(
            f"  [{name}] n={s.n} mean_dist={s.mean_dist:.3f} "
            f"median_dist={s.median_dist:.3f} reid_rate={s.reid_rate:.3f}"
        )

    write_per_frame_csv(output_dir / "reid_per_frame.csv", all_rows)
    write_summary_csv(output_dir / "reid_summary.csv", summaries)
    plot_summary(output_dir / "reid_summary.png", summaries)
    print(f"[eval-reid] wrote summary to {output_dir / 'reid_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
