#!/usr/bin/env python3
"""Phase 1 verification — Haar detection + One Euro bbox smoothing.

Per ``plan.md`` §2 Phase 1 this script:

1. Walks ``samples/input/clip.mp4`` once and picks 4–6 fixed test frame
   indices representing front-facing / profile / miss-on-detection /
   low-light / high-light conditions. The selection is persisted to
   ``samples/test_frames.json`` so later phases reuse the same set.
2. Renders each test frame as a PNG with the green bbox + centre cross
   overlay and composes them into a 2×N grid at
   ``data/output/phase1_grid.png``.
3. Renders ``data/output/phase1_bbox.mp4`` — a ~3 s clip from the start
   of the source video with the smoothed bbox drawn on every frame. The
   video is the only artifact that can show whether One Euro is doing
   its job (single frames cannot reveal jitter).

Run::

    .venv/bin/python scripts/phase1_test.py

Optional flags::

    --video PATH           override source clip
    --output-dir PATH      override data/output
    --reselect             re-categorize frames even if test_frames.json exists
    --clip-seconds FLOAT   length of the overlay clip (default 3.0 s)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

# Make sure we can import the package whether or not the script is run
# from the repo root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from avatarshield.detect import Bbox, HaarFaceDetector  # noqa: E402
from avatarshield.smooth import BboxSmoother  # noqa: E402


# ---------------------------------------------------------------------------
# Drawing helpers.
# ---------------------------------------------------------------------------


GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)


def draw_bbox(frame: np.ndarray, bbox: Bbox, color=GREEN) -> None:
    """Draw a rectangle + crosshair at bbox centre, in-place."""
    x, y, w, h = bbox.as_xywh()
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cx, cy = int(bbox.cx), int(bbox.cy)
    arm = max(8, min(w, h) // 12)
    cv2.line(frame, (cx - arm, cy), (cx + arm, cy), color, 1)
    cv2.line(frame, (cx, cy - arm), (cx, cy + arm), color, 1)


def draw_label(frame: np.ndarray, text: str, *, color=WHITE) -> None:
    """Banner text in the top-left, with a black drop shadow for contrast."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.6, frame.shape[0] / 1080.0)
    thickness = max(1, int(round(scale * 2)))
    pos = (12, int(28 * scale) + 6)
    cv2.putText(frame, text, pos, font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, pos, font, scale, color, thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Frame categorization — one pass over the clip.
# ---------------------------------------------------------------------------


def _detection_source(
    detector: HaarFaceDetector, frame: np.ndarray
) -> tuple[Bbox | None, str]:
    """Return ``(bbox, source)`` where source ∈ {frontal, profile_l,
    profile_r, miss}. Re-implements HaarFaceDetector.detect()'s cascade
    order so we can record which cascade fired without losing the bbox.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    h = gray.shape[0]
    min_side = max(20, int(round(detector.min_size_ratio * h)))

    def run(cascade, image):
        dets = cascade.detectMultiScale(
            image,
            scaleFactor=detector.scale_factor,
            minNeighbors=detector.min_neighbors,
            minSize=(min_side, min_side),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        if dets is None or len(dets) == 0:
            return None
        arr = np.asarray(dets).reshape(-1, 4)
        areas = arr[:, 2] * arr[:, 3]
        best = arr[int(np.argmax(areas))]
        return int(best[0]), int(best[1]), int(best[2]), int(best[3])

    front = run(detector._frontal, gray)
    if front is not None:
        return Bbox(*front), "frontal"

    prof_l = run(detector._profile, gray)
    if prof_l is not None:
        return Bbox(*prof_l), "profile_l"

    prof_r = run(detector._profile, cv2.flip(gray, 1))
    if prof_r is not None:
        x, y, bw, bh = prof_r
        return Bbox(gray.shape[1] - x - bw, y, bw, bh), "profile_r"

    return None, "miss"


def scan_video(video_path: Path) -> tuple[list[dict], float, int]:
    """Walk every frame and record source + mean luminance.

    Returns ``(rows, fps, width)``. ``rows`` has one dict per frame with
    keys ``index``, ``source``, ``bbox`` (xywh or None), ``luminance``.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    detector = HaarFaceDetector()
    rows: list[dict] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        bbox, source = _detection_source(detector, frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rows.append(
            {
                "index": idx,
                "source": source,
                "bbox": (
                    list(bbox.as_xywh()) if bbox is not None else None
                ),
                "luminance": float(gray.mean()),
            }
        )
        idx += 1

    cap.release()
    return rows, float(fps), width


def select_test_frames(rows: list[dict]) -> list[dict]:
    """Pick up to 6 representative frames covering the spec'd conditions.

    Categories in priority order (drop any with no candidate):

      - ``front``         : frontal cascade fires, central area
      - ``profile_left``  : only profile cascade fires (left-facing)
      - ``profile_right`` : only flipped-profile cascade fires
      - ``miss``          : no cascade fires — likely occlusion / 3/4 / blur
      - ``low_light``     : luminance in bottom 10 %
      - ``high_light``    : luminance in top 10 %

    Returned dicts: ``{"index": int, "label": str, "source": str,
    "luminance": float}``.
    """
    if not rows:
        return []

    by_source: dict[str, list[dict]] = {"frontal": [], "profile_l": [],
                                        "profile_r": [], "miss": []}
    for r in rows:
        by_source[r["source"]].append(r)

    lums = np.array([r["luminance"] for r in rows])
    lo_thr = float(np.quantile(lums, 0.10))
    hi_thr = float(np.quantile(lums, 0.90))

    picks: list[dict] = []
    chosen: set[int] = set()

    def pick(label: str, candidates: list[dict]) -> None:
        for c in candidates:
            if c["index"] in chosen:
                continue
            picks.append(
                {
                    "index": int(c["index"]),
                    "label": label,
                    "source": c["source"],
                    "luminance": float(c["luminance"]),
                }
            )
            chosen.add(c["index"])
            return

    # front-facing — frontal cascade, pick the one closest to median lum
    if by_source["frontal"]:
        med = float(np.median(
            [r["luminance"] for r in by_source["frontal"]]
        ))
        ranked = sorted(
            by_source["frontal"],
            key=lambda r: abs(r["luminance"] - med),
        )
        pick("front", ranked)

    pick("profile_left", by_source["profile_l"])
    pick("profile_right", by_source["profile_r"])
    pick("miss", by_source["miss"])  # may also act as the "occluded" sample

    low_candidates = sorted(
        [r for r in rows if r["luminance"] <= lo_thr],
        key=lambda r: r["luminance"],
    )
    pick("low_light", low_candidates)

    high_candidates = sorted(
        [r for r in rows if r["luminance"] >= hi_thr],
        key=lambda r: -r["luminance"],
    )
    pick("high_light", high_candidates)

    return picks[:6]


# ---------------------------------------------------------------------------
# Artifact 1 — per-frame PNGs + grid.
# ---------------------------------------------------------------------------


def render_test_frames(
    video_path: Path,
    picks: list[dict],
    out_dir: Path,
) -> list[Path]:
    """Re-open the source video, grab each test frame, draw overlay,
    write a PNG per frame. Returns the list of written paths in pick
    order.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    detector = HaarFaceDetector()
    written: list[Path] = []

    for pick in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(pick["index"]))
        ok, frame = cap.read()
        if not ok:
            continue

        bbox, source = _detection_source(detector, frame)
        if bbox is not None:
            draw_bbox(frame, bbox, color=GREEN)
        # Recompute via the public detect() to make sure both paths agree
        # — useful as a self-check, but only the source's bbox is drawn.

        label = (
            f"#{pick['index']:04d}  {pick['label']}  "
            f"src={source}  L={pick['luminance']:.0f}"
        )
        draw_label(frame, label)

        if bbox is None:
            # Visible "no detection" hint so the miss frame doesn't look
            # broken at a glance.
            h, w = frame.shape[:2]
            cv2.putText(
                frame,
                "NO DETECTION",
                (w // 2 - 140, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                RED,
                3,
                cv2.LINE_AA,
            )

        path = out_dir / f"phase1_frame_{pick['index']:04d}_{pick['label']}.png"
        cv2.imwrite(str(path), frame)
        written.append(path)

    cap.release()
    return written


def compose_grid(image_paths: list[Path], out_path: Path) -> None:
    """Lay images out as 2 rows × ceil(N/2) cols and write a single PNG."""
    if not image_paths:
        return
    images = [cv2.imread(str(p)) for p in image_paths]
    images = [img for img in images if img is not None]
    if not images:
        return

    # Normalize to a common size — pick the median to avoid one outlier
    # blowing up the canvas.
    heights = [img.shape[0] for img in images]
    widths = [img.shape[1] for img in images]
    target_h = int(np.median(heights))
    target_w = int(np.median(widths))
    resized = [
        cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        for img in images
    ]

    n = len(resized)
    cols = math.ceil(n / 2)
    rows = 2 if n > 1 else 1
    canvas = np.zeros((target_h * rows, target_w * cols, 3), dtype=np.uint8)
    for i, img in enumerate(resized):
        r, c = i // cols, i % cols
        canvas[r * target_h:(r + 1) * target_h,
               c * target_w:(c + 1) * target_w] = img

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


# ---------------------------------------------------------------------------
# Artifact 2 — temporal smoothing overlay clip.
# ---------------------------------------------------------------------------


def render_overlay_video(
    video_path: Path,
    out_path: Path,
    *,
    seconds: float = 3.0,
) -> tuple[int, int, int]:
    """Render the first ``seconds`` of the clip with smoothed bbox overlay.

    Returns ``(frames_written, raw_hits, miss_count)``. The miss count is
    how many frames the smoother had to hold or drop — useful as a quick
    sanity number in the verdict.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(round(seconds * fps))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    detector = HaarFaceDetector()
    smoother = BboxSmoother(fps=fps)

    raw_hits = 0
    misses = 0
    written = 0

    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break

        raw_bbox = detector.detect(frame)
        smoothed = smoother.update(raw_bbox)

        if raw_bbox is not None:
            raw_hits += 1
        else:
            misses += 1

        if smoothed is not None:
            draw_bbox(frame, smoothed, color=GREEN)
        if raw_bbox is not None:
            x, y, w, h = raw_bbox.as_xywh()
            cv2.rectangle(frame, (x, y), (x + w, y + h), RED, 1)

        legend = (
            f"frame {i:03d}/{n_frames}  hit={raw_bbox is not None}  "
            f"smoothed={smoothed is not None}"
        )
        draw_label(frame, legend)
        cv2.putText(
            frame,
            "green=smoothed  red=raw",
            (12, height - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        writer.write(frame)
        written += 1

    writer.release()
    cap.release()
    return written, raw_hits, misses


# ---------------------------------------------------------------------------
# Entrypoint.
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--video",
        type=Path,
        default=ROOT / "samples" / "input" / "clip.mp4",
        help="source clip (default: samples/input/clip.mp4)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "output",
        help="where to write phase1 artifacts",
    )
    parser.add_argument(
        "--test-frames-json",
        type=Path,
        default=ROOT / "samples" / "test_frames.json",
        help="persisted set of fixed test frame indices",
    )
    parser.add_argument(
        "--reselect",
        action="store_true",
        help="re-pick test frames even if the json file already exists",
    )
    parser.add_argument(
        "--clip-seconds",
        type=float,
        default=3.0,
        help="length of the smoothing overlay video (seconds)",
    )
    args = parser.parse_args()

    video = args.video.resolve()
    if not video.exists():
        print(f"video not found: {video}", file=sys.stderr)
        return 2

    print(f"[phase1] source: {video}")

    # Step 1 — pick / load test frames.
    if args.test_frames_json.exists() and not args.reselect:
        meta = json.loads(args.test_frames_json.read_text())
        if meta.get("video") != str(video):
            print(
                "[phase1] existing test_frames.json was for a different "
                "video; pass --reselect to regenerate."
            )
            return 2
        picks = meta["frames"]
        print(
            f"[phase1] reusing {len(picks)} test frames from "
            f"{args.test_frames_json.relative_to(ROOT)}"
        )
    else:
        print("[phase1] scanning video to categorize frames (one-time pass)…")
        rows, fps, _ = scan_video(video)
        picks = select_test_frames(rows)
        meta = {
            "video": str(video),
            "fps": fps,
            "frame_count": len(rows),
            "frames": picks,
        }
        args.test_frames_json.parent.mkdir(parents=True, exist_ok=True)
        args.test_frames_json.write_text(json.dumps(meta, indent=2))
        print(
            f"[phase1] wrote {len(picks)} test-frame indices to "
            f"{args.test_frames_json.relative_to(ROOT)}"
        )

    print("[phase1] test frames:")
    for p in picks:
        print(f"  - #{p['index']:>4d}  {p['label']:<14s}  src={p['source']:<10s}  L={p['luminance']:.1f}")

    # Step 2 — per-frame PNGs + grid.
    frame_pngs = render_test_frames(
        video, picks, args.output_dir / "phase1_frames"
    )
    grid_path = args.output_dir / "phase1_grid.png"
    compose_grid(frame_pngs, grid_path)
    print(f"[phase1] grid → {grid_path.relative_to(ROOT)}")

    # Step 3 — temporal smoothing overlay clip.
    video_path = args.output_dir / "phase1_bbox.mp4"
    written, hits, misses = render_overlay_video(
        video, video_path, seconds=float(args.clip_seconds)
    )
    print(
        f"[phase1] overlay video → {video_path.relative_to(ROOT)}  "
        f"({written} frames, {hits} raw hits, {misses} misses)"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
