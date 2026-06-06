#!/usr/bin/env python3
"""Phase 2 verification — skin mask, head ellipse, feathered ROI.

Per ``plan.md`` §2 Phase 2 the artifact for this phase is a 4-panel image
showing, for a representative frame, the original, the HSV ∩ YCbCr skin
mask, the geometric head ellipse, and the feathered head ROI (the alpha
that Phase 5 will use to composite the cel-shade back onto the frame).

The script reuses the test frames picked in Phase 1
(``samples/test_frames.json``). For each frame:

1. Detect the largest face with :class:`HaarFaceDetector`.
2. Build the skin mask with :func:`avatarshield.detect.skin_mask` (HSV ∩
   YCbCr, morph close 5x5 + open 3x3, largest connected component
   intersecting the bbox).
3. Build the head ellipse mask (top +60 %, side +20 %, bottom +5 %).
4. Feather it (Gaussian, default 41 px).

We write a per-frame 4-panel PNG and a single grid stacking every frame on
its own row. The grid is the headline artifact (``phase2_masks.png``).

Run::

    .venv/bin/python scripts/phase2_test.py

Optional flags::

    --video PATH        override source clip
    --output-dir PATH   override data/output
    --feather-px INT    head ROI feather radius (default: 41)
    --frame INT         render only this frame index (skip the grid)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from avatarshield.detect import (  # noqa: E402
    Bbox,
    HaarFaceDetector,
    feather_mask,
    head_ellipse_mask,
    skin_mask,
)


# ---------------------------------------------------------------------------
# Drawing helpers.
# ---------------------------------------------------------------------------


GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def draw_bbox(frame: np.ndarray, bbox: Bbox, color=GREEN) -> None:
    x, y, w, h = bbox.as_xywh()
    # Thicker outline so the bbox is legible on full-body 1080p frames where
    # the face is small relative to the canvas.
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)


def draw_panel_label(frame: np.ndarray, text: str) -> None:
    """Title strip on top of a panel (always-readable contrast)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.7, frame.shape[0] / 720.0)
    thickness = max(1, int(round(scale * 2)))
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    pad = int(8 * scale)
    bar_h = th + 2 * pad
    cv2.rectangle(frame, (0, 0), (frame.shape[1], bar_h), BLACK, thickness=-1)
    cv2.putText(
        frame,
        text,
        (pad, pad + th),
        font,
        scale,
        WHITE,
        thickness,
        cv2.LINE_AA,
    )


def draw_frame_label(frame: np.ndarray, text: str) -> None:
    """Bottom-of-frame caption (used on the originals panel)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.55, frame.shape[0] / 1080.0)
    thickness = max(1, int(round(scale * 2)))
    pos = (12, frame.shape[0] - 14)
    cv2.putText(frame, text, pos, font, scale, BLACK, thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, pos, font, scale, WHITE, thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Panel builders — each returns a BGR H×W×3 image with the same shape as the
# input frame, so they tile cleanly into the per-frame row.
# ---------------------------------------------------------------------------


def panel_original(frame_bgr: np.ndarray, bbox: Bbox | None, label: str) -> np.ndarray:
    """Frame + bbox + caption."""
    out = frame_bgr.copy()
    if bbox is not None:
        draw_bbox(out, bbox)
    draw_panel_label(out, "1. original")
    draw_frame_label(out, label)
    return out


def panel_skin_mask(mask_u8: np.ndarray) -> np.ndarray:
    """Skin mask as a 3-channel white-on-black image."""
    rgb = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)
    # Hershey fonts are ASCII only, so we avoid the ∩ glyph here.
    draw_panel_label(rgb, "2. skin mask  (HSV AND YCbCr, morph close+open, CC)")
    return rgb


def panel_head_ellipse(mask_u8: np.ndarray) -> np.ndarray:
    """Head ellipse mask as a 3-channel white-on-black image."""
    rgb = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)
    draw_panel_label(rgb, "3. head ellipse")
    return rgb


def panel_feathered_roi(
    frame_bgr: np.ndarray, alpha: np.ndarray
) -> np.ndarray:
    """Frame × feathered alpha — what Phase 5 will modify."""
    a = np.clip(alpha, 0.0, 1.0)[..., None]
    blended = frame_bgr.astype(np.float32) * a
    out = np.clip(blended, 0, 255).astype(np.uint8)
    draw_panel_label(out, "4. feathered ROI (frame × alpha)")
    return out


# ---------------------------------------------------------------------------
# Row builder — one row = 4 panels for one frame.
# ---------------------------------------------------------------------------


def build_panels_for_frame(
    frame_bgr: np.ndarray,
    bbox: Bbox | None,
    label: str,
    *,
    feather_px: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return the 4-panel row + a small stats dict for the verdict line."""
    h, w = frame_bgr.shape[:2]
    skin = skin_mask(frame_bgr, bbox)
    if bbox is not None:
        head = head_ellipse_mask(bbox, (w, h))
    else:
        head = np.zeros((h, w), dtype=np.uint8)
    feathered = feather_mask(head, feather_px)

    panels = [
        panel_original(frame_bgr, bbox, label),
        panel_skin_mask(skin),
        panel_head_ellipse(head),
        panel_feathered_roi(frame_bgr, feathered),
    ]
    row = np.concatenate(panels, axis=1)

    stats = {
        "skin_coverage": float(np.count_nonzero(skin)) / float(skin.size),
        "head_coverage": float(np.count_nonzero(head)) / float(head.size),
        "alpha_mass": float(feathered.sum()) / float(feathered.size),
    }
    return row, stats


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
        help="where to write phase2 artifacts",
    )
    parser.add_argument(
        "--test-frames-json",
        type=Path,
        default=ROOT / "samples" / "test_frames.json",
        help="frame indices picked in Phase 1 (reused here)",
    )
    parser.add_argument(
        "--feather-px",
        type=int,
        default=41,
        help="Gaussian feather radius for the head ROI (default: 41)",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help="render only this single frame index (skip the grid)",
    )
    args = parser.parse_args()

    video = args.video.resolve()
    if not video.exists():
        print(f"video not found: {video}", file=sys.stderr)
        return 2

    # Resolve which frames to render.
    if args.frame is not None:
        picks = [
            {
                "index": int(args.frame),
                "label": f"custom_{int(args.frame):04d}",
                "source": "?",
                "luminance": float("nan"),
            }
        ]
    else:
        if not args.test_frames_json.exists():
            print(
                f"missing {args.test_frames_json}; run scripts/phase1_test.py first.",
                file=sys.stderr,
            )
            return 2
        meta = json.loads(args.test_frames_json.read_text())
        picks = meta["frames"]

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"could not open video: {video}", file=sys.stderr)
        return 2

    detector = HaarFaceDetector()
    out_dir = args.output_dir
    per_frame_dir = out_dir / "phase2_frames"
    per_frame_dir.mkdir(parents=True, exist_ok=True)

    rows: list[np.ndarray] = []
    row_labels: list[str] = []

    try:
        for pick in picks:
            idx = int(pick["index"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                print(f"[phase2] could not read frame {idx}; skipping")
                continue

            bbox = detector.detect(frame)
            label = (
                f"#{idx:04d}  {pick['label']}  src={pick.get('source', '?')}"
            )
            row, stats = build_panels_for_frame(
                frame, bbox, label, feather_px=args.feather_px
            )
            rows.append(row)
            row_labels.append(pick["label"])

            png_name = (
                f"phase2_frame_{idx:04d}_{pick['label']}_masks.png"
            )
            cv2.imwrite(str(per_frame_dir / png_name), row)
            print(
                f"[phase2] #{idx:>4d} {pick['label']:<14s}  "
                f"bbox={'yes' if bbox is not None else 'no '}  "
                f"skin={stats['skin_coverage']:.3f}  "
                f"head={stats['head_coverage']:.3f}  "
                f"alpha_mass={stats['alpha_mass']:.3f}"
            )
    finally:
        cap.release()

    if not rows:
        print("[phase2] no rows produced", file=sys.stderr)
        return 1

    # Headline artifact — one 4-panel row per test frame, stacked vertically.
    grid = np.concatenate(rows, axis=0)
    grid_path = out_dir / "phase2_masks.png"
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(grid_path), grid)
    print(f"[phase2] grid → {grid_path.relative_to(ROOT)} ({grid.shape[1]}x{grid.shape[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
