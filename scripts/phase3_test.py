#!/usr/bin/env python3
"""Phase 3 verification — palette recolor on the skin region (no cel-shade).

Per ``plan.md`` §2 Phase 3 the artifacts for this phase are:

1. ``data/output/phase3_skin_recolor.mp4`` — the full source clip with the
   skin mask Reinhard-shifted toward the chosen theme's ``skin_lab``. No
   cel-shade is applied yet, so the result intentionally looks "off" — the
   point is to prove the Lab mean shift is wired correctly and to feel out
   a sensible ``palette_strength_skin``.
2. ``data/output/phase3_skin_recolor.png`` — a 3-panel grid over the test
   frames picked in Phase 1 (original | skin mask | recolored). Easier to
   eyeball than scrubbing the video.

Pipeline used here:

    Haar detect → BboxSmoother (One Euro on cx,cy,w,h)
                → skin_mask (HSV ∩ YCbCr + morph + largest CC inside bbox)
                → apply_skin_recolor (frame mean Lab → theme.skin_lab, strength)
                → write

Run::

    .venv/bin/python scripts/phase3_test.py
    .venv/bin/python scripts/phase3_test.py --theme porcelain-pink --strength 0.55

Flags::

    --video PATH       override source clip
    --output-dir PATH  override data/output
    --theme NAME       theme name under assets/themes/<name>.json
    --strength FLOAT   palette_strength_skin (default 0.55, per RenderConfig)
    --mask-feather-px  Gaussian feather applied to the skin mask before recolor
    --no-video         skip MP4 render (only emit the grid PNG)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from avatarshield.detect import (  # noqa: E402
    HaarFaceDetector,
    feather_mask,
    head_ellipse_mask,
    skin_mask,
)
from avatarshield.palette import (  # noqa: E402
    Theme,
    apply_skin_recolor,
    load_theme,
)
from avatarshield.smooth import BboxSmoother  # noqa: E402


# ---------------------------------------------------------------------------
# Drawing helpers — kept in sync with phase1/phase2 panels so the artifacts
# stack cleanly in the report.
# ---------------------------------------------------------------------------


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def draw_panel_label(frame: np.ndarray, text: str) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.7, frame.shape[0] / 720.0)
    thickness = max(1, int(round(scale * 2)))
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    pad = int(8 * scale)
    bar_h = th + 2 * pad
    cv2.rectangle(frame, (0, 0), (frame.shape[1], bar_h), BLACK, thickness=-1)
    cv2.putText(
        frame, text, (pad, pad + th), font, scale, WHITE, thickness, cv2.LINE_AA
    )


def draw_frame_label(frame: np.ndarray, text: str) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.55, frame.shape[0] / 1080.0)
    thickness = max(1, int(round(scale * 2)))
    pos = (12, frame.shape[0] - 14)
    cv2.putText(frame, text, pos, font, scale, BLACK, thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, pos, font, scale, WHITE, thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Per-frame recolor — used by both the grid and the video.
# ---------------------------------------------------------------------------


def recolor_frame(
    frame_bgr: np.ndarray,
    detector: HaarFaceDetector,
    smoother: BboxSmoother,
    theme: Theme,
    *,
    strength: float,
    mask_feather_px: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return ``(recolored_bgr, soft_skin_mask, hit)``.

    ``hit`` is True when either the detector fired this frame or the
    smoother served a held bbox, i.e. a skin mask was actually computed
    and the recolor was applied. When False, the recolored output is just
    the input frame (no detection → no mask → passthrough).
    """
    raw = detector.detect(frame_bgr)
    bbox = smoother.update(raw)

    if bbox is None:
        empty = np.zeros(frame_bgr.shape[:2], dtype=np.float32)
        return frame_bgr.copy(), empty, False

    # Limit the skin mask to the head ellipse so background skin-coloured
    # regions (hands, neck, wood furniture) do not get recoloured.
    skin = skin_mask(frame_bgr, bbox)
    h, w = frame_bgr.shape[:2]
    head_bin = head_ellipse_mask(bbox, (w, h))
    skin = cv2.bitwise_and(skin, head_bin)

    # Light feather smooths the recolor boundary — without it the Lab mean
    # shift produces a visible edge wherever the binary mask drops to 0.
    soft = feather_mask(skin, mask_feather_px)
    recolored = apply_skin_recolor(frame_bgr, soft, theme, strength=strength)
    return recolored, soft, True


# ---------------------------------------------------------------------------
# Artifact 1 — 3-panel grid over the Phase 1 test frames.
# ---------------------------------------------------------------------------


def panel_original(frame_bgr: np.ndarray, label: str) -> np.ndarray:
    out = frame_bgr.copy()
    draw_panel_label(out, "1. original")
    draw_frame_label(out, label)
    return out


def panel_mask(soft_mask: np.ndarray) -> np.ndarray:
    m_u8 = np.clip(soft_mask * 255.0, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(m_u8, cv2.COLOR_GRAY2BGR)
    draw_panel_label(rgb, "2. skin mask (feathered)")
    return rgb


def panel_recolored(recolored_bgr: np.ndarray) -> np.ndarray:
    out = recolored_bgr.copy()
    draw_panel_label(out, "3. skin recolor (theme.skin_lab)")
    return out


def build_grid(
    video_path: Path,
    picks: list[dict],
    theme: Theme,
    *,
    strength: float,
    mask_feather_px: int,
) -> tuple[np.ndarray, list[dict]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {video_path}")

    detector = HaarFaceDetector()
    # Per-frame grid does not benefit from temporal smoothing — give each
    # pick its own short-lived smoother so the held-bbox policy never
    # carries state across unrelated frame indices.
    rows: list[np.ndarray] = []
    stats: list[dict] = []
    try:
        for pick in picks:
            idx = int(pick["index"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                print(f"[phase3] could not read frame {idx}; skipping")
                continue

            local_smoother = BboxSmoother(
                fps=cap.get(cv2.CAP_PROP_FPS) or 30.0
            )
            recolored, soft, hit = recolor_frame(
                frame,
                detector,
                local_smoother,
                theme,
                strength=strength,
                mask_feather_px=mask_feather_px,
            )
            label = (
                f"#{idx:04d}  {pick['label']}  src={pick.get('source', '?')}  "
                f"theme={theme.name}  s={strength:.2f}"
            )
            row = np.concatenate(
                [
                    panel_original(frame, label),
                    panel_mask(soft),
                    panel_recolored(recolored),
                ],
                axis=1,
            )
            rows.append(row)
            stats.append(
                {
                    "index": idx,
                    "label": pick["label"],
                    "hit": hit,
                    "mask_mass": float(soft.sum()) / float(soft.size),
                }
            )
            print(
                f"[phase3] #{idx:>4d} {pick['label']:<14s}  "
                f"hit={'yes' if hit else 'no '}  "
                f"mask_mass={stats[-1]['mask_mass']:.3f}"
            )
    finally:
        cap.release()

    if not rows:
        raise RuntimeError("no rows produced")

    grid = np.concatenate(rows, axis=0)
    return grid, stats


# ---------------------------------------------------------------------------
# Artifact 2 — full-clip MP4.
# ---------------------------------------------------------------------------


def render_full_clip(
    video_path: Path,
    out_path: Path,
    theme: Theme,
    *,
    strength: float,
    mask_feather_px: int,
) -> tuple[int, int]:
    """Walk the whole input clip and write the recoloured version.

    Returns ``(frames_written, miss_count)``. Misses are frames where
    neither detection nor smoother-hold produced a bbox; those are
    passed through unmodified.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    detector = HaarFaceDetector()
    smoother = BboxSmoother(fps=fps)

    t0 = time.time()
    written = 0
    misses = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            recolored, _soft, hit = recolor_frame(
                frame,
                detector,
                smoother,
                theme,
                strength=strength,
                mask_feather_px=mask_feather_px,
            )
            if not hit:
                misses += 1
            writer.write(recolored)
            written += 1
            if written % 60 == 0:
                elapsed = time.time() - t0
                rate = written / max(elapsed, 1e-6)
                print(
                    f"[phase3] {written}/{total}  {rate:.1f} fps  "
                    f"misses={misses}",
                    flush=True,
                )
    finally:
        writer.release()
        cap.release()

    return written, misses


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
        help="where to write phase3 artifacts",
    )
    parser.add_argument(
        "--test-frames-json",
        type=Path,
        default=ROOT / "samples" / "test_frames.json",
        help="frame indices picked in Phase 1 (reused for the grid)",
    )
    parser.add_argument(
        "--theme",
        default="porcelain-pink",
        help="theme name under assets/themes/<name>.json",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.55,
        help="palette_strength_skin (default: 0.55, matches RenderConfig)",
    )
    parser.add_argument(
        "--mask-feather-px",
        type=int,
        default=21,
        help="Gaussian feather radius for the skin mask before recolor "
        "(odd, default: 21)",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="skip the full MP4 render and only emit the grid PNG",
    )
    args = parser.parse_args()

    video = args.video.resolve()
    if not video.exists():
        print(f"video not found: {video}", file=sys.stderr)
        return 2

    theme = load_theme(args.theme)
    print(
        f"[phase3] theme={theme.name}  skin_lab={theme.skin_lab}  "
        f"strength={args.strength:.2f}"
    )

    # Step 1 — grid over the Phase 1 test frames.
    if not args.test_frames_json.exists():
        print(
            f"missing {args.test_frames_json}; run scripts/phase1_test.py first.",
            file=sys.stderr,
        )
        return 2
    picks = json.loads(args.test_frames_json.read_text())["frames"]
    grid, stats = build_grid(
        video,
        picks,
        theme,
        strength=args.strength,
        mask_feather_px=args.mask_feather_px,
    )
    grid_path = args.output_dir / "phase3_skin_recolor.png"
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(grid_path), grid)
    print(
        f"[phase3] grid → {grid_path.relative_to(ROOT)} "
        f"({grid.shape[1]}x{grid.shape[0]})"
    )

    # Step 2 — full-clip MP4 (the headline Phase 3 artifact).
    if not args.no_video:
        video_out = args.output_dir / "phase3_skin_recolor.mp4"
        written, misses = render_full_clip(
            video,
            video_out,
            theme,
            strength=args.strength,
            mask_feather_px=args.mask_feather_px,
        )
        print(
            f"[phase3] video → {video_out.relative_to(ROOT)}  "
            f"({written} frames, {misses} misses)"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
