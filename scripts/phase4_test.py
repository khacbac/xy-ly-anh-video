#!/usr/bin/env python3
"""Phase 4 verification — cel-shade the head ROI (no palette yet).

Per ``plan.md`` §2 Phase 4 the artifacts for this phase are:

1. ``data/output/phase4_celshade_only.mp4`` — the full source clip with
   :func:`avatarshield.cel_shade.cel_shade` applied to every frame and
   composited back onto the original using the feathered head ellipse
   mask. No palette recolor is performed — the goal is to confirm the
   cel-shade pipeline wires up cleanly with a single shared
   :class:`TemporalState` per video (no inter-frame "boiling").
2. ``data/output/phase4_celshade.png`` — a 3-panel grid over the test
   frames picked in Phase 1 (original | cel-shade whole frame |
   head-masked composite). Easier to eyeball than scrubbing the video.

Pipeline used here (matches Phase 3 minus the recolor)::

    Haar detect → BboxSmoother (One Euro on cx,cy,w,h)
                → head_ellipse_mask (top +60%, side +20%, bottom +5%)
                → feather_mask (Gaussian, 41 px)
                → cel_shade(whole frame, TemporalState shared per clip)
                → composite: out = cel * head_feather + frame * (1 - alpha)
                → write

Run::

    .venv/bin/python scripts/phase4_test.py
    .venv/bin/python scripts/phase4_test.py --ema 0.9 --bilateral-passes 4

Flags::

    --video PATH         override source clip
    --output-dir PATH    override data/output
    --frame INT          render only this single frame index (skip the grid)
    --feather-px INT     head ROI feather radius (default: 41)
    --ema FLOAT          TemporalState EMA, [0,1], higher = stickier centers
    --luminance-levels   K for L-channel posterize (default: 3)
    --chroma-levels      K for a/b posterize (default: 0 = off)
    --bilateral-passes   bilateral iterations inside cel_shade (default: 3)
    --edge-strength      XDoG line darkness, [0,1] (default: 0.85)
    --saturation-boost   HSV saturation multiplier after posterize
    --no-video           skip the full MP4 render and only emit the grid PNG

The cel_shade ``state_prefix`` is fixed to ``"phase4"`` so the K-means
center keys are namespaced (``phase4_L``, ``phase4_a``, ``phase4_b``) and
do not collide if a future phase shares the same TemporalState instance.
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

from avatarshield.cel_shade import TemporalState, cel_shade  # noqa: E402
from avatarshield.detect import (  # noqa: E402
    HaarFaceDetector,
    feather_mask,
    head_ellipse_mask,
)
from avatarshield.smooth import BboxSmoother  # noqa: E402


# ---------------------------------------------------------------------------
# Drawing helpers — kept in sync with phase1/phase2/phase3 panels.
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
# Per-frame cel-shade — used by both the grid and the video.
# ---------------------------------------------------------------------------


class CelShadeParams:
    """Container for the cel_shade kwargs surfaced by the CLI."""

    def __init__(
        self,
        *,
        luminance_levels: int,
        chroma_levels: int,
        bilateral_passes: int,
        edge_strength: float,
        saturation_boost: float,
    ) -> None:
        self.luminance_levels = int(luminance_levels)
        self.chroma_levels = int(chroma_levels)
        self.bilateral_passes = int(bilateral_passes)
        self.edge_strength = float(edge_strength)
        self.saturation_boost = float(saturation_boost)

    def as_kwargs(self) -> dict:
        return {
            "luminance_levels": self.luminance_levels,
            "chroma_levels": self.chroma_levels,
            "bilateral_passes": self.bilateral_passes,
            "edge_strength": self.edge_strength,
            "saturation_boost": self.saturation_boost,
        }


def composite_frame(
    frame_bgr: np.ndarray,
    detector: HaarFaceDetector,
    smoother: BboxSmoother,
    state: TemporalState,
    params: CelShadeParams,
    *,
    feather_px: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Return ``(composite, cel_full, soft_head_mask, hit)``.

    ``hit`` is True when either the detector fired this frame or the
    smoother served a held bbox. When False, the composite is the original
    frame (passthrough) and the head mask is all zeros — Phase 4 leaves
    miss frames untouched, exactly as Phase 3 did.

    The cel-shade is always computed on the whole frame so the
    :class:`TemporalState` K-means centers see consistent input statistics
    even on miss frames. We just don't *show* it on miss frames.
    """
    h, w = frame_bgr.shape[:2]
    cel = cel_shade(frame_bgr, state=state, state_prefix="phase4", **params.as_kwargs())

    raw = detector.detect(frame_bgr)
    bbox = smoother.update(raw)

    if bbox is None:
        empty = np.zeros((h, w), dtype=np.float32)
        return frame_bgr.copy(), cel, empty, False

    head_bin = head_ellipse_mask(bbox, (w, h))
    alpha = feather_mask(head_bin, feather_px)
    a = alpha[..., None]
    out = cel.astype(np.float32) * a + frame_bgr.astype(np.float32) * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8), cel, alpha, True


# ---------------------------------------------------------------------------
# Artifact 1 — 3-panel grid over the Phase 1 test frames.
# ---------------------------------------------------------------------------


def panel_original(frame_bgr: np.ndarray, label: str) -> np.ndarray:
    out = frame_bgr.copy()
    draw_panel_label(out, "1. original")
    draw_frame_label(out, label)
    return out


def panel_cel_full(cel_bgr: np.ndarray) -> np.ndarray:
    out = cel_bgr.copy()
    draw_panel_label(out, "2. cel_shade (whole frame)")
    return out


def panel_composite(composite_bgr: np.ndarray) -> np.ndarray:
    out = composite_bgr.copy()
    draw_panel_label(out, "3. head-masked composite")
    return out


def build_grid(
    video_path: Path,
    picks: list[dict],
    params: CelShadeParams,
    *,
    feather_px: int,
    ema: float,
) -> tuple[np.ndarray, list[dict]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {video_path}")

    detector = HaarFaceDetector()
    rows: list[np.ndarray] = []
    stats: list[dict] = []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    try:
        for pick in picks:
            idx = int(pick["index"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                print(f"[phase4] could not read frame {idx}; skipping")
                continue

            # Fresh smoother + state per grid frame: the picks are non-adjacent
            # so cross-frame temporal blending would average unrelated content
            # and the cel-shade would not reflect what a real video render
            # produces locally.
            local_smoother = BboxSmoother(fps=fps)
            local_state = TemporalState(ema=ema)
            composite, cel_full, _alpha, hit = composite_frame(
                frame,
                detector,
                local_smoother,
                local_state,
                params,
                feather_px=feather_px,
            )

            label = (
                f"#{idx:04d}  {pick['label']}  src={pick.get('source', '?')}  "
                f"L={params.luminance_levels}  bp={params.bilateral_passes}  "
                f"ema={ema:.2f}"
            )
            row = np.concatenate(
                [
                    panel_original(frame, label),
                    panel_cel_full(cel_full),
                    panel_composite(composite),
                ],
                axis=1,
            )
            rows.append(row)
            stats.append(
                {
                    "index": idx,
                    "label": pick["label"],
                    "hit": hit,
                }
            )
            print(
                f"[phase4] #{idx:>4d} {pick['label']:<14s}  "
                f"hit={'yes' if hit else 'no '}"
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
    params: CelShadeParams,
    *,
    feather_px: int,
    ema: float,
) -> tuple[int, int, float]:
    """Walk the whole input clip and write the cel-shaded version.

    Returns ``(frames_written, miss_count, elapsed_seconds)``. Misses are
    frames where neither detection nor smoother-hold produced a bbox;
    those are passed through unmodified. A *single* TemporalState is
    shared across every frame so K-means centers EMA-smooth properly.
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
    state = TemporalState(ema=ema)

    t0 = time.time()
    written = 0
    misses = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            composite, _cel, _alpha, hit = composite_frame(
                frame,
                detector,
                smoother,
                state,
                params,
                feather_px=feather_px,
            )
            if not hit:
                misses += 1
            writer.write(composite)
            written += 1
            if written % 60 == 0:
                elapsed = time.time() - t0
                rate = written / max(elapsed, 1e-6)
                print(
                    f"[phase4] {written}/{total}  {rate:.1f} fps  "
                    f"misses={misses}",
                    flush=True,
                )
    finally:
        writer.release()
        cap.release()

    return written, misses, time.time() - t0


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
        help="where to write phase4 artifacts",
    )
    parser.add_argument(
        "--test-frames-json",
        type=Path,
        default=ROOT / "samples" / "test_frames.json",
        help="frame indices picked in Phase 1 (reused for the grid)",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help="render only this single frame index (skip the grid)",
    )
    parser.add_argument(
        "--feather-px",
        type=int,
        default=41,
        help="Gaussian feather radius for the head ROI (default: 41, "
        "matches RenderConfig.head_feather_px)",
    )
    parser.add_argument(
        "--ema",
        type=float,
        default=0.85,
        help="TemporalState EMA (default: 0.85; raise to 0.9 if boiling)",
    )
    parser.add_argument(
        "--luminance-levels",
        type=int,
        default=3,
        help="K-means K for the L channel (default: 3 = shadow/mid/highlight)",
    )
    parser.add_argument(
        "--chroma-levels",
        type=int,
        default=0,
        help="K-means K for a/b channels (default: 0 = off; try 8 for comic look)",
    )
    parser.add_argument(
        "--bilateral-passes",
        type=int,
        default=3,
        help="bilateral iterations inside cel_shade (default: 3)",
    )
    parser.add_argument(
        "--edge-strength",
        type=float,
        default=0.85,
        help="XDoG line darkness in [0, 1] (default: 0.85)",
    )
    parser.add_argument(
        "--saturation-boost",
        type=float,
        default=1.25,
        help="HSV saturation multiplier after posterize (default: 1.25, "
        "matches cel_shade's default)",
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

    params = CelShadeParams(
        luminance_levels=args.luminance_levels,
        chroma_levels=args.chroma_levels,
        bilateral_passes=args.bilateral_passes,
        edge_strength=args.edge_strength,
        saturation_boost=args.saturation_boost,
    )
    print(
        f"[phase4] cel_shade params: L={params.luminance_levels} "
        f"chroma={params.chroma_levels} bilateral={params.bilateral_passes} "
        f"edge={params.edge_strength:.2f} sat={params.saturation_boost:.2f} "
        f"ema={args.ema:.2f} feather={args.feather_px}"
    )

    # Resolve which frames to render for the grid.
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
        picks = json.loads(args.test_frames_json.read_text())["frames"]

    # Step 1 — 3-panel grid over the Phase 1 test frames.
    grid, _stats = build_grid(
        video,
        picks,
        params,
        feather_px=args.feather_px,
        ema=args.ema,
    )
    grid_path = args.output_dir / "phase4_celshade.png"
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(grid_path), grid)
    print(
        f"[phase4] grid → {grid_path.relative_to(ROOT)} "
        f"({grid.shape[1]}x{grid.shape[0]})"
    )

    # Step 2 — full-clip MP4 (the headline Phase 4 artifact).
    if not args.no_video:
        video_out = args.output_dir / "phase4_celshade_only.mp4"
        written, misses, elapsed = render_full_clip(
            video,
            video_out,
            params,
            feather_px=args.feather_px,
            ema=args.ema,
        )
        rate = written / max(elapsed, 1e-6)
        print(
            f"[phase4] video → {video_out.relative_to(ROOT)}  "
            f"({written} frames, {misses} misses, "
            f"{rate:.1f} fps render)"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
