#!/usr/bin/env python3
"""Phase 5 verification — full pipeline (cel-shade + per-region palette).

Per ``plan.md`` §2 Phase 5 the artifacts for this phase are:

1. ``data/output/phase5_full.mp4`` — the full source clip with the
   complete pipeline applied (skin pre-pass → cel_shade → per-region
   Reinhard for skin/lips/eyes/brows/hair → composite). This is the
   demo clip planned for the presentation.
2. ``data/output/clip_sxs.mp4`` — same frames laid out side-by-side as
   ``original | full pipeline`` so the team can visually verify the
   transformation in one play-through.
3. ``data/output/phase5_full.png`` — a 4-panel grid over the test
   frames picked in Phase 1 (original | cel-shade whole | recolored
   region masks overlay | full composite).

Pipeline (matches ``plan.md`` §2 Phase 5 exactly)::

    Haar detect → BboxSmoother
                → head_ellipse + feather (alpha for final composite)
                → face_oval + hair_ring + feature_strip_masks (per-region)
                → skin_mask ∩ face_oval         (refined skin region)
                → skin pre-pass: gentle Lab snap toward theme.skin_lab
                                  (so cel_shade's K-means clusters land
                                   on bright/peachy bands, not natural
                                   skin bands)
                → cel_shade(whole frame, shared TemporalState)
                → post-pass Reinhard, in order:
                      skin   → theme.skin_lab     full Lab, 0.55
                      lips   → theme.lips_lab     full Lab, 0.75
                      brows  → theme.brows_lab    full Lab, 0.75
                      eyes   → theme.eyes_lab     full Lab, 0.75
                      hair   → theme.hair_lab     chroma-only, 0.55
                → composite stylized * alpha + frame * (1 - alpha)
                → write

Run::

    .venv/bin/python scripts/phase5_test.py
    .venv/bin/python scripts/phase5_test.py --theme porcelain-pink --no-sxs

Flags::

    --video PATH         override source clip
    --output-dir PATH    override data/output
    --theme NAME         theme name under assets/themes/<name>.json
    --frame INT          render only this single frame index (skip the grid)
    --feather-px INT     head ROI feather radius (default: 41)
    --ema FLOAT          TemporalState EMA (default: 0.85)
    --luminance-levels   K for L (default: 3)
    --bilateral-passes   bilateral iterations (default: 3)
    --edge-strength      XDoG line darkness (default: 0.85)
    --pre-skin-strength  skin pre-pass strength (default: 0.25)
    --skin-strength      post-pass skin Reinhard strength (default: 0.55)
    --features-strength  post-pass lips/eyes/brows strength (default: 0.75)
    --hair-strength      post-pass hair (chroma-only) strength (default: 0.55)
    --no-video           skip the full MP4 render
    --no-sxs             skip the side-by-side MP4
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
    face_oval_mask,
    feather_mask,
    feature_strip_masks,
    hair_ring_mask,
    head_ellipse_mask,
    skin_mask,
)
from avatarshield.palette import (  # noqa: E402
    Theme,
    apply_target_lab_recolor,
    load_theme,
)
from avatarshield.smooth import BboxSmoother  # noqa: E402


# ---------------------------------------------------------------------------
# Drawing helpers — kept in sync with phase1/2/3/4 panels.
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
# Per-frame full pipeline.
# ---------------------------------------------------------------------------


class PipelineConfig:
    """Container for the tuning knobs surfaced by the CLI."""

    def __init__(
        self,
        *,
        theme: Theme,
        feather_px: int,
        ema: float,
        luminance_levels: int,
        chroma_levels: int,
        bilateral_passes: int,
        edge_strength: float,
        saturation_boost: float,
        pre_skin_strength: float,
        skin_strength: float,
        features_strength: float,
        hair_strength: float,
        mask_feather_px: int,
    ) -> None:
        self.theme = theme
        self.feather_px = int(feather_px)
        self.ema = float(ema)
        self.luminance_levels = int(luminance_levels)
        self.chroma_levels = int(chroma_levels)
        self.bilateral_passes = int(bilateral_passes)
        self.edge_strength = float(edge_strength)
        self.saturation_boost = float(saturation_boost)
        self.pre_skin_strength = float(pre_skin_strength)
        self.skin_strength = float(skin_strength)
        self.features_strength = float(features_strength)
        self.hair_strength = float(hair_strength)
        self.mask_feather_px = int(mask_feather_px)

    def cel_kwargs(self) -> dict:
        return {
            "luminance_levels": self.luminance_levels,
            "chroma_levels": self.chroma_levels,
            "bilateral_passes": self.bilateral_passes,
            "edge_strength": self.edge_strength,
            "saturation_boost": self.saturation_boost,
        }


def _light_feather(mask_u8: np.ndarray, feather_px: int) -> np.ndarray:
    """Light Gaussian feather of a {0, 255} mask → float32 in [0, 1]."""
    if feather_px <= 0:
        return mask_u8.astype(np.float32) / 255.0
    k = int(feather_px) | 1
    blurred = cv2.GaussianBlur(mask_u8, (k, k), 0)
    return blurred.astype(np.float32) / 255.0


def process_frame(
    frame_bgr: np.ndarray,
    detector: HaarFaceDetector,
    smoother: BboxSmoother,
    state: TemporalState,
    config: PipelineConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray], bool]:
    """Return ``(composite, masks_debug, hit)``.

    ``masks_debug`` carries soft float32 masks used in the pipeline so
    the QA grid can overlay them. ``hit`` is True when either the
    detector fired this frame or the smoother served a held bbox.
    """
    h, w = frame_bgr.shape[:2]
    theme = config.theme

    # Cel-shade is computed unconditionally on every frame so the shared
    # TemporalState's K-means centers see consistent input statistics —
    # even when we don't show the cel-shaded output (miss frames).
    cel_full = cel_shade(
        frame_bgr, state=state, state_prefix="phase5", **config.cel_kwargs()
    )

    raw = detector.detect(frame_bgr)
    bbox = smoother.update(raw)

    if bbox is None:
        empty = np.zeros((h, w), dtype=np.float32)
        return frame_bgr.copy(), {
            "head_alpha": empty,
            "skin": empty,
            "lips": empty,
            "eyes": empty,
            "brows": empty,
            "hair": empty,
        }, False

    # 1) Geometric masks driven by the bbox.
    head_bin = head_ellipse_mask(bbox, (w, h))
    head_alpha = feather_mask(head_bin, config.feather_px)
    face_bin = face_oval_mask(bbox, (w, h))
    hair_bin = hair_ring_mask(bbox, (w, h))
    feature = feature_strip_masks(bbox, (w, h))

    # 2) Skin mask = HSV ∩ YCbCr ∩ face_oval — the intersection with the
    #    face oval throws out background skin-coloured patches (hands,
    #    wood) that the colour thresholds alone would catch.
    skin_bin = skin_mask(frame_bgr, bbox)
    skin_bin = cv2.bitwise_and(skin_bin, face_bin)

    # 3) Soften every region mask so the Reinhard transfer fades smoothly
    #    at boundaries (otherwise we get visible seams between regions).
    skin_soft = _light_feather(skin_bin, config.mask_feather_px)
    hair_soft = _light_feather(hair_bin, config.mask_feather_px)
    lips_soft = feature["lips"]   # already float32 with soft falloff
    eyes_soft = feature["eyes"]
    brows_soft = feature["brows"]

    # 4) Skin pre-pass — gentle Lab snap on the *raw* frame so that
    #    cel_shade's K-means clusters skin pixels into the theme's
    #    bright band rather than the natural-skin band. Matches the
    #    skin_palette_snap call inside cel_shade.anime_head_pass — same
    #    idea, but driven by the theme palette rather than a fixed
    #    anime target.
    prepped = apply_target_lab_recolor(
        frame_bgr,
        skin_soft,
        theme.skin_lab,
        strength=config.pre_skin_strength,
    )

    # 5) Cel-shade the pre-pass'd frame. We discard the cel_full
    #    computed above for the temporal-state warm-up; the *visible*
    #    cel-shade uses the pre-pass'd input so the bands land on the
    #    intended palette. The temporal state is shared — by the time
    #    pre-pass shifts the means slightly, the previous frame's
    #    centers already encode the right palette so the K-means
    #    initialisation stays stable.
    stylized = cel_shade(
        prepped, state=state, state_prefix="phase5", **config.cel_kwargs()
    )

    # 6) Per-region Reinhard post-pass on the cel-shaded frame.
    #    Order matters: skin first (fills the largest region with the
    #    theme's neutral tone), then features (override on top of the
    #    skin tone), then hair (chroma-only — keep the user's hair
    #    luminance / shading detail).
    stylized = apply_target_lab_recolor(
        stylized, skin_soft, theme.skin_lab,
        strength=config.skin_strength,
    )
    stylized = apply_target_lab_recolor(
        stylized, lips_soft, theme.lips_lab,
        strength=config.features_strength,
    )
    stylized = apply_target_lab_recolor(
        stylized, brows_soft, theme.brows_lab,
        strength=config.features_strength,
    )
    stylized = apply_target_lab_recolor(
        stylized, eyes_soft, theme.eyes_lab,
        strength=config.features_strength,
    )
    stylized = apply_target_lab_recolor(
        stylized, hair_soft, theme.hair_lab,
        strength=config.hair_strength,
        channels=(1, 2),
    )

    # 7) Composite onto the original. Only the head region is touched —
    #    body / background stay exactly as in the input.
    a = head_alpha[..., None]
    out = stylized.astype(np.float32) * a + frame_bgr.astype(np.float32) * (1.0 - a)
    composite = np.clip(out, 0, 255).astype(np.uint8)

    return composite, {
        "head_alpha": head_alpha,
        "skin": skin_soft,
        "lips": lips_soft,
        "eyes": eyes_soft,
        "brows": brows_soft,
        "hair": hair_soft,
    }, True


# ---------------------------------------------------------------------------
# Artifact 1 — 4-panel grid over the Phase 1 test frames.
# ---------------------------------------------------------------------------


def panel_original(frame_bgr: np.ndarray, label: str) -> np.ndarray:
    out = frame_bgr.copy()
    draw_panel_label(out, "1. original")
    draw_frame_label(out, label)
    return out


def panel_cel(cel_bgr: np.ndarray) -> np.ndarray:
    out = cel_bgr.copy()
    draw_panel_label(out, "2. cel_shade")
    return out


def panel_masks_overlay(
    frame_bgr: np.ndarray, masks: dict[str, np.ndarray]
) -> np.ndarray:
    """Tint each region a distinctive colour over a darkened original.

    Provides at-a-glance verification that the per-region masks line up
    with the actual face: skin → green, lips → red, eyes → cyan,
    brows → yellow, hair → magenta.
    """
    base = (frame_bgr.astype(np.float32) * 0.35).clip(0, 255)
    colours = {
        "skin": (0, 200, 0),
        "lips": (0, 0, 255),
        "eyes": (255, 255, 0),
        "brows": (0, 255, 255),
        "hair": (255, 0, 255),
    }
    out = base.copy()
    for name, colour in colours.items():
        m = masks.get(name)
        if m is None:
            continue
        m3 = m[..., None].astype(np.float32)
        layer = np.array(colour, dtype=np.float32).reshape(1, 1, 3)
        out = out * (1.0 - m3) + layer * m3
    out = np.clip(out, 0, 255).astype(np.uint8)
    draw_panel_label(out, "3. region masks (skin/lips/eyes/brows/hair)")
    return out


def panel_full(composite_bgr: np.ndarray) -> np.ndarray:
    out = composite_bgr.copy()
    draw_panel_label(out, "4. full pipeline composite")
    return out


def build_grid(
    video_path: Path,
    picks: list[dict],
    config: PipelineConfig,
) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {video_path}")

    detector = HaarFaceDetector()
    rows: list[np.ndarray] = []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    try:
        for pick in picks:
            idx = int(pick["index"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                print(f"[phase5] could not read frame {idx}; skipping")
                continue

            # Fresh state per grid frame so non-adjacent picks don't
            # contaminate the K-means EMA.
            local_smoother = BboxSmoother(fps=fps)
            local_state = TemporalState(ema=config.ema)

            # Compute a standalone cel_shade for the panel (vs the
            # process_frame call below which recomputes it internally —
            # cheap on a single frame).
            cel_panel = cel_shade(
                frame, state=local_state, state_prefix="phase5_panel",
                **config.cel_kwargs(),
            )
            composite, masks, hit = process_frame(
                frame, detector, local_smoother, local_state, config
            )

            label = (
                f"#{idx:04d}  {pick['label']}  src={pick.get('source', '?')}  "
                f"theme={config.theme.name}  "
                f"hit={'yes' if hit else 'no '}"
            )
            row = np.concatenate(
                [
                    panel_original(frame, label),
                    panel_cel(cel_panel),
                    panel_masks_overlay(frame, masks),
                    panel_full(composite),
                ],
                axis=1,
            )
            rows.append(row)
            print(
                f"[phase5] #{idx:>4d} {pick['label']:<14s}  "
                f"hit={'yes' if hit else 'no '}  "
                f"skin={masks['skin'].sum() / masks['skin'].size:.3f}  "
                f"hair={masks['hair'].sum() / masks['hair'].size:.3f}"
            )
    finally:
        cap.release()

    if not rows:
        raise RuntimeError("no rows produced")

    return np.concatenate(rows, axis=0)


# ---------------------------------------------------------------------------
# Artifact 2 — full-clip MP4 + optional side-by-side.
# ---------------------------------------------------------------------------


def render_full_clip(
    video_path: Path,
    out_path: Path,
    config: PipelineConfig,
    *,
    side_by_side_path: Path | None = None,
) -> tuple[int, int, float]:
    """Walk the whole clip; write the composite MP4 (and an optional sxs).

    Returns ``(frames_written, miss_count, elapsed_seconds)``. A *single*
    :class:`TemporalState` is shared across every frame so K-means
    centers EMA-smooth properly — same policy as Phase 4.
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

    sxs_writer: cv2.VideoWriter | None = None
    if side_by_side_path is not None:
        side_by_side_path.parent.mkdir(parents=True, exist_ok=True)
        sxs_writer = cv2.VideoWriter(
            str(side_by_side_path), fourcc, fps, (width * 2, height)
        )

    detector = HaarFaceDetector()
    smoother = BboxSmoother(fps=fps)
    state = TemporalState(ema=config.ema)

    t0 = time.time()
    written = 0
    misses = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            composite, _masks, hit = process_frame(
                frame, detector, smoother, state, config
            )
            if not hit:
                misses += 1
            writer.write(composite)
            if sxs_writer is not None:
                sxs = np.concatenate([frame, composite], axis=1)
                sxs_writer.write(sxs)
            written += 1
            if written % 60 == 0:
                elapsed = time.time() - t0
                rate = written / max(elapsed, 1e-6)
                print(
                    f"[phase5] {written}/{total}  {rate:.1f} fps  "
                    f"misses={misses}",
                    flush=True,
                )
    finally:
        writer.release()
        if sxs_writer is not None:
            sxs_writer.release()
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
        help="where to write phase5 artifacts",
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
    parser.add_argument("--frame", type=int, default=None)
    parser.add_argument("--feather-px", type=int, default=41)
    parser.add_argument("--ema", type=float, default=0.85)
    parser.add_argument("--luminance-levels", type=int, default=3)
    parser.add_argument("--chroma-levels", type=int, default=0)
    parser.add_argument("--bilateral-passes", type=int, default=3)
    parser.add_argument("--edge-strength", type=float, default=0.85)
    parser.add_argument("--saturation-boost", type=float, default=1.25)
    parser.add_argument(
        "--pre-skin-strength",
        type=float,
        default=0.25,
        help="skin pre-pass Lab snap (default: 0.25)",
    )
    parser.add_argument(
        "--skin-strength",
        type=float,
        default=0.55,
        help="post-pass skin Reinhard strength (default: 0.55)",
    )
    parser.add_argument(
        "--features-strength",
        type=float,
        default=0.75,
        help="post-pass lips/eyes/brows strength (default: 0.75)",
    )
    parser.add_argument(
        "--hair-strength",
        type=float,
        default=0.55,
        help="post-pass hair chroma-only strength (default: 0.55)",
    )
    parser.add_argument(
        "--mask-feather-px",
        type=int,
        default=21,
        help="light Gaussian feather (px) on skin/hair masks before recolor",
    )
    parser.add_argument(
        "--no-video", action="store_true", help="skip the full MP4 render"
    )
    parser.add_argument(
        "--no-sxs", action="store_true", help="skip the side-by-side MP4"
    )
    args = parser.parse_args()

    video = args.video.resolve()
    if not video.exists():
        print(f"video not found: {video}", file=sys.stderr)
        return 2

    theme = load_theme(args.theme)
    config = PipelineConfig(
        theme=theme,
        feather_px=args.feather_px,
        ema=args.ema,
        luminance_levels=args.luminance_levels,
        chroma_levels=args.chroma_levels,
        bilateral_passes=args.bilateral_passes,
        edge_strength=args.edge_strength,
        saturation_boost=args.saturation_boost,
        pre_skin_strength=args.pre_skin_strength,
        skin_strength=args.skin_strength,
        features_strength=args.features_strength,
        hair_strength=args.hair_strength,
        mask_feather_px=args.mask_feather_px,
    )
    print(
        f"[phase5] theme={theme.name}  "
        f"skin_lab={theme.skin_lab}  hair_lab={theme.hair_lab}  "
        f"lips_lab={theme.lips_lab}"
    )
    print(
        f"[phase5] strengths: pre_skin={config.pre_skin_strength:.2f} "
        f"skin={config.skin_strength:.2f} features={config.features_strength:.2f} "
        f"hair={config.hair_strength:.2f}  ema={config.ema:.2f}"
    )

    # Resolve which frames the grid renders.
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

    grid = build_grid(video, picks, config)
    grid_path = args.output_dir / "phase5_full.png"
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(grid_path), grid)
    print(
        f"[phase5] grid → {grid_path.relative_to(ROOT)} "
        f"({grid.shape[1]}x{grid.shape[0]})"
    )

    if not args.no_video:
        video_out = args.output_dir / "phase5_full.mp4"
        sxs_out = None if args.no_sxs else args.output_dir / "clip_sxs.mp4"
        written, misses, elapsed = render_full_clip(
            video, video_out, config, side_by_side_path=sxs_out
        )
        rate = written / max(elapsed, 1e-6)
        print(
            f"[phase5] video → {video_out.relative_to(ROOT)}  "
            f"({written} frames, {misses} misses, {rate:.1f} fps render)"
        )
        if sxs_out is not None:
            print(f"[phase5] sxs   → {sxs_out.relative_to(ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
