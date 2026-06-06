"""AvatarShield render pipeline (pure-IVP501).

Public API:
    render_video(video_path, output_path, config=None) -> RenderResult
    render_preview_frame(video_path, output_path, *, frame_index, config=None)
        -> PreviewResult
    process_frame(frame_bgr, state, config) -> np.ndarray

CLI:
    python -m avatarshield.render \\
        --video samples/input/clip.mp4 \\
        --theme porcelain-pink \\
        --output data/output/clip_ivp.mp4

    python -m avatarshield.render \\
        --video samples/input/clip.mp4 \\
        --theme porcelain-pink \\
        --output data/output/clip_ivp.png \\
        --frame-index 120

Pipeline (matches ``plan.md`` §2 Phase 5)::

    Haar detect ─► BboxSmoother (One Euro on cx, cy, w, h)
                ─► head_ellipse + feather   (alpha for final composite)
                ─► face_oval + hair_ring + feature_strip_masks
                ─► skin_mask(HSV ∩ YCbCr) ∩ face_oval
                ─► skin pre-pass: gentle Lab snap toward theme.skin_lab
                ─► cel_shade(state shared, K-means EMA across frames)
                ─► per-region Reinhard:
                     skin   → theme.skin_lab     full Lab,    0.55
                     lips   → theme.lips_lab     full Lab,    0.75
                     brows  → theme.brows_lab    full Lab,    0.75
                     eyes   → theme.eyes_lab     full Lab,    0.75
                     hair   → theme.hair_lab     chroma only, 0.55
                ─► composite stylized * head_alpha + frame * (1 - head_alpha)
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .cel_shade import TemporalState, cel_shade
from .detect import (
    Bbox,
    HaarFaceDetector,
    face_oval_mask,
    feather_mask,
    feature_strip_masks,
    hair_ring_mask,
    head_ellipse_mask,
    skin_mask,
)
from .palette import Theme, apply_target_lab_recolor, load_theme
from .smooth import BboxSmoother
from .track import BoxTracker
from .video_io import VideoReader, VideoWriter


# ---------------------------------------------------------------------------
# Public dataclasses.
# ---------------------------------------------------------------------------


@dataclass
class RenderConfig:
    """Knob bundle for the pure-IVP render pipeline (spec.md §5.1)."""

    theme: str = "porcelain-pink"

    # Haar detector. ``haar_min_size_ratio`` and ``haar_min_skin_coverage``
    # mirror the matching :class:`detect.HaarFaceDetector` knobs; the
    # defaults track those in ``detect.py`` so the render pipeline and
    # bare-detector callers (tests / eval scripts) stay in sync.
    haar_scale_factor: float = 1.1
    haar_min_neighbors: int = 5
    haar_min_size_ratio: float = 0.05
    haar_min_skin_coverage: float = 0.30

    # BboxSmoother / One Euro.
    bbox_hold_frames: int = 6
    smooth_min_cutoff: float = 1.0
    smooth_beta: float = 0.02
    # Spatial / size sanity gates on raw Haar detections (see smooth.py
    # for the rationale). ``None`` disables either gate; defaults reject
    # implausible jumps (≥ 1× bbox size per frame, scaled by held count)
    # and implausible size jumps (> 2× / < 0.5× the running size EMA).
    bbox_max_jump_ratio: float | None = 1.0
    bbox_max_size_ratio: float | None = 2.0

    # Correlation-filter tracker fallback when Haar misses (see track.py).
    # ``tracker_enabled = False`` falls back to passthrough-on-miss.
    # ``tracker_max_age`` caps tracker drift — once the predictor has
    # run this many frames without a fresh Haar re-acquire we drop it
    # and let the head ellipse stay un-stylized until detection locks
    # back on.
    tracker_enabled: bool = True
    tracker_kind: str = "csrt"
    tracker_max_age: int = 30

    # Head ellipse extension (fractions of bbox).
    head_top_extend: float = 0.60
    head_side_extend: float = 0.20
    head_bottom_extend: float = 0.05
    head_feather_px: int = 41

    # Cel-shade.
    luminance_levels: int = 3
    chroma_levels: int = 0
    bilateral_passes: int = 3
    saturation_boost: float = 1.30
    edge_strength: float = 0.85
    state_ema: float = 0.85

    # Per-region Reinhard strengths.
    palette_strength_skin_pre: float = 0.25
    palette_strength_skin: float = 0.55
    palette_strength_features: float = 0.75
    palette_strength_hair: float = 0.55

    # Light feather on region masks before Reinhard transfer.
    region_mask_feather_px: int = 21


@dataclass
class RenderResult:
    output_path: str
    frame_count: int
    fps: float
    duration_s: float


@dataclass
class PreviewResult:
    output_path: str
    frame_index: int
    frame_count: int
    fps: float
    width: int
    height: int


@dataclass
class RenderState:
    """Per-clip mutable state carried across frames.

    Holds the One Euro bbox smoother, the cel-shade K-means EMA centers
    so K-means doesn't re-permute its labels every frame (would otherwise
    produce visible "boiling" / flicker), and the correlation-filter
    tracker that bridges Haar misses between successful detections.
    """

    smoother: BboxSmoother
    temporal: TemporalState
    detector: HaarFaceDetector = field(default_factory=HaarFaceDetector)
    tracker: BoxTracker = field(default_factory=BoxTracker)


# ---------------------------------------------------------------------------
# Theme + style scalar resolution.
# ---------------------------------------------------------------------------


def _resolve_theme_and_style(config: RenderConfig) -> tuple[Theme, dict]:
    """Load the named theme and merge its style scalars into cel-shade kwargs.

    A theme JSON may override the four style scalars (saturation_boost /
    edge_strength / luminance_levels / chroma_levels). When a theme value
    differs from the dataclass default, it wins — this matches the Phase 6
    preview script where each theme can dial its own posterise strength.
    """
    theme = load_theme(config.theme)
    defaults = RenderConfig()
    cel_kwargs = {
        "luminance_levels": (
            theme.luminance_levels
            if theme.luminance_levels != defaults.luminance_levels
            else config.luminance_levels
        ),
        "chroma_levels": (
            theme.chroma_levels
            if theme.chroma_levels != defaults.chroma_levels
            else config.chroma_levels
        ),
        "bilateral_passes": config.bilateral_passes,
        "edge_strength": (
            theme.edge_strength
            if theme.edge_strength != defaults.edge_strength
            else config.edge_strength
        ),
        "saturation_boost": (
            theme.saturation_boost
            if theme.saturation_boost != defaults.saturation_boost
            else config.saturation_boost
        ),
    }
    return theme, cel_kwargs


def _light_feather(mask_u8: np.ndarray, feather_px: int) -> np.ndarray:
    """Light Gaussian feather of a ``{0, 255}`` mask → float32 ``[0, 1]``."""
    if feather_px <= 0:
        return mask_u8.astype(np.float32) / 255.0
    k = int(feather_px) | 1
    blurred = cv2.GaussianBlur(mask_u8, (k, k), 0)
    return blurred.astype(np.float32) / 255.0


def _new_state(config: RenderConfig, fps: float) -> RenderState:
    detector = HaarFaceDetector(
        scale_factor=config.haar_scale_factor,
        min_neighbors=config.haar_min_neighbors,
        min_size_ratio=config.haar_min_size_ratio,
        min_skin_coverage=config.haar_min_skin_coverage,
    )
    smoother = BboxSmoother(
        fps=fps,
        min_cutoff=config.smooth_min_cutoff,
        beta=config.smooth_beta,
        hold_frames=config.bbox_hold_frames,
        max_jump_ratio=config.bbox_max_jump_ratio,
        max_size_ratio=config.bbox_max_size_ratio,
    )
    tracker = BoxTracker(kind=config.tracker_kind)
    return RenderState(
        smoother=smoother,
        temporal=TemporalState(ema=config.state_ema),
        detector=detector,
        tracker=tracker,
    )


# ---------------------------------------------------------------------------
# Per-frame pipeline.
# ---------------------------------------------------------------------------


def process_frame(
    frame_bgr: np.ndarray,
    state: RenderState,
    config: RenderConfig,
    *,
    theme: Theme | None = None,
    cel_kwargs: dict | None = None,
) -> np.ndarray:
    """Stylize one BGR frame end-to-end and return the composite.

    ``theme`` / ``cel_kwargs`` are optional — when omitted the helper
    resolves them via :func:`_resolve_theme_and_style`. The full-video
    driver caches them once per clip to avoid re-reading the theme JSON
    every frame.
    """
    if theme is None or cel_kwargs is None:
        theme, cel_kwargs = _resolve_theme_and_style(config)

    h, w = frame_bgr.shape[:2]

    # 0) Cel-shade unconditionally on every frame so the shared TemporalState
    #    sees consistent input statistics, even when we ultimately drop the
    #    stylized output for a miss frame.
    _ = cel_shade(
        frame_bgr, state=state.temporal, state_prefix="render", **cel_kwargs
    )

    raw_haar: Bbox | None = state.detector.detect(frame_bgr)
    from_haar = raw_haar is not None
    raw: Bbox | None = raw_haar

    # When Haar misses, ask the correlation-filter tracker (if active and
    # below the drift cap) for a prediction. The smoother's gates still
    # run on the tracker output — a tracker that drifts onto the body
    # gets size/jump-gated like any other false positive.
    if (
        config.tracker_enabled
        and raw_haar is None
        and state.tracker.active
    ):
        if state.tracker.age < config.tracker_max_age:
            raw = state.tracker.update(frame_bgr)
        else:
            state.tracker.reset()

    bbox = state.smoother.update(raw)

    # Re-anchor the tracker only on a fresh, gated Haar accept. ``held_count
    # == 0`` distinguishes a new accept from a held bbox (held → > 0). On
    # a gate-rejected Haar candidate the smoother holds or returns None,
    # so we deliberately do not re-init the tracker on the bad bbox.
    if (
        config.tracker_enabled
        and from_haar
        and bbox is not None
        and state.smoother.held_count == 0
    ):
        state.tracker.init(frame_bgr, bbox)

    if bbox is None:
        # Smoother gave up. Drop the tracker too so it doesn't keep
        # following a stale patch when detection eventually re-locks.
        if config.tracker_enabled:
            state.tracker.reset()
        return frame_bgr.copy()

    # 1) Geometric masks driven by the bbox.
    head_bin = head_ellipse_mask(
        bbox,
        (w, h),
        top_extend=config.head_top_extend,
        side_extend=config.head_side_extend,
        bottom_extend=config.head_bottom_extend,
    )
    head_alpha = feather_mask(head_bin, config.head_feather_px)
    face_bin = face_oval_mask(bbox, (w, h))
    hair_bin = hair_ring_mask(
        bbox,
        (w, h),
        top_extend=config.head_top_extend,
        side_extend=config.head_side_extend,
        bottom_extend=config.head_bottom_extend,
    )
    feature = feature_strip_masks(bbox, (w, h))

    # 2) Skin mask = HSV ∩ YCbCr ∩ face_oval. The intersection throws out
    #    skin-coloured patches outside the face (hands, wood).
    skin_bin = skin_mask(frame_bgr, bbox)
    skin_bin = cv2.bitwise_and(skin_bin, face_bin)

    # 3) Light feather on region masks so the Reinhard transfer fades at
    #    boundaries instead of leaving visible seams.
    fp = config.region_mask_feather_px
    skin_soft = _light_feather(skin_bin, fp)
    hair_soft = _light_feather(hair_bin, fp)
    lips_soft = feature["lips"]
    eyes_soft = feature["eyes"]
    brows_soft = feature["brows"]

    # 4) Skin pre-pass — gentle Lab snap so cel_shade's K-means clusters land
    #    on the theme's bright band rather than the natural skin band.
    prepped = apply_target_lab_recolor(
        frame_bgr,
        skin_soft,
        theme.skin_lab,
        strength=config.palette_strength_skin_pre,
    )

    # 5) Cel-shade the pre-pass'd frame; share state with the warm-up call so
    #    the EMA centers stay locked to the theme palette.
    stylized = cel_shade(
        prepped, state=state.temporal, state_prefix="render", **cel_kwargs
    )

    # 6) Per-region Reinhard post-pass. Order matters: skin first (largest
    #    area, neutral tone), then features (override on top), then hair
    #    chroma-only (preserve user's hair luminance / shading detail).
    stylized = apply_target_lab_recolor(
        stylized, skin_soft, theme.skin_lab,
        strength=config.palette_strength_skin,
    )
    stylized = apply_target_lab_recolor(
        stylized, lips_soft, theme.lips_lab,
        strength=config.palette_strength_features,
    )
    stylized = apply_target_lab_recolor(
        stylized, brows_soft, theme.brows_lab,
        strength=config.palette_strength_features,
    )
    stylized = apply_target_lab_recolor(
        stylized, eyes_soft, theme.eyes_lab,
        strength=config.palette_strength_features,
    )
    stylized = apply_target_lab_recolor(
        stylized, hair_soft, theme.hair_lab,
        strength=config.palette_strength_hair,
        channels=(1, 2),
    )

    # 7) Composite onto the original; body / background pixels stay intact.
    a = head_alpha[..., None]
    out = (
        stylized.astype(np.float32) * a
        + frame_bgr.astype(np.float32) * (1.0 - a)
    )
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Full-clip render.
# ---------------------------------------------------------------------------


def render_video(
    video_path: str | Path,
    output_path: str | Path,
    config: RenderConfig | None = None,
    *,
    progress: bool = True,
) -> RenderResult:
    """Render the whole clip through the pure-IVP pipeline."""
    config = config or RenderConfig()
    theme, cel_kwargs = _resolve_theme_and_style(config)

    reader = VideoReader(video_path)
    meta = reader.meta
    writer = VideoWriter(
        output_path, fps=meta.fps, size=(meta.width, meta.height)
    )
    state = _new_state(config, meta.fps)

    t0 = time.time()
    frames_written = 0
    try:
        for idx, frame in enumerate(reader):
            composite = process_frame(
                frame, state, config, theme=theme, cel_kwargs=cel_kwargs
            )
            writer.write(composite)
            frames_written += 1
            if progress and idx % 30 == 0:
                _log_progress(idx, meta.frame_count, t0)
    finally:
        reader.close()
        writer.close()

    elapsed = time.time() - t0
    if progress:
        print(
            f"[done] {frames_written}/{meta.frame_count} frames "
            f"in {elapsed:.1f}s -> {output_path}"
        )

    return RenderResult(
        output_path=str(output_path),
        frame_count=frames_written,
        fps=meta.fps,
        duration_s=frames_written / meta.fps if meta.fps else 0.0,
    )


#: Maximum number of frames to sample when auto-picking a preview frame.
#: 24 evenly-spaced samples across the clip is enough to find a Haar hit
#: on the test fixtures without running the heavy cascade on every frame.
_PREVIEW_SCAN_BUDGET = 24


def _auto_preview_index(
    reader: VideoReader, detector: HaarFaceDetector
) -> int:
    """Pick a preview frame where the detector has a chance of locking on.

    Scans up to :data:`_PREVIEW_SCAN_BUDGET` evenly-spaced frames and
    returns the first one Haar accepts. Falls back to the middle frame
    when none of the sampled frames produces a hit — the caller still
    gets a deterministic preview, it just won't be stylized (which is
    the honest signal that detection is failing on this clip).
    """
    total = reader.meta.frame_count
    if total <= 0:
        return 0
    midpoint = total // 2
    if total <= _PREVIEW_SCAN_BUDGET:
        candidates = list(range(total))
    else:
        step = max(1, total // _PREVIEW_SCAN_BUDGET)
        candidates = list(range(0, total, step))[:_PREVIEW_SCAN_BUDGET]
    # Probe from the middle outward — viewers expect the preview to come
    # from "somewhere in the clip" rather than the first second, and an
    # early frame is often still in the camera-warm-up zone.
    candidates.sort(key=lambda i: abs(i - midpoint))
    for idx in candidates:
        try:
            frame = reader.read_frame(idx)
        except IndexError:
            continue
        if detector.detect(frame) is not None:
            return idx
    return midpoint


def render_preview_frame(
    video_path: str | Path,
    output_path: str | Path,
    *,
    frame_index: int | None = None,
    config: RenderConfig | None = None,
) -> PreviewResult:
    """Render a single composite frame (PNG) for editor preview / tuning.

    When ``frame_index`` is omitted, scans the clip and picks the first
    frame where the Haar detector locks on (see :func:`_auto_preview_index`).
    This avoids the failure mode where the hard-coded middle frame happens
    to be a detection miss and the editor shows an un-stylized preview.
    Pass an explicit ``frame_index`` to override the scan.

    The pipeline state is fresh (no temporal warm-up) so the preview
    reflects what a single frame would look like rendered in isolation.
    """
    config = config or RenderConfig()
    theme, cel_kwargs = _resolve_theme_and_style(config)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    reader = VideoReader(video_path)
    meta = reader.meta
    try:
        if meta.frame_count <= 0:
            raise RuntimeError(f"video has no frames: {video_path}")
        if frame_index is None:
            probe = HaarFaceDetector(
                scale_factor=config.haar_scale_factor,
                min_neighbors=config.haar_min_neighbors,
                min_size_ratio=config.haar_min_size_ratio,
                min_skin_coverage=config.haar_min_skin_coverage,
            )
            idx = _auto_preview_index(reader, probe)
        else:
            idx = frame_index
        idx = max(0, min(meta.frame_count - 1, idx))
        frame = reader.read_frame(idx)
    finally:
        reader.close()

    state = _new_state(config, meta.fps)
    composite = process_frame(
        frame, state, config, theme=theme, cel_kwargs=cel_kwargs
    )
    cv2.imwrite(str(out), composite)

    return PreviewResult(
        output_path=str(out),
        frame_index=idx,
        frame_count=meta.frame_count,
        fps=meta.fps,
        width=meta.width,
        height=meta.height,
    )


def _log_progress(idx: int, total: int, t0: float) -> None:
    elapsed = time.time() - t0
    rate = (idx + 1) / max(elapsed, 1e-6)
    eta = (max(0, total - idx - 1)) / max(rate, 1e-6)
    print(
        f"[frame {idx + 1}/{total}] {rate:.1f} fps · eta {eta:5.1f}s",
        flush=True,
    )


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    defaults = RenderConfig()
    p = argparse.ArgumentParser(
        prog="python -m avatarshield.render",
        description="AvatarShield render pipeline — video + theme → MP4.",
    )
    p.add_argument("--video", required=True, type=Path, help="source MP4")
    p.add_argument(
        "--theme",
        default=defaults.theme,
        help="theme name under assets/themes/<name>.json",
    )
    p.add_argument(
        "--output",
        required=True,
        type=Path,
        help="output path; .mp4 → full video, .png → single-frame preview",
    )
    p.add_argument(
        "--frame-index",
        type=int,
        default=None,
        help="render a single frame (PNG); implied when --output ends in .png",
    )
    p.add_argument(
        "--haar-min-neighbors",
        type=int,
        default=defaults.haar_min_neighbors,
        help=(
            "Haar minNeighbors gate — higher rejects more textured-background "
            "false positives (default: 5)."
        ),
    )
    p.add_argument(
        "--haar-min-size-ratio",
        type=float,
        default=defaults.haar_min_size_ratio,
        help=(
            "Smallest face to look for, as a fraction of frame height "
            "(default: 0.05). Lower catches far-subject vertical clips."
        ),
    )
    p.add_argument(
        "--haar-min-skin-coverage",
        type=float,
        default=defaults.haar_min_skin_coverage,
        help=(
            "Reject Haar candidates whose HSV ∩ YCbCr skin fraction "
            "falls below this (default: 0.30). Higher kills sweater / "
            "fabric false positives."
        ),
    )
    p.add_argument("--head-feather-px", type=int, default=defaults.head_feather_px)
    p.add_argument("--ema", type=float, default=defaults.state_ema)
    p.add_argument("--luminance-levels", type=int, default=defaults.luminance_levels)
    p.add_argument("--chroma-levels", type=int, default=defaults.chroma_levels)
    p.add_argument(
        "--bilateral-passes", type=int, default=defaults.bilateral_passes
    )
    p.add_argument("--edge-strength", type=float, default=defaults.edge_strength)
    p.add_argument(
        "--saturation-boost", type=float, default=defaults.saturation_boost
    )
    p.add_argument(
        "--skin-pre-strength",
        type=float,
        default=defaults.palette_strength_skin_pre,
        help="strength of skin pre-pass Lab snap (default: 0.25)",
    )
    p.add_argument(
        "--skin-strength",
        type=float,
        default=defaults.palette_strength_skin,
    )
    p.add_argument(
        "--features-strength",
        type=float,
        default=defaults.palette_strength_features,
    )
    p.add_argument(
        "--hair-strength",
        type=float,
        default=defaults.palette_strength_hair,
    )
    p.add_argument(
        "--region-mask-feather-px",
        type=int,
        default=defaults.region_mask_feather_px,
    )
    p.add_argument(
        "--quiet", action="store_true", help="suppress per-frame progress"
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = RenderConfig(
        theme=args.theme,
        haar_min_neighbors=args.haar_min_neighbors,
        haar_min_size_ratio=args.haar_min_size_ratio,
        haar_min_skin_coverage=args.haar_min_skin_coverage,
        head_feather_px=args.head_feather_px,
        state_ema=args.ema,
        luminance_levels=args.luminance_levels,
        chroma_levels=args.chroma_levels,
        bilateral_passes=args.bilateral_passes,
        edge_strength=args.edge_strength,
        saturation_boost=args.saturation_boost,
        palette_strength_skin_pre=args.skin_pre_strength,
        palette_strength_skin=args.skin_strength,
        palette_strength_features=args.features_strength,
        palette_strength_hair=args.hair_strength,
        region_mask_feather_px=args.region_mask_feather_px,
    )

    is_png = args.output.suffix.lower() == ".png"
    if args.frame_index is not None or is_png:
        result = render_preview_frame(
            args.video,
            args.output,
            frame_index=args.frame_index,
            config=config,
        )
        if not args.quiet:
            print(
                f"[preview] frame {result.frame_index}/{result.frame_count - 1} "
                f"-> {result.output_path}"
            )
        return 0

    render_video(args.video, args.output, config, progress=not args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
