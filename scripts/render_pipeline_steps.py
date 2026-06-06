#!/usr/bin/env python3
"""Render labeled step-by-step pipeline visuals for one hero frame.

Writes individual PNGs plus a composite grid under ``data/output/demo_steps/``
for presentation and ``STUDY.md`` walkthroughs. Each image shows one pipeline
stage on the same source frame so the audience can follow cause → effect.

Run::

    .venv/bin/python scripts/render_pipeline_steps.py
    .venv/bin/python scripts/render_pipeline_steps.py --frame 262 --theme porcelain-pink

Outputs::

    data/output/demo_steps/
        00_original.png … 12_temporal_ema.png
        demo_steps_grid.png
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from avatarshield import detect as detect_mod  # noqa: E402
from avatarshield.cel_shade import (  # noqa: E402
    TemporalState,
    cel_shade,
    iterated_bilateral,
    kmeans_quantize_channel,
    xdog,
)
from avatarshield.detect import (  # noqa: E402
    Bbox,
    HaarFaceDetector,
    face_oval_mask,
    feather_mask,
    feature_strip_masks,
    hair_ring_mask,
    head_ellipse_mask,
    skin_mask,
)
from avatarshield.palette import apply_target_lab_recolor, load_theme  # noqa: E402
from avatarshield.smooth import BboxSmoother  # noqa: E402
from avatarshield.video_io import VideoReader  # noqa: E402
from scripts.phase5_test import PipelineConfig, draw_panel_label  # noqa: E402


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)


@dataclass
class StepImage:
    filename: str
    label: str
    image: np.ndarray
    grid_order: int  # lower = earlier in the composite grid


def draw_bbox(frame: np.ndarray, bbox: Bbox, color=GREEN) -> None:
    x, y, w, h = bbox.as_xywh()
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)


def mask_to_rgb(mask_u8: np.ndarray) -> np.ndarray:
    """Visualize a binary mask as white-on-black."""
    return cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)


def labeled_panel(image: np.ndarray, label: str) -> np.ndarray:
    out = image.copy()
    draw_panel_label(out, label)
    return out


def _light_feather(mask_u8: np.ndarray, feather_px: int) -> np.ndarray:
    if feather_px <= 0:
        return mask_u8.astype(np.float32) / 255.0
    k = int(feather_px) | 1
    blurred = cv2.GaussianBlur(mask_u8, (k, k), 0)
    return blurred.astype(np.float32) / 255.0


def build_cel_ladder(
    frame_bgr: np.ndarray,
    config: PipelineConfig,
    *,
    state: TemporalState | None = None,
    state_prefix: str = "demo",
) -> dict[str, np.ndarray]:
    """Return cel-shade intermediates: bilateral → kmeans → xdog → full core."""
    kwargs = config.cel_kwargs()

    flat = iterated_bilateral(
        frame_bgr,
        passes=kwargs["bilateral_passes"],
    )

    lab = cv2.cvtColor(flat, cv2.COLOR_BGR2LAB)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    L_quant = kmeans_quantize_channel(
        L,
        k=kwargs["luminance_levels"],
        state=state,
        key=f"{state_prefix}_L",
    )
    lab_quant = np.stack([L_quant, a, b], axis=-1).astype(np.uint8)
    posterized = cv2.cvtColor(lab_quant, cv2.COLOR_LAB2BGR)

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    edge_mask = xdog(gray)
    strength = float(kwargs["edge_strength"])
    line_factor = (1.0 - strength) + strength * edge_mask
    with_xdog = np.clip(
        posterized.astype(np.float32) * line_factor[..., None], 0, 255
    ).astype(np.uint8)

    full_core = cel_shade(
        frame_bgr, state=state, state_prefix=state_prefix, **kwargs
    )

    return {
        "bilateral": flat,
        "kmeans": posterized,
        "xdog": with_xdog,
        "cel_core": full_core,
    }


def spatial_filters_panel(frame_bgr: np.ndarray, bbox: Bbox) -> np.ndarray:
    """Three-up Gaussian / median / bilateral on the face crop."""
    x, y, w, h = bbox.as_xywh()
    pad_x = int(w * 0.25)
    pad_y = int(h * 0.35)
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(frame_bgr.shape[1], x + w + pad_x)
    y1 = min(frame_bgr.shape[0], y + h + pad_y)
    crop = frame_bgr[y0:y1, x0:x1]

    gauss = cv2.GaussianBlur(crop, (0, 0), 3.0)
    median = cv2.medianBlur(crop, 5)
    bilateral = cv2.bilateralFilter(crop, 9, 50, 7)

    panels = [
        labeled_panel(gauss, "Gaussian blur"),
        labeled_panel(median, "Median blur"),
        labeled_panel(bilateral, "Bilateral (edge-preserving)"),
    ]
    return np.concatenate(panels, axis=1)


def warm_cel_state(
    reader: VideoReader,
    frame_idx: int,
    config: PipelineConfig,
    *,
    lookback: int = 30,
) -> TemporalState:
    """Prime K-means EMA centers from preceding frames."""
    state = TemporalState(ema=config.ema)
    start = max(0, frame_idx - lookback)
    for i in range(start, frame_idx):
        frame = reader.read_frame(i)
        cel_shade(
            frame,
            state=state,
            state_prefix="demo_warm",
            **config.cel_kwargs(),
        )
    return state


def render_steps(
    frame_bgr: np.ndarray,
    fps: float,
    config: PipelineConfig,
    reader: VideoReader,
    frame_idx: int,
    *,
    warmup_frames: int = 30,
) -> list[StepImage]:
    """Build every labeled step image for ``frame_bgr``."""
    h, w = frame_bgr.shape[:2]
    detector = HaarFaceDetector()
    smoother = BboxSmoother(fps=fps)
    raw = detector.detect(frame_bgr)
    bbox = smoother.update(raw)

    steps: list[StepImage] = []

    original = frame_bgr.copy()
    if bbox is not None:
        draw_bbox(original, bbox)
    steps.append(
        StepImage("00_original.png", "0. original + Haar bbox", original, 0)
    )

    if bbox is None:
        steps.append(
            StepImage(
                "99_no_face.png",
                "detector miss — remaining steps skipped",
                labeled_panel(frame_bgr, "no face detected"),
                99,
            )
        )
        return steps

    hsv_mask = detect_mod._skin_mask_hsv(frame_bgr)
    ycc_mask = detect_mod._skin_mask_ycbcr(frame_bgr)
    skin_full = skin_mask(frame_bgr, bbox)
    face_bin = face_oval_mask(bbox, (w, h))
    skin_refined = cv2.bitwise_and(skin_full, face_bin)

    head_bin = head_ellipse_mask(bbox, (w, h))
    head_alpha = feather_mask(head_bin, config.feather_px)
    feather_preview = (
        frame_bgr.astype(np.float32) * head_alpha[..., None]
        + frame_bgr.astype(np.float32) * (1.0 - head_alpha[..., None]) * 0.25
    ).astype(np.uint8)

    hair_bin = hair_ring_mask(bbox, (w, h))
    feature = feature_strip_masks(bbox, (w, h))
    skin_soft = _light_feather(skin_refined, config.mask_feather_px)
    hair_soft = _light_feather(hair_bin, config.mask_feather_px)
    lips_soft = feature["lips"]
    eyes_soft = feature["eyes"]
    brows_soft = feature["brows"]

    prepped = apply_target_lab_recolor(
        frame_bgr,
        skin_soft,
        config.theme.skin_lab,
        strength=config.pre_skin_strength,
    )

    ladder_off = build_cel_ladder(
        prepped, config, state=None, state_prefix="demo_off"
    )

    warmed = warm_cel_state(
        reader, frame_idx, config, lookback=warmup_frames
    )
    ladder_on = build_cel_ladder(
        prepped, config, state=warmed, state_prefix="demo_on"
    )

    stylized = ladder_on["cel_core"]
    stylized = apply_target_lab_recolor(
        stylized, skin_soft, config.theme.skin_lab, strength=config.skin_strength
    )
    stylized = apply_target_lab_recolor(
        stylized, lips_soft, config.theme.lips_lab,
        strength=config.features_strength,
    )
    stylized = apply_target_lab_recolor(
        stylized, brows_soft, config.theme.brows_lab,
        strength=config.features_strength,
    )
    stylized = apply_target_lab_recolor(
        stylized, eyes_soft, config.theme.eyes_lab,
        strength=config.features_strength,
    )
    stylized = apply_target_lab_recolor(
        stylized,
        hair_soft,
        config.theme.hair_lab,
        strength=config.hair_strength,
        channels=(1, 2),
    )

    a = head_alpha[..., None]
    composite = np.clip(
        stylized.astype(np.float32) * a + frame_bgr.astype(np.float32) * (1.0 - a),
        0,
        255,
    ).astype(np.uint8)

    temporal_panel = np.concatenate(
        [
            labeled_panel(ladder_off["cel_core"], "cel-shade - EMA off"),
            labeled_panel(ladder_on["cel_core"], f"cel-shade - EMA {config.ema:.2f}"),
        ],
        axis=1,
    )

    steps.extend(
        [
            StepImage(
                "01_hsv_mask.png",
                "1. HSV skin threshold",
                mask_to_rgb(hsv_mask),
                1,
            ),
            StepImage(
                "02_ycbcr_mask.png",
                "2. YCbCr skin threshold",
                mask_to_rgb(ycc_mask),
                2,
            ),
            StepImage(
                "03_skin_mask.png",
                "3. HSV AND YCbCr + morph + CC + face oval",
                mask_to_rgb(skin_refined),
                3,
            ),
            StepImage(
                "04_spatial_filters.png",
                "4. spatial filters (face crop)",
                spatial_filters_panel(frame_bgr, bbox),
                4,
            ),
            StepImage(
                "05_head_feather.png",
                "5. head ellipse x feather alpha",
                feather_preview,
                5,
            ),
            StepImage(
                "06_bilateral.png",
                "6. iterated bilateral x3",
                ladder_off["bilateral"],
                6,
            ),
            StepImage(
                "07_kmeans_L.png",
                "7. Lab K-means posterize L (3 bands)",
                ladder_off["kmeans"],
                7,
            ),
            StepImage(
                "08_xdog.png",
                "8. XDoG line-art overlay",
                ladder_off["xdog"],
                8,
            ),
            StepImage(
                "09_cel_shade_core.png",
                "9. cel-shade core (+ HSV saturation)",
                ladder_off["cel_core"],
                9,
            ),
            StepImage(
                "10_reinhard_regions.png",
                "10. Reinhard per region (skin/lips/eyes/brows/hair)",
                stylized,
                10,
            ),
            StepImage(
                "11_composite.png",
                "11. alpha composite onto original",
                composite,
                11,
            ),
            StepImage(
                "12_temporal_ema.png",
                "12. temporal stability — EMA off vs on",
                temporal_panel,
                12,
            ),
        ]
    )
    return steps


def build_grid(steps: list[StepImage], cols: int = 4) -> np.ndarray:
    """Lay grid-eligible steps out in ``grid_order`` (skip miss marker)."""
    ordered = sorted(
        (s for s in steps if s.grid_order < 90),
        key=lambda s: s.grid_order,
    )
    if not ordered:
        raise RuntimeError("no steps to grid")

    target_h, target_w = ordered[0].image.shape[:2]

    def fit_panel(img: np.ndarray) -> np.ndarray:
        if img.shape[0] == target_h and img.shape[1] == target_w:
            return img
        return cv2.resize(
            img, (target_w, target_h), interpolation=cv2.INTER_AREA
        )

    panels = [
        labeled_panel(fit_panel(s.image), s.label) for s in ordered
    ]
    h, w = panels[0].shape[:2]
    padded = list(panels)
    rows = (len(padded) + cols - 1) // cols
    while len(padded) < rows * cols:
        padded.append(np.zeros((h, w, 3), dtype=np.uint8))

    grid_rows = [
        np.concatenate(padded[r * cols : (r + 1) * cols], axis=1)
        for r in range(rows)
    ]
    return np.concatenate(grid_rows, axis=0)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render step-by-step pipeline demo PNGs for one frame."
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=ROOT / "samples" / "input" / "clip.mp4",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=262,
        help="hero frame index (default: 262 — frontal pick from Phase 6)",
    )
    parser.add_argument(
        "--theme",
        default="porcelain-pink",
        help="theme under assets/themes/<name>.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "output" / "demo_steps",
    )
    parser.add_argument(
        "--grid-cols",
        type=int,
        default=4,
        help="columns in demo_steps_grid.png (default: 4)",
    )
    parser.add_argument(
        "--feather-px",
        type=int,
        default=41,
    )
    parser.add_argument(
        "--ema",
        type=float,
        default=0.85,
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=30,
        help="preceding frames used to warm K-means EMA for step 12",
    )
    args = parser.parse_args()

    theme = load_theme(args.theme)
    config = PipelineConfig(
        theme=theme,
        feather_px=args.feather_px,
        ema=args.ema,
        luminance_levels=theme.luminance_levels,
        chroma_levels=theme.chroma_levels,
        bilateral_passes=3,
        edge_strength=theme.edge_strength,
        saturation_boost=theme.saturation_boost,
        pre_skin_strength=0.25,
        skin_strength=0.55,
        features_strength=0.75,
        hair_strength=0.55,
        mask_feather_px=7,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with VideoReader(args.video) as reader:
        frame = reader.read_frame(args.frame)
        fps = reader.meta.fps
        steps = render_steps(
            frame,
            fps,
            config,
            reader,
            args.frame,
            warmup_frames=args.warmup_frames,
        )

    for step in steps:
        path = args.output_dir / step.filename
        cv2.imwrite(str(path), step.image)
        print(f"[demo_steps] {path.relative_to(ROOT)}")

    if any(s.grid_order < 90 for s in steps):
        grid = build_grid(steps, cols=args.grid_cols)
        grid_path = args.output_dir / "demo_steps_grid.png"
        cv2.imwrite(str(grid_path), grid)
        print(
            f"[demo_steps] grid -> {grid_path.relative_to(ROOT)} "
            f"({grid.shape[1]}x{grid.shape[0]})"
        )

    print(
        f"[demo_steps] done - frame {args.frame}, theme={theme.name}, "
        f"{len(steps)} images"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
