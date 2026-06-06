#!/usr/bin/env python3
"""Phase 6 — render one preview frame per theme + a comparison grid.

For each theme JSON under ``assets/themes/`` this script:

1. Loads the theme through :func:`avatarshield.palette.load_theme`.
2. Builds a :class:`scripts.phase5_test.PipelineConfig` that honours the
   theme's style scalars (``saturation_boost`` / ``edge_strength`` /
   ``luminance_levels`` / ``chroma_levels``) — spec §6.1.
3. Renders a single frame of ``samples/input/clip.mp4`` through the
   Phase 5 pipeline (cel-shade + per-region Reinhard + composite).
4. Writes the composite to ``assets/themes/previews/<name>.png``.

It also assembles a ``data/output/phase6_grid.png`` with the original
frame and the five themed previews side-by-side, labelled, for the
report and ``assets/themes/README.md``.

Run::

    .venv/bin/python scripts/phase6_render_previews.py
    .venv/bin/python scripts/phase6_render_previews.py --frame 262
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from avatarshield.cel_shade import TemporalState  # noqa: E402
from avatarshield.detect import HaarFaceDetector  # noqa: E402
from avatarshield.palette import (  # noqa: E402
    Theme,
    list_available_themes,
    load_theme,
)
from avatarshield.smooth import BboxSmoother  # noqa: E402
from scripts.phase5_test import (  # noqa: E402
    PipelineConfig,
    draw_panel_label,
    process_frame,
)


# Spec §6.3 order — the canonical catalogue ordering used in the report.
CATALOGUE_ORDER = [
    "porcelain-pink",
    "tan-amber",
    "ivory-violet",
    "bronze-teal",
    "peach-noir",
]


def build_config(theme: Theme, args) -> PipelineConfig:
    """Build a PipelineConfig honouring the theme's style scalars."""
    return PipelineConfig(
        theme=theme,
        feather_px=args.feather_px,
        ema=args.ema,
        luminance_levels=theme.luminance_levels,
        chroma_levels=theme.chroma_levels,
        bilateral_passes=args.bilateral_passes,
        edge_strength=theme.edge_strength,
        saturation_boost=theme.saturation_boost,
        pre_skin_strength=args.pre_skin_strength,
        skin_strength=args.skin_strength,
        features_strength=args.features_strength,
        hair_strength=args.hair_strength,
        mask_feather_px=args.mask_feather_px,
    )


def render_theme_preview(
    frame_bgr: np.ndarray,
    fps: float,
    theme: Theme,
    args,
) -> np.ndarray:
    """Run a single-frame Phase 5 pass for this theme."""
    config = build_config(theme, args)
    detector = HaarFaceDetector()
    smoother = BboxSmoother(fps=fps)
    state = TemporalState(ema=config.ema)
    composite, _masks, hit = process_frame(
        frame_bgr, detector, smoother, state, config
    )
    if not hit:
        print(
            f"[phase6] WARNING: theme={theme.name} — detector missed; "
            "preview will show the unmodified frame.",
            file=sys.stderr,
        )
    return composite


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--video",
        type=Path,
        default=ROOT / "samples" / "input" / "clip.mp4",
        help="source clip (default: samples/input/clip.mp4)",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=262,
        help="frame index to render (default: 262 — frontal pick from Phase 1)",
    )
    parser.add_argument(
        "--themes-dir",
        type=Path,
        default=ROOT / "assets" / "themes",
        help="folder of theme JSONs",
    )
    parser.add_argument(
        "--previews-dir",
        type=Path,
        default=ROOT / "assets" / "themes" / "previews",
        help="where to write per-theme preview PNGs",
    )
    parser.add_argument(
        "--grid-out",
        type=Path,
        default=ROOT / "data" / "output" / "phase6_grid.png",
        help="comparison grid (original + 5 themes)",
    )
    # Pipeline knobs forwarded from phase5_test defaults; theme overrides the
    # three style scalars (sat / edge / lum / chroma).
    parser.add_argument("--feather-px", type=int, default=41)
    parser.add_argument("--ema", type=float, default=0.85)
    parser.add_argument("--bilateral-passes", type=int, default=3)
    parser.add_argument("--pre-skin-strength", type=float, default=0.25)
    parser.add_argument("--skin-strength", type=float, default=0.55)
    parser.add_argument("--features-strength", type=float, default=0.75)
    parser.add_argument("--hair-strength", type=float, default=0.55)
    parser.add_argument("--mask-feather-px", type=int, default=21)
    args = parser.parse_args()

    video = args.video.resolve()
    if not video.exists():
        print(f"video not found: {video}", file=sys.stderr)
        return 2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"could not open video: {video}", file=sys.stderr)
        return 2
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.frame))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"could not read frame {args.frame}", file=sys.stderr)
        return 2

    available = set(list_available_themes(args.themes_dir))
    names = [n for n in CATALOGUE_ORDER if n in available]
    extras = sorted(available - set(CATALOGUE_ORDER))
    names.extend(extras)
    if not names:
        print(f"no theme JSONs under {args.themes_dir}", file=sys.stderr)
        return 2

    args.previews_dir.mkdir(parents=True, exist_ok=True)
    args.grid_out.parent.mkdir(parents=True, exist_ok=True)

    panels: list[np.ndarray] = []
    original_panel = frame.copy()
    draw_panel_label(original_panel, "0. original")
    panels.append(original_panel)

    for i, name in enumerate(names, start=1):
        theme = load_theme(name, themes_dir=args.themes_dir)
        preview = render_theme_preview(frame, fps, theme, args)
        preview_path = args.previews_dir / f"{name}.png"
        cv2.imwrite(str(preview_path), preview)
        labelled = preview.copy()
        draw_panel_label(labelled, f"{i}. {theme.name}")
        panels.append(labelled)
        print(
            f"[phase6] {name:<14s} → {preview_path.relative_to(ROOT)}  "
            f"skin={tuple(int(v) for v in theme.skin_lab)} "
            f"hair={tuple(int(v) for v in theme.hair_lab)} "
            f"sat={theme.saturation_boost:.2f} edge={theme.edge_strength:.2f} "
            f"lum={theme.luminance_levels}"
        )

    # Lay panels out as a 2×3 grid (original + 5 themes).
    cols = 3
    rows = (len(panels) + cols - 1) // cols
    h, w = panels[0].shape[:2]
    while len(panels) < rows * cols:
        panels.append(np.zeros_like(panels[0]))
    grid_rows = [
        np.concatenate(panels[r * cols : (r + 1) * cols], axis=1)
        for r in range(rows)
    ]
    grid = np.concatenate(grid_rows, axis=0)
    cv2.imwrite(str(args.grid_out), grid)
    print(
        f"[phase6] grid → {args.grid_out.relative_to(ROOT)} "
        f"({grid.shape[1]}x{grid.shape[0]})"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
