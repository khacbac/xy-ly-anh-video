#!/usr/bin/env python3
"""Render an input video through the pure-IVP pipeline (CLI helper).

Thin convenience wrapper around :func:`avatarshield.render.render_video`
and :func:`avatarshield.render.render_preview_frame` that picks sensible
defaults for ad-hoc tests / demo prep.

Usage::

    .venv/bin/python scripts/render_clip.py
    .venv/bin/python scripts/render_clip.py --theme tan-amber
    .venv/bin/python scripts/render_clip.py --frame-index 262
    .venv/bin/python scripts/render_clip.py \\
        --video samples/input/clip.mp4 \\
        --theme bronze-teal \\
        --output data/output/demo.mp4

When ``--output`` is omitted the script writes
``data/output/<video_stem>_<theme>.mp4`` (or ``.png`` if a frame index is
requested).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from avatarshield.render import (  # noqa: E402
    RenderConfig,
    render_preview_frame,
    render_video,
)


def _default_output(
    video: Path, theme: str, frame_index: int | None
) -> Path:
    stem = video.stem
    suffix = ".png" if frame_index is not None else ".mp4"
    return ROOT / "data" / "output" / f"{stem}_{theme}{suffix}"


def main(argv: list[str] | None = None) -> int:
    defaults = RenderConfig()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--video",
        type=Path,
        default=ROOT / "samples" / "input" / "clip.mp4",
        help="source clip (default: samples/input/clip.mp4)",
    )
    parser.add_argument(
        "--theme",
        default=defaults.theme,
        help="theme name under assets/themes/<name>.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "output path (.mp4 or .png); "
            "defaults to data/output/<stem>_<theme>"
        ),
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=None,
        help="render a single frame (PNG) instead of the whole clip",
    )
    parser.add_argument("--ema", type=float, default=defaults.state_ema)
    parser.add_argument(
        "--bilateral-passes", type=int, default=defaults.bilateral_passes
    )
    parser.add_argument(
        "--skin-strength",
        type=float,
        default=defaults.palette_strength_skin,
    )
    parser.add_argument(
        "--features-strength",
        type=float,
        default=defaults.palette_strength_features,
    )
    parser.add_argument(
        "--hair-strength",
        type=float,
        default=defaults.palette_strength_hair,
    )
    parser.add_argument(
        "--quiet", action="store_true", help="suppress per-frame progress"
    )
    args = parser.parse_args(argv)

    video = args.video.resolve()
    if not video.exists():
        print(f"video not found: {video}", file=sys.stderr)
        return 2

    output = args.output or _default_output(
        video, args.theme, args.frame_index
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    config = RenderConfig(theme=args.theme)
    config.state_ema = args.ema
    config.bilateral_passes = args.bilateral_passes
    config.palette_strength_skin = args.skin_strength
    config.palette_strength_features = args.features_strength
    config.palette_strength_hair = args.hair_strength

    def _pretty(path: str) -> str:
        p = Path(path).resolve()
        try:
            return str(p.relative_to(ROOT))
        except ValueError:
            return str(p)

    is_png = output.suffix.lower() == ".png"
    if args.frame_index is not None or is_png:
        result = render_preview_frame(
            video, output, frame_index=args.frame_index, config=config
        )
        if not args.quiet:
            print(
                f"[preview] theme={args.theme} "
                f"frame={result.frame_index} -> {_pretty(result.output_path)}"
            )
        return 0

    result = render_video(video, output, config, progress=not args.quiet)
    if not args.quiet:
        print(
            f"[render] theme={args.theme} "
            f"frames={result.frame_count} -> {_pretty(result.output_path)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
