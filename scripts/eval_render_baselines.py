#!/usr/bin/env python3
"""Phase 8 — render the 5 evaluation baselines from one source clip.

Per ``spec.md`` §7.5 / ``plan.md`` Phase 8, every privacy / utility metric
in the report is computed against a fixed set of five renders of the same
clip:

  1. control       — passthrough (no processing)
  2. blur          — Gaussian σ=12 over the full frame                (S05)
  3. mosaic        — 8×8 block averaging over the full frame           (S02)
  4. ivp-per-frame — AvatarShield pipeline, temporal smoothing OFF
  5. ivp-full      — AvatarShield pipeline, temporal smoothing ON

The two "ivp" runs share the cel-shade / per-region Reinhard pipeline from
``avatarshield.render.process_frame``. ``ivp-per-frame`` differs from the
default render only in that the One Euro bbox smoother and K-means EMA are
both disabled (``state_ema=0.0`` + a fresh ``RenderState`` is built every
frame), so the report can show the visible "boiling" we get without S10 /
temporal stability.

Output layout (under ``data/output/phase8/<clip-stem>/``)::

    control.mp4
    blur.mp4
    mosaic.mp4
    ivp_per_frame.mp4
    ivp_full.mp4
    grid_frame_<idx>.png   # 2×3 grid (orig + 5 baselines) at frame idx

Run::

    .venv/bin/python scripts/eval_render_baselines.py
    .venv/bin/python scripts/eval_render_baselines.py --video samples/input/clip.mp4 --theme porcelain-pink
    .venv/bin/python scripts/eval_render_baselines.py --baselines control,blur,ivp_full --no-grid
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from avatarshield.cel_shade import TemporalState  # noqa: E402
from avatarshield.detect import HaarFaceDetector  # noqa: E402
from avatarshield.render import (  # noqa: E402
    RenderConfig,
    RenderState,
    _new_state,
    _resolve_theme_and_style,
    process_frame,
)
from avatarshield.smooth import BboxSmoother  # noqa: E402
from avatarshield.video_io import VideoReader, VideoWriter  # noqa: E402


BASELINE_ORDER = [
    "control",
    "blur",
    "mosaic",
    "ivp_per_frame",
    "ivp_full",
]

BASELINE_LABELS = {
    "control": "Control (original)",
    "blur": "Gaussian blur (sigma=12)",
    "mosaic": "8x8 mosaic",
    "ivp_per_frame": "AvatarShield-IVP (per-frame)",
    "ivp_full": "AvatarShield-IVP (full)",
}


@dataclass
class BaselineResult:
    name: str
    output_path: Path
    frame_count: int
    elapsed_s: float

    @property
    def fps(self) -> float:
        return self.frame_count / self.elapsed_s if self.elapsed_s > 0 else 0.0


# ---------------------------------------------------------------------------
# Simple privacy baselines — Gaussian blur and mosaic.
# ---------------------------------------------------------------------------


def gaussian_blur_frame(frame_bgr: np.ndarray, sigma: float = 12.0) -> np.ndarray:
    """Whole-frame Gaussian blur — S05 spatial filtering baseline."""
    k = int(max(3, 2 * round(3 * sigma) + 1))  # 3*sigma rule, force odd
    return cv2.GaussianBlur(frame_bgr, (k, k), sigma)


def mosaic_frame(frame_bgr: np.ndarray, block: int = 8) -> np.ndarray:
    """Pixelate the whole frame in ``block``-pixel squares — S02 baseline."""
    h, w = frame_bgr.shape[:2]
    small = cv2.resize(
        frame_bgr,
        (max(1, w // block), max(1, h // block)),
        interpolation=cv2.INTER_AREA,
    )
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


# ---------------------------------------------------------------------------
# Per-baseline frame processors.
# ---------------------------------------------------------------------------


def _build_processor(
    name: str, config: RenderConfig, fps: float
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a ``frame -> frame`` callable for the named baseline.

    The IVP closures cache the resolved theme + cel-shade kwargs so the
    theme JSON is read once per clip rather than per frame. ``ivp_per_frame``
    additionally rebuilds the ``RenderState`` on every frame, which kills
    both the bbox One Euro smoother and the K-means EMA (the two temporal
    stabilisers in the pipeline) so the report can show the resulting
    flicker / boiling against ``ivp_full``.
    """
    if name == "control":
        return lambda frame: frame.copy()

    if name == "blur":
        return lambda frame: gaussian_blur_frame(frame, sigma=12.0)

    if name == "mosaic":
        return lambda frame: mosaic_frame(frame, block=8)

    if name in ("ivp_per_frame", "ivp_full"):
        theme, cel_kwargs = _resolve_theme_and_style(config)
        if name == "ivp_full":
            state = _new_state(config, fps)

            def _run_full(frame: np.ndarray) -> np.ndarray:
                return process_frame(
                    frame, state, config, theme=theme, cel_kwargs=cel_kwargs
                )

            return _run_full

        # ivp_per_frame — disable bbox smoother + K-means EMA by handing
        # a fresh BboxSmoother + TemporalState to process_frame every frame.
        # The detector itself is heavy to construct (loads two cascade XMLs),
        # so we share one across the loop.
        per_frame_config = RenderConfig(**{**config.__dict__, "state_ema": 0.0})
        detector = HaarFaceDetector(
            scale_factor=per_frame_config.haar_scale_factor,
            min_neighbors=per_frame_config.haar_min_neighbors,
            min_size_ratio=per_frame_config.haar_min_size_ratio,
            min_skin_coverage=per_frame_config.haar_min_skin_coverage,
        )

        def _run_per_frame(frame: np.ndarray) -> np.ndarray:
            fresh = RenderState(
                smoother=BboxSmoother(
                    fps=fps,
                    min_cutoff=per_frame_config.smooth_min_cutoff,
                    beta=per_frame_config.smooth_beta,
                    hold_frames=per_frame_config.bbox_hold_frames,
                ),
                temporal=TemporalState(ema=0.0),
                detector=detector,
            )
            return process_frame(
                frame, fresh, per_frame_config, theme=theme, cel_kwargs=cel_kwargs
            )

        return _run_per_frame

    raise ValueError(f"unknown baseline: {name}")


# ---------------------------------------------------------------------------
# Per-baseline video render.
# ---------------------------------------------------------------------------


def render_baseline(
    name: str,
    video_path: Path,
    output_path: Path,
    *,
    config: RenderConfig,
    progress_every: int = 60,
) -> BaselineResult:
    reader = VideoReader(video_path)
    meta = reader.meta
    writer = VideoWriter(output_path, fps=meta.fps, size=(meta.width, meta.height))
    processor = _build_processor(name, config, meta.fps)

    t0 = time.time()
    frames = 0
    try:
        for idx, frame in enumerate(reader):
            writer.write(processor(frame))
            frames += 1
            if progress_every and idx % progress_every == 0:
                elapsed = time.time() - t0
                rate = (idx + 1) / max(elapsed, 1e-6)
                print(
                    f"  [{name}] frame {idx + 1}/{meta.frame_count} "
                    f"({rate:.1f} fps)",
                    flush=True,
                )
    finally:
        reader.close()
        writer.close()

    elapsed = time.time() - t0
    print(f"  [{name}] done {frames} frames in {elapsed:.1f}s "
          f"({frames / max(elapsed, 1e-6):.2f} fps) -> {output_path}")
    return BaselineResult(name=name, output_path=output_path,
                          frame_count=frames, elapsed_s=elapsed)


# ---------------------------------------------------------------------------
# Side-by-side grid (2x3) for visual sanity check.
# ---------------------------------------------------------------------------


def _draw_label(panel: np.ndarray, text: str) -> np.ndarray:
    """Black bar with white text overlaid at the top of ``panel``."""
    out = panel.copy()
    h, w = out.shape[:2]
    bar_h = max(28, h // 22)
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.55, out, 0.45, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.45, bar_h / 60.0)
    cv2.putText(
        out, text, (10, int(bar_h * 0.72)),
        font, scale, (255, 255, 255), 2, cv2.LINE_AA,
    )
    return out


def _read_frame_from(video_path: Path, frame_index: int) -> np.ndarray:
    reader = VideoReader(video_path)
    try:
        return reader.read_frame(frame_index)
    finally:
        reader.close()


def build_grid(
    source_video: Path,
    baselines: list[BaselineResult],
    frame_index: int,
    output_path: Path,
    *,
    cols: int = 3,
) -> None:
    """Compose a 2x3 grid PNG: original + 5 baselines at ``frame_index``."""
    panels: list[tuple[str, np.ndarray]] = []
    panels.append(("Original", _read_frame_from(source_video, frame_index)))
    for b in baselines:
        if b.name == "control":
            continue  # control == original; would be redundant in the grid
        panels.append((BASELINE_LABELS[b.name], _read_frame_from(b.output_path, frame_index)))

    cell_h, cell_w = panels[0][1].shape[:2]
    rows = (len(panels) + cols - 1) // cols
    grid = np.zeros((cell_h * rows, cell_w * cols, 3), dtype=np.uint8)
    for i, (label, frame) in enumerate(panels):
        if frame.shape[:2] != (cell_h, cell_w):
            frame = cv2.resize(frame, (cell_w, cell_h))
        r, c = divmod(i, cols)
        grid[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w] = _draw_label(
            frame, label
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), grid)
    print(f"  [grid] -> {output_path}")


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _parse_baselines(arg: str | None) -> list[str]:
    if not arg:
        return list(BASELINE_ORDER)
    requested = [x.strip() for x in arg.split(",") if x.strip()]
    unknown = [x for x in requested if x not in BASELINE_ORDER]
    if unknown:
        raise SystemExit(
            f"unknown baseline(s): {unknown}. "
            f"choose from: {', '.join(BASELINE_ORDER)}"
        )
    return requested


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--video",
        type=Path,
        default=ROOT / "samples" / "input" / "clip.mp4",
        help="source clip (default: samples/input/clip.mp4)",
    )
    parser.add_argument(
        "--theme",
        default="porcelain-pink",
        help="theme name passed to the IVP renders (default: porcelain-pink)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="output dir (default: data/output/phase8/<clip-stem>/)",
    )
    parser.add_argument(
        "--baselines",
        default=None,
        help=(
            "comma-separated subset of "
            + ",".join(BASELINE_ORDER)
            + " (default: all)"
        ),
    )
    parser.add_argument(
        "--grid-frame",
        type=int,
        default=None,
        help="frame index for the side-by-side grid (default: middle frame)",
    )
    parser.add_argument(
        "--no-grid",
        action="store_true",
        help="skip the grid PNG",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-render baselines even when the MP4 already exists",
    )
    args = parser.parse_args(argv)

    video = args.video.resolve()
    if not video.exists():
        print(f"video not found: {video}", file=sys.stderr)
        return 2

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else ROOT / "data" / "output" / "phase8" / video.stem
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    config = RenderConfig(theme=args.theme)
    names = _parse_baselines(args.baselines)

    print(f"[eval] source: {video}")
    print(f"[eval] output: {output_dir}")
    print(f"[eval] baselines: {', '.join(names)}")
    print(f"[eval] theme: {args.theme}")

    results: list[BaselineResult] = []
    for name in names:
        out_path = output_dir / f"{name}.mp4"
        if out_path.exists() and not args.force:
            # Reuse existing render — populate a stub result so the grid
            # step still has output paths to read from.
            print(f"  [{name}] reusing existing {out_path}")
            with VideoReader(out_path) as r:
                frame_count = r.meta.frame_count
            results.append(
                BaselineResult(name=name, output_path=out_path,
                               frame_count=frame_count, elapsed_s=0.0)
            )
            continue
        print(f"  [{name}] rendering -> {out_path}")
        results.append(render_baseline(name, video, out_path, config=config))

    if args.no_grid or not results:
        return 0

    with VideoReader(video) as r:
        total_frames = r.meta.frame_count
    frame_index = (
        args.grid_frame if args.grid_frame is not None else max(0, total_frames // 2)
    )
    frame_index = max(0, min(total_frames - 1, frame_index))
    grid_path = output_dir / f"grid_frame_{frame_index}.png"
    build_grid(video, results, frame_index, grid_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
