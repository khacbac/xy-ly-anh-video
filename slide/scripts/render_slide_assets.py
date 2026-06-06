#!/usr/bin/env python3
"""Compose slide-ready images (head-cropped + landscape) from the Yejin clip.

Produces wide, slide-friendly composites that fit a 16:9 layout without
vertical cut-off. All output goes to ``slide/assets/`` next to the deck.

Inputs:
  - samples/input/13-year-old-Yejin-solo-dance.mp4 (source video for face crop)
  - data/output/phase8/13-year-old-Yejin-solo-dance/{control,blur,mosaic,ivp_per_frame,ivp_full}.mp4

Outputs:
  - slide/assets/slide_pipeline_strip.png   (5-step pipeline strip, head-cropped)
  - slide/assets/slide_celshade_head.png    (head-cropped cel-shade core view)
  - slide/assets/slide_reinhard_head.png    (head-cropped reinhard view)
  - slide/assets/slide_baselines_strip.png  (5-baseline horizontal strip, head-cropped)
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

SLIDE_DIR = Path(__file__).resolve().parent.parent
ROOT = SLIDE_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from avatarshield.detect import HaarFaceDetector  # noqa: E402


OUT_DIR = SLIDE_DIR / "assets"
DEMO_DIR = ROOT / "data" / "output" / "demo_steps"
PHASE8_YEJIN = ROOT / "data" / "output" / "phase8" / "13-year-old-Yejin-solo-dance"
SOURCE_VIDEO = ROOT / "samples" / "input" / "13-year-old-Yejin-solo-dance.mp4"

# Tuned for the Yejin clip — frame 143 has a clean frontal face mid-clip.
HERO_FRAME = 143
# Head crop: square box around the detected face, padded 2.2x for hair+shoulders.
HEAD_PAD = 2.2
# Output panel size after head crop (square).
PANEL_SIDE = 540


def read_frame(video_path: Path, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame {frame_index} from {video_path}")
    return frame


def head_crop(frame: np.ndarray, bbox_xywh: tuple[int, int, int, int],
              pad: float = HEAD_PAD, target: int = PANEL_SIDE) -> np.ndarray:
    """Square crop centered on bbox with `pad`x expansion, then resize."""
    x, y, w, h = bbox_xywh
    cx, cy = x + w / 2, y + h / 2
    side = int(max(w, h) * pad)
    H, W = frame.shape[:2]
    half = side // 2
    # Clamp center so the square fits the frame; if it doesn't, pad with black.
    x0 = int(round(cx - half))
    y0 = int(round(cy - half))
    x1 = x0 + side
    y1 = y0 + side
    # Pad if out-of-bounds.
    pad_l = max(0, -x0)
    pad_t = max(0, -y0)
    pad_r = max(0, x1 - W)
    pad_b = max(0, y1 - H)
    if any((pad_l, pad_t, pad_r, pad_b)):
        frame = cv2.copyMakeBorder(frame, pad_t, pad_b, pad_l, pad_r,
                                   cv2.BORDER_CONSTANT, value=(0, 0, 0))
        x0 += pad_l; x1 += pad_l
        y0 += pad_t; y1 += pad_t
    crop = frame[y0:y1, x0:x1]
    return cv2.resize(crop, (target, target), interpolation=cv2.INTER_AREA)


def draw_label(panel: np.ndarray, text: str) -> np.ndarray:
    out = panel.copy()
    h, w = out.shape[:2]
    bar_h = max(28, h // 18)
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.65, out, 0.35, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.55, bar_h / 55.0)
    cv2.putText(out, text, (10, int(bar_h * 0.72)),
                font, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def horizontal_strip(panels: list[tuple[str, np.ndarray]], gap: int = 8) -> np.ndarray:
    """Compose labeled panels left-to-right with thin black gutters."""
    labelled = [draw_label(p, lbl) for lbl, p in panels]
    h = labelled[0].shape[0]
    w = labelled[0].shape[1]
    total_w = w * len(labelled) + gap * (len(labelled) - 1)
    strip = np.zeros((h, total_w, 3), dtype=np.uint8)
    for i, p in enumerate(labelled):
        x = i * (w + gap)
        strip[:, x:x + w] = p
    return strip


def main() -> int:
    if not PHASE8_YEJIN.exists():
        print(f"missing phase8 dir: {PHASE8_YEJIN}", file=sys.stderr)
        print("run scripts/eval_render_baselines.py --video samples/input/13-year-old-Yejin-solo-dance.mp4 first",
              file=sys.stderr)
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Locate the face bbox once from the control (== original) frame.
    detector = HaarFaceDetector()
    orig = read_frame(PHASE8_YEJIN / "control.mp4", HERO_FRAME)
    bbox = detector.detect(orig)
    if bbox is None:
        # Fall back to nearby frames if HERO_FRAME misses.
        for delta in (1, 2, 3, -1, -2, -3, 5, -5, 10, -10):
            trial = read_frame(PHASE8_YEJIN / "control.mp4", HERO_FRAME + delta)
            bbox = detector.detect(trial)
            if bbox is not None:
                orig = trial
                print(f"  [info] using frame {HERO_FRAME + delta} (HERO_FRAME missed)")
                break
        if bbox is None:
            print("could not detect a face near HERO_FRAME", file=sys.stderr)
            return 3
    bbox_xywh = bbox.as_xywh()
    print(f"  [info] head bbox @ frame {HERO_FRAME}: {bbox_xywh}")

    # 2) Build the 5-baseline horizontal strip (head-cropped).
    baseline_specs = [
        ("Original",            "control.mp4"),
        ("Gaussian blur σ=12",  "blur.mp4"),
        ("8x8 mosaic",          "mosaic.mp4"),
        ("IVP (no smoothing)",  "ivp_per_frame.mp4"),
        ("IVP (full)",          "ivp_full.mp4"),
    ]
    panels = []
    for label, name in baseline_specs:
        f = read_frame(PHASE8_YEJIN / name, HERO_FRAME)
        panels.append((label, head_crop(f, bbox_xywh)))
    strip = horizontal_strip(panels)
    out_baselines = OUT_DIR / "slide_baselines_strip.png"
    cv2.imwrite(str(out_baselines), strip)
    print(f"  [write] {out_baselines}  ({strip.shape[1]}x{strip.shape[0]})")

    # 3) Build the pipeline-overview horizontal strip (5 key steps, head-cropped).
    #    Re-use existing demo_steps PNGs which are already rendered at full body.
    pipeline_specs = [
        ("Original",           DEMO_DIR / "00_original.png"),
        ("Skin mask",          DEMO_DIR / "03_skin_mask.png"),
        ("Bilateral x3",       DEMO_DIR / "06_bilateral.png"),
        ("K-means L",          DEMO_DIR / "07_kmeans_L.png"),
        ("Composite",          DEMO_DIR / "11_composite.png"),
    ]
    pipeline_panels = []
    for label, path in pipeline_specs:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  [warn] missing {path}, skipping pipeline strip")
            return 4
        # Detect bbox per-image for the original-aspect frames (all 1080x1920).
        # The skin mask is binary white-on-black; use the original bbox as a proxy.
        if label == "Skin mask":
            bb = bbox_xywh
        else:
            det = detector.detect(img)
            bb = det.as_xywh() if det is not None else bbox_xywh
        pipeline_panels.append((label, head_crop(img, bb)))
    pipeline_strip = horizontal_strip(pipeline_panels)
    out_pipeline = OUT_DIR / "slide_pipeline_strip.png"
    cv2.imwrite(str(out_pipeline), pipeline_strip)
    print(f"  [write] {out_pipeline}  ({pipeline_strip.shape[1]}x{pipeline_strip.shape[0]})")

    # 4) Head-cropped detail views for cel-shade core and reinhard regions.
    for src_name, out_name in [
        ("09_cel_shade_core.png", "slide_celshade_head.png"),
        ("10_reinhard_regions.png", "slide_reinhard_head.png"),
    ]:
        src = DEMO_DIR / src_name
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  [warn] missing {src}, skipping")
            continue
        det = detector.detect(img)
        bb = det.as_xywh() if det is not None else bbox_xywh
        crop = head_crop(img, bb, pad=2.6, target=720)
        out = OUT_DIR / out_name
        cv2.imwrite(str(out), crop)
        print(f"  [write] {out}  ({crop.shape[1]}x{crop.shape[0]})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
