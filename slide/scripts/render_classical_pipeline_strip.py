"""Render an 8-panel head-cropped strip that maps 1-1 to the Classical IVP
Pipeline diagram boxes on slide 5.

Panels (each labeled across the top):
  1. Frame + Haar           — 00_original.png with green bbox drawn
  2. Skin Mask              — 03_skin_mask.png  (HSV ∩ YCbCr + morphology)
  3. Head Ellipse           — 05_head_feather.png (Gaussian feather)
  4. Iterated Bilateral × 3 — 06_bilateral.png
  5. Lab K-means            — 07_kmeans_L.png
  6. XDoG Line Art          — 08_xdog.png
  7. Reinhard per region    — 10_reinhard_regions.png
  8. Composite              — 11_composite.png

Writes:
  slide/assets/slide_classical_strip.png
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

DEMO = ROOT / "data" / "output" / "demo_steps"
OUT = SLIDE_DIR / "assets" / "slide_classical_strip.png"

PAD = 2.2
PANEL = 540
GAP = 10

PANELS = [
    ("Frame + Haar",           "00_original.png",          True),
    ("Skin Mask",              "03_skin_mask.png",         False),
    ("Head Ellipse",           "05_head_feather.png",      False),
    ("Iterated Bilateral × 3", "06_bilateral.png",         False),
    ("Lab K-means",            "07_kmeans_L.png",          False),
    ("Cel-Shade + Reinhard",   "10_reinhard_regions.png",  False),
    ("Composite",              "11_composite.png",         False),
]


def head_crop(frame: np.ndarray, bbox_xywh: tuple[int, int, int, int],
              pad: float = PAD, target: int = PANEL) -> np.ndarray:
    x, y, w, h = bbox_xywh
    cx, cy = x + w / 2, y + h / 2
    side = int(max(w, h) * pad)
    H, W = frame.shape[:2]
    half = side // 2
    x0 = int(round(cx - half))
    y0 = int(round(cy - half))
    x1, y1 = x0 + side, y0 + side
    pad_l, pad_t = max(0, -x0), max(0, -y0)
    pad_r, pad_b = max(0, x1 - W), max(0, y1 - H)
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
    bar_h = max(32, h // 14)
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.72, out, 0.28, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.6, bar_h / 50.0)
    (tw, th), _ = cv2.getTextSize(text, font, scale, 2)
    while tw > w - 16 and scale > 0.4:
        scale -= 0.05
        (tw, th), _ = cv2.getTextSize(text, font, scale, 2)
    x = (w - tw) // 2
    y = int(bar_h * 0.72)
    cv2.putText(out, text, (x, y), font, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def draw_bbox(panel: np.ndarray, bbox_xywh: tuple[int, int, int, int],
              color=(80, 240, 80), thickness=4) -> np.ndarray:
    out = panel.copy()
    x, y, w, h = bbox_xywh
    cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
    return out


def main() -> int:
    if not DEMO.exists():
        print(f"missing demo_steps: {DEMO}", file=sys.stderr)
        return 2

    base = cv2.imread(str(DEMO / "00_original.png"), cv2.IMREAD_COLOR)
    if base is None:
        print(f"missing {DEMO / '00_original.png'}", file=sys.stderr)
        return 3

    detector = HaarFaceDetector()
    bbox = detector.detect(base)
    if bbox is None:
        print("Haar miss on 00_original.png — falling back to centered bbox", file=sys.stderr)
        H, W = base.shape[:2]
        side = min(H, W) // 3
        cx, cy = W // 2, H // 3
        bbox_xywh = (cx - side // 2, cy - side // 2, side, side)
    else:
        bbox_xywh = bbox.as_xywh()

    panels: list[np.ndarray] = []
    for label, fname, draw_box in PANELS:
        path = DEMO / fname
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"missing {path}, skipping", file=sys.stderr)
            continue
        if draw_box:
            img = draw_bbox(img, bbox_xywh)
        crop = head_crop(img, bbox_xywh)
        panels.append(crop)

    h = panels[0].shape[0]
    w = panels[0].shape[1]
    total_w = w * len(panels) + GAP * (len(panels) - 1)
    strip = np.full((h, total_w, 3), 255, dtype=np.uint8)
    for i, p in enumerate(panels):
        x = i * (w + GAP)
        strip[:, x:x + w] = p

    OUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT), strip)
    print(f"  [write] {OUT}  ({strip.shape[1]}x{strip.shape[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
