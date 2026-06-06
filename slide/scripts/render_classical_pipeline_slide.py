"""Render a corrected `Classical IVP Pipeline` slide PNG.

Fixes versus the original slide 5 of `... - enriched.pptx`:
  - drop the "Track A" framing (project pivoted to single-track pure-IVP501)
  - place One Euro right after Haar detect (on bbox), not at the end
  - move EMA inside the Cel-Shade Core (on K-means cluster centers)
  - call out Reinhard as `per region` with hair chroma-only
  - strip all IVP501 lesson-slide references (`S05` / `S06` / `S07` / `S08` / `S10`)
  - embed a head-cropped visual strip under the title so the abstract diagram
    is anchored to real pipeline outputs

Writes:
  slide/assets/slide_classical_pipeline.png  (3840x2160)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

SLIDE_DIR = Path(__file__).resolve().parent.parent
OUT = SLIDE_DIR / "assets" / "slide_classical_pipeline.png"
STRIP = SLIDE_DIR / "assets" / "slide_classical_strip.png"

NAVY = "#0d2a52"
NAVY_DARK = "#08193a"
ACCENT = "#1e63d5"
ACCENT_LIGHT = "#e3edff"
TEXT_LIGHT = "#ffffff"
TEXT_DARK = "#1a1a1a"
SUB_TEXT = "#4d4d4d"


def box(ax, x, y, w, h, title, sub=None, *, face=NAVY, title_color=TEXT_LIGHT,
        sub_color=ACCENT_LIGHT, title_size=11.5, sub_size=8.5, rad=0.04):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.0,rounding_size={rad}",
        linewidth=0, facecolor=face, edgecolor="none",
    )
    ax.add_patch(patch)
    if sub is None:
        ax.text(x + w / 2, y + h / 2, title,
                ha="center", va="center",
                color=title_color, fontsize=title_size, fontweight="bold")
    else:
        ax.text(x + w / 2, y + h * 0.66, title,
                ha="center", va="center",
                color=title_color, fontsize=title_size, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.30, sub,
                ha="center", va="center",
                color=sub_color, fontsize=sub_size)


def arrow(ax, x1, y1, x2, y2, *, color=NAVY_DARK, lw=1.6, style="-|>"):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=14,
        linewidth=lw, color=color, shrinkA=2, shrinkB=2,
    )
    ax.add_patch(a)


def main() -> int:
    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=200)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56.25)
    ax.set_axis_off()
    fig.patch.set_facecolor("white")

    ax.text(4, 52.5, "Classical IVP Pipeline",
            color=NAVY_DARK, fontsize=26, fontweight="bold")
    ax.add_patch(FancyBboxPatch(
        (4, 50.9), 92, 0.25,
        boxstyle="round,pad=0,rounding_size=0",
        facecolor=ACCENT, edgecolor="none",
    ))

    strip_labels = [
        "Frame + Haar",
        "Skin Mask",
        "Head Ellipse",
        "Iterated Bilateral × 3",
        "Lab K-means",
        "Cel-Shade + Reinhard",
        "Composite",
    ]
    if STRIP.exists():
        strip_img = imread(STRIP)
        sh, sw = strip_img.shape[:2]
        strip_w = 88.0
        strip_h = strip_w * (sh / sw)
        strip_x0 = (100 - strip_w) / 2
        strip_y1 = 49.8
        strip_y0 = strip_y1 - strip_h
        ax.imshow(
            strip_img,
            extent=(strip_x0, strip_x0 + strip_w, strip_y0, strip_y1),
            aspect="auto", zorder=2, interpolation="lanczos",
        )
        n = len(strip_labels)
        panel_w = strip_w / n
        for i, lbl in enumerate(strip_labels):
            cx = strip_x0 + (i + 0.5) * panel_w
            ax.text(cx, strip_y0 - 0.6, lbl,
                    ha="center", va="top",
                    color=NAVY_DARK, fontsize=10.5, fontweight="bold")

    row1_y = 30.5
    row1_h = 4.2
    row1_gap = 1.8
    row1 = [
        ("Frame", "BGR/RGB image"),
        ("Haar Detect", "frontal + profile"),
        ("One Euro", "bbox (cx,cy,w,h)"),
        ("Skin Mask", "HSV ∩ YCbCr"),
        ("Morphology", "clean + CC"),
        ("Head Ellipse", "Gaussian feather"),
    ]
    row1_w = (92 - row1_gap * (len(row1) - 1)) / len(row1)
    row1_x0 = 4
    centers_row1 = []
    for i, (title, sub) in enumerate(row1):
        cx = row1_x0 + i * (row1_w + row1_gap)
        box(ax, cx, row1_y, row1_w, row1_h, title, sub,
            title_size=11.5, sub_size=8.5)
        centers_row1.append((cx, cx + row1_w))

    for i in range(len(row1) - 1):
        x1 = centers_row1[i][1]
        x2 = centers_row1[i + 1][0]
        arrow(ax, x1, row1_y + row1_h / 2, x2, row1_y + row1_h / 2)

    core_y = 14.5
    core_h = 12.5
    core_x = 4
    core_w = 92
    ax.add_patch(FancyBboxPatch(
        (core_x, core_y), core_w, core_h,
        boxstyle="round,pad=0,rounding_size=0.6",
        facecolor=ACCENT_LIGHT, edgecolor=ACCENT, linewidth=1.6,
    ))
    ax.text(core_x + 1.6, core_y + core_h - 1.4, "Cel-Shade Core",
            color=NAVY_DARK, fontsize=14, fontweight="bold")
    ax.text(core_x + 1.6, core_y + core_h - 2.8,
            "runs inside the head ellipse — edge-preserving stylization",
            color=SUB_TEXT, fontsize=9.2, style="italic")

    inner_y = core_y + 1.4
    inner_h = 5.0
    inner_gap = 1.4
    inner_x0 = core_x + 1.6
    inner = [
        ("Iterated Bilateral × 3", "edge-preserving, kills micro-texture"),
        ("Lab K-means", "posterize L → 3 cluster bands"),
        ("XDoG Line Art", "extended DoG contour overlay"),
        ("HSV Saturation Lift", "theme-driven chroma boost"),
    ]
    inner_w = (core_w - 2 * 1.6 - inner_gap * (len(inner) - 1)) / len(inner)
    inner_centers = []
    for i, (title, sub) in enumerate(inner):
        cx = inner_x0 + i * (inner_w + inner_gap)
        box(ax, cx, inner_y, inner_w, inner_h, title, sub,
            face=NAVY, title_size=10.8, sub_size=8.4, rad=0.25)
        inner_centers.append((cx, cx + inner_w))
    for i in range(len(inner) - 1):
        x1 = inner_centers[i][1]
        x2 = inner_centers[i + 1][0]
        arrow(ax, x1, inner_y + inner_h / 2, x2, inner_y + inner_h / 2,
              color=ACCENT, lw=1.4)

    ema_x = core_x + 1.6
    ema_w = core_w - 3.2
    ema_y = core_y + 0.15
    ax.add_patch(FancyBboxPatch(
        (ema_x, ema_y), ema_w, 1.05,
        boxstyle="round,pad=0,rounding_size=0.15",
        facecolor="#fff3cd", edgecolor="#d6a700", linewidth=1.0,
    ))
    ax.text(ema_x + ema_w / 2, ema_y + 0.52,
            "EMA α=0.85 on K-means cluster centers across frames — keeps bands from boiling",
            ha="center", va="center", color="#5a4400", fontsize=9.4, fontweight="bold")

    row3_y = 4.5
    row3_h = 4.6
    row3 = [
        ("Reinhard per region",
         "Lab transfer · skin / lips / eyes / brows · hair chroma-only"),
        ("Alpha Composite",
         "feathered head ellipse → original frame"),
        ("Output Frame",
         "stylized identity, body & background untouched"),
    ]
    row3_gap = 2.4
    row3_w = (92 - row3_gap * (len(row3) - 1)) / len(row3)
    row3_x0 = 4
    centers_row3 = []
    for i, (title, sub) in enumerate(row3):
        cx = row3_x0 + i * (row3_w + row3_gap)
        box(ax, cx, row3_y, row3_w, row3_h, title, sub,
            title_size=12.5, sub_size=9.0)
        centers_row3.append((cx, cx + row3_w))
    for i in range(len(row3) - 1):
        x1 = centers_row3[i][1]
        x2 = centers_row3[i + 1][0]
        arrow(ax, x1, row3_y + row3_h / 2, x2, row3_y + row3_h / 2, lw=1.8)

    arrow(ax,
          centers_row1[-1][0] + row1_w / 2, row1_y,
          core_x + core_w / 2, core_y + core_h,
          color=NAVY_DARK, lw=1.8)
    arrow(ax,
          core_x + core_w / 2, core_y,
          centers_row3[0][0] + row3_w / 2, row3_y + row3_h,
          color=NAVY_DARK, lw=1.8)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.2)
    plt.close(fig)
    print(f"  [write] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
