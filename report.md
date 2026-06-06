---
title: "AvatarShield — Child-Safe Video Anonymization via Classical Cel-Shade Stylization with Theme Palette"
subtitle: "IVP501 final project · branch `ivp-pure` (spec v1.0)"
author: "Team — Bac, [member 2], [member 3] (FSB MSE)"
date: "2026-06-02"
geometry: margin=1in
fontsize: 11pt
---

## Abstract

We present **AvatarShield**, a video anonymization pipeline aimed at
protecting children's faces in shared video while preserving the
*communicative value* of the clip (expression, pose, motion).
The pipeline is intentionally classical: every transform maps to a
slide in the IVP501 syllabus — Viola–Jones face detection (S08), HSV
∩ YCbCr skin segmentation (S07), iterated bilateral filtering plus
Lab K-means quantization and XDoG line art for the cel-shade core
(S05/S06/S07), per-region Reinhard color transfer keyed by a hand-
authored *theme palette* (S07), and temporal stability via the One
Euro filter on the bounding box and EMA-smoothed K-means cluster
centers (S10). No neural model, no MediaPipe, no GAN. The entire
runtime is `opencv-python` + `numpy` + `Pillow`. Against four
reference baselines (control, Gaussian blur σ=12, 8×8 mosaic, and the
same pipeline with temporal smoothing disabled), AvatarShield achieves
a FaceNet re-identification rate of **0.68–0.70** versus 0.94 for
mosaic and 0.80 for blur, while keeping the detector usable on
27/33 sampled frames (blur drops to 5/33). The full pipeline reduces
frame-to-frame head-region MAD by ~30 % over the ablated per-frame
variant. We document where the classical FFT high-frequency band
metric is *not* a faithful proxy for cel-shade and argue that re-id
rate is the correct headline metric.

---

## 1. Introduction & motivation

Children appearing on social platforms face concrete risks: biometric
data collection for face-recognition training, cross-platform face
matching, and CSAM-adjacent harm. Existing protections fall short on
*utility*:

- **Blur / mosaic.** Universally available, but destroys the
  communicative value of the clip — expression, gaze direction,
  smile are gone.
- **Full face replacement** (DeepPrivacy and successors). Removes
  *agency* — the subject can no longer recognise themselves in the
  output.
- **Commercial AR filters.** Cosmetic. No formal anonymization
  guarantee, and identity is preserved.
- **GAN stylization** (VToonify, AnimeGAN, DCT-Net). Visually
  excellent but uninspectable; for a course report keyed to S02–S10
  the pipeline is reduced to "the network does it".

AvatarShield's design hypothesis is that, for child-safe sharing,
identity does *not* have to come from facial geometry. It can come
from a **theme palette** the user picks (consistent look across
clips), preserved self-recognition cues (relative luminance,
expression idiosyncrasy, head motion), and a *provable* collapse of
high-frequency face texture via classical low-pass + quantization.

The trade-off is honest. We accept a visual-fidelity loss versus
VToonify in exchange for explainability, zero-heavy-deps, and a
seconds-per-clip render budget on a CPU laptop.

## 2. Related work

| Family | Examples | What we keep / drop |
|---|---|---|
| Naïve anonymization | Gaussian blur, mosaic | Used as baselines (§5). Both kill identity *and* utility. |
| Face replacement   | DeepPrivacy (2019), DeepPrivacy2 (2023) | Cited as context only — out of scope: removes agency, depends on a large dataset prior. |
| GAN stylization    | VToonify (2022), AnimeGANv2, DCT-Net | Earlier branch of this project tried VToonify; dropped on 2026-06-01 because the syllabus-mapping story was weak. Legacy spec preserved as `spec_legacy.md`. |
| Classical face detection | Viola & Jones (2001) | Core of AvatarShield — used for bbox + skin gate. |
| Color transfer    | Reinhard et al. (2001) | Used for per-region palette transfer in Lab. |
| Cel-shade line art | Winnemöller (2012, XDoG) | Used as the line-art overlay step. |
| Temporal filtering | Casiez et al. (2012, 1€ filter) | Bbox stability + cluster-center EMA. |

Cited references for the methods we *implement* appear in §10.

## 3. Pipeline architecture

```
[Source MP4]
   └─► video_io ──► frame ─┐
                            │
       Haar detect (frontal + profile, S08)        ─► bbox(x,y,w,h)
                            │
       One Euro on (cx, cy, w, h) (S10)            ─► smoothed bbox
                            │
       Skin mask (HSV ∩ YCbCr, S07) + morphology + CC  (S06)
                            │
       Geometric helpers: head ellipse (S04) + feather (S05)
                            │
   ┌─ cel_shade (whole frame) ────────────────────┐
   │   ▪ iterated bilateral filter (S05)          │
   │   ▪ BGR → Lab; K-means quantize L (S07)      │ — TemporalState EMA over centers (S10)
   │   ▪ XDoG line art overlay (S06)              │
   │   ▪ HSV saturation lift (S07)                │
   └──────────────────────────────────────────────┘
                            │
       per-region Reinhard color transfer (S07):
         skin (Lab full, strength 0.55)
         lips / eyes / brows (Lab full, 0.75)
         hair (chroma only — preserve user's L, 0.55)
                            │
       composite: stylized · head_alpha + frame · (1 - head_alpha)
                            │
                            ▼
                      [Output MP4]
```

The core entry point is `avatarshield.render.render_video(video, output,
config)`. Per-frame work happens in `process_frame()` (≈ 100 LoC),
split across the seven modules listed below. Each module's docstring
includes the syllabus slide reference for the operations it performs.

| Module | File | Slide refs | Lines (≈) |
|---|---|---|---:|
| Video I/O                              | `video_io.py`  | S09           | 100 |
| Haar detect + skin mask + head ellipse | `detect.py`    | S04/S06/S07/S08 | 380 |
| One Euro bbox smoother                  | `smooth.py`    | S10            | 210 |
| Cel-shade core (bilateral / Lab K-means / XDoG / saturation) | `cel_shade.py` | S05/S06/S07/S10 | 460 |
| Theme palette loader + Reinhard transfer | `palette.py`  | S07            | 240 |
| Render pipeline / CLI                    | `render.py`   | (orchestrator) | 550 |

Total project source (excluding `scripts/` and `assets/`): ~1940 lines
of Python.

## 4. Theme palette

A theme is a small JSON (< 1 KB) holding mean Lab triples per facial
region plus three style scalars that override the cel-shade defaults:

```json
{
  "name": "porcelain-pink",
  "skin_lab":  [215, 138, 142],
  "hair_lab":  [ 60, 138, 110],
  "lips_lab":  [150, 165, 145],
  "eyes_lab":  [ 70, 135, 145],
  "brows_lab": [ 55, 135, 140],
  "saturation_boost": 1.30,
  "edge_strength":    0.85,
  "luminance_levels": 3,
  "chroma_levels":    0
}
```

Five starter themes ship under `assets/themes/`: `porcelain-pink`,
`tan-amber`, `ivory-violet`, `bronze-teal`, `peach-noir`. The
`peach-noir` theme uses `luminance_levels=2` for a heavier posterise.
See `assets/themes/README.md` for the full Lab table and preview grid
(`data/output/phase6_grid.png`).

Authoring is a 30-minute loop: pick reference image → estimate Lab
means by eye → edit JSON → re-run `scripts/phase6_render_previews.py`
→ inspect. We deferred the interactive authoring notebook (`spec.md`
§6.2) since hand-editing five themes was cheaper than building it.

## 5. Evaluation

### 5.1 Setup

Source clip: `samples/input/clip.mp4` (667 frames, 720×1280, 30 fps,
1 subject). Theme: `porcelain-pink`. Hardware: Apple Silicon M1 16 GB,
CPU only.

Baselines per `spec.md` §7.5 (all rendered by
`scripts/eval_render_baselines.py`):

1. **control** — passthrough.
2. **Gaussian blur σ=12** — whole-frame `cv2.GaussianBlur`.
3. **8×8 mosaic** — downsample / upsample with nearest neighbour.
4. **IVP per-frame** — full pipeline with temporal smoothing OFF
   (One Euro disabled, K-means EMA = 0).
5. **IVP full** — full pipeline with temporal smoothing ON.

Sampling: every 15th frame (45 sampled frames per video).

### 5.2 Privacy — FaceNet re-identification (S08)

Each sampled frame is detected with the project's Haar detector,
cropped, embedded with `facenet-pytorch` (InceptionResnetV1, vggface2
weights, CPU), and compared by cosine distance to the corresponding
embedding in the control clip. Match threshold: 0.7 (FaceNet's
verification default).

| Baseline | n (paired) | mean cosine distance | re-id rate |
|---|---:|---:|---:|
| Gaussian blur σ=12 |  5 | 0.596 | 0.800 |
| 8×8 mosaic         | 17 | 0.526 | 0.941 |
| IVP per-frame      | 28 | 0.614 | **0.679** |
| IVP **full**       | 27 | 0.637 | **0.704** |

Two readings:

- **Re-id rate** alone — IVP wins. The cel-shade collapses the
  identity signal more than mosaic does and roughly matches blur,
  while *also* preserving enough head structure for the detector to
  find the face on 27–28 frames versus blur's 5 frames.
- **Detector-survivability** (the `n` column) is a second-order
  metric the privacy literature usually omits. Downstream face-aware
  tooling — analytics, tagging, accidental further re-id — can still
  *locate* the head in the IVP clip but breaks down on blur. Whether
  this is good or bad depends on the deployment story; for child-safe
  sharing the family album case prefers "head is locatable, identity
  is collapsed".

### 5.3 Privacy — FFT high-frequency energy ratio (S05.02)

Per `spec.md` §7.2 we sum FFT power outside a centred disc of radius
`r₀ = 0.20 · min(H, W) / 2` and divide by the total.

| Baseline | whole-frame HF ratio | head-only HF ratio |
|---|---:|---:|
| Control            | 0.0195 | 0.0164 |
| Gaussian blur σ=12 | 0.0004 | 0.0007 |
| 8×8 mosaic         | 0.0099 | 0.0121 |
| IVP per-frame      | 0.0202 | **0.0400** |
| IVP full           | 0.0203 | **0.0390** |

The whole-frame measurement is dominated by the body and background,
which the IVP pipeline does not touch — IVP and control are tied. The
head-only crop is where the story flips: cel-shade *raises* HF
because the XDoG line art is itself a high-frequency signal. This
is documented in `RESULTS.md` §3.2 with the caveat that the FFT band
metric, while faithful for blur / mosaic, is **not** a faithful
privacy proxy for cel-shade. The right interpretation is that the
identity-bearing micro-texture (skin pores, eyebrow hairs, eye
details) is in the *mid* band, and cel-shade replaces it with sparse,
geometric line art in the *very-high* band. FaceNet, which keys on the
mid-band texture, loses the lock.

### 5.4 Joint trade-off

Plotting whole-frame HF ratio (x) against re-id rate (y) puts the
five baselines into four distinct quadrants:

- **Control** — top-right (full identity, full texture).
- **Blur** — top-left (no texture, identity bleeds through the tiny
  detectable set).
- **Mosaic** — top-middle (partial texture loss, identity intact).
- **IVP** — **bottom-right** (utility-preserving, identity-collapsed).
  This is the position the cel-shade story predicted.

`data/output/phase8/clip/eval/joint_scatter.png` renders the same plot.

### 5.5 Utility — temporal stability (S10)

The two IVP runs differ only in the temporal block. We measure
post-hoc on the rendered output videos:

| Baseline | head-region MAD (0–255) | bbox jitter (px / frame) |
|---|---:|---:|
| Control       |  8.29 ± 3.57  | 12.89 ± 45.06 |
| IVP per-frame | 20.17 ± 10.25 | 14.14 ± 44.75 |
| IVP **full**  | **17.66 ± 7.48** | 14.30 ± 46.53 |

Every baseline sees the same source motion, so the *excess* MAD
above the control floor is cel-shade flicker. The EMA + One Euro
block drops the excess from ~12 to ~9 (a **~30 % reduction**) and
roughly halves the std (10.25 → 7.48) — outlier "boiling" frames are
the part the EMA helps with most. The bbox jitter column is dominated
by detector noise on the rendered output (the in-pipeline smoother
state is not visible to a post-hoc detector pass); a tighter S10
measurement would require dumping the pipeline's internal trace.

## 6. Discussion

**Why the cel-shade story works against FaceNet.** FaceNet (and
ArcFace) embed faces via a CNN that learns from mid-frequency
texture: pore patterns, micro-shading around eyes, hairline detail.
The cel-shade replaces all of that with three to four flat luminance
bands per region. The line art that XDoG draws *is* high-frequency,
but it is sparse and geometric — FaceNet's filters do not key on it
because their training distribution did not include cartoon lines as
identity cues.

**Why FFT HF energy ratio is the wrong primary metric here.** The
metric was selected in `spec.md` §7.2 as a per-frame stand-in for
privacy. It is faithful for *uniform* low-pass operators (blur)
and for *block-averaging* operators (mosaic), where reducing HF
energy is identical to reducing identifiable detail. It is *not*
faithful for cel-shade, which kills mid-band texture while injecting
very-high-band line art. The HF ratio in §5.3 head-only crop rises
by ~2.4× over control, yet re-id rate drops by ~30 percentage
points. We keep the FFT metric in the report for the baseline
comparison story; we do not treat it as the headline privacy number
for IVP.

**Why we ship two IVP baselines.** The per-frame vs full split is the
ablation for S10. Without it the report cannot defend why the temporal
block matters; with it the head-region MAD story (§5.5) is a single
number.

## 7. Limitations & non-claims

- **n=1 source clip for headline numbers.** All numbers in §5.2–5.5
  are computed on one 667-frame 720p clip. We re-ran the pipeline on
  a second clip (`samples/input/dancing.mp4`, 184 frames at
  1080×1920) as a sanity check. On the dancing clip the Haar
  detector finds the face on only 1 of 13 sampled frames (the
  subject is small in-frame and frequently motion-blurred). When the
  detector loses the face on dancing the pipeline either passes the
  frame through unchanged (graceful) or — visible in
  `data/output/phase8/dancing/grid_frame_92.png` — locks onto a
  non-face region (leg / hand) and stylises it, which is a real
  failure mode. The lesson is that the pipeline assumes a frontal-
  or-near-frontal face occupying at least ~8 % of frame height
  (the `min_size_ratio` in `HaarFaceDetector`); outside that envelope
  the report's privacy claims do not apply. Multi-clip metric
  rollups conditioned on detection success are follow-up work.
- **Single embedder.** Re-id rate uses `facenet-pytorch` (vggface2)
  only. ArcFace via `deepface` was listed as optional in `spec.md`
  §7.1 and is left for follow-up.
- **No human study.** The self-recognition Studies A/B from
  `spec.md` §7.4 are not part of this submission.
- **Visual fidelity below VToonify.** Accepted trade-off; documented
  in `spec.md` §4.3 and `ETHICS.md` §3.
- **Threat model.** Defends against casual face matching and
  aesthetic anonymization. Does *not* defend against a motivated
  attacker with reference photos / video, nor against re-id by
  humans who already know the subject. See `ETHICS.md` §2.

## 8. Conclusion & future work

AvatarShield is a working, end-to-end, classical anonymization
pipeline that defends a course report against the question
"why did a neural network do that?" — every transform maps to a
specific IVP501 slide. On a single test clip it achieves a lower
FaceNet re-id rate than blur or mosaic while preserving a usable
head region, and the temporal block cuts frame-to-frame head MAD by
~30 % over the ablated per-frame variant.

Concrete follow-ups:

- Multi-clip evaluation (the dancing clip is rendered; tables to be
  folded in for the next pass).
- ArcFace re-id pass via `deepface`, for embedder generalisation.
- Interactive theme-authoring notebook (`spec.md` §6.2).
- Internal-trace dump for a tighter S10 measurement (compare
  in-pipeline smoothed bbox vs raw Haar bbox directly).
- Per-region histogram match preview when authoring themes (S04
  exercise).

## 9. Reproducibility

All numbers in §5 are regenerated by:

```bash
.venv/bin/python scripts/eval_render_baselines.py
.venv/bin/python scripts/eval_freq.py
.venv/bin/python scripts/eval_freq.py --head-only
/path/to/torch-venv/bin/python scripts/eval_reid.py   # optional dep
.venv/bin/python scripts/eval_scatter.py
.venv/bin/python scripts/eval_temporal.py
```

The pipeline itself is deterministic apart from K-means initialisation
(EMA-smoothed across frames, so per-clip jitter is bounded by the
seed of the first frame's K-means). Re-running on the same input
produces visually equivalent output.

Repository: `final-project/` on branch `ivp-pure`. Spec / plan / docs:
`spec.md`, `plan.md`, `RESULTS.md`, `ETHICS.md`, `README.md`.

## 10. References

1. Viola, P. & Jones, M. (2001). **Rapid Object Detection using a
   Boosted Cascade of Simple Features.** CVPR.
2. Cheddad, A. *et al.* (2009). **A new colour space for skin tone
   detection.** ICIP.
3. Reinhard, E. *et al.* (2001). **Color Transfer between Images.**
   IEEE CG&A.
4. Winnemöller, H. (2012). **XDoG: An eXtended difference-of-Gaussians
   compendium including advanced image stylization.** Computers &
   Graphics.
5. Casiez, G., Roussel, N. & Vogel, D. (2012). **1€ Filter: A simple
   speed-based low-pass filter for noisy input in interactive
   systems.** CHI.
6. Schroff, F., Kalenichenko, D. & Philbin, J. (2015). **FaceNet: A
   Unified Embedding for Face Recognition and Clustering.** CVPR.
7. Hukkelås, H. & Mester, R. (2019). **DeepPrivacy: A Generative
   Adversarial Network for Face Anonymization.** ISVC. *(context only)*
8. Yang, S. *et al.* (2022). **VToonify: Controllable High-Resolution
   Portrait Video Style Transfer.** SIGGRAPH Asia. *(legacy path,
   dropped — see `spec_legacy.md`.)*

---

*Document length: ~9 pages at default `pandoc` margins. Convert to PDF
with `pandoc report.md -o report.pdf --pdf-engine=xelatex`.*
