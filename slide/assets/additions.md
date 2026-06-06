---
marp: true
theme: default
paginate: false
size: 16:9
style: |
  section {
    font-family: 'Inter', 'Helvetica Neue', 'Arial', sans-serif;
    font-weight: 500;
    background: #ffffff;
    color: #0e0e0e;
    padding: 36px 50px;
    -webkit-font-smoothing: antialiased;
    text-rendering: optimizeLegibility;
  }
  h1 {
    color: #0d3a66;
    font-size: 46px;
    font-weight: 800;
    border-bottom: 4px solid #0d3a66;
    padding-bottom: 8px;
    margin-bottom: 20px;
  }
  h2 {
    color: #0d3a66;
    font-size: 30px;
    font-weight: 700;
    margin-top: 10px;
    margin-bottom: 6px;
  }
  table {
    margin: 10px auto;
    font-size: 24px;
    border-collapse: collapse;
  }
  th {
    background: #0d3a66;
    color: #ffffff;
    font-weight: 700;
    padding: 10px 16px;
  }
  td {
    padding: 9px 16px;
    border-bottom: 1px solid #b8b8b8;
    color: #111;
  }
  strong, b {
    color: #0d3a66;
    font-weight: 800;
  }
  .cols {
    display: flex;
    gap: 30px;
  }
  .col {
    flex: 1;
  }
  .callout {
    background: #eaf1f9;
    border-left: 5px solid #0d3a66;
    padding: 8px 16px;
    margin-top: 8px;
    font-size: 19px;
    font-weight: 500;
    color: #0f0f0f;
  }
  ul, ol {
    font-size: 24px;
    line-height: 1.45;
    color: #0f0f0f;
  }
  .small {
    font-size: 20px;
    color: #3a3a3a;
    font-weight: 500;
  }
  img[alt~="center"] {
    display: block;
    margin: 0 auto;
  }
---

# Why Pure-Classical IVP?

<div class="cols">
<div class="col" style="flex:1.1;">

## Neural approaches fall short

- **GAN stylization** — polished but *inexplicable*
- **Full-face replacement** — loses *agency*
- **AR filters** — cosmetic, not formal
- **Black-box** — every pixel un-defendable

## Pure-classical advantages

- **OpenCV + NumPy + Pillow** — 0 neural
- **CPU-only** ~2.7 fps on M1
- **Fully explainable** per pixel
- **Identity reborn** from theme + self-recognition cues

</div>
<div class="col" style="flex:1;">

![h:470 center](before_after.png)

<div class="small" style="text-align:center;">Same frame · cel-shade core + porcelain-pink palette</div>

</div>
</div>

---

# Pipeline Overview

![w:1000 center](slide_pipeline_strip.png)

| Stage | Module |
|---|---|
| Detect | Haar cascade (frontal + profile) |
| Skin mask | HSV ∩ YCbCr + morphology + connected components |
| Cel-shade core | Bilateral × 3 → Lab K-means → XDoG → HSV lift |
| Recolor | Reinhard transfer in Lab, **per region** (hair chroma-only) |
| Stabilize | One Euro on bbox + EMA α=0.85 on K-means centers |
| Composite | Alpha blend ellipse onto original frame |

---

# 5 Theme Palettes × Per-Region Reinhard

![w:820 center](themes_swatch.png)

<div class="cols" style="margin-top:0;">
<div class="col">

**Palette structure** — 5 Lab triples (skin · lips · eyes · brows · hair) + 4 style scalars (luminance levels · edge strength · saturation boost · chroma levels) per theme. Reinhard transfer in **Lab space, per region**.

</div>
<div class="col">

**Hair: chroma-only** — preserves user's L → self-recognition cue intact. Subject sees a *stylized themselves*, not a stranger; strangers see a scrambled identity.

</div>
</div>

---

# Temporal Stability — One Euro + EMA

<div class="cols">
<div class="col" style="flex:1.1;">

![h:380 center](12_temporal_ema.png)
<div class="small" style="text-align:center;">EMA off (left) vs EMA α=0.85 (right) — same source, no boiling</div>

</div>
<div class="col" style="flex:1;">

## Two anti-flicker mechanisms

- **One Euro filter** on bbox `(cx, cy, w, h)` — adaptive low-pass, hold-last-good ≤ 6 frames
- **EMA α=0.85** on the 3 Lab K-means cluster centers — posterize bands locked across frames

## Ablation (head-region MAD, n=667 frames)

| Baseline | MAD |
|---|:---:|
| Control | 8.29 ± 3.57 |
| IVP **OFF** | 20.17 ± 10.25 |
| IVP **ON** | **17.66 ± 7.48** |

→ Excess MAD drops **~30 %**, std **halved**

</div>
</div>

---

# Privacy — FaceNet Re-ID + Detector Survivability

![w:1050 center](slide_baselines_strip.png)
<div class="small" style="text-align:center;">Same frame · control · blur σ=12 · 8×8 mosaic · IVP no-smooth · IVP full</div>

<div class="cols">
<div class="col" style="flex:1.2;">

<div class="small">FaceNet (vggface2), cos d > 0.7, every 15th frame, n=33 on <code>clip.mp4</code></div>

| Baseline | n (paired) | mean dist | re-id rate |
|---|:---:|:---:|:---:|
| Gaussian blur σ=12      |  5 | 0.596 | 0.800 |
| 8×8 mosaic              | 17 | 0.526 | 0.941 |
| **AvatarShield IVP**    | 24 | 0.673 | **0.667** |

</div>
<div class="col">

- **IVP: lowest re-id (0.667) AND highest detector survivability (24/33)**
- Blur "wins" re-id only because detector dies (5/33) — false win
- IVP keeps head locate-able *and* pushes re-id below blur and mosaic

</div>
</div>

---

# Joint Trade-Off — FFT HF Ratio × Re-ID Rate

<div class="cols">
<div class="col" style="flex:1.3;">

![h:480 center](slide_joint_scatter.png)

</div>
<div class="col">

## Reading the scatter

- **Blur** — top-left: maximum texture suppression, but detector dies
- **Mosaic** — middle: blocks both axes moderately
- **Control / IVP** — top-right: XDoG line art **is** HF signal by design

## Why re-id is the headline

- HF ratio alone rewards blur unfairly
- Re-id with detector-survivability count tells the honest story
- IVP wins the joint trade-off without faking either axis

</div>
</div>

---

# Limitations & Non-Claims

<div class="cols">
<div class="col">

## What this does NOT claim

- **n=1 source clip** for headline numbers (dancing clip fails when face < 5 % of frame height — Haar limit)
- **Single embedder** — FaceNet vggface2 only; ArcFace deferred
- **No human study** — self-recognition Studies A/B deferred
- **Threat model** — defends *casual* face matching only

</div>
<div class="col">

## Out of scope

- Motivated attacker with reference photo
- Re-identification by people who already know the subject
- Real-time / on-device deployment (2.7 fps CPU)
- Production privacy guarantee (research prototype)

</div>
</div>

<div class="callout">
AvatarShield ≠ K-anonymity guarantee. It is <b>visual-story-preserving anonymization</b> for child-safe social sharing, defended by transparent classical IVP operators — not by black-box trust.
</div>
