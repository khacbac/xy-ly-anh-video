# AvatarShield — Project Specification

> **Course:** IVP501 — Image and Video Processing (FSB, MSE)
> **Working title:** AvatarShield: Child-Safe Video Anonymization via Classical Cel-Shade Stylization with Theme Palette
> **Team size:** 3 members
> **Spec version:** v1.0 — pure-IVP501 path (replaces v0.5)
> **Branch:** `ivp-pure`
> **Supersedes:** `spec_legacy.md` (MediaPipe + avatar PNG + VToonify path — kept for historical context only)

-----

## 0. What changed from v0.5

| Aspect | Legacy (v0.5) | This spec (v1.0) |
|---|---|---|
| Face detection | MediaPipe Face Landmarker (468 pts) | OpenCV Haar cascade (frontal + profile) |
| Head segmentation | BiSeNet face parsing | Skin range (HSV+YCbCr) + morphology + face oval |
| Stylization | VToonify GAN (per preset checkpoint) | Cel-shade pipeline (bilateral + Lab K-means + XDoG) |
| Identity / look source | Avatar PNG (VRoid export) | Theme palette JSON (Lab stats per region) |
| Compositing | Similarity warp + Poisson + alpha | Feathered alpha onto cel-shaded head region |
| Heavy deps | `mediapipe`, `torch`, `dlib`, BiSeNet ckpt, VToonify ckpts | `opencv-python`, `numpy`, `Pillow` only |
| Runtime on M1 (clip ~10 s) | ~minutes (async job) | ~seconds (synchronous) |
| Pedagogical mapping | Partial (most heavy lifting inside neural nets) | Direct 1-1 to slides S02–S10 |

**Driving rationale.** Pin the implementation to the IVP501 syllabus so every transform in the pipeline maps to a specific slide. Goal: students can explain *every pixel* end-to-end, and the report defends the pipeline as a course artifact rather than a thin wrapper around third-party models.

-----

## 1. Đề tài tóm lược

Hệ thống xử lý video tự động che danh tính khuôn mặt người dùng (target: trẻ em) bằng cách **stylize vùng đầu** thành phong cách anime cel-shaded, dưới một *theme palette* mà user chọn. Theme palette là một file JSON chứa thống kê Lab cho da / tóc / môi / mắt, không phải ảnh PNG. Pipeline chỉ dùng các phép biến đổi cổ điển trong khóa IVP501 (lọc không gian / tần số, K-means quantization, color-space transform, edge detection, morphology, geometric warp, temporal filtering).

**Phân biệt rõ:**

- KHÔNG phải face filter Snapchat-style (overlay không xóa identity).
- KHÔNG phải deepfake / face-swap (không impersonate người khác).
- KHÔNG phải GAN stylization (không VToonify / AnimeGAN / DCT-Net).
- LÀ identity-preserving anonymization qua classical IVP, identity được "ghi" bằng theme palette + self-recognition cues.

-----

## 2. Motivation

Trẻ em xuất hiện trên social media đối mặt với rủi ro:

- Biometric data collection cho face recognition training.
- Cross-platform face matching và doxxing.
- CSAM-adjacent harm.

Các approach hiện tại không đủ:

- **Blur / mosaic:** phá vỡ communicative value của video.
- **Full face replacement (DeepPrivacy, etc.):** xóa cảm giác *agency* và *self-identity*.
- **Commercial face filters (Snapchat, TikTok):** thẩm mỹ, không formal anonymization guarantee.
- **GAN stylization (VToonify):** chất lượng visual cao, nhưng không giải thích được bản chất bên trong → không phù hợp report theo khóa IVP501.

**Insight cốt lõi.** Với child viewer, identity không cần đến từ facial geometry. Identity có thể đến từ:

1. *Theme palette* user chọn (consistent look across clips).
2. *Self-recognition cues* preserved bằng classical color transfer: relative luminance, expression idiosyncrasy, head motion.
3. *Anonymization* đạt được bằng cách collapse high-frequency face texture qua bilateral + posterize — provable bằng FFT spectrum (S05.02).

-----

## 3. Problem Formulation

**Input:** Video chứa khuôn mặt user + tên theme (e.g. `porcelain-pink`).
**Output:** Video với vùng đầu được cel-shade + recolor theo theme, satisfying:

| Constraint | Yêu cầu | Đo lường | Slide |
|---|---|---|---|
| Privacy — FR re-id | Face recognition không match output với original | Re-id rate (FaceNet / ArcFace) | S08 |
| Privacy — frequency | High-frequency face texture bị suppress | Ratio of HF FFT energy (cel-shaded / original) ≤ 0.5 | S05 |
| Utility — pose | Head pose preserved | Bbox center jitter (One Euro smoothed); rotation roughly | S04, S10 |
| Self-recognition | User tự nhận ra "đó là tôi" | Self-rating 5-point + blind A/B | — |
| Temporal stability | Không "boiling" / flicker | EMA-stability metric over K-means centers | S10 |

-----

## 4. Scope

### 4.1 In scope

- **Face detection:** OpenCV `cv2.CascadeClassifier` (Haar frontal + profile), single largest face per frame. **S08** feature-based detection.
- **Bbox temporal smoothing:** One Euro filter applied to `(cx, cy, w, h)` per frame; hold-last-good for ≤ 6 missed frames. **S10**.
- **Skin segmentation:** HSV range ∩ YCbCr range, then morphological close + open, then connected components → keep largest blob inside the Haar bbox. **S06, S07**.
- **Head region mask:** geometric ellipse extending the Haar bbox upward (hair) + sides (ears), feathered. **S06**.
- **Cel-shade core:** iterated bilateral filter on a downsampled copy → Lab color-space split → K-means quantize on L (and optionally a,b) → XDoG line art overlay → HSV saturation lift. Temporal EMA on K-means centers per video. **S05, S06, S07**.
- **Theme palette recolor:** Reinhard color transfer in Lab between (frame skin / hair / lips / eyes) and the theme JSON's stored stats. Chroma-only on hair (preserve user's hair shading). **S07**.
- **Compositing:** feathered alpha blend of stylized head region onto the original frame. **S05** (Gaussian feather), **S04** (geometry).
- **Video I/O:** OpenCV `VideoCapture` / `VideoWriter`, MP4 (H.264) input/output. **S09**.
- **Evaluation:**
  - FFT high-frequency energy ratio per frame (**S05.02**).
  - FaceNet / ArcFace re-id on sampled frames (**S08**).
  - Baselines: no processing, Gaussian blur (**S05**), 8×8 mosaic (**S02**), AvatarShield-IVP per-frame vs smoothed.
- **Demo:** CLI + notebook. Optional: minimal FastAPI `/ivp/render` endpoint and an Expo screen wired to it (reuses scaffolding from the legacy app).

### 4.2 Out of scope

- MediaPipe / dlib / any 68- or 468-point landmark inference.
- Avatar PNG warp / paste; VRoid pipeline; mesh / piecewise-affine warps.
- BiSeNet face parsing; any pixel-level neural segmentation.
- VToonify / AnimeGAN / DCT-Net / pSp encoder; any GAN stylization.
- Reversible anonymization; production deployment.
- Voice anonymization, body identity anonymization.
- Real-time mobile inference; on-device acceleration.
- Context-based identification defense (background, social graph, voice).
- User studies with actual minors.

### 4.3 Explicit non-claims

- KHÔNG claim production-ready hay commercial-grade.
- KHÔNG claim chống motivated attacker với reference photos.
- KHÔNG claim self-recognition cho stranger.
- KHÔNG claim visual fidelity ngang một GAN stylization (VToonify) — chấp nhận trade visual quality cho explainability + zero heavy deps.

### 4.4 Course knowledge ↔ implementation mapping (IVP501)

*Thứ tự học gợi ý: S02–S03 → S04 → S05 → S06 → S07 → S08 → S09–S10.*

| Knowledge area | Course material | Project module / feature |
|---|---|---|
| Raster image / video, pixel ops | S01–S03 | `video_io.py`, frame iteration |
| Pixel neighborhoods, kernels | S02, S05 | Bilateral / Gaussian / median in cel_shade |
| Geometric transforms | S04 | Haar bbox → head ellipse + feathered ROI |
| Histograms & tonal adjustment | S04 | Per-region histogram preview when authoring themes |
| Spatial filtering (conv, bilateral) | S05 | `cel_shade.iterated_bilateral` |
| Frequency-domain filtering (FFT) | S05.02 | Evaluation notebook: HF energy ratio |
| Morphology (open, close, dilate, erode) | S06 | Skin mask cleanup, hair-ring refinement |
| Edge detection (Canny, DoG, XDoG) | S06 | `cel_shade.xdog` and `face_aware_line_art` |
| Color spaces (RGB / Lab / HSV / YCbCr) | S07 | Skin segmentation, palette transfer, saturation lift |
| Color quantization (K-means) | S07 | `cel_shade.kmeans_quantize_channel` |
| Color transfer (Reinhard) | S07 | `palette.apply_theme` Lab transfer per region |
| Features & feature-based detection | S08 | Haar cascade frontal + profile |
| Embedding-based privacy eval | S08 | FaceNet / ArcFace re-id rate |
| Compression (MP4 / H.264) | S08 | OpenCV / ffmpeg encode quality |
| Video temporal perception | S09 | Motivation for bbox + center EMA |
| Temporal filtering | S10 | One Euro filter on bbox; EMA on K-means centers |

Mỗi module trong `avatarshield/` (v1.0) phải có docstring chỉ rõ slide tham chiếu.

-----

## 5. Architecture

```
[Source MP4]
   └─► video_io ─► frame ─┐
                          │
   ┌────────── detect (Haar frontal + profile) ──────────┐
   │     ▼                                                │
   │  bbox (x,y,w,h) ─► One Euro smoother ─► bbox_s       │ (S08, S10)
   │     │                                                │
   │     ▼                                                │
   │  skin mask (HSV ∩ YCbCr) ─► morph close/open ─► CC   │ (S06, S07)
   │     │                                                │
   │     ▼                                                │
   │  head ellipse ROI + feather                          │ (S04, S05)
   └─────────────────────────────────────────────────────┘
                          │
   ┌─ cel_shade head crop ─┐                              (S05, S06, S07)
   │   ▪ iter. bilateral   │
   │   ▪ Lab posterize     │  ← TemporalState (EMA centers, per video)  (S10)
   │   ▪ XDoG line art     │
   │   ▪ saturation lift   │
   └───────────────────────┘
                          │
   ┌─ palette.apply ───────┐  (S07)
   │   per-region Lab      │  theme JSON: {skin, hair, lips, eyes, brows}
   │   Reinhard transfer   │
   └───────────────────────┘
                          │
   ┌─ composite ──────────┐  (S05 feather, S04 affine)
   │   alpha = feathered  │
   │   head_mask          │
   └──────────────────────┘
                          │
                          ▼
                    [Output MP4]
```

### 5.1 Public API (target)

```python
# avatarshield/render.py

@dataclass
class RenderConfig:
    theme: str = "porcelain-pink"           # name → themes/<name>.json
    # detect
    haar_scale_factor: float = 1.1
    haar_min_neighbors: int = 4
    bbox_hold_frames: int = 6
    # smoothing
    smooth_min_cutoff: float = 1.0
    smooth_beta: float = 0.02
    # head ellipse extension (relative to bbox)
    head_top_extend: float = 0.6
    head_side_extend: float = 0.20
    head_bottom_extend: float = 0.05
    head_feather_px: int = 41
    # cel-shade (forwarded to cel_shade.anime_head_pass)
    luminance_levels: int = 3
    chroma_levels: int = 0
    bilateral_passes: int = 3
    saturation_boost: float = 1.30
    edge_strength: float = 0.85
    # palette transfer strengths
    palette_strength_skin: float = 0.55
    palette_strength_features: float = 0.75
    palette_strength_hair: float = 0.55     # chroma-only

@dataclass
class RenderResult:
    output_path: str
    frame_count: int
    fps: float
    duration_s: float
    misses: int
    avg_hf_ratio: float | None              # optional eval pass-through

def render_video(
    video_path: str | Path,
    output_path: str | Path,
    config: RenderConfig | None = None,
    *,
    progress: bool = True,
) -> RenderResult: ...

def process_frame(
    frame_bgr: np.ndarray,
    state: "RenderState",
    config: RenderConfig,
) -> np.ndarray: ...
```

Note: signature drops `avatar_path`. Theme is selected by name and resolved against `assets/themes/`.

### 5.2 Module breakdown

| Module | File | Responsibility | Status |
|---|---|---|---|
| M1 I/O | `video_io.py` | Decode / encode MP4 | keep (no change) |
| M2 Detect | `detect.py` | Haar cascades + skin mask + head ellipse mask | **new** |
| M3 Smooth | `smooth.py` | One Euro on bbox (cx, cy, w, h) | **rewrite** (simpler than landmark version) |
| M4 Cel-shade | `cel_shade.py` | Bilateral + Lab posterize + XDoG + region boosters | keep as-is |
| M5 Palette | `palette.py` | Load theme JSON, Reinhard recolor per region | **new** |
| M6 Composite | `composite.py` | Feathered alpha blend | keep, slim |
| M7 Entrypoint | `render.py` | `render_video()`, `process_frame()`, CLI | **rewrite** |
| M8 Evaluation | `notebooks/` + `scripts/eval_*.py` | FFT HF energy, re-id, baselines | **new** |
| M9 Themes | `assets/themes/*.json` | 5 hand-authored theme palettes | **new** |

### 5.3 Repository layout (target)

```
final-project/
├── avatarshield/
│   ├── __init__.py
│   ├── render.py
│   ├── video_io.py
│   ├── detect.py
│   ├── smooth.py
│   ├── cel_shade.py
│   ├── palette.py
│   └── composite.py
├── assets/
│   └── themes/
│       ├── porcelain-pink.json
│       ├── tan-amber.json
│       ├── ivory-violet.json
│       ├── bronze-teal.json
│       └── peach-noir.json
├── samples/input/            # local test media (gitignored)
├── data/output/              # renders (gitignored)
├── notebooks/                # eval (HF ratio, re-id, baselines)
├── scripts/                  # render / eval / theme-author helpers
├── requirements.txt          # opencv-python, numpy, Pillow (+ scipy for eval)
├── spec.md                   # this file
├── spec_legacy.md            # historical
├── plan.md
├── plan_legacy.md            # historical
└── README.md
```

### 5.4 Module deletion checklist (legacy → drop)

Files to be removed in Phase 7 of `plan.md`:

- `avatarshield/landmarks.py` — MediaPipe
- `avatarshield/parsing.py` — BiSeNet
- `avatarshield/morph.py` — mesh warp on landmarks
- `avatarshield/align.py` — similarity transform on 468-point sets
- `avatarshield/anime.py` — wraps `cv2.stylization`; small useful bits migrate into `cel_shade.py` or are simply dropped
- `avatarshield/stylize.py` — VToonify path
- `avatarshield/mask.py` — landmark/oval polygon masks (functionality folded into `detect.py` head ellipse)
- `third_party/` — VToonify, dlib, BiSeNet sources
- `checkpoints/` — all model weights
- `scripts/animegan_*.py`, `scripts/vtoonify_*.py`, `scripts/combo_render_video.py`
- `api/server.py` stylize routes (or full API) — see Open Decisions §13

`scripts/cel_shade_render.py` and `scripts/cel_shade_test.py` are kept (they were already exercising the cel-shade core) and renamed / merged into `scripts/render_clip.py`.

-----

## 6. Theme palette

### 6.1 Format

```json
{
  "name": "porcelain-pink",
  "description": "Bright porcelain skin, warm undertone, magenta hair, hazel eyes.",
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

- Lab triples follow OpenCV's `cv2.COLOR_BGR2LAB` convention: `L ∈ [0, 255]`, `a, b` centered at 128.
- `hair_lab` only needs `(a, b)` accuracy — `L` is informative but ignored at recolor time (chroma-only transfer).
- The three style scalars (`saturation_boost`, `edge_strength`, `luminance_levels`) override the matching defaults in `RenderConfig` if present in the JSON, so a theme can ship its own look (e.g. higher-contrast cel = lower `luminance_levels`).

### 6.2 Authoring workflow

`notebooks/author_theme.ipynb`:

1. Load a reference image (any source — VRoid render, illustration, etc.).
2. Manually mask regions (polygons / brush) **or** let user click sample points.
3. Compute mean Lab per region.
4. Preview cel-shade + Reinhard on a sample face.
5. Tune `saturation_boost`, `edge_strength`, `luminance_levels` interactively.
6. Save to `assets/themes/<name>.json`.

### 6.3 Starter set (target)

| Theme | Skin tone | Hair | Eyes | Style tag |
|---|---|---|---|---|
| porcelain-pink | Bright, warm | Magenta-black | Hazel | High-key shoujo |
| tan-amber | Warm, deeper | Brown-amber | Honey | Studio Ghibli mid |
| ivory-violet | Cool ivory | Violet | Pale blue | Fantasy |
| bronze-teal | Bronze | Black-teal sheen | Teal | Sporty |
| peach-noir | Peach-neutral | Inky black | Deep brown | Noir / monochromatic |

Each theme JSON is < 1 KB; replaces the 5 PNG avatars of the legacy spec.

### 6.4 Customization (light, stretch)

User can override `skin_lab.L` to preserve their own brightness (self-recognition cue) while still taking the theme's chroma — single CLI flag `--keep-user-L`.

-----

## 7. Evaluation framework

### 7.1 Privacy — face recognition

- FaceNet (`facenet-pytorch`) and ArcFace via `deepface` (optional, Colab GPU if local CPU too slow).
- Re-identification rate on sampled frames (every 15th frame): top-1 match against `{original, distractors}`.
- Targets (aspirational): re-id rate < 5 %; embedding distance > 0.7.

### 7.2 Privacy — frequency-domain (NEW, from S05.02)

For each sampled frame `f`:

```
F(u,v)    = fftshift(fft2(L(f)))           # luminance only
HF energy = Σ |F(u,v)|²   over { (u,v) : √(u²+v²) > r₀ }
Total     = Σ |F(u,v)|²
HF ratio  = HF energy / Total
```

`r₀` = 0.20 × min(H, W) (Nyquist fraction). Report:

- `HF ratio` (mean ± std) — original vs cel-shaded vs blur baseline.
- Expectation: cel-shade ≥ 50 % reduction vs original; Gaussian blur σ=12 also reduces but at the cost of completely flattening the face. Cel-shade keeps low-mid frequency contour (utility) while suppressing high-frequency texture (privacy).

This is the eval-side payoff of choosing classical methods: students implement S05.02's exact FFT band code and use it as a privacy metric.

### 7.3 Utility

- Bbox center jitter (px per frame) with vs without One Euro.
- Optical-flow magnitude inside head ROI (sanity: shouldn't explode after stylization).

### 7.4 Self-recognition

- **Study A:** team self-rating (Likert 5).
- **Study B:** blind A/B with adult volunteers (optional, n ≥ 10).

### 7.5 Baselines

1. No processing (control)
2. Gaussian blur σ=12 (**S05**)
3. 8 × 8 pixel mosaic (**S02**)
4. AvatarShield-IVP — per-frame (no temporal smoothing)
5. AvatarShield-IVP — full (One Euro + EMA centers)

All five renderable from the same `scripts/eval_render_baselines.py` driver.

-----

## 8. Tech stack

### 8.1 Core runtime

Python **3.10+**. `requirements.txt`:

```
opencv-python>=4.9
numpy>=1.26
Pillow>=10.2
```

That's the entire runtime. No `torch`, no `mediapipe`, no `dlib`, no `modelscope`.

### 8.2 Optional (evaluation only)

Installed ad-hoc for notebooks / scripts in `notebooks/` and `scripts/eval_*`:

- `matplotlib`, `jupyter`
- `scipy` (FFT-2D convenience, though `numpy.fft` is sufficient)
- `deepface` and / or `facenet-pytorch` for re-id metric (usually run on Colab)
- `ffmpeg` (system) for any side-by-side compositing in the report

### 8.3 Design & ops tools

- **Git** — version control. Branch `ivp-pure` for v1.0.
- Theme authoring: any image editor + the `author_theme.ipynb` notebook.
- Demo media: `samples/input/` (gitignored).

-----

## 9. Team roles

| Role | Primary | Secondary |
|---|---|---|
| **R1 — Detection + Smoothing** | `detect.py`, `smooth.py`, integration into `render.py` | Bbox visual QA |
| **R2 — Stylization + Themes** | Tune `cel_shade.py` params, `palette.py`, 5 theme JSONs | `author_theme.ipynb` |
| **R3 — Evaluation + Report** | FFT notebook, re-id notebook, baseline scripts, written report | User study (Study A/B) |

Sync points: end of Phase 3 (palette recolor working), end of Phase 5 (full pipeline), end of Phase 8 (eval).

-----

## 10. Deliverables

### 10.1 Course

- Proposal, presentation slides, Q&A note, written report (≤ 10 pages), code repository.

### 10.2 Repository artifacts

- `avatarshield/` — pipeline (M1–M7)
- `assets/themes/` — 5 JSON themes
- `samples/input/` — test media (local, consent documented; not committed)
- `data/output/` — renders (gitignored)
- `notebooks/eval_freq.ipynb` — FFT HF ratio
- `notebooks/eval_reid.ipynb` — FaceNet / ArcFace
- `notebooks/author_theme.ipynb` — theme authoring
- `report.pdf`, `slides.pptx`, `evaluation_results.csv`

### 10.3 Documentation

- `README.md` — setup, CLI, disclaimers, theme catalog
- `ETHICS.md` — safety framing, consent, IP (carried over)
- `RESULTS.md` — eval summary (to author after Phase 8)
- `plan.md` — phased implementation checklist
- `spec_legacy.md`, `plan_legacy.md` — kept for context

-----

## 11. Risk register

| ID | Risk | Mitigation |
|---|---|---|
| R1 | Haar miss-rate on profile / occluded faces | Profile cascade fallback; `bbox_hold_frames=6` to coast through misses; fall through to passthrough frame on extended miss |
| R2 | Cel-shade "boiling" / flicker frame-to-frame | Temporal EMA on K-means centers (already in `cel_shade.TemporalState`); One Euro on bbox; optional 2-frame median on output |
| R3 | Visual quality below VToonify baseline | Acceptable for course — report frames the trade-off pedagogically; theme parameters tuned per-clip |
| R4 | Skin tone collapse under heavy palette snap | Cap `palette_strength_skin ≤ 0.65`; keep user's L when `--keep-user-L` |
| R5 | FFT eval inconclusive / overlapping with blur baseline | Pair with re-id metric; report joint scatter (HF ratio × re-id) — cel-shade should sit at low re-id with mid HF ratio, blur at low re-id with very low HF ratio + visible utility loss |
| R6 | Skin segmentation fails on dark skin under low light | Document tested skin-tone range; add `--skin-range tight/wide` flag; allow fully geometric mask fallback (face ellipse only) |
| R7 | Theme JSON authoring is fiddly | Provide `author_theme.ipynb` + 5 ready-to-use starters; bound the JSON schema with a small validator in `palette.py` |
| R8 | Member unavailable | Cross-train across M2 / M5; cel_shade already exists so tuning is parallelizable |
| R9 | Live demo failure | Pre-record backup clip per theme |

Removed vs legacy: VToonify checkpoint download, GPU availability, BiSeNet ckpt licensing, MediaPipe model file integrity, DeepPrivacy fallback complexity.

-----

## 12. Ethical considerations

- No minor participants in studies; adults only.
- Research prototype — not a deployable child-safety product.
- Theme palettes are numeric (Lab triples) — no third-party character IP involved.
- Threat model and non-claims documented in `README.md` / `ETHICS.md`.
- MIT (or equivalent) code license.
- Dual-use disclosure: classical anonymization cannot impersonate identity (no avatar PNG, no GAN) → reduced misuse surface vs the legacy path.

-----

## 13. Open decisions

- [ ] API surface — keep FastAPI `/ivp/render` endpoint for the existing Expo app, or revert to CLI + notebook only? Default: minimal `/ivp/render` (single endpoint, synchronous response), Expo screen simplified to "pick theme + render".
- [ ] `--keep-user-L` self-recognition slot: ship on by default, or as opt-in flag?
- [ ] Profile-face detection: extend Haar with `haarcascade_profileface.xml`, or accept that side views fall through to passthrough?
- [ ] Theme authoring: ship the notebook from day one (Phase 6) or after Phase 5 once the pipeline is visually stable?
- [ ] Re-id metric provider: `facenet-pytorch` (lightweight, CPU OK) vs `deepface` (heavier, multi-backbone) — recommend `facenet-pytorch` for portability.

-----

## 14. References (working bibliography)

**Core methods (pure-IVP):**

- Viola & Jones (2001) — Rapid Object Detection using a Boosted Cascade of Simple Features (Haar).
- Reinhard et al. (2001) — Color Transfer between Images.
- Winnemöller et al. (2012) — XDoG: An eXtended difference-of-Gaussians compendium.
- Pérez et al. (2003) — Poisson Image Editing (referenced but unused in v1.0).
- Casiez et al. (2012) — 1 € filter.

**Privacy / context:**

- DeepPrivacy (2019), DeepPrivacy2 (2023) — for related-work comparison only.
- Proteus Effect (Yee & Bailenson, 2007) — self-recognition framing.
- UNICEF digital safety guidance (to add).

**Removed vs legacy:** VToonify (2022), AnimeGANv2, DCT-Net, BiSeNet face parsing, MediaPipe Face Landmarker — no longer in scope.

-----

**Spec status:** v1.0 — pure-IVP501 path. Implementation tracked in `plan.md` on branch `ivp-pure`. Legacy spec preserved as `spec_legacy.md`.
