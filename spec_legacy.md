# AvatarShield — Project Specification (LEGACY)

> ⚠️ **LEGACY — superseded by `spec.md` v1.0 on branch `ivp-pure`.**
> This document describes the previous pipeline: MediaPipe Face Landmarker (468 points) + avatar PNG warp + BiSeNet parsing + VToonify head-region GAN stylization. It is kept for historical context only — do not implement against it.
> The active spec moves to a pure-IVP501 path (Haar cascade + skin segmentation + classical cel-shade + theme palette JSON), removing every neural-network dependency except a single Haar cascade. See `spec.md` and `plan.md` for the live design.
>
> **Course:** IVP501 — Image and Video Processing (FSB, MSE)
> **Working title:** AvatarShield: Child-Safe Video Anonymization via Stylistic Avatar Replacement with Self-Recognition Cues
> **Team size:** 3 members
> **Spec version:** v0.5 — render pipeline only (video + avatar image → output video)
> **Implementation status:** docs + samples; Python package not yet scaffolded — see `plan.md`.

-----

## 1. Đề tài tóm lược

Hệ thống xử lý video tự động thay thế khuôn mặt người dùng (target: trẻ em) bằng avatar anime stylized do user chọn từ gallery curated. Mục tiêu: đảm bảo biometric identity không bị thu thập hay match qua face recognition AI, đồng thời giữ lại các *self-recognition cues* (eye color, skin tone, expression idiosyncrasy) để user tự nhận ra avatar là phiên bản của mình.

**Phân biệt rõ:**

- KHÔNG phải face filter Snapchat-style (overlay không xóa identity).
- KHÔNG phải deepfake/face-swap (không impersonate người khác).
- LÀ identity-preserving anonymization với stylistic avatar replacement.

-----

## 2. Motivation

Trẻ em xuất hiện trên social media đối mặt với rủi ro:

- Biometric data collection cho face recognition training.
- Cross-platform face matching và doxxing.
- CSAM-adjacent harm.

Các approach hiện tại không đủ:

- **Blur/mosaic:** phá vỡ communicative value của video.
- **Full face replacement (DeepPrivacy, etc.):** xóa cảm giác *agency* và *self-identity* của user trong content của chính mình.
- **Commercial face filters (Snapchat, TikTok):** thiên về thẩm mỹ, không có formal anonymization guarantee.

**Insight cốt lõi:** Đối với child viewer, identity không cần đến từ facial geometry. Identity có thể đến từ:

1. *Persistent stylistic avatar* user chọn.
2. *Expression mapping* (stretch): avatar phản ứng theo user qua blendshapes.
3. *Self-recognition cues*: thuộc tính (eye color, skin tone) mà user nhận ra là của mình, nhưng không đủ cho face recognition AI match.

-----

## 3. Problem Formulation

**Input:** Video chứa khuôn mặt user + ảnh avatar reference (PNG).
**Output:** Video với face region được thay thế bằng avatar stylized, satisfying:

| Constraint | Yêu cầu | Đo lường |
|------------|---------|----------|
| Privacy | Face recognition không match output với original | Re-id rate (FaceNet / ArcFace) |
| Utility — expression | Avatar phản ứng biểu cảm user (stretch) | Blendshape MSE |
| Utility — pose | Head pose preserved | Rotation error (degrees) |
| Self-recognition | User tự nhận ra avatar là mình | Self-rating 5-point + blind A/B |
| Coherence (stretch) | Body/background đồng nhất với avatar head | Visual quality survey |

-----

## 4. Scope

### 4.1 In scope

- **Face region anonymization** với avatar PNG (thiết kế nguồn VRoid Studio, export 2D cho pipeline).
- **Avatar gallery:** 5 PNG curated, neutral pose, license sạch.
- **MediaPipe Face Landmarker** — landmarks + blendshapes per frame.
- **Geometric pipeline:** similarity transform, scale matching (`align.py`).
- **Mask + color:** face oval mask, Gaussian feather, histogram / mean color match.
- **Compositing:** alpha blend (+ optional Poisson).
- **Temporal smoothing:** Kalman / One Euro trên landmark tracks (`smooth.py`).
- **Stretch (implemented):** whole-frame cartoonization via AnimeGANv2 paprika + optional VToonify pixar face boost — see `scripts/combo_render_video.py`. Out-of-band of the avatar-PNG pipeline.
- **Demo:** CLI + notebook (Gradio optional).
- **Evaluation:** privacy, utility, self-recognition (+ classical blur baseline).

### 4.2 Out of scope

- Mobile app, backend API, job queue, cloud upload (Boogiz-style orchestration).
- Real-time headless Three.js / VRM render **per frame** (MVP dùng PNG + warp).
- Voice anonymization, reversible anonymization, production deployment.
- Body identity anonymization beyond cartoon stylization (stretch).
- Context-based identification (background, social graph, voice).
- User study with actual minors (adult proxy participants only).
- Motivated attacker với reference photos.

### 4.3 Explicit non-claims

- KHÔNG claim self-recognition cho stranger nhận ra user.
- KHÔNG claim chống bạn bè/người quen nhận ra qua context.
- KHÔNG claim production-ready hay commercial-grade.

### 4.4 Course knowledge ↔ implementation mapping (IVP501)

*Bảng dưới map slide/khóa IVP501 (`S00`–`S10`, syllabus) sang module trong repo. Thứ tự học gợi ý: S04 → S07 → S06 → S09–S10 → S08 → S05.*

| Knowledge area | Course material | Project module / feature |
|----------------|-----------------|-------------------------|
| Raster image & video | S01–S03 | `video_io.py`, frame pipeline, face ROI |
| Pixel neighborhoods | S02 | Mask feather, skin region |
| Geometric transforms | S04 | `align.py` — similarity warp, scale |
| Histograms & tonal adjustment | S04 | `color.py` — lighting on skin region |
| Convolution & filtering | S05 | Gaussian blur **baseline** in eval |
| Morphological ops | S06 | `mask.py` — clean holes, dilate/erode optional |
| Segmentation | S06 | Face oval mask from landmark indices |
| Color spaces | S07 | BGR workflow; skin / scene harmony |
| Features & embeddings | S08 | Landmarks; FaceNet/ArcFace in eval notebooks |
| Compression | S08 | MP4 encode quality (OpenCV / ffmpeg) |
| Video temporal perception | S09 | Motivation for `smooth.py` |
| Temporal filtering | S10 | Kalman / One Euro in `smooth.py` |

-----

## 5. Architecture (render pipeline only)

**Core contracts.** Two render paths share a job folder layout under
`data/api_jobs/<job_id>/`:

```
# Avatar-PNG (synchronous)
render_ai_video(source_video, avatar_image) → output_video

# Head-region stylization (async via API)
render_stylized_video(source_video, config) → output_video
```

```
Avatar-PNG path:
[Source video MP4] → video_io → landmarks → smooth → align → mask + color →
                     composite (alpha + Poisson) → video_io → [Output MP4]

Stylize path (avatarshield.stylize):
[Source video MP4] → dlib head crop (EMA bbox) → VToonify (5 preset ckpts) →
                     BiSeNet parsing mask (head classes) → Reinhard color
                     match → composite + temporal blend → [Output MP4]
```

Both are exposed by the same FastAPI server (`api/server.py`). Stylize
renders are heavy (~minutes on M1) and run as in-memory async jobs the
client polls via `GET /stylize/render/{job_id}/status`.

Optional **external backend** (`backend="external"`) cùng signature
`render_ai_video()` để so sánh chất lượng — không bắt buộc MVP.

### 5.1 Module breakdown

| Module | File | Responsibility |
|--------|------|----------------|
| M1 Preprocessing | `video_io.py`, `landmarks.py` | Decode/encode MP4; Face Landmarker |
| M2 Assets | `assets/avatars/*.png` | 5 curated avatar PNG (VRoid source) |
| M3 Compositing | `align.py`, `mask.py`, `color.py`, `composite.py` | Warp, mask, color, blend |
| M4 Temporal | `smooth.py` | Landmark smoothing |
| M5 Entrypoint | `render.py` | `render_ai_video()`, `process_frame()`, CLI |
| M5b Stylization | `stylize.py` | `render_stylized_preview()` / `render_stylized_video()` — head-region VToonify + BiSeNet parsing mask. Wired into the API as the primary user filter. |
| M6 Evaluation | `notebooks/`, optional `scripts/eval_*.py` | Privacy, utility, baselines |
| M-debug | `scripts/animegan_*.py`, `scripts/combo_render_video.py`, `scripts/vtoonify_*.py` | Offline sweep / POC scripts. Kept for tuning; not on the API path. |

### 5.2 Repository layout

```
final-project/
├── avatarshield/          # M1, M3–M5
├── assets/avatars/        # M2 — *.png
├── samples/input/         # local test media (gitignored blobs)
├── data/output/           # renders (gitignored)
├── notebooks/             # M6
├── scripts/               # thin CLI helpers (optional)
├── requirements.txt       # core runtime
├── plan.md
├── spec.md
└── README.md
```

-----

## 6. Avatar gallery

### 6.1 Source

5 avatars **tự thiết kế VRoid Studio** → export **PNG** frontal/neutral cho pipeline. License: team-owned; ghi trong report.

### 6.2 Runtime assets (MVP)

```
assets/avatars/
├── 01.png
├── 02.png
├── …
├── 05.png
└── README.md    # naming, VRoid export settings, optional meta
```

Optional (không chặn render): `meta.json` per avatar, VRM source archived ngoài repo hoặc Drive.

### 6.3 Diversity matrix (target)

| Avatar | Gender presentation | Hair | Eyes | Style tag |
|--------|---------------------|------|------|-----------|
| A1 | Femme | Black | Brown | Realistic-anime |
| A2 | Masc | Brown | Blue | Realistic-anime |
| A3 | Neutral | Pink | Green | Stylized-cute |
| A4 | Femme | Silver | Purple | Fantasy |
| A5 | Masc | Blonde | Hazel | Sporty/casual |

### 6.4 Customization (light, stretch)

Preset eye color / skin tone trên PNG hoặc post-color trong `color.py` — không bắt buộc MVP.

-----

## 7. Evaluation framework

### 7.1 Privacy

- Re-identification rate: FaceNet / ArcFace trên frame sample vs original + distractors.
- Targets (aspirational): re-id rate < 5%; embedding distance > 0.7.

### 7.2 Utility

- Blendshape MSE, head pose error, landmark jitter (with vs without `smooth.py`).

### 7.3 Self-recognition

- **Study A:** team self-rating (Likert 5-point).
- **Study B:** blind A/B với adult volunteers (optional, n≥10).

### 7.4 Baselines

1. No processing (control)
2. Gaussian blur (S05 classical)
3. Black box overlay
4. DeepPrivacy2 (SOTA, Colab GPU if needed)
5. AvatarShield — per-frame vs smoothed

-----

## 8. Tech stack

### 8.1 Core runtime (required)

Python **3.10+**. Dependencies in `requirements.txt`:

```
opencv-python>=4.9
mediapipe>=0.10
numpy>=1.26
Pillow>=10.2
```

### 8.2 Optional (evaluation & stretch)

Install khi làm M6 / stretch — không nằm trong `requirements.txt` core:

- `matplotlib`, `jupyter` — notebooks
- `deepface` hoặc `face-recognition` — re-id metrics
- `torch`, `scikit-image` — DeepPrivacy2 / AnimeGAN baselines (thường chạy Colab)
- `gradio` — optional web demo

### 8.3 Design & ops tools

- **VRoid Studio** — avatar design → PNG export
- **Git** — version control
- **Colab / Kaggle** — GPU cho SOTA baseline và heavy eval (free tier)

-----

## 9. Team roles

| Role | Primary | Secondary |
|------|---------|-----------|
| **R1 — Pipeline + eval** | M1, M4, M6 | Smoke tests, `render.py` |
| **R2 — Assets + compositing** | M2, M3 | VRoid PNG export, visual QA |
| **R3 — Report + coordination** | Deliverables, repo health | User study admin, stretch M-stretch |

**All:** code review, sync, Study A participation.

-----

## 10. Deliverables

### 10.1 Course

- Proposal, presentation slides, Q&A note, written report (≤10 pages), code repository.

### 10.2 Repository artifacts

- `avatarshield/` — pipeline (M1–M5)
- `assets/avatars/` — 5 PNG
- `samples/input/` — test media (local, consent documented; not committed)
- `data/output/` — renders (gitignored)
- `notebooks/` — eval
- `report.pdf`, `slides.pptx`, `evaluation_results.csv` (outside repo or release tag)

### 10.3 Documentation

- `README.md` — setup, CLI, disclaimers
- `ETHICS.md` — safety framing, consent, IP (to author)
- `RESULTS.md` — eval summary (to author after M6)
- `plan.md` — phased implementation checklist

-----

## 11. Risk register

| ID | Risk | Mitigation |
|----|------|------------|
| R1 | Scope creep | Lock spec v0.5; team vote for additions |
| R2 | Avatar PNG quality / consistency | VRoid export checklist; single lighting template |
| R3 | AnimeGAN stretch fails | Stretch only; drop from report claims |
| R4 | Member unavailable | Cross-train one backup per module |
| R5 | Live demo failure | Pre-record backup clip |
| R6 | DeepPrivacy2 deps heavy | Colab notebook or HF Space |
| R7 | Study B low n | Report limitations; reduce n |
| R8 | Third-party avatar IP | VRoid self-designed only |
| R9 | “Self-recognition” challenged | Cite Proteus Effect; document non-claims |
| R10 | No local CUDA | Colab for GPU baselines |

-----

## 12. Ethical considerations

- No minor participants in studies; adults only.
- Research prototype — not a deployable child-safety product.
- VRoid avatars team-owned; no copyrighted characters.
- Threat model and non-claims documented in README / `ETHICS.md`.
- MIT (or equivalent) code license for transparency.
- Dual-use: discuss misuse (impersonation) in report; non-realistic avatars reduce spoof risk.

-----

## 13. Open decisions

- [ ] Team member names assigned to R1–R3
- [ ] VRoid art lead
- [ ] Local-only vs optional `backend="external"` adapter (see `plan.md` D2)
- [ ] DeepPrivacy2: local vs Colab fallback
- [ ] Gradio demo: yes/no

-----

## 14. References (working bibliography)

**Core methods:** DeepPrivacy (2019); DeepPrivacy2 (2023); expression-preserving anonymization (2025); KFAAR (2025).

**Techniques:** MediaPipe (2019); Proteus Effect (Yee & Bailenson, 2007); Poisson editing (2003).

**Child safety context:** (to add) UNICEF digital safety; biometric protection for minors.

-----

**Spec status:** v0.5 — aligned with `plan.md` and `requirements.txt` (core). Previous code removed; implementation follows Phase 0 in `plan.md`.
