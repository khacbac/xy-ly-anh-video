# AvatarShield — Kế hoạch triển khai (pure-IVP501)

**Tham chiếu:** `spec.md` v1.0 · Khóa IVP501  
**Branch:** `ivp-pure` (cắt từ `master`)  
**Phạm vi:** render video pipeline thuần OpenCV/numpy — Haar detect + cel-shade + theme palette. Không model ngoài (không MediaPipe, không VToonify, không BiSeNet).

**Trạng thái khởi đầu:** `cel_shade.py` đã có sẵn và là core của pipeline. Các module landmark/parsing/vtoonify còn nằm trong cây code (legacy) — sẽ bị xoá ở Phase 7. `plan_legacy.md` chứa kế hoạch cũ.

---

## 0. Tóm tắt

**Input:** video gốc (người quay) + tên theme (e.g. `porcelain-pink`).  
**Output:** video mới — vùng đầu được cel-shade + recolor theo theme JSON; phần body / background giữ nguyên (mild cel-shade nhẹ optional).

**Deliverable CLI:**

```bash
python -m avatarshield.render \
  --video   samples/input/clip.mp4 \
  --theme   porcelain-pink \
  --output  data/output/clip_ivp.mp4
```

**Deliverable preview (single frame, for tuning):**

```bash
python -m avatarshield.render \
  --video   samples/input/clip.mp4 \
  --theme   porcelain-pink \
  --output  data/output/clip_ivp.png \
  --frame-index 120
```

---

## 1. Kiến trúc (rút gọn từ `spec.md` §5)

```
[MP4]
  └─► video_io
        └─► frame ─► Haar detect ─► bbox + One Euro ─┐
                  ─► skin mask (HSV ∩ YCbCr + morph)─┤
                                                      ▼
                                             head ellipse ROI (feathered)
                                                      │
                  cel_shade (bilateral + Lab K-means + XDoG)
                                                      │
                  palette.apply (Reinhard per region) │
                                                      │
                  composite (alpha blend onto frame)  │
                                                      ▼
                                                 [Output MP4]
```

Module layout target (xem `spec.md` §5.2–5.3):

```
avatarshield/
├── render.py          # entrypoint + CLI         (rewrite)
├── video_io.py        # decode/encode             (keep)
├── detect.py          # Haar + skin + head ellipse (new)
├── smooth.py          # One Euro on bbox          (rewrite — simpler)
├── cel_shade.py       # already there             (keep)
├── palette.py         # theme JSON + Reinhard     (new)
└── composite.py       # feathered alpha blend     (slim)

assets/themes/         # 5 JSON theme palettes      (new)
```

---

## 2. Lộ trình theo phase

Thứ tự implement — không gắn deadline; tick khi xong. Mỗi phase phải có 1 visual artifact để team xác nhận trước khi sang phase kế tiếp.

### Phase 0 — Branch + docs ✅

- [x] `git checkout -b ivp-pure`
- [x] `spec.md` v0.5 → `spec_legacy.md` (LEGACY banner)
- [x] `plan.md` cũ → `plan_legacy.md` (LEGACY banner)
- [x] `spec.md` v1.0 mới (pure-IVP501)
- [x] `plan.md` mới (file này)

**Sync point.** Team đọc spec v1.0 và confirm trước khi viết code.

---

### Phase 1 — Detect + bbox tracking

**Học:** S08 (Haar / feature-based detection), S10 (One Euro filter).

Module: `avatarshield/detect.py`, `avatarshield/smooth.py`, sửa nhẹ `render.py`.

- [ ] `detect.py` — wrapper quanh `cv2.CascadeClassifier` (frontal + profile)
  - Input: BGR frame.
  - Output: `Bbox(x, y, w, h)` hoặc `None`.
  - Strategy: chạy frontal trước; nếu miss, chạy profile.
  - Pick largest bbox khi multi-face.
- [ ] `smooth.py` — One Euro filter trên 4 chiều `(cx, cy, w, h)`.
  - Tham số mặc định: `min_cutoff=1.0`, `beta=0.02`.
  - Hold-last-good policy: nếu miss < 6 frame, dùng lại bbox cũ; ≥ 6 thì trả None.
- [ ] `render.py` — viết lại skeleton `render_video()` chỉ chạy detect + smooth, vẽ bbox màu xanh lên frame, ghi ra video.

**Artifact:** `data/output/phase1_bbox.mp4` — clip có overlay bbox theo head, mượt qua các frame.

**Sync point.** Visual check: bbox không "nhảy", không lệch khi user quay đầu nhẹ.

---

### Phase 2 — Skin mask + head ellipse ROI

**Học:** S06 (morphology, connected components), S07 (color spaces).

Module: `avatarshield/detect.py` (mở rộng).

- [ ] Skin range trong HSV: `H ∈ [0, 25] ∪ [160, 179]`, `S ∈ [40, 200]`, `V ∈ [60, 255]`.
- [ ] Skin range trong YCbCr: `Y ∈ [60, 255]`, `Cb ∈ [85, 135]`, `Cr ∈ [135, 180]`.
- [ ] Mask cuối = `mask_HSV ∩ mask_YCbCr`, sau đó:
  - `cv2.morphologyEx(MORPH_CLOSE, 5x5)` lấp lỗ trong da.
  - `cv2.morphologyEx(MORPH_OPEN, 3x3)` xoá nhiễu rời rạc.
  - Connected components, giữ blob lớn nhất giao với Haar bbox.
- [ ] `head_ellipse_mask(bbox)` — vẽ ellipse trên ảnh đen, kéo dài lên trên 60 %, ngang 20 % mỗi bên, xuống 5 % (tóc + cằm).
- [ ] Gaussian feather mask với `head_feather_px=41` → `float32` trong `[0, 1]`.

**Artifact:** `data/output/phase2_masks.png` — 4-panel: original | skin mask | head ellipse | feathered head ROI.

---

### Phase 3 — Palette recolor (chưa cel-shade)

**Học:** S07 (Lab color space, Reinhard color transfer).

Module: `avatarshield/palette.py` (new).

- [ ] `Theme` dataclass: `skin_lab`, `hair_lab`, `lips_lab`, `eyes_lab`, `brows_lab`, `saturation_boost`, `edge_strength`, `luminance_levels`.
- [ ] `load_theme(name) -> Theme` — đọc `assets/themes/<name>.json`, validate schema, raise `ValueError` nếu thiếu key.
- [ ] `apply_skin_recolor(frame_bgr, skin_mask, theme, strength) -> frame_bgr`
  - Convert frame → Lab.
  - Tính mean Lab của vùng `skin_mask > 0.5`.
  - Reinhard transfer (mean + std) về `theme.skin_lab` (chỉ shift mean nếu thiếu std target — std giữ nguyên).
  - Blend với mask + strength.
- [ ] 1 theme JSON tạm thời: `assets/themes/porcelain-pink.json` (giá trị placeholder, sẽ tune ở Phase 6).

**Artifact:** `data/output/phase3_skin_recolor.mp4` — clip với vùng da chuyển tông theo theme, chưa cel-shade. Trông "kỳ kỳ" là đúng — bước này chỉ test pipeline.

---

### Phase 4 — Cel-shade integration ✅

**Học:** S05 (bilateral filter), S07 (Lab + K-means quantize), S06 (XDoG edge detection), S10 (TemporalState EMA centers).

Module: `scripts/phase4_test.py` (driver, mirrors Phase 1–3 pattern). Wiring goes into `render.py` proper at Phase 5; this phase only proves the cel-shade pipeline works against the existing detect/smooth/head-mask scaffold.

- [x] Khởi tạo `TemporalState(ema=0.85)` 1 lần / video, share qua tất cả frame.
- [x] Mỗi frame: gọi `cel_shade(...)` trên TOÀN frame với params `luminance_levels=3`, `chroma_levels=0`, `bilateral_passes=3`, `edge_strength=0.85`, `saturation_boost=1.25`.
- [x] Composite chỉ vùng head: `out = cel * head_feather + frame * (1 - head_feather)` (chưa palette).

**Artifact:** `data/output/phase4_celshade_only.mp4` (full clip) + `data/output/phase4_celshade.png` (3-panel grid: original | cel_shade whole frame | head-masked composite).

**Tuning:** `luminance_levels=3`, `bilateral_passes=3`, `edge_strength=0.85`, `state.ema=0.85` — verified against the six Phase 1 test frames; no visible "boiling" in the 667-frame clip. Raise `--ema 0.9` if a darker / faster clip flickers.

---

### Phase 5 — Full pipeline (cel-shade + palette) ✅

Module: `scripts/phase5_test.py` (driver) + geometric helpers added to `avatarshield/detect.py` (`face_oval_mask`, `hair_ring_mask`, `feature_strip_masks`). Hợp nhất Phase 3 + Phase 4 — chưa rewrite `render.py` proper (vẫn dùng pattern test-script tới Phase 7 cleanup).

- [x] Pipeline thứ tự: skin recolor pre-pass (snap palette nhẹ, strength 0.25) → cel_shade(state shared) → palette per-region post-pass → composite onto frame.
  - Lý do pre-pass: K-means cluster sẽ "snap" về bands sáng/peachy thay vì band da tự nhiên. Giống logic trong `cel_shade.anime_head_pass` đã làm.
- [x] Per-region post-pass:
  - Skin: full Lab Reinhard, `strength=0.55`. Mask = `skin_mask(HSV ∩ YCbCr) ∩ face_oval_mask` + light feather.
  - Lips: full Lab Reinhard trên vùng môi (Gaussian band tại `0.75 · bbox_h`), `strength=0.75`.
  - Eyes / brows: Gaussian band tại `0.45 · bbox_h` và `0.40 · bbox_h`, `strength=0.75`.
  - Hair: chroma-only (chỉ a, b), `strength=0.55`, vùng = `hair_ring_mask` = `head_ellipse \ face_oval` + light feather.
- [x] Composite: `out = stylized_recolored * head_feather + frame * (1 - head_feather)`.
- [x] Side-by-side: `--no-sxs` flag điều khiển — mặc định bật, output `clip_sxs.mp4` (original | full pipeline).

**Artifact:** `data/output/phase5_full.mp4`, `clip_sxs.mp4`, `phase5_full.png` (4-panel grid: original | cel_shade | region masks overlay | composite). Đây là demo dự kiến cho buổi presentation. Hiệu ứng porcelain-pink còn nhẹ do theme values là placeholder của Phase 3 — Phase 6 sẽ tune lại.

**Sync point.** Visual check 3 người: chấp nhận hay quay lại tune?

---

### Phase 6 — Theme authoring (5 themes) ✅

Module: `scripts/phase6_render_previews.py`, `assets/themes/*.json`, `assets/themes/README.md`. Notebook flow promised in `spec.md` §6.2 was deferred — the 5 themes are hand-authored from the spec §6.3 brief and validated visually through the preview script (cheaper than building the notebook for a 5-theme catalogue we tune once).

- [x] 5 theme JSON theo bảng `spec.md` §6.3:
  - `porcelain-pink.json` — re-authored (Phase 3 placeholder values replaced)
  - `tan-amber.json`
  - `ivory-violet.json`
  - `bronze-teal.json`
  - `peach-noir.json` (heavier posterise: `luminance_levels=2`, `saturation_boost=0.95`)
- [x] `scripts/phase6_render_previews.py` — reuses `phase5_test.PipelineConfig` + `process_frame`; per-theme style scalars (`saturation_boost`, `edge_strength`, `luminance_levels`, `chroma_levels`) are now wired through from the JSON.
- [x] Preview artifacts:
  - `assets/themes/previews/<name>.png` × 5 (one frontal frame per theme)
  - `data/output/phase6_grid.png` — 2×3 grid (original + 5 themes)
- [x] `assets/themes/README.md` — JSON schema reminder, Lab values table, embedded previews, re-render command, authoring pointer.
- [ ] (Stretched) `notebooks/author_theme.ipynb` interactive authoring flow — defer to Phase 8 if there's bandwidth; tuning is currently fast enough by hand-editing JSON + re-running the preview script.

**Artifact:** 5 JSON + 5 preview PNGs + `data/output/phase6_grid.png` + `assets/themes/README.md`.

**Sync point.** Visual check 3 người trên `phase6_grid.png`: chấp nhận palette hay cần tune lại Lab values? Re-run script + edit JSON cho đến khi pass.

---

### Phase 7 — Cleanup (xoá legacy) ✅ (app/web rework deferred)

Module deletion theo `spec.md` §5.4. Làm 1 commit riêng để diff rõ ràng.

- [x] Xoá:
  - `avatarshield/landmarks.py`
  - `avatarshield/parsing.py`
  - `avatarshield/morph.py`
  - `avatarshield/align.py`
  - `avatarshield/anime.py`
  - `avatarshield/stylize.py`
  - `avatarshield/mask.py`
  - (bonus) `avatarshield/color.py`, `avatarshield/composite.py` — sau khi `render.py` được viết lại không còn dùng.
- [x] Xoá thư mục:
  - `third_party/` (VToonify, dlib, BiSeNet)
  - `checkpoints/`
- [x] Xoá scripts cũ:
  - `scripts/animegan_poc.py`, `scripts/animegan_render_video.py`
  - `scripts/vtoonify_face_video.py`, `scripts/vtoonify_style_sweep.py`
  - `scripts/combo_render_video.py`
  - `scripts/cel_shade_render.py`, `scripts/cel_shade_test.py`
  - `scripts/post_render.sh`
- [x] `scripts/render_clip.py` — thin wrapper quanh `avatarshield.render.render_video` / `render_preview_frame`, defaults sensible.
- [x] Rewrite `avatarshield/render.py` → `RenderConfig` / `RenderState` / `render_video` / `render_preview_frame` / `process_frame` matching spec §5.1. CLI: `python -m avatarshield.render --video --theme --output [--frame-index]`. `avatarshield/__init__.py` re-exports only the new API.
- [x] `avatarshield/smooth.py` — bỏ `LandmarkSmoother` (đã hết caller).
- [x] `requirements.txt` slim down:
  ```
  opencv-python>=4.9
  numpy>=1.26
  Pillow>=10.2
  ```
- [x] Bỏ `mediapipe`, `torch`, `dlib` khỏi `requirements.txt` + `api/requirements.txt`. (Còn lại cho API: `fastapi`, `uvicorn`, `python-multipart`.)
- [x] API (`api/server.py`):
  - Decision (`spec.md` §13 D1): **giữ minimal `/ivp/render` (synchronous)**.
  - Endpoints mới: `/health`, `/ivp/themes`, `/ivp/preview`, `/ivp/preview/{job_id}/reroll`, `/ivp/render/{job_id}`, `/ivp/render`, `/preview/{job_id}.png`, `/output/{job_id}.mp4`.
  - Bỏ toàn bộ stylize + avatar-PNG routes.
  - `api/README.md` + `api/requirements.txt` đã update.
- [ ] Expo app (`app/`) — **DEFERRED**. App hiện wire tới `/preview`, `/render`, `/stylize/*` (legacy) qua nhiều screen + store. Cần re-write `EditorScreen`/`PreviewScreen`/`api.ts`/`filters` thành "chọn theme → render đồng bộ"; risk cao cho demo nên để Phase 7 follow-up commit riêng.
- [ ] Web client (`web/`) — **DEFERRED** cùng lý do; chỉ chạy avatar-PNG flow legacy. Nếu không demo, có thể xoá thư mục sau khi quyết.
- [x] Update `README.md` — disclaimer mới, CLI mới, theme catalog, lưu ý app/web deferred.

**Sync point.** `git diff --stat master..ivp-pure` cho thấy net deletions ≥ net additions (kỳ vọng repo gọn đi). User cần eye-check một frame preview (theme bất kỳ) trước khi commit.

---

### Phase 8 — Evaluation + báo cáo

**Học:** S05.02 (FFT band filtering), S08 (FaceNet / ArcFace embedding), S05 baseline Gaussian.

Module: `scripts/eval_*.py` (preferred over notebooks per Phase 6 rationale — deterministic + cheap to regenerate), `RESULTS.md`, `ETHICS.md`.

- [x] `scripts/eval_render_baselines.py` — driver 1 lệnh render 5 phiên bản:
  1. control (passthrough)            — `control.mp4`
  2. Gaussian blur σ=12                — `blur.mp4`
  3. 8×8 mosaic                        — `mosaic.mp4`
  4. AvatarShield-IVP per-frame        — `ivp_per_frame.mp4` (state_ema=0, fresh smoother per frame)
  5. AvatarShield-IVP full (smoothing) — `ivp_full.mp4`
  Outputs land under `data/output/phase8/<clip-stem>/`. Also assembles a 2×3 grid PNG at the middle frame for visual sanity.
- [x] `scripts/eval_freq.py` — FFT HF energy ratio (`spec.md` §7.2).
  - Whole-frame mode + `--head-only` mode (head ellipse crop only, fair per-region measurement).
  - Writes `hf_ratio_summary.csv` / `hf_ratio_summary_head.csv` + bar / timeline PNGs.
  - Finding: HF ratio is a faithful proxy for blur / mosaic but NOT for cel-shade (XDoG line art is itself high-frequency); see RESULTS.md §3.2.
- [x] `scripts/eval_reid.py` — FaceNet re-id rate (`facenet-pytorch`).
  - Sample 1 / 15 frame; per-frame face crop via the project's Haar detector; cosine-distance threshold 0.7 (spec §7.1).
  - Optional dependency — when missing, the script prints a SKIP banner and writes stub CSV.
  - Distractor-set evaluation deferred (single-clip re-id-to-self anchor was sufficient for the headline metric).
- [x] `scripts/eval_scatter.py` — joint HF × re-id scatter PNG for RESULTS.md §4.
- [x] `scripts/eval_temporal.py` — bbox jitter + head-region MAD for the temporal-stability ablation in RESULTS.md §5.
- [x] Second-clip sanity run (`samples/input/dancing.mp4`, 184 frames @ 1080×1920) — baselines + FFT + re-id + temporal rendered into `data/output/phase8/dancing/`. Headline finding: Haar detection drops to 1/13 sampled frames at the prior `min_size_ratio = 0.08` default, so metric tables are clip-only; report §7 documents this as a failure-mode envelope. **2026-06-03 update:** investigating `samples/input/yellow.mp4` exposed that the same default also drove 59 % of accepted hits onto the torso (knit sweater passing the 10 % skin gate). Detector defaults retuned to `min_size_ratio = 0.05 / min_neighbors = 5 / min_skin_coverage = 0.30` and exposed via `RenderConfig.haar_*` + CLI; on yellow this lifts Haar coverage 22 → 115 frames and drops torso false positives 13 → 3.
- [x] `RESULTS.md` — full write-up: setup, re-id table, HF tables (whole + head-only), joint scatter, temporal stability §5, reproduction commands, caveats.
- [x] `ETHICS.md` — research-status, threat model in / out of scope, non-claims, data / consent, dual-use disclosure, license.
- [x] `report.md` (≤ 10 trang, pandoc-convertible) + `slide/FaceVeil_Dual_Track_Anonymization v2 (main) - enriched.pptx` (built by `slide/scripts/build_enriched_pptx.py`) — markdown + pptx deliverables.
- [x] Pre-record demo clip (backup nếu live demo lỗi) — already have `data/output/phase5_full.mp4` + `clip_sxs.mp4` + `phase8/clip/ivp_full.mp4` + `phase8/dancing/ivp_full.mp4`; pick one at presentation time.

**Sync point.** Báo cáo nháp xong, team review trước khi nộp.

---

### Phase 9 — Identity-obscuring preset (`cartoon`) — REMOVED

Trước đây Phase 9 thêm radial face warp + cartoon-eye overlay (`avatarshield/obscure.py`) và 2 preset `baseline`/`cartoon` để hạ re-id rate xuống 0.111. Để giữ phạm vi dự án gọn và explainable theo đúng toolbox IVP501 cel-shade, đã **bỏ Phase 9** khỏi pipeline: code (`obscure.py`, `PRESETS`, `config_for_preset`, `list_presets`, knob `face_warp_strength` / `eye_occlusion`), endpoint API (`/ivp/presets` + `preset` Form field) và preset selector trong app đều bị xoá. `RenderConfig` chỉ còn cel-shade + theme palette (Phase 5/6 look).

Baseline re-id của Phase 8 (ivp_full ≈ 0.704) là số headline cuối cùng.

---

## 3. Public API (target — đã chốt trong `spec.md` §5.1)

```python
# avatarshield/render.py

@dataclass
class RenderConfig:
    theme: str = "porcelain-pink"
    haar_scale_factor: float = 1.1
    haar_min_neighbors: int = 4
    bbox_hold_frames: int = 6
    smooth_min_cutoff: float = 1.0
    smooth_beta: float = 0.02
    head_top_extend: float = 0.6
    head_side_extend: float = 0.20
    head_bottom_extend: float = 0.05
    head_feather_px: int = 41
    luminance_levels: int = 3
    chroma_levels: int = 0
    bilateral_passes: int = 3
    saturation_boost: float = 1.30
    edge_strength: float = 0.85
    palette_strength_skin: float = 0.55
    palette_strength_features: float = 0.75
    palette_strength_hair: float = 0.55

def render_video(
    video_path: str | Path,
    output_path: str | Path,
    config: RenderConfig | None = None,
    *,
    progress: bool = True,
) -> RenderResult: ...

def process_frame(
    frame_bgr: np.ndarray,
    state: RenderState,
    config: RenderConfig,
) -> np.ndarray: ...
```

`RenderState` ôm: `OneEuro` smoother (`BboxSmoother`), `TemporalState` (cel-shade EMA),
`BoxTracker` (CSRT fallback khi Haar miss).

---

## 4. Phân công nhóm (3 người)

| Vai trò | Phase chính | Phase phụ |
|---|---|---|
| **A — Detection + Smoothing** | Phase 1, 2 | hỗ trợ Phase 5 (integration), Phase 7 (cleanup detect side) |
| **B — Stylization + Themes** | Phase 3, 4, 6 | tune `cel_shade.py`, viết `palette.py`, `author_theme.ipynb` |
| **C — Evaluation + Docs** | Phase 8 | viết `README.md` + `RESULTS.md`, demo clip, slides |

Sync points (đã nêu): cuối Phase 1, 5, 8.

---

## 5. Quyết định mở (mirror `spec.md` §13)

| # | Câu hỏi | Gợi ý |
|---|---|---|
| D1 | API surface? | Minimal `/ivp/render` (1 endpoint, synchronous) — giữ Expo screen đơn giản |
| D2 | `--keep-user-L` self-recognition slot mặc định bật? | **Bật** — đây là cốt lõi self-recognition theo spec |
| D3 | Haar profile fallback? | **Có** — `haarcascade_profileface.xml` |
| D4 | Theme authoring notebook ship Phase 6 hay sớm hơn? | Phase 6 (sau khi pipeline ổn ở Phase 5) |
| D5 | Re-id metric provider? | `facenet-pytorch` — nhẹ, CPU OK |

---

## 6. Definition of Done

- [x] `render_video(video, output) → MP4` với 5 theme đều render được trên `clip.mp4` (Phase 6 grid). Chưa cover 3 clip khác nhau — Phase 8 caveat (`dancing.mp4` Haar fail), 2 clip còn lại pending.
- [x] Debug single-frame artifact ở mỗi phase (Phase 1–5) — `scripts/phase{1..5}_test.py` + Phase 6 grid.
- [x] Temporal stability: side-by-side per-frame vs smoothed (`ivp_per_frame.mp4` vs `ivp_full.mp4`) cho thấy không boiling — RESULTS.md §5.
- [x] FFT HF energy ratio: not the right metric for cel-shade (XDoG line art = high-freq); RESULTS.md §3.2 ghi rõ, re-id là primary privacy number.
- [x] Re-id rate (Phase 8): ivp_full ≈ 0.704.
- [x] Repo gọn: `requirements.txt` core ≤ 5 dòng (opencv + numpy + Pillow), không còn `third_party/` hay `checkpoints/`. Torch + facenet-pytorch chỉ cài vào `.venv` cho eval, không list trong `requirements.txt`.
- [x] Report + slides + demo clip pre-recorded (`report.md`, `slide/FaceVeil_Dual_Track_Anonymization v2 (main) - enriched.pptx`, `data/output/phase8/clip/ivp_full.mp4`).

---

## 7. Next actions (immediate)

1. Đọc lại `spec.md` v1.0 + xác nhận §13 open decisions.
2. Phase 1: scaffold `detect.py` + `smooth.py`, viết test với 1 clip mẫu.
3. Phase 2: skin mask + head ellipse, artifact 4-panel PNG.
4. Sync 3 người sau Phase 2 trước khi bắt cel-shade integration.

---

## 8. Notes — chuyển đổi từ legacy

- `cel_shade.py` đã có sẵn và CHẠY ĐƯỢC (xem `scripts/cel_shade_render.py`). Phase 4 chỉ là wiring, không phải viết lại.
- `smooth.py` cũ là `LandmarkSmoother(num_points=468, ...)` — rewrite cho 4-D vector (bbox).
- `composite.py` cũ có hybrid Poisson; v1.0 chỉ cần alpha blend.
- `video_io.py` không cần thay đổi.
- `requirements.txt` cũ nặng — chỉnh ở Phase 7.
- Nếu tuning Phase 4 vẫn không vượt được "blur baseline aesthetic": review nhóm, có thể quay lại bật `chroma_levels > 0` trong cel-shade hoặc tăng `head_top_extend` cho tóc rộng hơn.
