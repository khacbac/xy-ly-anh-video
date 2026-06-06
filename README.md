# AvatarShield — Final Project (IVP501)

**Scope (v1.0, pure-IVP501):** render a video where the head region is
re-stylized with a **theme palette + cel-shade** built from classical
OpenCV / numpy primitives only — Haar detect, HSV ∩ YCbCr skin mask,
ellipse + One Euro tracker, bilateral + Lab K-means quantize + XDoG edges,
Reinhard color transfer. **No neural models, no MediaPipe, no torch.**

## Core feature

```bash
# Cel-shade + theme palette render.
python -m avatarshield.render \
  --video  samples/input/clip.mp4 \
  --theme  porcelain-pink \
  --output data/output/clip_ivp.mp4
```

Single-frame preview (auto-detected by `.png` output suffix, or via
`--frame-index N`):

```bash
python -m avatarshield.render \
  --video  samples/input/clip.mp4 \
  --theme  porcelain-pink \
  --output data/output/clip_ivp.png \
  --frame-index 262
```

`scripts/render_clip.py` is a thin convenience wrapper around the same
entrypoint with sensible defaults (`samples/input/clip.mp4` →
`data/output/<stem>_<theme>.mp4`).

## Themes

Five hand-authored palettes ship under `assets/themes/`:

| theme            | tone                                          |
|------------------|-----------------------------------------------|
| `porcelain-pink` | porcelain skin, magenta lips, cool brunette   |
| `tan-amber`      | warm tan, amber hair, deep brown brows        |
| `ivory-violet`   | ivory skin, violet hair, plum lips            |
| `bronze-teal`    | bronze skin, teal hair, deep red lips         |
| `peach-noir`     | peach skin, blue-black hair, heavier posterise|

See [`assets/themes/README.md`](assets/themes/README.md) for the Lab
values + preview grid, and [`scripts/phase6_render_previews.py`](scripts/phase6_render_previews.py)
to re-render previews after editing a theme JSON.

## Documentation

| File                                  | Purpose                                        |
|---------------------------------------|------------------------------------------------|
| [`spec.md`](spec.md)                  | Scope, architecture, evaluation, course mapping|
| [`plan.md`](plan.md)                  | Phased implementation checklist                |
| [`RESULTS.md`](RESULTS.md)            | Phase 8 evaluation — re-id + HF ratio + scatter|
| [`ETHICS.md`](ETHICS.md)              | Research status, threat model, non-claims      |
| [`report.md`](report.md)              | Final report draft (≤10 pages, pandoc-ready)   |
| [`slide/`](slide/)                    | Final deck (`FaceVeil_Dual_Track_Anonymization v2 (main) - enriched.pptx`) + asset PNGs + build scripts |
| [`spec_legacy.md`](spec_legacy.md)    | v0.5 legacy spec (VToonify / MediaPipe)        |
| [`plan_legacy.md`](plan_legacy.md)    | v0.5 legacy plan                               |

## Evaluation (Phase 8)

Render the five reference baselines and compute the privacy / utility
metrics from `spec.md` §7:

```bash
.venv/bin/python scripts/eval_render_baselines.py    # control / blur / mosaic / ivp x 2
.venv/bin/python scripts/eval_freq.py                # FFT HF energy ratio
.venv/bin/python scripts/eval_freq.py --head-only    # same metric, head crop only
.venv/bin/python scripts/eval_scatter.py             # joint HF x re-id scatter

# Re-id needs facenet-pytorch + torch (kept out of the core .venv).
# Install into a side venv and point at it:
/path/to/torch-venv/bin/python scripts/eval_reid.py
```

Artifacts land under `data/output/phase8/<clip-stem>/eval/`. See
[`RESULTS.md`](RESULTS.md) for the full write-up.

## Repository layout

```
final-project/
├── avatarshield/        # render pipeline (cel_shade, detect, palette, smooth)
├── api/                 # FastAPI wrapper — sync /ivp/render endpoint
├── app/                 # Expo (React Native) client — wired to /ivp/* with theme selector
├── assets/themes/       # 5 theme JSONs + previews
├── samples/input/       # local test media (gitignored)
├── data/output/         # rendered videos (gitignored)
├── scripts/             # phase test drivers + render_clip CLI
├── requirements.txt
├── spec.md / plan.md
└── README.md
```

## Setup

```bash
cd final-project
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python **3.10+** recommended. Core dependencies: `opencv-python`, `numpy`,
`Pillow` (3 packages, no GPU stack).

## HTTP API (optional)

A minimal synchronous render endpoint lives in `api/server.py`:

```bash
cd final-project
source .venv/bin/activate
pip install -r api/requirements.txt
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

```bash
# Cel-shade + theme render.
curl -X POST http://localhost:8000/ivp/render \
  -F video=@samples/input/clip.mp4 \
  -F theme=tan-amber

# List available themes.
curl http://localhost:8000/ivp/themes
```

See [`api/README.md`](api/README.md) for the full contract.

## Disclaimer

Research prototype for IVP501. **Not a production privacy / safety
product.** No third-party model weights, no character IP, no biometric
verification claim. See `spec.md` §4.3 (non-claims).

## Client status

* **`app/` (Expo / React Native)** — wired to the `/ivp/*` endpoints
  with a theme picker (5 themes from §6.3) inside the editor. This is
  the only first-party client; the web client has been removed (mobile
  is the demo target).
