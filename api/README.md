# AvatarShield API

HTTP wrapper around the pure-IVP render pipeline (`avatarshield.render`).
A single synchronous theme-based path, per `spec.md` §13 D1. The legacy
avatar-PNG and VToonify stylize endpoints were removed in Phase 7.

## Run

```bash
cd final-project
source .venv/bin/activate
pip install -r api/requirements.txt
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

The server binds to `0.0.0.0` so a phone / simulator on the same LAN can reach
it (use your host's LAN IP, e.g. `http://192.168.1.10:8000`).

## Endpoints

### `GET /health`
Liveness probe → `{ "status": "ok" }`.

### `GET /ivp/themes`
List available theme names from `assets/themes/*.json`.

### `POST /ivp/preview`

`multipart/form-data`:

| field         | type       | required | notes                              |
|---------------|------------|----------|------------------------------------|
| `video`       | mp4/mov/m4v| yes      | source video, ≤100 MB              |
| `theme`       | str        | no       | default `porcelain-pink`           |
| `frame_index` | int        | no       | else middle frame                  |

Returns `{ job_id, preview_url, theme, frame_index, frame_count, fps, width, height }`.

### `POST /ivp/preview/{job_id}/reroll`
Body may include `theme` / `frame_index` to re-render the preview for an
existing job at a different theme or frame.

### `POST /ivp/render/{job_id}`
Commit a previewed job to a full video render. **Synchronous.** Body may
include `theme` to override the saved one. Returns
`{ job_id, output_url, theme, frame_count, fps, duration_s }`.

### `POST /ivp/render` (one-shot)
`multipart/form-data` with `video` + optional `theme`. Synchronous render
in one request. Convenient for CLI / curl use.

### Static file serving

| route                       | purpose             |
|-----------------------------|---------------------|
| `GET /preview/{job_id}.png` | preview PNG         |
| `GET /output/{job_id}.mp4`  | final rendered MP4  |

## Notes

- Job artifacts live in `final-project/data/api_jobs/<job_id>/`.
  Layout: `input.<ext>`, `preview.png`, `output.mp4`, `theme.txt`.
- CORS is wide-open; tighten for non-dev deployments.
- No auth, no persistent queue.
