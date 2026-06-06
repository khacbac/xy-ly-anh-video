"""AvatarShield HTTP API — pure-IVP501 build.

A single synchronous theme-based render path (spec.md §13 D1). The legacy
avatar-PNG and VToonify stylize endpoints were removed in Phase 7.

Endpoints
---------
GET  /health                       — liveness probe
GET  /ivp/themes                   — list available theme names
POST /ivp/preview                  — multipart: video, [theme], [frame_index]
                                     → {job_id, preview_url, ...}
POST /ivp/preview/{job_id}/reroll  — re-render preview at a different frame
POST /ivp/render/{job_id}          — commit a previewed job (sync MP4 render)
POST /ivp/render                   — one-shot: video → MP4 (sync)
GET  /preview/{job_id}.png         — preview PNG
GET  /output/{job_id}.mp4          — rendered MP4

The render pipeline is cel-shade + theme palette only (Phase 5/6 look);
there is no preset selector.

Run
---
    cd final-project
    source .venv/bin/activate
    pip install -r api/requirements.txt
    python -m uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from avatarshield import (
    RenderConfig,
    render_preview_frame,
    render_video,
)
from avatarshield.palette import list_available_themes, load_theme

logger = logging.getLogger("avatarshield.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
JOBS_DIR = PROJECT_ROOT / "data" / "api_jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v"}
MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB

app = FastAPI(title="AvatarShield API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _suffix_or_default(filename: str | None, default: str) -> str:
    if not filename:
        return default
    return Path(filename).suffix.lower() or default


async def _save_upload(upload: UploadFile, dest: Path) -> int:
    written = 0
    with dest.open("wb") as fh:
        while chunk := await upload.read(1024 * 1024):
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                fh.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"{upload.filename or 'file'} exceeds "
                        f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB"
                    ),
                )
            fh.write(chunk)
    return written


def _public_output_url(request: Request, job_id: str) -> str:
    return str(request.url_for("get_output", job_id=job_id))


def _public_preview_url(request: Request, job_id: str) -> str:
    return str(request.url_for("get_preview", job_id=job_id))


def _validate_job_id(job_id: str) -> None:
    if not job_id.isalnum() or len(job_id) > 64:
        raise HTTPException(status_code=400, detail="invalid job_id")


def _classify_render_error(exc: BaseException) -> HTTPException:
    msg = str(exc).strip() or exc.__class__.__name__
    lower = msg.lower()
    if "no face" in lower or "no frames" in lower:
        return HTTPException(status_code=422, detail=msg)
    return HTTPException(status_code=500, detail=msg)


def _coerce_theme(theme: str) -> str:
    try:
        load_theme(theme)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return theme


def _locate_job_video(job_dir: Path) -> Path:
    candidates = sorted(
        p
        for p in job_dir.glob("input.*")
        if p.suffix.lower() in ALLOWED_VIDEO_SUFFIXES
    )
    if not candidates:
        raise HTTPException(status_code=404, detail="job not found")
    return candidates[0]


def _saved_theme(job_dir: Path) -> str:
    cfg = job_dir / "theme.txt"
    if not cfg.is_file():
        return RenderConfig().theme
    return cfg.read_text().strip() or RenderConfig().theme


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ivp/themes")
async def themes() -> JSONResponse:
    return JSONResponse({"themes": list_available_themes()})


@app.post("/ivp/preview")
async def preview(
    request: Request,
    video: UploadFile = File(..., description="Source video (mp4/mov)"),
    theme: str = Form(default=RenderConfig().theme),
    frame_index: int | None = Form(default=None),
) -> JSONResponse:
    video_suffix = _suffix_or_default(video.filename, ".mp4")
    if video_suffix not in ALLOWED_VIDEO_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"video must be one of {sorted(ALLOWED_VIDEO_SUFFIXES)}, "
                f"got {video_suffix}"
            ),
        )
    theme = _coerce_theme(theme)

    job_id = uuid.uuid4().hex
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    video_path = job_dir / f"input{video_suffix}"
    preview_path = job_dir / "preview.png"

    try:
        video_bytes = await _save_upload(video, video_path)
        logger.info(
            "preview %s received video=%d bytes theme=%s frame=%s",
            job_id, video_bytes, theme, frame_index,
        )
        config = RenderConfig(theme=theme)
        result = await asyncio.to_thread(
            render_preview_frame,
            video_path,
            preview_path,
            frame_index=frame_index,
            config=config,
        )
    except HTTPException:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    except Exception as exc:
        logger.exception("preview %s failed", job_id)
        shutil.rmtree(job_dir, ignore_errors=True)
        raise _classify_render_error(exc) from exc

    (job_dir / "theme.txt").write_text(theme)

    return JSONResponse(
        {
            "job_id": job_id,
            "preview_url": _public_preview_url(request, job_id),
            "theme": theme,
            "frame_index": result.frame_index,
            "frame_count": result.frame_count,
            "fps": result.fps,
            "width": result.width,
            "height": result.height,
        }
    )


@app.post("/ivp/preview/{job_id}/reroll")
async def reroll_preview(
    request: Request,
    job_id: str,
    theme: str | None = Form(default=None),
    frame_index: int | None = Form(default=None),
) -> JSONResponse:
    _validate_job_id(job_id)
    job_dir = JOBS_DIR / job_id
    if not job_dir.is_dir():
        raise HTTPException(status_code=404, detail="job not found")

    video_path = _locate_job_video(job_dir)
    chosen_theme = _coerce_theme(theme or _saved_theme(job_dir))
    preview_path = job_dir / "preview.png"

    try:
        config = RenderConfig(theme=chosen_theme)
        result = await asyncio.to_thread(
            render_preview_frame,
            video_path,
            preview_path,
            frame_index=frame_index,
            config=config,
        )
    except Exception as exc:
        logger.exception("preview reroll %s failed", job_id)
        raise _classify_render_error(exc) from exc

    (job_dir / "theme.txt").write_text(chosen_theme)

    return JSONResponse(
        {
            "job_id": job_id,
            "preview_url": _public_preview_url(request, job_id),
            "theme": chosen_theme,
            "frame_index": result.frame_index,
            "frame_count": result.frame_count,
            "fps": result.fps,
            "width": result.width,
            "height": result.height,
        }
    )


@app.post("/ivp/render/{job_id}")
async def render_from_job(
    request: Request,
    job_id: str,
    theme: str | None = Form(default=None),
) -> JSONResponse:
    _validate_job_id(job_id)
    job_dir = JOBS_DIR / job_id
    if not job_dir.is_dir():
        raise HTTPException(status_code=404, detail="job not found")

    video_path = _locate_job_video(job_dir)
    chosen_theme = _coerce_theme(theme or _saved_theme(job_dir))
    output_path = job_dir / "output.mp4"

    try:
        config = RenderConfig(theme=chosen_theme)
        result = await asyncio.to_thread(
            render_video,
            video_path,
            output_path,
            config,
            progress=False,
        )
    except Exception as exc:
        logger.exception("job %s render failed", job_id)
        raise _classify_render_error(exc) from exc

    (job_dir / "theme.txt").write_text(chosen_theme)

    return JSONResponse(
        {
            "job_id": job_id,
            "output_url": _public_output_url(request, job_id),
            "theme": chosen_theme,
            "frame_count": result.frame_count,
            "fps": result.fps,
            "duration_s": result.duration_s,
        }
    )


@app.post("/ivp/render")
async def render(
    request: Request,
    video: UploadFile = File(..., description="Source video (mp4/mov)"),
    theme: str = Form(default=RenderConfig().theme),
) -> JSONResponse:
    video_suffix = _suffix_or_default(video.filename, ".mp4")
    if video_suffix not in ALLOWED_VIDEO_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"video must be one of {sorted(ALLOWED_VIDEO_SUFFIXES)}, "
                f"got {video_suffix}"
            ),
        )
    theme = _coerce_theme(theme)

    job_id = uuid.uuid4().hex
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    video_path = job_dir / f"input{video_suffix}"
    output_path = job_dir / "output.mp4"

    try:
        video_bytes = await _save_upload(video, video_path)
        logger.info(
            "job %s received video=%d bytes theme=%s",
            job_id, video_bytes, theme,
        )
        config = RenderConfig(theme=theme)
        result = await asyncio.to_thread(
            render_video,
            video_path,
            output_path,
            config,
            progress=False,
        )
    except HTTPException:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    except Exception as exc:
        logger.exception("job %s render failed", job_id)
        shutil.rmtree(job_dir, ignore_errors=True)
        raise _classify_render_error(exc) from exc

    (job_dir / "theme.txt").write_text(theme)

    return JSONResponse(
        {
            "job_id": job_id,
            "output_url": _public_output_url(request, job_id),
            "theme": theme,
            "frame_count": result.frame_count,
            "fps": result.fps,
            "duration_s": result.duration_s,
        }
    )


@app.get("/preview/{job_id}.png", name="get_preview")
async def get_preview(job_id: str) -> FileResponse:
    _validate_job_id(job_id)
    path = JOBS_DIR / job_id / "preview.png"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="preview not found")
    return FileResponse(path, media_type="image/png", filename=f"{job_id}.png")


@app.get("/output/{job_id}.mp4", name="get_output")
async def get_output(job_id: str) -> FileResponse:
    _validate_job_id(job_id)
    path = JOBS_DIR / job_id / "output.mp4"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="output not found")
    return FileResponse(path, media_type="video/mp4", filename=f"{job_id}.mp4")
