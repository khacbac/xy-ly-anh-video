"""Theme palette + Reinhard recolor (S07 — color spaces & color transfer).

A *theme* is a small JSON descriptor that encodes the target look of the
rendered video as a per-region triple in CIE-Lab (OpenCV's
``cv2.COLOR_BGR2LAB`` convention: ``L ∈ [0, 255]``, ``a`` and ``b``
centred at 128). Themes replace the avatar PNG of the legacy spec — they
ship the *palette* of the look without any pixel data, which is what
lets the v1.0 pipeline stay model-free.

Why mean-only shift and not full Reinhard
-----------------------------------------
Reinhard et al. (2001) propose matching both the mean and the std of the
reference. We only have a single Lab triple per region in the theme JSON
(no std), so the canonical Reinhard recipe is reduced here to a *mean
shift* — pulling the frame's mean Lab toward the target mean while
keeping the frame's natural std. This preserves shading detail inside
the skin region; only the tint moves.

References
----------
Reinhard, E. et al. (2001). *Color Transfer between Images.* IEEE CG&A.
(Course slide S07.)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Theme dataclass + loader.
# ---------------------------------------------------------------------------


_REGION_KEYS = ("skin_lab", "hair_lab", "lips_lab", "eyes_lab", "brows_lab")
_SCALAR_DEFAULTS = {
    "saturation_boost": 1.3,
    "edge_strength": 0.85,
    "luminance_levels": 3,
    "chroma_levels": 0,
}


@dataclass(frozen=True)
class Theme:
    """A theme palette read from ``assets/themes/<name>.json``.

    All ``*_lab`` triples follow OpenCV's BGR2LAB convention. For ``hair_lab``
    the ``L`` channel is informative but unused at recolor time — hair is
    recolored chroma-only so the user's hair brightness survives.
    """

    name: str
    description: str
    skin_lab: tuple[float, float, float]
    hair_lab: tuple[float, float, float]
    lips_lab: tuple[float, float, float]
    eyes_lab: tuple[float, float, float]
    brows_lab: tuple[float, float, float]
    saturation_boost: float = _SCALAR_DEFAULTS["saturation_boost"]
    edge_strength: float = _SCALAR_DEFAULTS["edge_strength"]
    luminance_levels: int = _SCALAR_DEFAULTS["luminance_levels"]
    chroma_levels: int = _SCALAR_DEFAULTS["chroma_levels"]


def _coerce_lab(name: str, value) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(
            f"theme field {name!r}: expected [L, a, b], got {value!r}"
        )
    out = tuple(float(v) for v in value)
    for v in out:
        if not (0.0 <= v <= 255.0):
            raise ValueError(
                f"theme field {name!r}: channel {v} outside [0, 255]"
            )
    return out  # type: ignore[return-value]


def _default_themes_dir() -> Path:
    # ``assets/themes/`` lives at the repo root (one level above this package).
    return Path(__file__).resolve().parent.parent / "assets" / "themes"


def load_theme(name: str, *, themes_dir: Path | str | None = None) -> Theme:
    """Load and validate ``assets/themes/<name>.json``.

    Raises ``FileNotFoundError`` when the file is missing and ``ValueError``
    when the schema is incomplete or malformed.
    """
    base = Path(themes_dir) if themes_dir is not None else _default_themes_dir()
    path = base / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"theme not found: {path}")

    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"theme {name!r}: invalid JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise ValueError(f"theme {name!r}: top-level must be an object")

    missing = [k for k in _REGION_KEYS if k not in data]
    if missing:
        raise ValueError(
            f"theme {name!r}: missing required keys {missing!r}"
        )

    return Theme(
        name=str(data.get("name", name)),
        description=str(data.get("description", "")),
        skin_lab=_coerce_lab("skin_lab", data["skin_lab"]),
        hair_lab=_coerce_lab("hair_lab", data["hair_lab"]),
        lips_lab=_coerce_lab("lips_lab", data["lips_lab"]),
        eyes_lab=_coerce_lab("eyes_lab", data["eyes_lab"]),
        brows_lab=_coerce_lab("brows_lab", data["brows_lab"]),
        saturation_boost=float(
            data.get("saturation_boost", _SCALAR_DEFAULTS["saturation_boost"])
        ),
        edge_strength=float(
            data.get("edge_strength", _SCALAR_DEFAULTS["edge_strength"])
        ),
        luminance_levels=int(
            data.get("luminance_levels", _SCALAR_DEFAULTS["luminance_levels"])
        ),
        chroma_levels=int(
            data.get("chroma_levels", _SCALAR_DEFAULTS["chroma_levels"])
        ),
    )


def list_available_themes(themes_dir: Path | str | None = None) -> list[str]:
    """Return theme names discoverable under ``assets/themes/``."""
    base = Path(themes_dir) if themes_dir is not None else _default_themes_dir()
    if not base.exists():
        return []
    return sorted(p.stem for p in base.glob("*.json"))


# ---------------------------------------------------------------------------
# Recolor primitive — mean shift toward a target Lab triple.
# ---------------------------------------------------------------------------


def _coerce_mask(mask: np.ndarray, frame_shape: tuple[int, int]) -> np.ndarray:
    """Return a float32 mask in ``[0, 1]`` with the frame's H×W shape."""
    if mask is None:
        raise ValueError("apply_recolor: mask is required")
    if mask.shape[:2] != frame_shape:
        raise ValueError(
            f"apply_recolor: mask shape {mask.shape[:2]} does not match "
            f"frame shape {frame_shape}"
        )
    m = mask.astype(np.float32)
    # uint8 binary masks come in as {0, 255}; rescale to [0, 1].
    if m.max() > 1.0:
        m = m / 255.0
    return np.clip(m, 0.0, 1.0)


def apply_target_lab_recolor(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    target_lab: Iterable[float],
    *,
    strength: float = 0.55,
    channels: tuple[int, ...] = (0, 1, 2),
) -> np.ndarray:
    """Pull the mean Lab inside ``mask`` toward ``target_lab``.

    The transform is a partial mean shift per channel (no std rescaling —
    the theme only ships a single Lab triple per region). ``strength``
    blends between identity (0.0) and full shift to the target mean (1.0).
    The shift is applied through the soft mask so the recolor fades at
    the boundary instead of producing a hard edge.

    Channel filter — pass ``channels=(1, 2)`` for chroma-only transfer
    (hair, where the user's L should survive).
    """
    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("apply_recolor: empty frame")
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(
            f"apply_recolor: expected H x W x 3 BGR frame, got shape "
            f"{frame_bgr.shape}"
        )

    h, w = frame_bgr.shape[:2]
    m = _coerce_mask(mask, (h, w))
    mass = float(m.sum())
    if mass < 1.0:
        # Empty region — nothing to recolor without hallucinating stats.
        return frame_bgr.copy()

    s = float(np.clip(strength, 0.0, 1.0))
    if s <= 0.0:
        return frame_bgr.copy()

    target = tuple(float(v) for v in target_lab)
    if len(target) != 3:
        raise ValueError(f"target_lab must be a 3-tuple, got {target!r}")

    src_lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    out = src_lab.copy()

    # Channel-wise mean shift, weighted by the soft mask.
    for ch in channels:
        ch_vals = src_lab[..., ch]
        mean = float((ch_vals * m).sum() / mass)
        shift = (target[ch] - mean) * s
        out[..., ch] = ch_vals + shift * m

    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)


def apply_skin_recolor(
    frame_bgr: np.ndarray,
    skin_mask: np.ndarray,
    theme: Theme,
    *,
    strength: float = 0.55,
) -> np.ndarray:
    """Shift the frame's skin region toward ``theme.skin_lab``.

    Thin convenience wrapper used by Phase 3 (skin-only recolor, no
    cel-shade yet). Phase 5 will call :func:`apply_target_lab_recolor`
    directly for each region (skin, lips, eyes, brows, hair).
    """
    return apply_target_lab_recolor(
        frame_bgr,
        skin_mask,
        theme.skin_lab,
        strength=strength,
        channels=(0, 1, 2),
    )


__all__ = [
    "Theme",
    "load_theme",
    "list_available_themes",
    "apply_target_lab_recolor",
    "apply_skin_recolor",
]
