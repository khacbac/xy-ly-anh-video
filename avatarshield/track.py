"""Classical correlation-filter tracker — bridges Haar detection misses.

Why a tracker on top of Haar
----------------------------
The Viola–Jones cascade is single-frame: every frame is detected from
scratch. On the dance / motion-blur / small-face clips that recall
drops below 15 %, so the head ellipse only paints onto ~1 in 7 frames
and the rest of the video sees the un-stylized original. A correlation
filter trained on the last Haar hit can carry the bbox forward through
the missing frames by aligning local image features against the patch
captured at init time — no DNN, no per-clip re-training.

This module wraps OpenCV's built-in trackers; we default to CSRT
(Lukežič et al., 2017 — channel & spatial reliability, built on HOG +
colour names + a spatial reliability map) and expose KCF (Henriques
et al., 2015 — kernelized correlation filter with circulant-matrix
factorisation) as a faster fallback. Both sit squarely inside the
IVP501 syllabus's "feature-based tracking" block (S08); neither
trains parameters at run-time after ``init``.

Lifecycle
---------
``init(frame, bbox)`` is called on every fresh Haar accept. ``update``
is called on each miss frame between Haar hits and returns a predicted
``Bbox`` (or ``None`` if the tracker has internally lost track). The
caller is responsible for capping ``age`` — correlation filters drift
after roughly a second of motion blur, so a 15–30 frame cap is the
standard practice.

References
----------
Lukežič, A., Vojíř, T., Čehovin Zajc, L., Matas, J. & Kristan, M.
(2017). *Discriminative Correlation Filter with Channel and Spatial
Reliability.* CVPR.

Henriques, J. F., Caseiro, R., Martins, P. & Batista, J. (2015).
*High-Speed Tracking with Kernelized Correlation Filters.* IEEE PAMI.
"""

from __future__ import annotations

import cv2
import numpy as np

from .detect import Bbox


def _resolve_tracker_factory(name: str):
    """Find an OpenCV factory for ``TrackerCSRT`` / ``TrackerKCF``.

    The CSRT / KCF trackers live in the ``tracking`` contrib module, so
    they are only present in ``opencv-contrib-python`` (not the slimmer
    ``opencv-python`` wheel). The factory name has also moved between
    releases: older builds expose ``cv2.TrackerCSRT_create`` at the top
    level, while 4.5.1+ relocated it to ``cv2.legacy.TrackerCSRT_create``.
    Probe both so the pipeline runs across the OpenCV versions we see on
    macOS and Windows.
    """
    candidates = (
        getattr(cv2, f"Tracker{name}_create", None),
        getattr(getattr(cv2, "legacy", None), f"Tracker{name}_create", None),
    )
    for factory in candidates:
        if factory is not None:
            return factory
    raise RuntimeError(
        f"OpenCV build is missing Tracker{name}_create. The CSRT/KCF "
        "trackers ship only in opencv-contrib-python — install it with "
        "`pip install opencv-contrib-python` (uninstall opencv-python first)."
    )


class BoxTracker:
    """Correlation-filter tracker wrapper (CSRT or KCF).

    Stateful: a ``BoxTracker`` instance follows one head across a clip
    and is re-initialised from each fresh Haar detection. ``age``
    counts the number of ``update`` calls since the last ``init`` — the
    caller uses it to bound drift (the tracker becomes unreliable
    after ~1 second of motion blur).

    Parameters
    ----------
    kind:
        ``"csrt"`` (default) or ``"kcf"``. CSRT is more accurate but
        ~3× slower per frame; KCF is faster but loses lock on fast
        rotation.
    """

    def __init__(self, kind: str = "csrt") -> None:
        self.kind = kind.lower()
        if self.kind not in {"csrt", "kcf"}:
            raise ValueError(f"unknown tracker kind: {kind!r}")
        self._tracker: cv2.Tracker | None = None
        self._age: int = 0

    def _new_tracker(self) -> cv2.Tracker:
        factory = _resolve_tracker_factory("CSRT" if self.kind == "csrt" else "KCF")
        return factory()

    @property
    def active(self) -> bool:
        """True if the tracker has been initialised and is following a bbox."""
        return self._tracker is not None

    @property
    def age(self) -> int:
        """Number of ``update`` calls since the last ``init``."""
        return self._age

    def reset(self) -> None:
        """Drop tracker state. Next ``init`` will create a fresh tracker."""
        self._tracker = None
        self._age = 0

    def init(self, frame_bgr: np.ndarray, bbox: Bbox) -> None:
        """Re-anchor the tracker on ``bbox`` inside ``frame_bgr``.

        Called whenever Haar lands a fresh accept — we throw away any
        previous tracker state so the filter is locked to the gated
        face position rather than continuing to follow a drifting
        prediction.
        """
        self._tracker = self._new_tracker()
        x, y, w, h = bbox.as_xywh()
        # OpenCV trackers expect a tuple of ints / floats, not a Bbox.
        self._tracker.init(frame_bgr, (int(x), int(y), int(w), int(h)))
        self._age = 0

    def update(self, frame_bgr: np.ndarray) -> Bbox | None:
        """Ask the tracker for the predicted bbox in ``frame_bgr``.

        Returns ``None`` when the tracker has not been initialised or
        when its own confidence has collapsed (OpenCV reports
        ``ok=False``). On a successful step ``age`` is incremented
        so the caller can drop the tracker after a fixed budget.
        """
        if self._tracker is None:
            return None
        ok, box = self._tracker.update(frame_bgr)
        if not ok:
            self.reset()
            return None
        x, y, w, h = box
        self._age += 1
        return Bbox(int(round(x)), int(round(y)), int(round(w)), int(round(h)))


__all__ = ["BoxTracker"]
