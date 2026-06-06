"""Haar-cascade face detection + skin / head ROI masks.

The module covers two distinct steps of the pipeline (see ``spec.md`` §5):

* **Face detection — S08.** :class:`HaarFaceDetector` wraps OpenCV's
  pre-trained Viola–Jones cascades (frontal + profile) and returns the
  largest face as a :class:`Bbox` (or ``None``).
* **Skin segmentation — S06, S07.** :func:`skin_mask` thresholds the
  frame in HSV ∩ YCbCr, cleans the result with morphology, and keeps the
  largest connected component that overlaps the Haar bbox.
* **Head ROI — S04, S06.** :func:`head_ellipse_mask` paints a single
  ellipse centred on the bbox, extended upward to catch hair, sideways
  to catch ears, and slightly downward to catch the chin. The result is
  a binary mask; :func:`feather_mask` adds a Gaussian falloff so the
  composite at Phase 5 blends smoothly into the surrounding frame.

Why Haar and not a neural detector
----------------------------------
Per ``spec.md`` §4.1 the project is pinned to the IVP501 syllabus.
``S08`` covers the Viola–Jones cascade of boosted Haar-like features, so the
detector here is the textbook reference: integral image + rectangular
features + AdaBoost cascade, no learning at run-time, no GPU.

Strategy per frame
------------------
1. Run the frontal cascade.
2. On miss, run the profile cascade (left-facing profile is the only one
   the OpenCV model encodes — for right-facing faces we flip the frame
   horizontally and run the same cascade again).
3. When multiple candidates survive, keep the largest by area — the project
   processes one subject per clip and a tighter face usually wins over
   background patches.

Why HSV ∩ YCbCr for skin and not just one
-----------------------------------------
HSV captures hue + saturation cleanly (so neutral / desaturated skin tones
read as skin) but is brittle under shading: a shadowed cheek dips below
``V=60`` and drops out. YCbCr collapses the lighting dimension into ``Y``
and keeps the chroma ``Cb / Cr`` essentially illumination-invariant, so it
catches the shadowed pixels HSV missed — but it also fires on muted oranges
in the background. The intersection (Cheddad et al., 2009 and the bulk of
the skin-detection literature this module follows) keeps the agreement and
drops both kinds of false positives.

References
----------
Viola, P. & Jones, M. (2001). *Rapid Object Detection using a Boosted Cascade
of Simple Features.* CVPR. (Course slide S08.)

Cheddad, A. et al. (2009). *A new colour space for skin tone detection.*
ICIP. (Course slide S07.)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Public dataclass.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Bbox:
    """Axis-aligned face bounding box in pixel coordinates.

    Fields match the OpenCV cascade output order ``(x, y, w, h)`` so a Bbox
    can be unpacked directly into ``cv2.rectangle`` / slicing without
    reshuffling.
    """

    x: int
    y: int
    w: int
    h: int

    # Convenience accessors — used by the smoother (cx, cy, w, h) and by
    # any downstream ROI / ellipse computation in later phases.
    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    @property
    def area(self) -> int:
        return int(self.w) * int(self.h)

    def as_xywh(self) -> tuple[int, int, int, int]:
        return int(self.x), int(self.y), int(self.w), int(self.h)


# ---------------------------------------------------------------------------
# Detector.
# ---------------------------------------------------------------------------


def _largest(detections: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return the (x, y, w, h) with the largest area, or ``None`` if empty."""
    if detections is None or len(detections) == 0:
        return None
    # cv2 returns either a numpy array of shape (N, 4) or an empty tuple.
    arr = np.asarray(detections).reshape(-1, 4)
    if arr.size == 0:
        return None
    areas = arr[:, 2] * arr[:, 3]
    best = arr[int(np.argmax(areas))]
    return int(best[0]), int(best[1]), int(best[2]), int(best[3])


class HaarFaceDetector:
    """Largest-face detector built on Haar cascades (frontal + profile).

    Parameters
    ----------
    scale_factor:
        ``cv2.CascadeClassifier.detectMultiScale`` image-pyramid step.
        ``1.1`` (10 % per scale) is the textbook default — slower than
        ``1.2`` but catches in-between sizes that ``1.2`` skips.
    min_neighbors:
        Minimum overlapping detections a candidate must collect before it
        is reported. ``5`` is the working default — strict enough to drop
        textured-background false positives (the profile cascade loves
        firing on a knit sweater) while still tolerant enough to keep
        legitimate hits on small far-subject faces.
    min_size_ratio:
        Smallest face to look for, expressed as a fraction of frame
        height. ``0.05`` ≈ 5 % covers the standing-subject vertical
        clips in ``samples/input/`` where the face occupies ~5–9 % of
        the frame; smaller floors than this start to admit eye- and
        nose-sized noise blobs.

    Notes
    -----
    The class is stateless across frames apart from the cached cascades —
    temporal continuity is the smoother's job (see ``smooth.py``).
    """

    def __init__(
        self,
        *,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size_ratio: float = 0.05,
        # Skin-coverage gate — reject candidate bboxes whose HSV ∩ YCbCr
        # skin fraction inside the box is below this. Defends against
        # false positives from textured backgrounds (graffiti, brick,
        # foliage, knit fabric) when the frontal cascade misclassifies
        # a high-contrast patch as a face — especially likely when the
        # subject's eyes are closed and the cascade falls back to the
        # profile model. A real face has ≥ 30–50 % skin inside its bbox;
        # a graffiti / fabric / sweater patch usually sits below 20 %.
        # 0.30 is the cross-over that empirically drops chest / torso
        # false positives on ``samples/input/yellow.mp4`` without losing
        # legitimate hits on the other clips.
        min_skin_coverage: float = 0.30,
        verify_skin: bool = True,
        # `alt2` is the OpenCV-recommended frontal cascade for natural
        # video — better recall than `default` on tilted heads / closed
        # eyes, with comparable precision.
        frontal_cascade: str = "haarcascade_frontalface_alt2.xml",
        profile_cascade: str = "haarcascade_profileface.xml",
    ) -> None:
        self.scale_factor = float(scale_factor)
        self.min_neighbors = int(min_neighbors)
        self.min_size_ratio = float(min_size_ratio)
        self.min_skin_coverage = float(min_skin_coverage)
        self.verify_skin = bool(verify_skin)

        haar_dir = cv2.data.haarcascades
        self._frontal = self._load(haar_dir, frontal_cascade)
        self._profile = self._load(haar_dir, profile_cascade)

    @staticmethod
    def _load(haar_dir: str, name: str) -> cv2.CascadeClassifier:
        path = os.path.join(haar_dir, name)
        cascade = cv2.CascadeClassifier(path)
        if cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade: {path}")
        return cascade

    # -------- internal helpers --------

    def _detect_multi(
        self, cascade: cv2.CascadeClassifier, gray: np.ndarray, min_side: int
    ) -> np.ndarray:
        return cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(min_side, min_side),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

    def _skin_coverage(
        self, frame_bgr: np.ndarray, candidate: tuple[int, int, int, int]
    ) -> float:
        """Cheap HSV ∩ YCbCr skin fraction inside ``candidate`` bbox.

        Intentionally does NOT call the full :func:`skin_mask` (no morph,
        no connected components, no largest-CC filter) — this gate runs
        on every detector candidate so it has to be O(bbox_pixels) and
        allocation-free. The colour-threshold AND of HSV + YCbCr is the
        same primitive :func:`skin_mask` builds on, so a face that
        survives the full pipeline always passes the gate; a graffiti
        / brick / foliage false positive does not.
        """
        x, y, bw, bh = candidate
        H, W = frame_bgr.shape[:2]
        x0 = max(0, int(x))
        y0 = max(0, int(y))
        x1 = min(W, int(x) + int(bw))
        y1 = min(H, int(y) + int(bh))
        if x1 <= x0 or y1 <= y0:
            return 0.0
        roi = frame_bgr[y0:y1, x0:x1]
        hsv_mask = _skin_mask_hsv(roi)
        ycc_mask = _skin_mask_ycbcr(roi)
        skin = cv2.bitwise_and(hsv_mask, ycc_mask)
        area = float(skin.size)
        if area <= 0.0:
            return 0.0
        return float(np.count_nonzero(skin)) / area

    def _pick_verified(
        self,
        frame_bgr: np.ndarray,
        detections: np.ndarray | None,
        *,
        flip_x_max: int | None = None,
    ) -> tuple[int, int, int, int] | None:
        """Return the largest detection whose skin coverage passes the gate.

        Sorted by area (largest first) so the face wins when it shares a
        cascade hit list with a smaller background false positive.
        ``flip_x_max`` is the frame width — set it to convert flipped-
        profile coordinates back into the original image space *before*
        the skin check (the skin colour ROI must match the bbox).
        """
        if detections is None or len(detections) == 0:
            return None
        arr = np.asarray(detections).reshape(-1, 4)
        if arr.size == 0:
            return None
        # Sort by area descending. The largest plausible face usually
        # corresponds to the subject; if it fails the skin gate (e.g.
        # graffiti), we still get a shot at smaller candidates below.
        areas = arr[:, 2] * arr[:, 3]
        order = np.argsort(-areas)
        for idx in order:
            x, y, bw, bh = int(arr[idx, 0]), int(arr[idx, 1]), int(arr[idx, 2]), int(arr[idx, 3])
            if flip_x_max is not None:
                x_check = flip_x_max - x - bw
                cand = (x_check, y, bw, bh)
            else:
                cand = (x, y, bw, bh)
            if not self.verify_skin:
                return cand
            cov = self._skin_coverage(frame_bgr, cand)
            if cov >= self.min_skin_coverage:
                return cand
        return None

    # -------- public --------

    def detect(self, frame_bgr: np.ndarray) -> Bbox | None:
        """Detect the largest face in ``frame_bgr`` (BGR uint8 H×W×3).

        Returns ``None`` when neither cascade fires *and passes the skin-
        coverage gate*. Order of attempts: frontal → left-profile →
        right-profile (frame flipped). The first attempt that yields a
        skin-verified detection wins; we do not vote between cascades —
        flipping back and forth between frontal/profile bbox sizes is
        exactly the jitter the temporal smoother is meant to eat, so
        picking one source per frame keeps the input clean.

        Skin-coverage gate
        ------------------
        Each candidate bbox returned by ``detectMultiScale`` is checked
        against a cheap HSV ∩ YCbCr skin mask cropped to the bbox; the
        candidate is rejected when skin pixels cover less than
        ``min_skin_coverage`` of the bbox area. This kills the common
        eye-closed failure mode where the profile cascade fires on a
        textured background (graffiti / fabric / foliage) and produces
        a bogus bbox far from the actual head.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return None

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        h, w = gray.shape[:2]
        min_side = max(20, int(round(self.min_size_ratio * h)))

        # 1) Frontal cascade — covers ~front-facing and mild 3/4 views.
        best = self._pick_verified(
            frame_bgr, self._detect_multi(self._frontal, gray, min_side)
        )
        if best is not None:
            return Bbox(*best)

        # 2) Profile cascade — left-facing profile as the model is trained.
        best = self._pick_verified(
            frame_bgr, self._detect_multi(self._profile, gray, min_side)
        )
        if best is not None:
            return Bbox(*best)

        # 3) Profile cascade on a horizontally flipped frame — recovers
        #    right-facing profiles. We flip x back to the original
        #    coordinate system *inside* _pick_verified so the skin gate
        #    samples the correct ROI.
        flipped = cv2.flip(gray, 1)
        best = self._pick_verified(
            frame_bgr,
            self._detect_multi(self._profile, flipped, min_side),
            flip_x_max=w,
        )
        if best is not None:
            return Bbox(*best)

        return None


# ---------------------------------------------------------------------------
# Phase 2 — skin mask, head ellipse mask, feathering.
# ---------------------------------------------------------------------------


# Skin range constants — copied verbatim from ``plan.md`` §2 Phase 2 so the
# numbers in code, plan, and report stay locked together. Tuned for the
# samples in ``samples/input/`` (warm indoor lighting, light-to-medium tone);
# Phase 6 (theme authoring) does not re-tune these — themes only shift the
# *target* palette, not the *source* skin detector.

# HSV bounds. The hue split covers the two wrap-around halves of the warm
# skin band (reds near 0, magentas near 179 in OpenCV's 0–179 H scale). The
# saturation lower bound rejects gray walls and unshaven gray-blue regions;
# the upper bound rejects very saturated lips / clothing. The V lower bound
# rejects deep shadow / hair.
_HSV_H_LO_A, _HSV_H_HI_A = 0, 25
_HSV_H_LO_B, _HSV_H_HI_B = 160, 179
_HSV_S_LO, _HSV_S_HI = 40, 200
_HSV_V_LO, _HSV_V_HI = 60, 255

# YCbCr bounds. The classical Cb / Cr "skin cluster" sits roughly at
# (Cb≈100, Cr≈155); the box around it (Cb 85–135, Cr 135–180) is a
# standard textbook range. The Y lower bound rejects pure black.
_YCBCR_Y_LO, _YCBCR_Y_HI = 60, 255
_YCBCR_CB_LO, _YCBCR_CB_HI = 85, 135
_YCBCR_CR_LO, _YCBCR_CR_HI = 135, 180


def _skin_mask_hsv(frame_bgr: np.ndarray) -> np.ndarray:
    """Binary skin mask from HSV thresholding (uint8, {0, 255})."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    # Two hue ranges — OpenCV uses H ∈ [0, 179], so warm skin straddles 0 / 179
    # and we need an OR of the two boxes.
    mask_a = cv2.inRange(
        hsv,
        np.array([_HSV_H_LO_A, _HSV_S_LO, _HSV_V_LO], dtype=np.uint8),
        np.array([_HSV_H_HI_A, _HSV_S_HI, _HSV_V_HI], dtype=np.uint8),
    )
    mask_b = cv2.inRange(
        hsv,
        np.array([_HSV_H_LO_B, _HSV_S_LO, _HSV_V_LO], dtype=np.uint8),
        np.array([_HSV_H_HI_B, _HSV_S_HI, _HSV_V_HI], dtype=np.uint8),
    )
    return cv2.bitwise_or(mask_a, mask_b)


def _skin_mask_ycbcr(frame_bgr: np.ndarray) -> np.ndarray:
    """Binary skin mask from YCbCr thresholding (uint8, {0, 255})."""
    ycc = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    # OpenCV's YCrCb channel order is Y, Cr, Cb (not Y, Cb, Cr).
    y = ycc[..., 0]
    cr = ycc[..., 1]
    cb = ycc[..., 2]
    mask = (
        (y >= _YCBCR_Y_LO) & (y <= _YCBCR_Y_HI)
        & (cb >= _YCBCR_CB_LO) & (cb <= _YCBCR_CB_HI)
        & (cr >= _YCBCR_CR_LO) & (cr <= _YCBCR_CR_HI)
    )
    return (mask.astype(np.uint8)) * 255


def _largest_cc_intersecting_bbox(
    mask_u8: np.ndarray, bbox: "Bbox | None"
) -> np.ndarray:
    """Keep the single largest connected component that overlaps ``bbox``.

    When ``bbox`` is ``None`` (e.g. on a brief detection miss the caller still
    wants the raw skin mask), return the largest component overall — this
    is the same behaviour the standalone skin-mask literature uses.

    The mask is treated as 8-connected. Components are filtered by *area
    inside the bbox*: a tiny background region whose convex hull happens to
    clip the bbox corner gets dropped because its actual pixel count inside
    the bbox is small, while the face component dominates.
    """
    if mask_u8.size == 0:
        return mask_u8.copy()

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8
    )
    if num_labels <= 1:
        return np.zeros_like(mask_u8)

    if bbox is None:
        # No anchor — fall back to the largest component overall, excluding
        # the background label 0.
        areas = stats[1:, cv2.CC_STAT_AREA]
        best = int(np.argmax(areas)) + 1
        return ((labels == best).astype(np.uint8)) * 255

    x0, y0, w, h = bbox.as_xywh()
    H, W = mask_u8.shape[:2]
    x1 = max(0, min(W, x0 + w))
    y1 = max(0, min(H, y0 + h))
    x0 = max(0, min(W, x0))
    y0 = max(0, min(H, y0))
    if x1 <= x0 or y1 <= y0:
        return np.zeros_like(mask_u8)

    roi_labels = labels[y0:y1, x0:x1]
    overlap_ids = np.unique(roi_labels)
    overlap_ids = overlap_ids[overlap_ids != 0]
    if overlap_ids.size == 0:
        return np.zeros_like(mask_u8)

    # Pick the component whose pixel count *inside the bbox* is largest.
    # This biases toward the face blob even when a much larger arm/torso
    # blob exists in the same frame.
    best_id = -1
    best_count = -1
    for lid in overlap_ids:
        count = int(np.count_nonzero(roi_labels == lid))
        if count > best_count:
            best_count = count
            best_id = int(lid)

    return ((labels == best_id).astype(np.uint8)) * 255


def skin_mask(
    frame_bgr: np.ndarray,
    bbox: "Bbox | None" = None,
    *,
    close_ksize: int = 5,
    open_ksize: int = 3,
) -> np.ndarray:
    """Binary skin mask for ``frame_bgr`` ({0, 255} uint8, HxW).

    Pipeline (matches ``plan.md`` §2 Phase 2):

    1. ``mask_HSV`` ∧ ``mask_YCbCr`` (pixel-wise AND).
    2. ``MORPH_CLOSE`` with a 5×5 rectangle → fills small holes inside skin
       regions (eye, mouth, hair-line gaps).
    3. ``MORPH_OPEN`` with a 3×3 rectangle → removes isolated false-positive
       noise (single-pixel hair / lip blobs left after the close).
    4. Connected components — keep the single component whose overlap with
       ``bbox`` is largest. The face blob wins even when an arm or hand is
       also visible.

    When ``bbox`` is ``None``, step 4 falls back to the globally largest
    component. This keeps the helper usable in standalone tests / notebooks
    where no detector has run.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("skin_mask: empty frame")
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(
            f"skin_mask: expected H x W x 3 BGR frame, got shape "
            f"{frame_bgr.shape}"
        )

    hsv_mask = _skin_mask_hsv(frame_bgr)
    ycc_mask = _skin_mask_ycbcr(frame_bgr)
    mask = cv2.bitwise_and(hsv_mask, ycc_mask)

    # Order matters: CLOSE first to seal interior holes (eyes / mouth) before
    # OPEN strips off speckle. Doing OPEN first would erode the seal points
    # before they have a chance to fuse.
    if close_ksize > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (close_ksize, close_ksize)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if open_ksize > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (open_ksize, open_ksize)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return _largest_cc_intersecting_bbox(mask, bbox)


def head_ellipse_mask(
    bbox: "Bbox",
    image_size: tuple[int, int],
    *,
    top_extend: float = 0.60,
    side_extend: float = 0.20,
    bottom_extend: float = 0.05,
) -> np.ndarray:
    """Binary head-region mask built from a single ellipse around ``bbox``.

    The face bbox covers the face but not the hair — extending it upward
    (default ``top_extend=0.60`` → +60 % of bbox height) approximates the
    top of the skull and most short / medium hair. Sideways extension
    (default ``side_extend=0.20`` per side) catches ears and side-burns,
    and a small bottom extension (``bottom_extend=0.05``) covers the chin
    when the cascade clips it tight.

    Parameters
    ----------
    bbox:
        Face bounding box (typically the smoother's output for the current
        frame).
    image_size:
        ``(width, height)`` of the frame the mask is drawn on.
    top_extend, side_extend, bottom_extend:
        Fractions of ``bbox`` height / width by which the ellipse extends
        past each side of the bbox. Defaults match ``plan.md`` §2 Phase 2.

    Returns
    -------
    Binary uint8 mask of shape ``(height, width)`` — ``255`` inside the
    ellipse, ``0`` elsewhere. The caller may pass this through
    :func:`feather_mask` to obtain a soft-falloff alpha.
    """
    width, height = int(image_size[0]), int(image_size[1])
    if width <= 0 or height <= 0:
        raise ValueError(
            f"head_ellipse_mask: bad image_size {image_size!r}"
        )

    x, y, w, h = bbox.as_xywh()
    top = float(max(0.0, top_extend))
    side = float(max(0.0, side_extend))
    bot = float(max(0.0, bottom_extend))

    # Extended rectangle.
    x_lo = x - side * w
    x_hi = x + w + side * w
    y_lo = y - top * h
    y_hi = y + h + bot * h

    cx = (x_lo + x_hi) * 0.5
    cy = (y_lo + y_hi) * 0.5
    # Semi-axes: width / 2 along x, height / 2 along y.
    ax = max(1.0, (x_hi - x_lo) * 0.5)
    ay = max(1.0, (y_hi - y_lo) * 0.5)

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(
        mask,
        center=(int(round(cx)), int(round(cy))),
        axes=(int(round(ax)), int(round(ay))),
        angle=0,
        startAngle=0,
        endAngle=360,
        color=255,
        thickness=-1,
    )
    return mask


def face_oval_mask(
    bbox: "Bbox",
    image_size: tuple[int, int],
    *,
    width_shrink: float = 0.05,
    height_shrink: float = 0.0,
) -> np.ndarray:
    """Binary mask of the face oval inscribed in ``bbox`` (no hair extension).

    Phase 5 needs to distinguish *face* pixels (skin, eyes, lips, brows)
    from *hair* pixels. :func:`head_ellipse_mask` already gives us the
    whole head ROI (face + hair); subtracting this face-only ellipse from
    that head ellipse yields the hair ring around the silhouette.

    The Haar bbox is a tight rectangle around the face — an inscribed
    ellipse approximates the actual face oval well enough for per-region
    color transfer. The two small ``*_shrink`` knobs let the caller pull
    the oval slightly inside the bbox (defaults: 5 % width inset, no
    height inset) so the hair ring picks up the temples without the face
    oval bleeding into the surrounding hair.
    """
    width, height = int(image_size[0]), int(image_size[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"face_oval_mask: bad image_size {image_size!r}")

    x, y, w, h = bbox.as_xywh()
    sx = float(np.clip(width_shrink, 0.0, 0.49))
    sy = float(np.clip(height_shrink, 0.0, 0.49))
    cx = x + w / 2.0
    cy = y + h / 2.0
    ax = max(1.0, (w * (1.0 - 2.0 * sx)) * 0.5)
    ay = max(1.0, (h * (1.0 - 2.0 * sy)) * 0.5)

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(
        mask,
        center=(int(round(cx)), int(round(cy))),
        axes=(int(round(ax)), int(round(ay))),
        angle=0,
        startAngle=0,
        endAngle=360,
        color=255,
        thickness=-1,
    )
    return mask


def hair_ring_mask(
    bbox: "Bbox",
    image_size: tuple[int, int],
    *,
    top_extend: float = 0.60,
    side_extend: float = 0.20,
    bottom_extend: float = 0.05,
    face_width_shrink: float = 0.05,
) -> np.ndarray:
    """Binary mask covering the hair region around the head silhouette.

    Defined geometrically as ``head_ellipse_mask − face_oval_mask`` — the
    extended head ellipse (which catches the hair on top and beside the
    face) minus the face oval (which is skin / lips / eyes / brows). The
    leftover ring is where hair lives.

    Phase 5 uses this to drive a chroma-only Reinhard transfer toward
    ``theme.hair_lab`` without touching the user's hair brightness.
    """
    head = head_ellipse_mask(
        bbox,
        image_size,
        top_extend=top_extend,
        side_extend=side_extend,
        bottom_extend=bottom_extend,
    )
    face = face_oval_mask(
        bbox, image_size, width_shrink=face_width_shrink
    )
    # Subtract face from head; both are {0, 255} so we use bitwise ops.
    return cv2.bitwise_and(head, cv2.bitwise_not(face))


# ---------------------------------------------------------------------------
# Per-feature soft masks driven by the bbox alone (no landmarks).
# ---------------------------------------------------------------------------
#
# Anthropometric layout of an upright face inside a tight Haar bbox:
#
#   y / h ≈ 0.40   →  brow line
#   y / h ≈ 0.45   →  pupil line
#   y / h ≈ 0.75   →  mouth line
#
# We render each feature as a horizontal band (a soft 1-D Gaussian along
# y, masked to the face oval along x). The bands are intentionally short
# in height — they only need to weight the per-region color transfer, not
# segment the feature precisely. Slight overlap is OK; the Reinhard
# transfer per region averages out any soft-edge contributions.


def _gauss_band(
    height: int,
    centre_y: float,
    sigma_y: float,
) -> np.ndarray:
    """1-D Gaussian along the y axis, peak = 1, shape (height,)."""
    ys = np.arange(height, dtype=np.float32)
    return np.exp(-0.5 * ((ys - centre_y) / max(1e-3, sigma_y)) ** 2)


def feature_strip_masks(
    bbox: "Bbox",
    image_size: tuple[int, int],
    *,
    lips_y_ratio: float = 0.75,
    lips_height_ratio: float = 0.10,
    eyes_y_ratio: float = 0.45,
    eyes_height_ratio: float = 0.08,
    brows_y_ratio: float = 0.40,
    brows_height_ratio: float = 0.05,
    face_width_shrink: float = 0.05,
) -> dict[str, np.ndarray]:
    """Soft horizontal-strip masks for lips / eyes / brows inside ``bbox``.

    Each returned mask is ``float32`` in ``[0, 1]`` with the same
    ``(height, width)`` shape as the frame:

      * A vertical Gaussian centred at ``*_y_ratio · bbox_h`` and σ ≈
        ``*_height_ratio · bbox_h / 2``. This produces a soft band that
        falls off smoothly above and below the feature line — exactly
        what the Reinhard transfer wants as a soft weight.
      * Constrained to the face oval (gated by :func:`face_oval_mask`)
        so the strip does not extend past the cheeks into the hair /
        background.

    Because the bands have no horizontal segmentation, the eye band
    covers both eyes plus the nose bridge between them, and the lip
    band covers the entire mouth + chin shadow. That is acceptable for
    a *color* transfer: the eye / lip regions of the theme have a
    similar shift to the chin / bridge, and the cel-shade pass has
    already pulled neighbouring pixels into the same band, so the soft
    extra coverage just makes the recolor feel less surgical.
    """
    width, height = int(image_size[0]), int(image_size[1])
    if width <= 0 or height <= 0:
        raise ValueError(
            f"feature_strip_masks: bad image_size {image_size!r}"
        )

    x, y, w, h = bbox.as_xywh()

    oval = face_oval_mask(
        bbox, image_size, width_shrink=face_width_shrink
    ).astype(np.float32) / 255.0

    def band(centre_ratio: float, height_ratio: float) -> np.ndarray:
        centre_y = y + centre_ratio * h
        sigma_y = max(1.0, height_ratio * h * 0.5)
        col = _gauss_band(height, centre_y, sigma_y)
        # Broadcast the 1-D column to a 2-D mask (constant along x).
        band2d = np.broadcast_to(col[:, None], (height, width))
        return (band2d * oval).astype(np.float32)

    return {
        "lips": band(lips_y_ratio, lips_height_ratio),
        "eyes": band(eyes_y_ratio, eyes_height_ratio),
        "brows": band(brows_y_ratio, brows_height_ratio),
    }


def feather_mask(mask_u8: np.ndarray, feather_px: int = 41) -> np.ndarray:
    """Gaussian-feather a binary mask into a float32 alpha in ``[0, 1]``.

    Larger ``feather_px`` → softer boundary, smoother composite at the cost
    of leaking style outside the head silhouette. ``feather_px=41`` (the
    default from ``RenderConfig.head_feather_px``) corresponds to a
    σ ≈ 6.7 px Gaussian and gives a ~20 px transition band that is wide
    enough to hide the ellipse rim but tight enough to keep the cel-shade
    contained to the head area.

    ``feather_px`` is coerced to an odd integer (OpenCV requirement).
    Non-positive values short-circuit to a hard {0, 1} conversion of the
    input mask — useful for QA panels where the raw silhouette should
    be visible.
    """
    if feather_px is None or feather_px <= 0:
        return (mask_u8 > 0).astype(np.float32)
    k = int(feather_px) | 1  # round up to next odd integer
    blurred = cv2.GaussianBlur(mask_u8, (k, k), 0)
    return blurred.astype(np.float32) / 255.0


__all__ = [
    "Bbox",
    "HaarFaceDetector",
    "skin_mask",
    "head_ellipse_mask",
    "face_oval_mask",
    "hair_ring_mask",
    "feature_strip_masks",
    "feather_mask",
]
