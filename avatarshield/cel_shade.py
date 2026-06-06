"""Cell-shading stylization (numpy + OpenCV only — no neural networks).

The goal is to make a real face read as a hand-painted anime cel: a small
set of flat color regions (skin shadow / midtone / highlight, plus hair
blocks) outlined by clean line work. This is the visual signature of
2D animation cels — gradients are *banded* into discrete steps rather
than smoothly interpolated.

Pipeline (each step is one cv2/numpy primitive — easy to step through):

    1. Iterated bilateral filter
       Smooths skin texture and small details while preserving silhouette
       edges. Running 2–4 passes on a *downsampled* image is the same
       trick `cv2.stylization()` uses internally — gives real flat regions
       instead of merely "soft" skin.

    2. BGR → Lab color space
       L (luminance) carries shading; a/b carry hue. Splitting them lets
       us posterize shading independently from hue — exactly how anime
       artists build cels (line art + flat fills + a couple of shadow
       tones).

    3. Posterize L via K-means
       Cluster the L channel into K levels (default 3 = shadow / mid /
       highlight). Each pixel's L is replaced by its cluster centroid.
       This is the bit that turns smooth gradients into discrete cells.

    4. Posterize a/b lightly (optional)
       Same trick on chroma but with more levels (8+). Skip it if you
       want skin tone to keep its natural variation.

    5. Lab → BGR

    6. XDoG (eXtended Difference of Gaussians) edge overlay
       Cleaner, more controllable line art than plain DoG. The classical
       Winnemöller (2012) formulation. Multiplied on top of the flat
       fills to darken edges.

All ops are vectorized OpenCV primitives — no per-pixel Python loops.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Temporal stability — EMA-smoothing of K-means cluster centers across frames.
# ---------------------------------------------------------------------------


@dataclass
class TemporalState:
    """Carries EMA-smoothed K-means centers between frames.

    Why this exists:
      cv2.kmeans with KMEANS_PP_CENTERS picks initial centers randomly,
      so frame-to-frame the cluster centers can jitter even when the
      underlying image barely changed. That jitter is what causes the
      "boiling" / flicker you see when you naively cel-shade a video
      with K-means posterize. We fix it by:

        1. Sorting centers (so center[0] is always the darkest band,
           center[1] the next, etc.) — kills label permutation flicker.
        2. EMA-blending each frame's sorted centers with the previous
           frame's stored centers (``new = ema·prev + (1-ema)·curr``).
        3. Re-quantizing the channel against the smoothed centers
           (simple nearest-center assignment), so the *output* uses
           stable band values.

    The state is keyed so multiple K-means calls in the same pipeline
    (whole-frame L, whole-frame a, whole-frame b, hair L, ...) each get
    their own series of smoothed centers.
    """

    ema: float = 0.8
    centers: dict[str, np.ndarray] = field(default_factory=dict)


def _temporal_centers(
    flat_data: np.ndarray,
    k: int,
    state: "TemporalState | None",
    key: str,
    *,
    attempts: int = 3,
    max_iter: int = 10,
    eps: float = 1.0,
) -> np.ndarray:
    """K-means and (optionally) EMA-blend the sorted centers via ``state``."""
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        int(max_iter),
        float(eps),
    )
    _, _, raw = cv2.kmeans(
        flat_data.astype(np.float32),
        int(k),
        None,
        criteria,
        int(attempts),
        cv2.KMEANS_PP_CENTERS,
    )
    centers = np.sort(raw.flatten())
    if state is not None:
        prev = state.centers.get(key)
        if prev is not None and prev.shape[0] == k:
            centers = prev * state.ema + centers * (1.0 - state.ema)
        state.centers[key] = centers
    return centers


def _quantize_with_centers(
    values: np.ndarray, centers: np.ndarray
) -> np.ndarray:
    """Replace each value with its nearest center (flat 1-D output)."""
    diffs = np.abs(values.reshape(-1, 1) - centers.reshape(1, -1))
    labels = diffs.argmin(axis=1)
    return centers[labels]


# ---------------------------------------------------------------------------
# Step 1 — iterated bilateral smoothing.
# ---------------------------------------------------------------------------


def iterated_bilateral(
    bgr: np.ndarray,
    *,
    passes: int = 3,
    d: int = 9,
    sigma_color: float = 50.0,
    sigma_space: float = 7.0,
    downsample: int = 2,
) -> np.ndarray:
    """Run bilateral filter ``passes`` times on a downsampled image.

    Bilateral filter averages nearby pixels but skips ones that differ
    too much in *color* — that's what preserves edges. A single pass
    smooths a little; 3+ passes turn skin into a true flat region.

    Working on a half-resolution copy is the standard speed trick:
    bilateral filter cost is ~O(d² · N), so halving N quadruples speed
    per pass with no visible quality loss after the upscale.
    """
    if downsample > 1:
        small = cv2.resize(
            bgr,
            (bgr.shape[1] // downsample, bgr.shape[0] // downsample),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = bgr

    for _ in range(max(1, passes)):
        small = cv2.bilateralFilter(small, d, sigma_color, sigma_space)

    if downsample > 1:
        return cv2.resize(
            small,
            (bgr.shape[1], bgr.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    return small


# ---------------------------------------------------------------------------
# Step 3 — K-means posterize on a single channel.
# ---------------------------------------------------------------------------


def kmeans_quantize_channel(
    channel: np.ndarray,
    k: int,
    *,
    state: "TemporalState | None" = None,
    key: str = "default",
    attempts: int = 3,
    max_iter: int = 10,
    eps: float = 1.0,
) -> np.ndarray:
    """Replace each pixel with its K-means cluster centroid (1-D channel).

    When ``state`` is given, the cluster centers are EMA-smoothed across
    successive calls keyed by ``key`` — that's what stops the cel bands
    from flickering frame-to-frame in a video render. Use the same
    ``state`` instance across the whole render loop and give each K-means
    site a unique ``key`` (e.g. ``"head_L"`` vs ``"body_L"``).

    Why K-means and not uniform quantization? K-means picks band
    boundaries from the data — clusters land where pixels actually
    concentrate (skin midtone, shadow under the chin, highlight on the
    nose). Uniform quantization slices through skin tones at fixed cut
    points and creates ugly banding.
    """
    if k <= 1:
        return channel
    flat = channel.reshape(-1, 1).astype(np.float32)
    centers = _temporal_centers(
        flat, k, state, key, attempts=attempts, max_iter=max_iter, eps=eps
    )
    quantized = _quantize_with_centers(flat, centers).reshape(channel.shape)
    return np.clip(quantized, 0, 255).astype(channel.dtype)


# ---------------------------------------------------------------------------
# Step 6 — XDoG line art.
# ---------------------------------------------------------------------------


def xdog(
    gray: np.ndarray,
    *,
    sigma: float = 0.8,
    k: float = 1.6,
    p: float = 30.0,
    eps: float = 0.5,
    phi: float = 10.0,
) -> np.ndarray:
    """Extended Difference of Gaussians line art (Winnemöller 2012).

    Formula
    -------
        D(x) = (1 + p)·G_σ(x) - p·G_{kσ}(x)
        T(x) = 1                      if D(x) >= eps
             = 1 + tanh(phi·(D - eps)) otherwise

    Intuition
    ---------
    Plain DoG is ``G_σ - G_{kσ}`` — a band-pass that responds to edges.
    XDoG adds two knobs:
      * ``p`` boosts the response (sharper, darker lines).
      * the soft threshold ``tanh(phi·(D - eps))`` lets you tune the
        line vs. flat-region split smoothly instead of a hard cutoff.
        Inside a strong edge, T → 0 (very dark line). In a flat region,
        T = 1 (no darkening). The transition between them is what makes
        XDoG lines feel "drawn" instead of speckled.

    Output is in [0, 1] — multiply against your color image to overlay.
    """
    g = gray.astype(np.float32) / 255.0
    g_sigma = cv2.GaussianBlur(g, (0, 0), sigma)
    g_ksigma = cv2.GaussianBlur(g, (0, 0), sigma * k)
    d = (1.0 + p) * g_sigma - p * g_ksigma
    # Where d is "edge-like" (high value), output 1 (no darkening).
    # Where d is "line-like" (low value), output ramps toward 0.
    edges = np.where(
        d >= eps,
        1.0,
        1.0 + np.tanh(phi * (d - eps)),
    )
    return np.clip(edges, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Pipeline composer.
# ---------------------------------------------------------------------------


def cel_shade(
    frame_bgr: np.ndarray,
    *,
    # Step 1 — flatten
    bilateral_passes: int = 3,
    bilateral_d: int = 9,
    bilateral_sigma_color: float = 60.0,
    bilateral_sigma_space: float = 7.0,
    bilateral_downsample: int = 2,
    # Step 3/4 — posterize
    luminance_levels: int = 3,
    chroma_levels: int = 0,
    # Step 5 — saturation
    saturation_boost: float = 1.25,
    # Step 6 — line art
    edge_strength: float = 0.85,
    xdog_sigma: float = 0.8,
    xdog_k: float = 1.6,
    xdog_p: float = 30.0,
    xdog_eps: float = 0.5,
    xdog_phi: float = 10.0,
    # Temporal stability — pass the same TemporalState across frames in
    # a video render. ``state_prefix`` namespaces the K-means keys so
    # multiple cel_shade calls (e.g. body + head) don't share centers.
    state: "TemporalState | None" = None,
    state_prefix: str = "frame",
) -> np.ndarray:
    """Apply the full cel-shade pipeline to a BGR uint8 frame.

    The defaults are tuned for a 720p portrait clip. Keep ``chroma_levels=0``
    on first try — quantizing chroma is the most opinionated step and can
    push skin tones into uncanny territory; turn it on (8–16) only if you
    want a flatter "comic book" feel.
    """
    if frame_bgr.dtype != np.uint8:
        frame_bgr = np.clip(frame_bgr, 0, 255).astype(np.uint8)

    # 1) Flatten textures, keep silhouette edges.
    flat = iterated_bilateral(
        frame_bgr,
        passes=bilateral_passes,
        d=bilateral_d,
        sigma_color=bilateral_sigma_color,
        sigma_space=bilateral_sigma_space,
        downsample=bilateral_downsample,
    )

    # 2) BGR → Lab so we can posterize shading separately from hue.
    lab = cv2.cvtColor(flat, cv2.COLOR_BGR2LAB)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    # 3) Posterize L into discrete shading bands (cel shading core).
    L_quant = kmeans_quantize_channel(
        L, k=luminance_levels, state=state, key=f"{state_prefix}_L"
    )

    # 4) Optional chroma posterize. Lighter touch — more levels.
    if chroma_levels > 1:
        a_quant = kmeans_quantize_channel(
            a, k=chroma_levels, state=state, key=f"{state_prefix}_a"
        )
        b_quant = kmeans_quantize_channel(
            b, k=chroma_levels, state=state, key=f"{state_prefix}_b"
        )
    else:
        a_quant, b_quant = a, b

    lab_quant = np.stack([L_quant, a_quant, b_quant], axis=-1).astype(np.uint8)

    # 5) Back to BGR, then saturation lift in HSV for the "anime palette" feel.
    posterized = cv2.cvtColor(lab_quant, cv2.COLOR_LAB2BGR)
    if saturation_boost != 1.0:
        hsv = cv2.cvtColor(posterized, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * float(saturation_boost), 0, 255)
        posterized = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    if edge_strength <= 0.0:
        return posterized

    # 6) XDoG line art overlay. Run on the *original* gray so we don't
    #    miss edges that the bilateral smoothing flattened.
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)  # cheap denoise so XDoG isn't speckly
    edge_mask = xdog(
        gray,
        sigma=xdog_sigma,
        k=xdog_k,
        p=xdog_p,
        eps=xdog_eps,
        phi=xdog_phi,
    )
    # edge_mask = 1 in flat regions, → 0 along lines.
    # Modulate strength: at strength=0 → no darkening; at strength=1 → full.
    line_factor = (1.0 - edge_strength) + edge_strength * edge_mask
    out = posterized.astype(np.float32) * line_factor[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Anime-face boosters — small per-region effects that read as "anime" beyond
# what whole-frame cel-shading can do. Each operates on a soft mask in [0,1]
# so the effect fades smoothly at boundaries instead of cutting hard.
# ---------------------------------------------------------------------------


def lift_lightness(
    bgr: np.ndarray,
    mask: np.ndarray,
    amount: float,
) -> np.ndarray:
    """Add ``amount`` (0–100) to Lab L inside ``mask`` (soft, in [0,1]).

    Real skin lives around L ≈ 60–75; anime skin sits noticeably higher
    (L ≈ 80–90) — that "porcelain glow" is a big part of why faces read
    as drawn rather than photographed. We just shift L upward where the
    mask says so; chroma is untouched so the hue still belongs to the
    person.
    """
    if amount == 0.0:
        return bgr
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] = np.clip(lab[..., 0] + amount * mask, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def skin_palette_snap(
    bgr: np.ndarray,
    skin_mask: np.ndarray,
    *,
    target_L: float = 215.0,
    target_a: float = 140.0,
    target_b: float = 145.0,
    strength: float = 0.55,
) -> np.ndarray:
    """Pull skin Lab values toward an anime-skin target (porcelain / peachy).

    OpenCV's Lab uses ``L ∈ [0, 255]`` (scaled from [0, 100]) and ``a, b``
    centered at 128 (a > 128 → red, b > 128 → yellow). Anime skin sits
    roughly at L ≈ 210–220, a ≈ 138–145, b ≈ 142–150 — bright, slightly
    warm, slightly yellow. Just lifting L makes faces "washed-out
    photograph"; shifting a/b toward warm hues is what unlocks the
    "porcelain doll" feel.

    Implementation is a linear blend per pixel inside ``skin_mask``:
        new = old + (target - old) · strength · mask

    Keep ``strength`` ≤ 0.7 — full snap erases the subject's natural
    undertone and all skin tones converge to one anime color.
    """
    if strength <= 0.0:
        return bgr
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    weight = (skin_mask * strength).astype(np.float32)
    for ch, target in enumerate((target_L, target_a, target_b)):
        lab[..., ch] = lab[..., ch] + (target - lab[..., ch]) * weight
    return cv2.cvtColor(
        np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR
    )


def boost_saturation_in_mask(
    bgr: np.ndarray,
    mask: np.ndarray,
    multiplier: float,
) -> np.ndarray:
    """Multiply HSV S by ``multiplier`` inside ``mask``, linear blend at edges.

    Used for the "eyes/lips/brows are way more saturated than skin" anime
    convention. Plain whole-frame saturation boost flattens that contrast
    because skin gets boosted too; we want a *differential* — punch the
    iris + lip + brow blocks specifically.
    """
    if multiplier == 1.0:
        return bgr
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    base_s = hsv[..., 1]
    boosted_s = np.clip(base_s * multiplier, 0, 255)
    hsv[..., 1] = base_s * (1.0 - mask) + boosted_s * mask
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def specular_bloom(
    bgr: np.ndarray,
    mask: np.ndarray,
    *,
    threshold: int = 200,
    strength: float = 0.35,
    sigma: float = 6.0,
) -> np.ndarray:
    """Lift the brightest pixels inside ``mask`` to fake the anime "shine".

    Recipe:
      1. Find pixels brighter than ``threshold`` (in grayscale).
      2. Blur that bitmap with a wide Gaussian → halo around highlights.
      3. Add halo back to the image, gated by ``mask`` so the bloom only
         lifts skin/hair specular points, not background lamps.

    This is what makes anime hair look glossy and noses look "lit". On
    real faces it tracks the natural specular spots (cheekbones, nose
    bridge, forehead) so it doesn't need geometry to land believably.
    """
    if strength <= 0.0:
        return bgr
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bright = (gray.astype(np.float32) - threshold).clip(min=0.0) / max(
        1.0, 255.0 - threshold
    )
    halo = cv2.GaussianBlur(bright, (0, 0), sigma)
    halo_masked = halo * mask * strength
    out = bgr.astype(np.float32) + halo_masked[..., None] * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def stylize_hair(
    bgr: np.ndarray,
    hair_ring_mask: np.ndarray,
    *,
    dark_l_max: float = 110.0,
    refine_morph_px: int = 5,
    refine_feather_px: int = 11,
    luminance_levels: int = 2,
    saturation_boost: float = 1.15,
    streak_strength: float = 0.6,
    streak_sigma: float = 5.0,
    streak_percentile: float = 78.0,
    state: "TemporalState | None" = None,
    state_prefix: str = "hair",
) -> np.ndarray:
    """Posterize hair into a few flat tones + add anime "streak" highlights.

    The ``hair_ring_mask`` covers the face_oval-extended ellipse minus the
    face — geometrically that's where hair lives, but it also catches some
    background pixels at the edges. We refine it in two steps:

      1. Inside the ring, threshold ``L < dark_l_max`` to keep only the
         pixels that actually look like dark hair. (This assumes dark hair
         — for light/blonde hair, raise the threshold or invert.)
      2. Morphological open + Gaussian feather to remove speckle and get
         a soft alpha-style mask.

    Then on the refined hair pixels:
      * K-means the L channel into ``luminance_levels`` (2 = base + highlight).
        Hair cells in anime are typically two flat blocks: the bulk color
        and the sheen on top.
      * Mildly boost saturation so blacks become "deep purple-black" / browns
        become more chromatic.
      * Specular streak: find pixels brighter than the ``streak_percentile``
        of hair L, blur them with a wide Gaussian, add back as a halo.

    Operates in-place on a copy and returns the new BGR uint8 image.
    """
    h, w = bgr.shape[:2]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[..., 0].astype(np.float32)

    # 1) Refine the ring to actual dark-hair pixels.
    in_ring = hair_ring_mask > 0.3
    is_dark = L < dark_l_max
    hair_bin = (in_ring & is_dark).astype(np.uint8) * 255
    if refine_morph_px > 1:
        k = refine_morph_px | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        hair_bin = cv2.morphologyEx(hair_bin, cv2.MORPH_OPEN, kernel)
    if refine_feather_px > 1:
        k = refine_feather_px | 1
        hair_soft = cv2.GaussianBlur(hair_bin, (k, k), 0).astype(np.float32) / 255.0
    else:
        hair_soft = hair_bin.astype(np.float32) / 255.0

    if hair_soft.sum() < 200:
        return bgr  # not enough hair detected — bail

    # 2) Posterize L on hair pixels only. K-means runs on a 1-D vector of
    #    just-hair L values; we then scatter the quantized values back.
    core = hair_soft > 0.5
    if core.sum() < 20:
        return bgr
    hair_L_vec = L[core].reshape(-1, 1).astype(np.float32)
    centers = _temporal_centers(
        hair_L_vec,
        max(2, luminance_levels),
        state,
        key=f"{state_prefix}_L",
    )
    L_quant = L.copy()
    L_quant[core] = _quantize_with_centers(hair_L_vec, centers)
    # Soft blend so the boundary between posterized hair and the rest fades.
    L_new = L * (1.0 - hair_soft) + L_quant * hair_soft
    lab[..., 0] = np.clip(L_new, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 3) Mild hair saturation boost — anime blacks read as cool-purple-black
    #    or warm-brown-black rather than dead gray.
    if saturation_boost != 1.0:
        out = boost_saturation_in_mask(out, hair_soft, saturation_boost)

    # 4) Specular streak halo on the brightest hair pixels.
    if streak_strength > 0.0:
        streak_threshold = float(np.percentile(hair_L_vec, streak_percentile))
        streak_bin = ((L > streak_threshold) & core).astype(np.float32)
        if streak_bin.any():
            streak_halo = cv2.GaussianBlur(streak_bin, (0, 0), streak_sigma)
            streak_halo = streak_halo * hair_soft  # confine to hair
            out = np.clip(
                out.astype(np.float32)
                + streak_halo[..., None] * 255.0 * streak_strength,
                0,
                255,
            ).astype(np.uint8)

    return out


def stylize_eyes(
    bgr: np.ndarray,
    eye_mask: np.ndarray,
    *,
    sclera_l_min: float = 110.0,
    sclera_target_L: float = 245.0,
    sclera_desat: float = 0.7,
    sclera_strength: float = 0.7,
    iris_l_max: float = 95.0,
    iris_saturation_boost: float = 2.0,
    iris_darken: float = 0.85,
) -> np.ndarray:
    """Whiten the sclera (eye whites) and saturate / darken the iris.

    Inside the eye polygon (``eye_mask``) we don't have separate sclera /
    iris masks. We synthesize them from luminance:

      * sclera pixels = bright (L > sclera_l_min)
      * iris pixels   = dark   (L < iris_l_max)

    The transitions between thresholds are softened so the masks aren't
    binary — gives a more painted feel at the iris boundary.

    Sclera step: pull L hard toward white, desaturate a/b toward neutral.
    Anime sclera is essentially pure white, no pinkish hue from blood
    vessels.

    Iris step: keep it dark (darken slightly) and crank saturation. That
    is how anime "intensifies" eye color even when the original was a
    natural brown / hazel.
    """
    if eye_mask.sum() < 4:
        return bgr
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    # Soft sclera/iris weights inside the eye polygon.
    sclera_w = np.clip((L - sclera_l_min) / 30.0, 0.0, 1.0) * eye_mask
    iris_w = np.clip((iris_l_max - L) / 25.0, 0.0, 1.0) * eye_mask

    # Sclera — pull L and a/b toward pure white / neutral chroma.
    s = sclera_w * sclera_strength
    lab[..., 0] = L + (sclera_target_L - L) * s
    # Desaturate chroma proportionally (target = 128 = neutral).
    lab[..., 1] = a + (128.0 - a) * (s * sclera_desat)
    lab[..., 2] = b + (128.0 - b) * (s * sclera_desat)

    # Iris — darken slightly (multiply L) so saturation boost has room.
    if iris_darken < 1.0:
        lab[..., 0] = lab[..., 0] * (1.0 - iris_w) + lab[..., 0] * iris_darken * iris_w

    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    if iris_saturation_boost != 1.0:
        out = boost_saturation_in_mask(out, iris_w, iris_saturation_boost)

    return out


def face_aware_line_art(
    frame_bgr: np.ndarray,
    skin_mask: np.ndarray,
    feature_mask: np.ndarray,
    *,
    base_strength: float = 0.75,
    skin_suppression: float = 0.9,
    feature_boost: float = 0.6,
) -> np.ndarray:
    """Multiplier image (HxWx1, in [0,1]) ready to multiply onto an RGB pass.

    Anime faces never have edges scattered across the cheek; they have
    edges only along the outline (jaw, hair, nose-tip) and the *features*
    (eyes, brows, lip line). Plain XDoG on the whole image puts edges
    everywhere texture changes, which speckles smooth skin and looks
    "noisy photo". The fix is region-aware:

      * Run a single XDoG once.
      * Push edges toward 1 (no darkening) inside the skin mask — kills
        the cheek/forehead speckle.
      * Run a second, bolder XDoG and only let it through inside the
        feature mask — gives strong outlines on lips/eyes/brows where
        anime really wants them.

    Returns a float32 multiplier with shape (H, W, 1) so the caller can
    do ``image * multiplier`` directly.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    base = xdog(gray, sigma=0.8, k=1.6, p=30.0, eps=0.5, phi=10.0)
    bold = xdog(gray, sigma=1.0, k=1.6, p=55.0, eps=0.35, phi=10.0)

    # 1) Suppress edges inside skin — push base toward 1 by skin_suppression
    #    fraction. skin=1 → multiplier 1 (clean); skin=0 → keep base edges.
    cleaned = base + (1.0 - base) * (skin_mask * skin_suppression)

    # 2) Override with bolder edges inside features.
    feature_mix = feature_mask * feature_boost
    merged = cleaned * (1.0 - feature_mix) + bold * feature_mix
    merged = np.clip(merged, 0.0, 1.0)

    # 3) Modulate overall strength: at base_strength=0 → no darkening at all.
    line_factor = (1.0 - base_strength) + base_strength * merged
    return line_factor[..., None]


def anime_head_pass(
    frame_bgr: np.ndarray,
    head_mask: np.ndarray,
    region_masks: dict[str, np.ndarray],
    *,
    # Optional masks for hair / eye stylization. Hair ring covers the
    # head-ellipse minus the face oval (mostly hair + some background);
    # we refine it internally via L-thresholding inside stylize_hair.
    hair_ring_mask: np.ndarray | None = None,
    # Strong cel-shade params used inside the head region.
    head_cel_params: dict | None = None,
    # Whole-frame milder cel-shade for body/background. None → keep original.
    body_cel_params: dict | None = None,
    # Pre-cel-shade skin palette snap — shifts skin Lab toward an anime target
    # before posterization, so K-means clusters land on bright/peachy bands
    # instead of natural skin bands.
    skin_palette_strength: float = 0.55,
    skin_palette_target_L: float = 215.0,
    skin_palette_target_a: float = 140.0,
    skin_palette_target_b: float = 145.0,
    # Hair-specific stylization (only runs if hair_ring_mask is provided).
    hair_params: dict | None = None,
    # Eye-specific stylization (sclera whitening + iris saturation).
    eye_params: dict | None = None,
    # Post-cel-shade boosters (operate inside the head).
    skin_lightness_lift: float = 0.0,
    feature_saturation_boost: float = 1.7,
    bloom_threshold: int = 200,
    bloom_strength: float = 0.45,
    bloom_sigma: float = 6.0,
    # Face-aware line work — replaces cel_shade's whole-frame XDoG inside head.
    face_line_strength: float = 0.8,
    face_line_skin_suppression: float = 0.9,
    face_line_feature_boost: float = 0.6,
    # Temporal stability — pass the same TemporalState across frames.
    state: "TemporalState | None" = None,
) -> np.ndarray:
    """Two-pass cel-shade with anime touches concentrated on the head.

    Why two passes?
      Whole-frame cel-shade with the head's aggressive settings would
      destroy background detail (2-level posterize on graffiti = ugly
      blobs). Conversely, the body's gentler settings on the head are
      not anime enough. Splitting it gives us two budgets — keep body
      readable, push head into cartoon territory.

    Composition order inside the head:
      1. Strong cel-shade (the cell-shading core).
      2. Lighten skin (Lab L lift) — porcelain face tone.
      3. Boost saturation in eyes / lips / brows only — anime feature pop.
      4. Specular bloom — glossy highlights on cheekbones / nose / hair.
      5. Alpha-blend with the body pass using the feathered head mask.

    Each step is a single op so you can comment any one out to see what
    it contributes.
    """
    # We disable cel_shade's built-in XDoG (which speckles cheeks) and add
    # face-aware line art at the end instead.
    defaults = dict(
        luminance_levels=2,
        chroma_levels=0,
        saturation_boost=1.6,
        bilateral_passes=4,
        bilateral_sigma_color=70.0,
        edge_strength=0.0,
        xdog_p=40.0,
        xdog_eps=0.4,
    )
    head_cel_params = {**defaults, **(head_cel_params or {})}
    # Force-disable whole-frame edges — they're now handled by face_aware_line_art.
    head_cel_params["edge_strength"] = 0.0

    if body_cel_params is None:
        body_pass = frame_bgr
    else:
        body_pass = cel_shade(
            frame_bgr, state=state, state_prefix="body", **body_cel_params
        )

    skin_mask = region_masks["skin"]

    # Step A — snap skin Lab toward an anime palette on the RAW frame
    # *before* cel_shade. Order matters: if we did this after K-means,
    # we would erase the discrete cell bands. Doing it before lets
    # K-means cluster the already-anime-tinted skin into bright bands.
    prepped = skin_palette_snap(
        frame_bgr,
        skin_mask,
        target_L=skin_palette_target_L,
        target_a=skin_palette_target_a,
        target_b=skin_palette_target_b,
        strength=skin_palette_strength,
    )

    head_pass = cel_shade(
        prepped, state=state, state_prefix="head", **head_cel_params
    )

    # Optional extra L lift after posterize (usually 0 now that palette
    # snap handles brightness — kept as an escape hatch for darker clips).
    if skin_lightness_lift != 0:
        head_pass = lift_lightness(head_pass, skin_mask, skin_lightness_lift)

    # Hair stylization — runs on the cel-shaded head_pass so the K-means
    # quantization stacks with cel_shade's whole-frame one. The refinement
    # inside stylize_hair (L < dark_l_max) ensures only actually-hair
    # pixels get touched, even when hair_ring_mask catches background.
    if hair_ring_mask is not None:
        head_pass = stylize_hair(
            head_pass,
            hair_ring_mask,
            state=state,
            state_prefix="hair",
            **(hair_params or {}),
        )

    feature_mask = np.clip(
        region_masks["lips"] + region_masks["eyes"] + region_masks["brows"],
        0.0,
        1.0,
    )
    head_pass = boost_saturation_in_mask(
        head_pass, feature_mask, feature_saturation_boost
    )

    # Eye-specific stylization (sclera + iris). Done after the broad
    # feature-saturation boost so the sclera doesn't get coloured by it.
    head_pass = stylize_eyes(
        head_pass, region_masks["eyes"], **(eye_params or {})
    )

    head_pass = specular_bloom(
        head_pass,
        head_mask,
        threshold=bloom_threshold,
        strength=bloom_strength,
        sigma=bloom_sigma,
    )

    # Face-aware line art — clean skin, bold feature outlines.
    if face_line_strength > 0:
        line_factor = face_aware_line_art(
            frame_bgr,
            skin_mask=skin_mask,
            feature_mask=feature_mask,
            base_strength=face_line_strength,
            skin_suppression=face_line_skin_suppression,
            feature_boost=face_line_feature_boost,
        )
        head_pass = np.clip(
            head_pass.astype(np.float32) * line_factor, 0, 255
        ).astype(np.uint8)

    alpha = head_mask[..., None]
    out = (
        head_pass.astype(np.float32) * alpha
        + body_pass.astype(np.float32) * (1.0 - alpha)
    )
    return np.clip(out, 0, 255).astype(np.uint8)
