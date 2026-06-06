"""Temporal smoothing for the face bounding box (S10 — 1€ filter).

Replaces the legacy ``LandmarkSmoother`` (which filtered 468 MediaPipe
landmark points). v1.0 of the pipeline only carries a single bounding box
per frame, so the smoother runs One Euro on the 4-D vector
``(cx, cy, w, h)`` and exposes a small hold-last-good policy on top.

Why a velocity-adaptive low-pass and not a plain EMA
----------------------------------------------------
Per slide ``S10``, video jitter has two regimes:

* Slow / static motion — we want aggressive smoothing to kill noise.
* Fast motion — we want low lag so the box tracks the head.

A constant-cutoff low-pass has to choose one trade-off. The One Euro
filter (Casiez et al., 2012) adapts the cutoff to the instantaneous
speed of the signal: ``cutoff = min_cutoff + beta · |dx/dt|``. Slow
movement → low cutoff → smooth output. Fast movement → high cutoff →
the filter "opens up" and follows.

References
----------
Casiez, G., Roussel, N., & Vogel, D. (2012). *1€ Filter: A Simple
Speed-based Low-pass Filter for Noisy Input in Interactive Systems.*
CHI 2012. (Course slide S10.)
"""

from __future__ import annotations

import math

from .detect import Bbox


# ---------------------------------------------------------------------------
# Scalar One Euro filter — same primitive the legacy module used.
# ---------------------------------------------------------------------------


def _alpha(cutoff: float, dt: float) -> float:
    """Low-pass smoothing coefficient for cutoff ``cutoff`` Hz over ``dt`` s.

    Derivation: standard RC low-pass with τ = 1 / (2π · f_c) and
    α = 1 / (1 + τ / dt). Used by both the signal pass and the
    derivative pass inside :class:`OneEuroFilter`.
    """
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """One Euro filter for a single scalar stream (Casiez et al., 2012).

    The filter keeps the previous output and the previous derivative
    estimate, low-passes the derivative at a fixed ``d_cutoff``, then
    low-passes the signal at a *speed-adapted* cutoff:

        cutoff(t) = min_cutoff + beta · |dx_hat(t)|

    Call :meth:`reset` to drop state (used after the bbox smoother gives
    up on a long miss; the next successful detection should be taken at
    face value rather than blended against a stale held position).
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.0,
        d_cutoff: float = 1.0,
    ) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._prev_x: float | None = None
        self._prev_dx: float = 0.0

    def reset(self) -> None:
        self._prev_x = None
        self._prev_dx = 0.0

    def __call__(self, x: float, dt: float) -> float:
        if self._prev_x is None:
            self._prev_x = float(x)
            self._prev_dx = 0.0
            return float(x)

        dx = (x - self._prev_x) / max(dt, 1e-6)
        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._prev_dx

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = _alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._prev_x

        self._prev_x = x_hat
        self._prev_dx = dx_hat
        return x_hat


# ---------------------------------------------------------------------------
# Bbox smoother — 4-D vector (cx, cy, w, h).
# ---------------------------------------------------------------------------


class BboxSmoother:
    """One Euro filter applied independently to ``(cx, cy, w, h)``.

    Why this representation and not ``(x, y, w, h)`` directly?
      Centre-and-size is the natural axis-aligned parameterisation: when
      the subject's head pans across the frame, only ``cx`` moves; ``x``
      moves *and* ``w`` is implicitly coupled to it. Smoothing centre and
      size independently keeps the box from squashing/stretching under
      noise.

    Hold-last-good policy
    ---------------------
    Haar cascades miss frames at 3/4 views, brief occlusions (e.g. an
    eye blink lets the frontal cascade lose its anchor), and motion
    blur. The smoother absorbs short gaps by re-emitting the last
    valid bbox up to ``hold_frames`` times (default 12, ≈ 200 ms at
    60 fps — long enough to bridge a natural blink plus a few warm-up
    frames after the cascade re-locks). Beyond that the subject has
    plausibly left the scene, so we return ``None`` and reset the
    filter state so re-acquisition isn't biased by a stale prior.

    Spatial / size sanity gates
    ---------------------------
    A face on screen cannot teleport between consecutive frames, and the
    head cannot suddenly triple in size. Haar (with the skin-coverage
    gate, see ``detect.py``) still emits two kinds of plausible-looking
    false positives on hard clips:

      * a body-sized blob fires after a long detection miss (profile
        cascade locks onto the torso silhouette);
      * a fist / leg / sock patch fires far from the last face position.

    Without a gate, ``bbox_hold_frames`` then re-emits these wrong
    bboxes for several extra frames, sticking the cel-shade onto the
    body or leg. We protect against both by rejecting a new detection
    that is implausible *given* the recent history:

      * ``max_size_ratio`` — reject when the candidate width / height is
        outside ``[1/r, r] · size_ema``; the size EMA survives across
        resets so a body blob after a 30-frame miss is still caught.
      * ``max_jump_ratio`` — when ``_last_good`` is still active,
        reject any new centre that travels more than ``ratio ·
        max(last_w, last_h) · (1 + held)`` since that last good bbox.
        After ``_last_good`` is dropped (long miss), fall back to a
        cold-start gate against ``_position_estimate`` — an EMA of
        valid centres that survives resets — with budget
        ``cold_start_jump_ratio · size_ema``.

    Both gates treat a rejected detection as a miss for accounting:
    the held budget ticks up and the last good bbox is re-emitted
    (or ``None`` once the budget runs out), so the rest of the
    pipeline sees the same passthrough behaviour as a clean Haar miss.

    Parameters
    ----------
    fps:
        Frame rate of the source clip. Sets ``dt = 1 / fps`` for the
        One Euro update — the filter's smoothing strength depends on it,
        so always pass the actual ``VideoCapture`` fps rather than a
        default.
    min_cutoff, beta, d_cutoff:
        See :class:`OneEuroFilter`. Defaults follow spec.md §5.1.
    hold_frames:
        Maximum number of consecutive ``None`` detections we coast through.
    max_jump_ratio:
        Spatial-continuity budget when ``_last_good`` is active.
        ``None`` disables the gate entirely.
    max_size_ratio:
        Size-continuity ratio. ``None`` disables the gate.
    cold_start_jump_ratio:
        Cold-start jump budget (in multiples of the running size EMA)
        used when ``_last_good`` has been dropped after a long miss but
        ``_position_estimate`` is still available. Defaults to ``3`` —
        a face can plausibly drift up to ~3 head-widths from the
        running anchor without the detection being a torso / leg false
        positive.
    size_ema:
        EMA weight on the new sample for ``_size_estimate`` updates.
    position_ema:
        EMA weight on the new sample for ``_position_estimate`` updates.
        Lower than ``size_ema`` (default ``0.3``) so a single outlier
        cannot drag the anchor across the frame.

    Notes
    -----
    The first valid bbox after construction (or after a reset) is passed
    through unsmoothed — there is no prior to blend against, and starting
    the filter from a zero state would drag the first reported bbox toward
    the origin for several frames.
    """

    def __init__(
        self,
        fps: float,
        *,
        min_cutoff: float = 1.0,
        beta: float = 0.02,
        d_cutoff: float = 1.0,
        hold_frames: int = 12,
        max_jump_ratio: float | None = 1.0,
        max_size_ratio: float | None = 2.0,
        cold_start_jump_ratio: float = 3.0,
        size_ema: float = 0.5,
        position_ema: float = 0.3,
    ) -> None:
        self.dt = 1.0 / max(float(fps), 1.0)
        self.hold_frames = int(hold_frames)
        self.max_jump_ratio = (
            None if max_jump_ratio is None else float(max_jump_ratio)
        )
        self.max_size_ratio = (
            None if max_size_ratio is None else float(max_size_ratio)
        )
        self.cold_start_jump_ratio = float(cold_start_jump_ratio)
        self.size_ema = float(size_ema)
        self.position_ema = float(position_ema)
        self._filters = [
            OneEuroFilter(min_cutoff, beta, d_cutoff) for _ in range(4)
        ]
        self._last_good: Bbox | None = None
        self._held: int = 0
        # Long-lived priors. Both survive ``reset`` so the gates still
        # have an anchor after a long detection miss — that anchor is
        # what catches body-sized blobs and leg / fist false positives
        # that fire when Haar locks onto something far from the face.
        self._size_estimate: float | None = None
        self._position_estimate: tuple[float, float] | None = None

    @property
    def held_count(self) -> int:
        """Number of consecutive held frames since the last fresh accept.

        ``0`` immediately after a fresh detection passes the gates and
        the filter; > 0 while we are coasting on the last good bbox;
        rises until ``hold_frames`` then triggers a reset.
        """
        return self._held

    @property
    def last_good(self) -> Bbox | None:
        """Most recent gate-passing, filter-output bbox (``None`` after reset)."""
        return self._last_good

    def reset(self) -> None:
        """Drop filter state. Next valid bbox is taken at face value.

        ``_size_estimate`` and ``_position_estimate`` deliberately
        survive — the cross-reset priors are what catch body-sized
        and leg / fist false positives that fire when Haar locks onto
        something far from the face after the face has been missing
        for many frames.
        """
        for f in self._filters:
            f.reset()
        self._last_good = None
        self._held = 0

    def _record_miss(self) -> Bbox | None:
        """Account a rejected / missing detection against the hold budget."""
        if self._last_good is not None and self._held < self.hold_frames:
            self._held += 1
            return self._last_good
        self.reset()
        return None

    def _passes_size_gate(self, bbox: Bbox) -> bool:
        if self.max_size_ratio is None or self._size_estimate is None:
            return True
        if self._size_estimate <= 0.0:
            return True
        size = float(max(bbox.w, bbox.h))
        ratio = size / self._size_estimate
        return (1.0 / self.max_size_ratio) <= ratio <= self.max_size_ratio

    def _passes_jump_gate(self, bbox: Bbox) -> bool:
        if self.max_jump_ratio is None:
            return True

        # Hot path: there is still an active last_good. Budget tracks
        # the recent bbox size and scales with held count.
        if self._last_good is not None and self._held < self.hold_frames:
            last = self._last_good
            dx = float(bbox.cx) - float(last.cx)
            dy = float(bbox.cy) - float(last.cy)
            distance = math.hypot(dx, dy)
            budget = (
                self.max_jump_ratio
                * float(max(last.w, last.h))
                * (1.0 + float(self._held))
            )
            return distance <= budget

        # Cold start: last_good was dropped on a long miss but the
        # position EMA persists. Gate against that anchor instead so
        # a leg / fist patch fired after the miss still gets rejected.
        if (
            self._position_estimate is None
            or self._size_estimate is None
            or self._size_estimate <= 0.0
        ):
            return True
        px, py = self._position_estimate
        dx = float(bbox.cx) - px
        dy = float(bbox.cy) - py
        distance = math.hypot(dx, dy)
        budget = self.cold_start_jump_ratio * self._size_estimate
        return distance <= budget

    def _update_size_estimate(self, bbox: Bbox) -> None:
        size = float(max(bbox.w, bbox.h))
        a = self.size_ema
        if self._size_estimate is None:
            self._size_estimate = size
        else:
            self._size_estimate = a * size + (1.0 - a) * self._size_estimate

    def _update_position_estimate(self, bbox: Bbox) -> None:
        cx = float(bbox.cx)
        cy = float(bbox.cy)
        a = self.position_ema
        if self._position_estimate is None:
            self._position_estimate = (cx, cy)
        else:
            px, py = self._position_estimate
            self._position_estimate = (
                a * cx + (1.0 - a) * px,
                a * cy + (1.0 - a) * py,
            )

    def update(self, bbox: Bbox | None) -> Bbox | None:
        """Push the latest detection and return the smoothed bbox.

        - ``bbox`` is ``None`` → emit the held bbox for up to
          ``hold_frames`` frames, then drop to ``None`` + reset.
        - ``bbox`` is a :class:`Bbox` → run the size + jump gates;
          rejected candidates are treated as misses. Surviving
          candidates are low-passed through One Euro on ``(cx, cy,
          w, h)`` and returned as a new :class:`Bbox`.
        """
        if bbox is None:
            return self._record_miss()

        if not self._passes_size_gate(bbox):
            return self._record_miss()
        if not self._passes_jump_gate(bbox):
            return self._record_miss()

        cx = bbox.cx
        cy = bbox.cy
        w = float(bbox.w)
        h = float(bbox.h)

        cx_s = self._filters[0](cx, self.dt)
        cy_s = self._filters[1](cy, self.dt)
        w_s = max(1.0, self._filters[2](w, self.dt))
        h_s = max(1.0, self._filters[3](h, self.dt))

        x_s = int(round(cx_s - w_s / 2.0))
        y_s = int(round(cy_s - h_s / 2.0))
        smoothed = Bbox(x_s, y_s, int(round(w_s)), int(round(h_s)))

        self._update_size_estimate(smoothed)
        self._update_position_estimate(smoothed)
        self._last_good = smoothed
        self._held = 0
        return smoothed


__all__ = ["OneEuroFilter", "BboxSmoother"]
