"""Heading estimation for a slow ground vehicle.

Why this exists
---------------
GPS course-over-ground (COG) is computed by extrapolating direction from
successive position fixes. A receiver that is barely moving has no direction to
extrapolate, so COG is documented to go erratic below roughly 0.75 knots
(~0.39 m/s). Our vehicle spends most of its life at or below that speed.

The old code trusted COG down to 0.35 m/s (i.e. *below* the cliff) and, when it
judged heading unusable, simply drove FORWARD blind. That is the primary reason
the car does not track a line from A to B.

Strategy here, best source first:

  1. FUSED    - if the GPS is a dead-reckoning unit (u-blox M8L) and the Pi
                forwards a fused heading, just use it. Valid at any speed.
  2. GYRO+COG - integrate the IMU yaw rate for a heading that is valid at ANY
                speed, and slowly anchor it to GPS COG whenever the vehicle is
                moving fast enough for COG to be trustworthy. This is the
                standard complementary filter for a slow ground robot, and it is
                what we use once the Pi forwards `imu.gyro_z`.
  3. COG-only - degraded fallback (what we have until the Pi forwards the IMU).
                Trust COG above the threshold; HOLD the last good heading below
                it rather than pretending we know nothing.

In every case `valid` tells the caller whether the estimate can be steered on.
An invalid heading must never be silently treated as "drive straight".
"""

from __future__ import annotations
import time
from collections import deque

from .geo import bearing_deg, haversine_m

# --- The COG == 0.0 sentinel ------------------------------------------------
# NMEA receivers leave the RMC course field EMPTY when they cannot determine a
# course (i.e. at low speed). The Pi's gps_reader turns that empty field into
# 0.0 — and 0.0 is indistinguishable from a genuine "due north".
#
# The Pi's own code knows this and guards against it (raspberry_pi/main.py:57:
# "0.0 is the gps_reader default when NMEA course field is empty" -> rejects).
# The laptop did NOT, because its check was `cog_deg is not None`, and 0.0 is
# not None. So the car silently concluded it was PERMANENTLY FACING NORTH, with
# full confidence, and steered on that. Confirmed in the real flight log:
# "hdg=0/cog" appears while the car is physically driving in another direction.
#
# We therefore refuse an exact 0.0 AND — more importantly — we no longer depend
# on the Pi's course field at all when we can compute the course ourselves from
# consecutive GPS positions. Course made good over a real displacement is the
# same quantity, derived from data we can sanity-check.
COG_ZERO_EPS = 1e-6

# Deriving course from position deltas needs a displacement that is large
# compared to GPS position noise (~1-2 m), or the bearing is just noise.
MIN_DISPLACEMENT_M = 4.0
POS_HISTORY_S = 12.0

# COG is unreliable below ~0.39 m/s (0.75 kn).
#
# ⚠ THE CAR CRUISES AT 0.5-0.8 m/s. That is only 1.3x-2x above the cliff — we are
# operating close to the edge of where GPS heading exists at all. Consequences:
#   * Set the trust threshold just above the cliff, not higher, or we'd reject
#     heading during normal cruise and never navigate.
#   * COG at this speed is NOISY (tens of degrees of jitter). It MUST be smoothed
#     — see COG_EMA_ALPHA. Feeding raw COG into the steering law would make the
#     car chase GPS noise.
#   * Drive at the TOP of the range (~0.8 m/s), not the bottom. Creeping at
#     0.5 m/s puts heading right on the edge of nonsense.
COG_TRUST_MPS = 0.45

# Low-pass on raw COG. Only used in the COG-only (no-gyro) mode, where there is
# no gyro to provide the high-frequency truth. Lower = smoother but laggier.
# 0.25 was the value the old controller used and it is a reasonable start.
COG_EMA_ALPHA = 0.25

# Complementary-filter gain: how hard each trustworthy COG sample pulls the
# gyro-integrated heading back toward it. Small = trust the gyro (smooth, drifts
# slowly); large = trust COG (noisy but drift-free). 0.05-0.15 is the usual band.
COG_ANCHOR_GAIN = 0.08

# If the gyro-integrated heading has gone this long with no COG anchor, its
# accumulated drift is no longer trustworthy.
GYRO_DRIFT_TIMEOUT_S = 30.0

# Below this yaw rate, treat the gyro as stationary (kills integration of bias).
GYRO_DEADBAND_DPS = 0.5

# --- Learning the gyro sign from GPS, instead of trusting a hand-rotation test ---
# A flipped sign makes the car steer AWAY from its target and diverge. The manual
# test was ambiguous (the sign changed three times as the board was twisted back
# and forth), so we let the car settle it itself against GPS ground truth.
SIGN_MIN_TURN_DPS = 8.0      # only learn from unambiguous turns, not noise/bias
SIGN_SAMPLES_NEEDED = 20     # agreeing samples before we trust the sign


def _wrap180(d: float) -> float:
    """Wrap degrees to (-180, +180]."""
    return (d + 540.0) % 360.0 - 180.0


def _wrap360(d: float) -> float:
    """Wrap degrees to [0, 360)."""
    return d % 360.0


class HeadingEstimator:
    """Best-available heading, in degrees [0,360), 0=N, 90=E.

    Feed it everything you have each tick; it picks the best source itself.
    """

    def __init__(self, gyro_sign: float | None = None):
        # gyro_sign: +1 if gyro_z > 0 means turning RIGHT (clockwise from above).
        #
        # A FLIPPED SIGN IS CATASTROPHIC: the car would steer AWAY from its target
        # and diverge, turning harder the more wrong it gets. So we do NOT guess,
        # and we do NOT rely on someone twisting the board by hand and eyeballing a
        # number (that test came back ambiguous — the sign flipped three times as
        # the board was rotated back and forth).
        #
        # Instead the car LEARNS it while driving: whenever GPS course is moving
        # AND the gyro feels a turn, compare the two signs. GPS is the ground truth
        # for "which way did we actually turn". After enough agreeing samples the
        # sign is known, with no human in the loop.
        #
        # Until then the gyro is NOT used for steering — we run on GPS course alone,
        # which is exactly what we do today. So this can only ever improve things.
        self.gyro_sign: float | None = gyro_sign      # None = not yet known
        self._sign_votes = 0.0                        # + = agrees, - = inverted
        self._sign_samples = 0
        self._prev_cog: float | None = None
        self._prev_cog_t: float | None = None

        self._heading: float | None = None
        self._valid = False
        self._source = "none"
        self._last_t: float | None = None
        self._last_anchor_t: float | None = None
        self._has_gyro = False
        self._pos: deque = deque()      # (t, lat, lon) for course-made-good

    # ---------- public ----------

    @property
    def valid(self) -> bool:
        return self._valid

    @property
    def heading(self) -> float | None:
        return self._heading if self._valid else None

    @property
    def source(self) -> str:
        return self._source

    def reset(self) -> None:
        """Call on every new run (GO). Stale heading across runs is dangerous.

        NOTE: the learned gyro_sign is deliberately NOT reset. It is a property of
        how the chip is bolted to the car, not of this run, and it takes real
        driving to learn. Throwing it away every GO would mean never having it.
        """
        self._heading = None
        self._valid = False
        self._source = "none"
        self._last_t = None
        self._last_anchor_t = None
        self._has_gyro = False
        self._pos.clear()
        self._prev_cog = None
        self._prev_cog_t = None

    @property
    def sign_status(self) -> str:
        if self.gyro_sign is None:
            return f"learning({self._sign_samples}/{SIGN_SAMPLES_NEEDED})"
        return f"{'+' if self.gyro_sign > 0 else '-'}1"

    def _learn_gyro_sign(self, cog_deg: float | None, gyro_z: float | None,
                         now: float) -> None:
        """Work out whether gyro_z > 0 means turning RIGHT, by watching GPS.

        GPS course is the ground truth for 'which way did we actually turn'. When
        the course is swinging AND the gyro feels a turn, the two must agree in
        sign. Enough agreeing samples and the sign is settled — no hand-rotation
        test, no human judgement, no chance of the car diverging on a bad guess.
        """
        if self.gyro_sign is not None or cog_deg is None or gyro_z is None:
            self._prev_cog, self._prev_cog_t = cog_deg, now
            return

        if self._prev_cog is not None and self._prev_cog_t is not None:
            dt = now - self._prev_cog_t
            if 0.05 < dt < 2.0:
                cog_rate = _wrap180(cog_deg - self._prev_cog) / dt   # deg/s, + = right
                # Only learn from UNAMBIGUOUS turns. A small wobble is GPS noise, and
                # a small gyro reading is bias — neither tells us anything.
                if abs(cog_rate) >= SIGN_MIN_TURN_DPS and abs(gyro_z) >= SIGN_MIN_TURN_DPS:
                    agree = 1.0 if (cog_rate > 0) == (gyro_z > 0) else -1.0
                    self._sign_votes += agree
                    self._sign_samples += 1
                    if (self._sign_samples >= SIGN_SAMPLES_NEEDED
                            and abs(self._sign_votes) >= SIGN_SAMPLES_NEEDED * 0.6):
                        self.gyro_sign = 1.0 if self._sign_votes > 0 else -1.0

        self._prev_cog, self._prev_cog_t = cog_deg, now

    def _course_made_good(self, lat: float | None, lon: float | None,
                          now: float) -> float | None:
        """Course computed from OUR OWN GPS position history.

        This is the same quantity the receiver's COG field is meant to hold, but
        derived from data we can check — so it is immune to the Pi's empty-field
        0.0 sentinel, and to any other quirk of its NMEA parsing.

        Only returns a bearing once the car has actually MOVED far enough that the
        displacement dominates GPS position noise (~1-2 m). Below that, the bearing
        between two fixes is just noise, which is exactly why COG is untrustworthy
        at low speed in the first place.
        """
        if lat is None or lon is None:
            return None
        self._pos.append((now, lat, lon))
        while self._pos and now - self._pos[0][0] > POS_HISTORY_S:
            self._pos.popleft()
        cur = (lat, lon)
        # Walk back to the most recent fix that is far enough away to give a bearing
        # that means something.
        for t, plat, plon in reversed(self._pos):
            if haversine_m((plat, plon), cur) >= MIN_DISPLACEMENT_M:
                return bearing_deg((plat, plon), cur)
        return None

    def update(self,
               cog_deg: float | None,
               speed_mps: float | None,
               lat: float | None = None,
               lon: float | None = None,
               gyro_z_dps: float | None = None,
               gyro_valid: bool = False,
               fused_deg: float | None = None,
               reversing: bool = False,
               now: float | None = None) -> float | None:
        """Advance the estimate. Returns the heading, or None if not yet valid.

        `reversing` MATTERS ENORMOUSLY. Course-over-ground is the direction the
        vehicle is TRAVELLING, not the way its nose points. Those agree only when
        driving forwards. While reversing, COG is the nose heading + 180.

        The first outdoor run failed partly on this: the obstacle layer kept forcing
        a reverse, and every reverse flipped the heading estimate by 180 degrees. The
        log showed heading swinging 132 -> 276 -> 341 -> 115 while the car sat in
        roughly one spot, and a heading error of -176 degrees. The controller was
        steering on a heading that was pointing backwards.

        We now correct COG by 180 while reversing, so the estimate always tracks the
        NOSE — which is the frame the steering law assumes.
        """
        now = now if now is not None else time.monotonic()
        dt = (now - self._last_t) if self._last_t is not None else 0.0
        self._last_t = now
        # A long stall (paused loop, GC, reconnect) makes dt meaningless for
        # integration — skip integrating rather than jumping the heading.
        if dt < 0.0 or dt > 1.0:
            dt = 0.0

        # ---- 1. Fused heading from a DR-capable GPS: unconditionally best. ----
        if fused_deg is not None:
            self._heading = _wrap360(float(fused_deg))
            self._valid = True
            self._source = "fused"
            self._last_anchor_t = now
            return self._heading

        # ---- Sanitise the Pi's course field, then fall back to our own. ----
        #
        # An EXACT 0.0 is the Pi's "NMEA course field was empty" sentinel, not a
        # heading. Accepting it is what convinced the car it permanently faced
        # north. Reject it — a genuine due-north course will read 359.9x / 0.3x,
        # essentially never a bit-exact 0.0, and if we lose one such sample the
        # course-made-good below covers us anyway.
        if cog_deg is not None and abs(float(cog_deg)) < COG_ZERO_EPS:
            cog_deg = None

        # Course computed from OUR OWN position history — independent of the Pi's
        # parsing entirely. Preferred when the receiver gives us nothing usable.
        cmg = self._course_made_good(lat, lon, now)
        used_cmg = False
        if cog_deg is None and cmg is not None:
            cog_deg = cmg
            used_cmg = True

        cog_ok = (
            cog_deg is not None
            and speed_mps is not None
            and speed_mps >= COG_TRUST_MPS
        )
        # Course-made-good already proves the car moved MIN_DISPLACEMENT_M, which is
        # stronger evidence of real motion than the (sometimes bogus) speed field.
        if used_cmg:
            cog_ok = True

        # COG is the direction of TRAVEL. Reversing means the nose points the other
        # way. Correct it, or every reverse manoeuvre inverts the heading estimate.
        if cog_ok and reversing:
            cog_deg = _wrap360(float(cog_deg) + 180.0)

        # ---- 2. Gyro integration (valid at ANY speed), anchored to COG. ----
        # Learn the sign from GPS first — the gyro is NOT used for steering until we
        # know which way is 'right'. Until then we fall through to COG/CMG, which is
        # what we do today, so this can only improve things, never regress them.
        self._learn_gyro_sign(cog_deg if cog_ok else None, gyro_z_dps, now)

        if gyro_valid and gyro_z_dps is not None and self.gyro_sign is not None:
            self._has_gyro = True
            rate = self.gyro_sign * float(gyro_z_dps)
            if abs(rate) < GYRO_DEADBAND_DPS:
                rate = 0.0

            if self._heading is None:
                # Can't integrate from nothing — need one COG anchor to start.
                if cog_ok:
                    self._heading = _wrap360(float(cog_deg))
                    self._valid = True
                    self._source = "gyro+cog"
                    self._last_anchor_t = now
                return self.heading

            self._heading = _wrap360(self._heading + rate * dt)

            if cog_ok:
                err = _wrap180(float(cog_deg) - self._heading)
                self._heading = _wrap360(self._heading + COG_ANCHOR_GAIN * err)
                self._last_anchor_t = now

            # Un-anchored gyro drifts without bound. Don't steer on a stale one.
            drift_age = (now - self._last_anchor_t) if self._last_anchor_t else 1e9
            self._valid = drift_age <= GYRO_DRIFT_TIMEOUT_S
            self._source = "gyro+cog" if self._valid else "gyro-drifted"
            return self.heading

        # ---- 3. COG only (no IMU). This is our actual operating mode. ----
        # Raw COG at 0.5-0.8 m/s is noisy — it is derived from small position
        # deltas, and at this speed those deltas are barely above the GPS noise
        # floor. Feed it straight into the steering law and the car chases the
        # noise. So low-pass it, taking the shortest arc so 359 -> 1 does not
        # swing the estimate 358 degrees the wrong way.
        if cog_ok:
            cog = _wrap360(float(cog_deg))
            if self._heading is None:
                self._heading = cog
            else:
                err = _wrap180(cog - self._heading)
                self._heading = _wrap360(self._heading + COG_EMA_ALPHA * err)
            self._valid = True
            self._source = "cmg" if used_cmg else "cog"
            self._last_anchor_t = now
            return self._heading

        # Too slow for COG and no gyro. HOLD the last known heading — it is the
        # best estimate we have (the car cannot have turned much while creeping)
        # — but mark WHY, and let it go stale so the caller can fail safe rather
        # than drive blind forever.
        if self._heading is not None:
            age = (now - self._last_anchor_t) if self._last_anchor_t else 1e9
            self._valid = age <= GYRO_DRIFT_TIMEOUT_S
            self._source = "held" if self._valid else "stale"
            return self.heading

        self._valid = False
        self._source = "none"
        return None
