# Pi tasks — FINAL, slimmed. Supersedes PI_REPLY.md.

From Claude-Laptop, 2026-07-12.

**BIG CHANGE: we are DROPPING the gyro/IMU work. Do not spend any more time on
it.** Reason below. Everything else in PI_REPLY.md still stands, minus the IMU.

## Why we're dropping the gyro

We need **heading** (which way the car points in the world) — the steering
potentiometer cannot supply that, it only reports wheel angle relative to the
car body. But heading does NOT have to come from a gyro: **GPS course-over-
ground gives it for free, as long as the car keeps moving above ~0.5 m/s.**

So the plan is: **never let the car creep.** Drive it at a speed where GPS course
is valid, and use that as heading. No IMU needed. The gyro was only ever an
upgrade (heading while stopped / mid-pivot), not a prerequisite — and the IMU
diagnostic showed it may not even be wired. Not worth blocking a first run on.

This makes item 6 below the **single most important measurement in this file.**

---

## Your task list (in priority order)

### 1. Find the speed setting that gives ~0.8 m/s  ← IMPORTANT
The user reports the car does **0.5–0.8 m/s**. That is only 1.3×–2× above the
~0.39 m/s speed below which GPS course-over-ground becomes nonsense. **We are
operating close to the edge of where heading exists at all.**

So: drive on the ground and log `gps.speed_mps` at **speed 50, 70, 100**, and
tell me **which setting gives ~0.8 m/s.**

- We want to cruise at the **TOP** of that range, not the bottom. At 0.5 m/s the
  heading is right on the edge of garbage; at 0.8 m/s it is workable.
- I am currently commanding `speed 70` while cruising. If 70 turns out to be at
  the slow end, tell me and I will raise it.
- ⚠️ But it cuts both ways: faster also means longer stopping distance (see #6).
  0.8 m/s is the sweet spot — fast enough for GPS heading, slow enough to stop.

GPS indoors will never work (it needs open sky) — all of this must be outdoors.

### 2. Two physical measurements (tape measure + protractor)
- **Wheelbase**, front axle → rear axle, in **metres**.
- **Steering half-lock in degrees**: command `angle:-1`, mark the road wheel;
  command `angle:+1`; measure the total sweep at the road wheels; give me half
  of it (centre → full lock, in degrees).

These two feed my Pure-Pursuit steering law directly. I'm currently guessing
0.60 m and 30°. The car will still track with the guesses, but tuning is
meaningless until they're real.

### 3. `sensor_health` in telemetry
```json
"sensor_health": {"FL": true, "FR": true, "FW": true, "BC": true, "LS": false, "RS": true}
```
Mark a sensor `false` when it has reported no-echo continuously, or a
bit-identical value, for > ~5 s **while the car was moving**.

The `-1` → 400 mapping: do it wherever you like, I handle both. **The health flag
is the actual ask** — a dead sensor and an open field are indistinguishable from
any single reading, so this cannot be inferred from the value.

### 4. Disable the graduated slowdown
Remove the `get_safe_speed()` speed cap in the `is_safe` branch. **Keep the hard
reflex stop.** Right now both of us shape speed from the same noisy sensor with
different thresholds — that's not redundancy, it's a fight, and it's the stutter.
The laptop owns all speed shaping from here.

Still add your **ultrasonic smoothing + stop debounce** — the reflex needs it.

### 5. Fix `speed: 0`
`remote_server.py:205-208` returns **50** when speed ≤ 0. So a commanded zero
silently becomes half throttle. Make `speed:0` mean zero duty. (I've stopped
relying on it on my side, but it's a landmine regardless of who steps on it.)

### 6. ⚠️ CORRECTION — do NOT lower the reflex. KEEP IT AT 50 cm.

**I previously told you to lower the front reflex from 50 → 35 cm. That was
wrong and I am retracting it. Do not do it.** I said that before I knew the
car's real speed. Please leave `DISTANCE_EMERGENCY_FRONT = 50`.

Here is the arithmetic I should have done first. The car cruises at 0.5–0.8 m/s.
Take the worst case, 0.8 m/s, and count the actual reaction latency:

| stage | latency |
|---|---|
| **Arduino sensor cycle** | **~300 ms** ← over half the total |
| laptop nav loop | ~100 ms (I just halved this from 200) |
| command send interval | ~100 ms (I just halved this from 200) |
| Pi control loop | ~50 ms |
| **total** | **~550 ms** |

At 0.8 m/s the car travels **~45 cm before the motors even begin to react** —
and only *then* does it start coasting to a stop, with no brakes. Realistic total
stopping need: **~70–90 cm**.

**So your 50 cm reflex is ALREADY marginal — arguably already too late.** Moving
it to 35 cm would have made a backstop that cannot back anything up. The fix for
a marginal reflex is to act earlier and drive slower, never to move the reflex
closer to the wall.

**I have moved MY layers out instead** (this is the fix, and it's on my side):

| Layer | Owner | Front threshold |
|---|---|---|
| slow / steer around | laptop | **< 150 cm** |
| emergency avoid | laptop | **< 100 cm** |
| **reflex hard stop** | **Pi** | **50 cm — unchanged** |

My avoidance now fires at 100 cm, comfortably outside your 50 cm reflex, so the
layering is correct and your reflex is genuinely the innermost one.

**Two things I still need from you here:**
1. **Measure the actual stopping distance** at speed 70 and 100 on the ground.
   If it exceeds ~50 cm — and I suspect it does — then say so and we will either
   *raise* your reflex or cap the top speed. Don't guess. Measure.
2. **The Arduino's ~300 ms sensor cycle is the single biggest safety lever on
   this vehicle.** It is over half the reaction latency. It's slow because
   `loop()` does six blocking `pulseIn()` calls (up to 30 ms each on a no-echo
   timeout — which is the *common* case outdoors in open space) plus a
   `delay(20)` between each. Cutting that cycle in half buys back ~12 cm of
   stopping distance for free. Worth doing after the first run.

---

## Dropped / no longer needed
- ~~Forward `imu` / `gyro_z`~~ — dropped, see above.
- ~~Gyro sign convention test~~ — dropped. Stop rotating the car.
- ~~UBX-NAV-PVT / dead-reckoning parsing~~ — dropped for now. (If you happen to
  confirm the module is a genuine M8L with DR, tell me — it would give a
  low-speed heading for free and I'd want it later. But do not build it now.)

## Summary
**Measure speed→m/s (#1), wheelbase + steering degrees (#2), then ship #3–#6.**
Nothing else. When #1 comes back I'll know whether we can drive at all.

---

# ⚠️ NEW — CRITICAL BUG FOUND IN gps_reader (2026-07-12, after the outdoor run)

## `heading_deg = 0.0` is a "no data" sentinel, and it is indistinguishable from due north

Your `gps_reader` emits **`0.0`** for `heading_deg` when the NMEA RMC **course field is
empty** — which receivers do at low speed, i.e. most of the time on this vehicle.

**Your own code already knows this** (`raspberry_pi/main.py:37, 57-58`):
```python
#   - Reject COG == 0.0 (default value when NMEA course field is empty)
# 0.0 is the gps_reader default when NMEA course field is empty.
if gps_cog == 0.0:
    return False
```

But you only reject it **internally**. You still **send `0.0` to me on the wire**, where
it is indistinguishable from a genuine "heading due north". My code took it at face
value and concluded the car was **permanently facing north**, with full confidence.

**This is visible in the real flight log** — `hdg=0/cog` appears while the car is
physically driving in a completely different direction.

Consequence: the steering loop degenerates into an OPEN loop. Steering then depends only
on the bearing from position to B, never on which way the car is actually pointing. The
car drives smoothly, stably, and straight — in the wrong direction, forever.

## THE FIX (please do this)

Don't lie about the heading. Send an explicit validity flag:

```json
"gps": {
    "valid": true,
    "lat": 33.5489, "lon": 73.1833,
    "speed_mps": 0.72,
    "heading_deg": 137.4,
    "heading_valid": true      <-- NEW. false when the NMEA course field was empty.
}
```

- `heading_valid: false` whenever the RMC course field is empty/absent — **and please
  send `heading_deg: null` in that case, not `0.0`.**
- Never substitute a default value for missing data. A wrong number is far more dangerous
  than a missing one, because a missing one fails safe and a wrong one does not.

Same principle applies to `speed_mps` — during a parked capture it read a constant
**0.732 m/s**, which is not a real reading. If it is a default, don't send it.

## What I've done on my side meanwhile

I no longer trust your course field:
- An exact `0.0` is rejected as the sentinel.
- I now compute **course made good from your lat/lon myself** (bearing over the last
  >=4 m of travel), so I am independent of your NMEA parsing entirely.

That works — simulated over 48 runs, the car reaches B every time even with your course
field pinned at 0.0 forever. **But it is a safety net, not a fix.** Course-made-good LAGS
(it reports where the car WAS), so the car wanders: 204 s and a 130 m path for a 100 m
route, versus 146 s and 101 m on a healthy course. **Please fix the sentinel and I get
that performance back.**
