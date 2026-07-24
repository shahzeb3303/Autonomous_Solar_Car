# Code Audit — Autonomous Solar Car (laptop side)

**Date:** 2026-07-12
**Branch:** `distance-controller`
**Status:** Findings below are CONFIRMED — I read the actual lines, not a summary.
A deeper multi-agent audit is still running; its extra findings get appended later.

Goal being audited against: **drive A→B autonomously (GPS + obstacle avoidance), and
manually.**

---

## 0. TL;DR — the three things that actually matter

1. **Autonomous steering is bang-bang, not proportional.** The controller computes a
   precise heading error and then throws it away. The car can only steer *full-left,
   dead-centre, or full-right*. This alone is enough to make A→B navigation fail.
2. **GPS course-over-ground cannot give you heading at your speed.** It is physically
   unreliable below ~0.39 m/s and you trust it at 0.35 m/s. Below that threshold the
   code drives **blind and straight**.
3. **Your git state holds three different, mutually-exclusive steering designs at once**
   (HEAD, the index, and the working tree). Whatever you *think* you are running,
   you are probably not running it.

---

## 1. THE GIT STATE IS TANGLED — fix this before anything else

Three different steering architectures coexist right now:

| Where | Steering design |
|---|---|
| `HEAD` (36c37213, 2026-06-09) | **Gyro yaw-rate closed loop** — "motor + MPU, **no pot**". Uses `KP_YAW * heading_err`. |
| **Staged index** (`git diff --cached`) | **Open-loop `steer_pos` integrator** — the old May-13 design. Guesses wheel position from elapsed time. |
| **Working tree** (what actually runs) | **Potentiometer closed loop** — sends `angle` to the Pi, reads back `steer_current_angle`. |

So `git stash`, `git checkout .`, or `git reset --hard` each land you on a *different
steering controller*. This is almost certainly why "things that were working stopped
working, and things that weren't started working."

**Also note:** `HANDOFF.md` is dated 2026-07-06 but the last commit is 2026-06-09, and
HANDOFF says "nothing is committed yet" — which is false. HANDOFF describes the *pot*
design; HEAD is the *gyro* design. The doc and the code disagree.

**Action: pick ONE steering design and commit it. Everything else is downstream of this.**

---

## 2. BLOCKER — autonomous steering is bang-bang (this is the A→B bug)

The pipeline destroys its own precision:

- [`nav/controller.py:198`](laptop/nav/controller.py#L198) computes `err` — a clean,
  continuous heading error in degrees. Good.
- [`nav/controller.py:205-210`](laptop/nav/controller.py#L205-L210) immediately
  **quantises it to a string**: `FORWARD` / `TURN_LEFT` / `TURN_RIGHT`. The magnitude is
  discarded. A 5° error and a 44° error produce the *identical* command.
- [`web_control.py:462-467`](laptop/web_control.py#L462-L467) then maps that string to an
  angle of exactly **`-1.0`, `0.0`, or `+1.0`** — i.e. full lock or dead centre.

With `TURN_ENTER_DEG = 45.0` ([controller.py:28](laptop/nav/controller.py#L28)), the car:
- drives **perfectly straight** while up to **45° off course**, correcting nothing;
- then **slams the wheel to full lock** the instant it crosses 45°;
- holds full lock until error drops under `TURN_EXIT_DEG = 15°`, then snaps back to straight.

That is a relay/bang-bang controller with a 30° hysteresis band. It **cannot** track a
line. It will weave, overshoot, and oscillate — exactly the "not moving correctly from
A to B" symptom.

**The fix is the standard pure-pursuit steering law** (confirmed against the literature):

```
δ = atan( 2 · L · sin(α) / L_d )
```

where `L` = wheelbase (m), `L_d` = lookahead distance (m), `α` = angle to the carrot
point (which is exactly the `err` the code already has). Then map `δ` → the `-1..+1`
`angle` command via the steering lock in degrees.

**Blocked on two physical numbers nobody has recorded: the WHEELBASE (m) and the
STEERING LOCK-TO-LOCK (degrees).** Both are in [PI_QUESTIONS.md](PI_QUESTIONS.md).

Note: the reverted gyro commit (`HEAD`) was the **only** version that ever used the
heading error proportionally (`w_des = KP_YAW * snap.heading_err_deg`). The working tree
is therefore **strictly worse at steering than the commit it replaced.**

---

## 3. BLOCKER — GPS heading is unusable at this vehicle's speed

- [`controller.py:30`](laptop/nav/controller.py#L30): `HEADING_MIN_SPEED_MPS = 0.35`.
- GPS course-over-ground is derived by extrapolating direction across successive fixes.
  Below roughly **0.75 knots ≈ 0.39 m/s** it is well documented to go erratic/nonsensical
  (a stationary receiver has no direction to extrapolate).
- **0.35 m/s is BELOW that cliff.** The code trusts heading exactly where it stops
  being trustworthy.

And when heading *is* deemed unusable, [`controller.py:191-195`](laptop/nav/controller.py#L191-L195)
returns a bare `FORWARD` — "heading lost, nudging fwd". **The car drives blind, in
whatever direction it happens to be pointing, with no correction.** For a slow vehicle
this is not an edge case; it is the normal operating regime.

The same blind-`FORWARD` happens during `CALIBRATING`
([controller.py:186-189](laptop/nav/controller.py#L186-L189)) for up to `CALIBRATE_S = 6.0`
seconds at startup.

**This cannot be tuned away. It needs an architectural fix — one of:**
- **(a)** Gyro-integrated heading, anchored/corrected to GPS COG only when speed is high
  enough to trust it. *(This is what commit `4311f15b` did — "gyro-only with GPS-COG
  anchoring" — and it was the right instinct. The working tree threw it away.)*
- **(b)** If the GPS is genuinely a **NEO-M8L** (the automotive dead-reckoning variant
  with a built-in IMU) and DR is actually enabled, it outputs a **fused heading valid at
  low speed** — strictly better than anything we compute. **Need to confirm the module.**

---

## 4. HIGH — the ML model is in the autonomous path and can override navigation

[`web_control.py:399-423`](laptop/web_control.py#L399-L423). The behaviour-cloned model
(`models/best.pt`, present, 4.6 MB, trained 2026-04-29) is **live in the nav loop** and
is allowed to overrule the GPS controller:

- `FORWARD` → `SLOW_DOWN` at confidence > 0.55
- `FORWARD` → `TURN_LEFT`/`TURN_RIGHT` at confidence > 0.50 **whenever GPS heading is
  unusable** — i.e. **exactly in the low-speed regime from §3, which is most of the time.**
- `FORWARD`/`SLOW_DOWN` → `REVERSE*` at confidence > 0.70

The code's own comment admits *"the model doesn't know where B is, so it cannot pick
direction toward the goal."* Yet at low speed it is **precisely what picks the turn
direction**. A behaviour-cloned model with no goal knowledge is steering your car toward
B. It has no idea where B is.

**Recommendation: turn the model OFF for A→B navigation** (`ENABLE_MODEL = False`,
[web_control.py:396](laptop/web_control.py#L396)) until GPS/gyro navigation works
standalone. It is an uncontrolled variable in the middle of your control loop.

---

## 5. HIGH — two obstacle-avoidance systems can fight each other

- **Laptop:** [`autonomous_hybrid.py`](laptop/autonomous_hybrid.py) `decide()` — thresholds
  `FRONT_EMERGENCY_CM = 30`, `FRONT_SLOW_CM = 70`, `SIDE_PINCH_CM = 25`, `BACK_SAFE_CM = 40`.
- **Pi:** `safety_governor.py` + `obstacle_monitor.get_safe_speed()` — per HANDOFF, its own
  thresholds (`DISTANCE_EMERGENCY_FRONT/BACK` = 50/30), plus graduated slowdown.

**The laptop says "turn aside at 70 cm"; the Pi says "hard stop at 50 cm."** These are
different policies over the same sensors. The laptop can command an avoidance manoeuvre
that the Pi simultaneously vetoes. Result: **command, veto, command, veto — which looks
exactly like the reported "jerky moves-stops-moves" driving.**

**Recommendation: the Pi owns ONLY the last-resort reflex stop. The laptop owns
avoidance strategy.** One brain decides; the other is a reflex. Currently both decide.

---

## 6. MEDIUM — confirmed smaller defects

| # | Issue | Location |
|---|---|---|
| 6.1 | **No newline framing on commands.** `sendall(json.dumps(payload).encode())` sends JSON with **no trailing delimiter**. The receiver *does* split on `\n`. If two commands coalesce in one TCP segment (Nagle), the Pi sees `{...}{...}` and cannot parse it. Works today only by luck of timing. | [web_control.py:177](laptop/web_control.py#L177) |
| 6.2 | **Speed-slider drag kills autonomous mode.** `/cmd` calls `_set_nav_active(False)` on *any* request — including a speed change. Nudging the slider mid-run silently aborts navigation. | [web_control.py:916-917](laptop/web_control.py#L916-L917) |
| 6.3 | **No laptop-side watchdog.** If the browser tab closes, `nav_loop` keeps running and keeps commanding motion. Only the Pi's 0.5 s watchdog saves you — and only if it is real. Nothing on the laptop notices the operator is gone. | [web_control.py:363](laptop/web_control.py#L363) |
| 6.4 | **`ARRIVE_M = 3.0`** — the car declares "arrived" within 3 m of B. That is larger than the car. Also, arrival is checked **only against the final waypoint**; intermediate waypoints are never "reached", they only shape the pure-pursuit path. | [controller.py:26](laptop/nav/controller.py#L26) |
| 6.5 | **`nav_loop` stores state on the function object** (`nav_loop._last_action`). This **persists across STOP→GO cycles** — a new run inherits the last run's action and commitment timer. | [web_control.py:433-434](laptop/web_control.py#L433-L434) |
| 6.6 | **`TURN_ANGLE = 0.6` is not used anywhere.** Named in HANDOFF as an uncalibrated placeholder — but grep shows it is **not in `ml/actions.py` at all**. The handoff is stale on this point. | `ml/actions.py` |
| 6.7 | **Steering deadband inconsistency.** Manual accumulates in steps of `MANUAL_STEER_STEP = 0.3`, so manual can command 0.3/0.6/0.9 — but **autonomous can only ever command 0.0/±1.0.** Manual mode is *more* proportional than autonomous mode. | [web_control.py:82](laptop/web_control.py#L82) |
| 6.8 | **Video stream shares the control link.** `/video_feed` streams MJPEG at ~15 fps over the same WiFi as the control commands, and `pollStatus` runs every 300 ms. On a hotspot this is a plausible cause of command gaps > the Pi's 0.5 s watchdog → auto-stop → jerky driving. | [web_control.py:1041-1051](laptop/web_control.py#L1041-L1051) |
| 6.9 | **Committed secret.** `laptop/secrets.yaml` (Google Maps API key) is in the repo. Rotate the key and gitignore the file. | `laptop/secrets.yaml` |

---

## 7. The `raspberry_pi/` folder in THIS repo is a decoy

It contains only `main.py`, `config.py`, `obstacle_monitor.py`, `sensor_reader.py`.
It is **missing** `motor_controller.py`, `steering_controller.py`, `remote_server.py`,
`safety_governor.py`, `gps_reader.py` — so it **cannot run**. The Pi runs a *different
repo* (`laptop_controlled_vehicle`). Anyone who reads `raspberry_pi/` here and believes
it will be debugging code that never executes.

**Recommendation: delete it, or add a README saying "NOT THE RUNNING CODE".**

---

## 8. What I still need before I can write the fix

The laptop code cannot answer these — they live on the Pi or on a tape measure.
Full prompt for Claude-Pi is in [PI_QUESTIONS.md](PI_QUESTIONS.md). The critical five:

1. **Does the Pi accept the `angle` field, and can it hold an arbitrary angle (e.g. 0.35)?**
   If it is bang-bang only, proportional steering is impossible and the whole design changes.
2. **Is the steering potentiometer real and working right now — ESP32 or Arduino A0?**
   (`HEAD` says "no pot". The working tree says "pot". Both cannot be true.)
3. **Wheelbase (metres) and steering lock-to-lock (degrees).** Required for the steering law.
4. **Which GPS module — M8N or M8L?** If M8L with DR enabled, low-speed heading is solved for free.
5. **Does the Pi independently veto motion?** Determines who owns obstacle avoidance.

---

## 9. Recommended order of work

1. **Untangle git.** Pick one steering design, commit it, get to a known state. *(Nothing
   else is meaningful until "what am I running" has an answer.)*
2. **Get the Pi contract** (PI_QUESTIONS.md).
3. **Fix heading** — gyro-integrated + GPS-COG anchoring, or M8L fused heading.
4. **Fix steering** — replace bang-bang with the pure-pursuit law.
5. **Disable the ML model** in the nav path until 3 and 4 work standalone.
6. **Split safety ownership** — Pi = reflex stop only; laptop = avoidance strategy.
7. *Then* tune (`ARRIVE_M`, lookahead, gains) against a car that actually tracks a line.
