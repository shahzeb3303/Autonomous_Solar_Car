# Prompt for Claude-Pi (paste this whole block into the Claude session on the Pi)

You are working on `~/laptop_controlled_vehicle` on the Raspberry Pi. I am the Claude
session on the laptop, working on the `vehicle_control` repo. We drive the same car:
your `main.py` is the TCP server on port 5555, my `web_control.py` is the client.

I am rewriting the laptop-side control stack so the car can drive **autonomously from
A to B** (GPS waypoints + obstacle avoidance) and **manually**. Before I write a single
line, I need the EXACT contract your side implements. Do not guess or describe what it
"should" do — read the actual source on the Pi and quote it.

Please answer ALL of the following with file:line citations and real code quotes.

---

## 1. Command channel (laptop -> Pi)

1.1 Paste the exact code that parses an incoming command JSON.
1.2 List EVERY key you accept, with its type, valid range, and what it does.
    In particular:
      - `command` — exact accepted string values?
      - `steer`   — exact accepted string values?
      - `speed`   — units? 0-100? PWM duty? Is 0 = stop?
      - `angle`   — **DO YOU ACCEPT THIS KEY AT ALL?** If yes:
                    * what range? (-1.0..+1.0?)
                    * which sign is RIGHT and which is LEFT?
                    * does it override `steer`, or is it ignored when `steer` is present?
                    * is it absolute wheel position, or a rate?
1.3 Framing: are commands newline-delimited? Length-prefixed? Do you rely on one JSON
    object per TCP segment? (The laptop currently does `sock.sendall(json.dumps(...))`
    with NO trailing newline — tell me if that breaks you, and what framing you want.)
1.4 What is your command rate expectation, and what happens if commands arrive faster
    or slower than that?

## 2. Watchdog / failsafe

2.1 What is the watchdog timeout value, and where is it in the code?
2.2 What EXACTLY happens on watchdog expiry — motors to zero? Steering recentered?
2.3 What happens if the TCP socket drops mid-drive while `command=FORWARD`?
2.4 Is there any way the car keeps moving after the laptop dies? Prove it either way.

## 3. Telemetry channel (Pi -> laptop)

3.1 Paste one REAL, VERBATIM telemetry line as it goes on the wire (run the server,
    capture an actual line, paste it). Not a schema — the real bytes.
3.2 Is it newline-delimited JSON? One object per line?
3.3 For EVERY field, give: exact key name, type, units, and range. I specifically need
    to know whether these exist, with their EXACT key names and nesting:
      - distances: `FL FR FW BC LS RS` — units (cm?), nesting (`distances.FL`?)
      - GPS: `valid`, `lat`, `lon`, `heading_deg`, `speed_mps`, `satellites` — exact
        names? nested under `gps`?
      - `steer_current_angle`  — **DOES THIS EXIST?** units? range? sign convention?
      - `imu` / `gyro_z`       — **DOES THIS EXIST?** units (deg/s?)? sign convention?
      - `steer_raw`            — raw potentiometer ADC? exists?
      - `actual_speed`
3.4 What is the telemetry rate (Hz)?

## 4. Steering hardware — THIS IS THE BIG ONE

The laptop repo's git history is contradictory. One commit says "gyro yaw-rate closed
loop, motor + MPU, **no pot**". The uncommitted working tree says "closed loop with a
**potentiometer**". The Arduino sketch has an uncommitted change adding a pot on A0.
I need the ground truth:

4.1 **Is there a steering potentiometer physically installed and working RIGHT NOW?**
4.2 If yes: is it read by the **ESP32** (`/dev/ttyACM0`) or by the **Arduino** (the one
    running `sensors_with_imu.ino`, publishing `steer_raw`)? Or both?
4.3 What are the current calibration constants (raw LEFT / CENTER / RIGHT values)?
4.4 Is there a closed-loop PD controller driving the steering to a commanded position?
    Paste it. What are KP/KD/MIN/MAX?
4.5 **Can I command an arbitrary steering angle like 0.35 and have the wheels go there
    and HOLD there?** Or does the Pi only understand bang-bang LEFT / RIGHT / STOP?
    This determines whether the laptop can do proportional steering at all.
4.6 Does the steering motor have a centering spring, or does it stay wherever it stops?
4.7 What is the steering **lock-to-lock range in real degrees** at the road wheels?
    (I need this for the pure-pursuit steering law. An estimate is fine — measure it.)
4.8 What is the **wheelbase** of the car in metres, front axle to rear axle? (Also
    needed for the steering law. Please measure.)

## 5. Drive motors

5.1 Speed: what does `speed: 80` physically mean? PWM duty? What is the minimum duty at
    which the car actually MOVES (deadband)? What speed in m/s does the car do at, say,
    50 and 80?
5.2 Is `MOTOR_INVERTED` set, i.e. does protocol `FORWARD` mean physical forward?
5.3 Can the car steer while stationary, or does it need to be rolling?

## 6. Sensors

6.1 Paste the code that parses the Arduino serial line.
6.2 What does a `-1` (or `-1.0`) distance mean, and how do you currently map it?
    Does `-1` reach the laptop as `-1`, or do you convert it (to 0? to 400?)?
6.3 Are LS/RS/BC/FL/FR/FW all currently returning real values? Which, if any, are dead
    right now?
6.4 Is there any smoothing/median filter on the ultrasonic readings, or is it raw?

## 7. Safety governor / obstacle logic ON THE PI

7.1 Does the Pi independently veto motion (i.e. would it refuse a `FORWARD` I send if an
    obstacle is close)? Paste that code and its thresholds.
7.2 Does the Pi do graduated slowdown? Paste it.
7.3 **This matters a lot**: if BOTH the Pi and the laptop are doing obstacle avoidance,
    they can fight each other. Tell me exactly what the Pi does so I can decide who owns
    the decision. My strong preference: **the Pi owns the hard safety stop (last-resort,
    reflex) and the laptop owns the avoidance strategy.** Tell me if the Pi's current
    behaviour is compatible with that.

## 8. GPS module

8.1 Which exact module is it — NEO-M8N? NEO-M8L? Something else? (Check the silkscreen
    on the board, and/or the UBX-MON-VER output.)
8.2 If it is an M8L (the automotive dead-reckoning variant), is dead reckoning actually
    ENABLED and calibrated? Does it output a fused heading that is valid at low speed?
    This would be much better than anything we can compute and changes my design.
8.3 What update rate is it configured for?
8.4 At what speed does the reported `heading_deg` become garbage? (Test: put the car on
    the ground, drive it slowly straight, log heading_deg vs speed_mps.)

## 9. Loop timing

9.1 What is the main loop rate on the Pi?
9.2 Is anything in the loop blocking (serial reads, sensor waits) that could stall it?
9.3 The laptop reports "jerky move-stop-move" driving. From the Pi side, is there
    anything that would cause that — watchdog firing, safety veto, serial stall?

---

## What I need back

A single markdown file `PI_CONTRACT.md` with:
- The verbatim command JSON schema you accept (and whether `angle` is real).
- The verbatim telemetry JSON schema you emit (a real captured line).
- The steering truth: pot or no pot, closed-loop or bang-bang, can it hold an angle.
- Wheelbase (m) and steering lock-to-lock (degrees).
- Watchdog timeout and failsafe behaviour.
- Who owns obstacle avoidance.

Be precise. If something does not exist, say "DOES NOT EXIST" plainly rather than
describing what it might do. I will design the laptop side directly against this, so a
wrong answer here becomes a car that drives into a wall.
