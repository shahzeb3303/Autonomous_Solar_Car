/*
 * ESP32 #2 — DEDICATED GYRO NODE.
 *
 * Reads the MPU-6500 and reports the yaw rate to the Pi as one JSON line per
 * cycle. Nothing else. It does not touch motors, ultrasonics, or the steering
 * potentiometer (that is ESP32 #1's job, on its own USB port).
 *
 * WHY THE CAR NEEDS THIS
 * ---------------------
 * The car cannot hold a straight line because it does not know which way it is
 * pointing. GPS course-over-ground only exists when the car is moving above
 * ~0.4 m/s, and this car lives right at that edge. Without a heading, navigation
 * is guesswork.
 *
 * The MPU-6500 has NO magnetometer (WHO_AM_I = 0x70 — it is a 6500 sold as a
 * "9250"), so there is no compass. But its GYRO works, and works well: it
 * saturates at +/-250 deg/s under rotation with only -0.30 deg/s of bias on Z.
 * Gyro + GPS is the standard heading solution for a slow ground vehicle: the gyro
 * is valid at ANY speed (even stopped, even pivoting), and GPS corrects its drift
 * whenever the car is moving fast enough to trust it.
 *
 * THE BUG THIS FIRMWARE EXISTS TO AVOID
 * ------------------------------------
 * The old Arduino firmware's mpuRead16() returned 0 on ANY failed I2C read. A
 * dead chip and a perfectly still chip therefore produced BYTE-IDENTICAL output.
 * Everyone concluded the IMU was broken. It never was — the bus was just flaky.
 *
 * So this firmware NEVER fabricates a value. If a read fails, it says so:
 *     {"gyro_z":0.00,"valid":false,"err":"i2c"}
 * `valid:false` means "I do not know", NOT "the car is not turning". The laptop
 * treats those very differently — one fails safe, the other steers on a lie.
 *
 * SIGN CONVENTION: none is applied. gyro_z is reported RAW, in deg/s, exactly as
 * the chip sees it. The laptop learns the sign itself by comparing against GPS
 * while driving (nav/heading.py), because the hand-rotation test was ambiguous
 * and a flipped sign would make the car steer away from its target and diverge.
 *
 * WIRING (ESP32 #2 <-> MPU-6500):
 *     VCC -> 3V3        (NOT 5V)
 *     GND -> GND
 *     SDA -> D21  (GPIO21)
 *     SCL -> D22  (GPIO22)
 *     AD0 -> GND        (I2C address 0x68)
 *     NCS -> 3V3        (selects I2C mode)
 *
 *   *** ALSO FIT PULL-UPS *** — the diagnostic showed repeated I2C READ FAILED:
 *     4.7k from SDA to 3V3
 *     4.7k from SCL to 3V3
 *   Keep SDA/SCL wires SHORT (<10cm) and AWAY from the motor wiring.
 *
 * OUTPUT (one line, ~50 Hz, 115200 baud):
 *     {"gyro_x":-0.95,"gyro_y":8.10,"gyro_z":-12.34,"yaw":137.4,"valid":true}
 */

#include <Wire.h>

#define SDA_PIN 21
#define SCL_PIN 18   // NOT 22 (not broken out) and NOT 35 (input-only, cannot
                     // drive the clock). 18 is output-capable and exposed.
#define MPU_ADDR     0x68
#define WHO_AM_I     0x75
#define PWR_MGMT_1   0x6B
#define PWR_MGMT_2   0x6C
#define GYRO_CONFIG  0x1B
#define GYRO_XOUT_H  0x43

const float GYRO_LSB_PER_DPS = 131.0;   // +/-250 dps full scale

// Per-axis bias, measured at boot. It MUST be per-axis: on this board Z reads
// -0.30 deg/s at rest but Y reads +8.10. One number for all three would be wrong.
float bias_x = 0, bias_y = 0, bias_z = 0;
bool  imu_ok = false;
float yaw_deg = 0.0;
unsigned long last_us = 0;

bool wr(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg); Wire.write(val);
  return Wire.endTransmission() == 0;
}

// Returns FALSE on failure. Never invents a value. This is the whole point.
bool rd(uint8_t reg, uint8_t n, uint8_t *buf) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((int)MPU_ADDR, (int)n) != n) return false;
  for (uint8_t i = 0; i < n; i++) buf[i] = Wire.read();
  return true;
}

bool readGyro(float *gx, float *gy, float *gz) {
  uint8_t b[6];
  if (!rd(GYRO_XOUT_H, 6, b)) return false;
  int16_t rx = (int16_t)((b[0] << 8) | b[1]);
  int16_t ry = (int16_t)((b[2] << 8) | b[3]);
  int16_t rz = (int16_t)((b[4] << 8) | b[5]);
  *gx = rx / GYRO_LSB_PER_DPS - bias_x;
  *gy = ry / GYRO_LSB_PER_DPS - bias_y;
  *gz = rz / GYRO_LSB_PER_DPS - bias_z;
  return true;
}

/* Measure the zero-rate offset — but ONLY if the car is genuinely still.
 *
 * If the car is moved while this runs, that motion is baked in as "zero" and the
 * heading then drifts for the entire run. So we check the spread of the samples
 * and refuse to accept a calibration taken while moving. */
bool calibrateBias() {
  const int N = 300;
  float sx = 0, sy = 0, sz = 0;
  float minz = 1e9, maxz = -1e9;
  int got = 0;

  for (int i = 0; i < N; i++) {
    uint8_t b[6];
    if (!rd(GYRO_XOUT_H, 6, b)) { delay(5); continue; }
    int16_t rz = (int16_t)((b[4] << 8) | b[5]);
    float z = rz / GYRO_LSB_PER_DPS;
    sx += (int16_t)((b[0] << 8) | b[1]) / GYRO_LSB_PER_DPS;
    sy += (int16_t)((b[2] << 8) | b[3]) / GYRO_LSB_PER_DPS;
    sz += z;
    if (z < minz) minz = z;
    if (z > maxz) maxz = z;
    got++;
    delay(5);
  }
  if (got < N / 2) return false;              // bus too unreliable to trust

  // A stationary gyro's yaw rate should barely move. A wide spread means the car
  // was being carried/bumped — reject rather than bake motion into the zero.
  if ((maxz - minz) > 5.0) return false;

  bias_x = sx / got;
  bias_y = sy / got;
  bias_z = sz / got;
  return true;
}

void setup() {
  Serial.begin(115200);
  delay(500);
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(50000);        // 50 kHz — slow and robust. The bus was flaky at 100k.
  delay(100);

  uint8_t who = 0;
  if (!rd(WHO_AM_I, 1, &who)) {
    Serial.println(F("{\"valid\":false,\"err\":\"no_i2c\"}"));
    return;                    // loop() will keep reporting valid:false
  }
  // 0x70 = MPU-6500, 0x71/0x73 = MPU-9250, 0x68 = MPU-6050. All have a usable gyro.
  if (who != 0x70 && who != 0x71 && who != 0x73 && who != 0x68) {
    Serial.print(F("{\"valid\":false,\"err\":\"who_am_i_0x"));
    Serial.print(who, HEX); Serial.println(F("\"}"));
    return;
  }

  wr(PWR_MGMT_1, 0x80); delay(120);   // reset
  wr(PWR_MGMT_1, 0x01); delay(50);    // wake, PLL clock
  wr(PWR_MGMT_2, 0x00); delay(20);    // all axes enabled
  wr(GYRO_CONFIG, 0x00); delay(50);   // +/-250 dps

  // HOLD THE CAR STILL DURING THIS (~1.5 s).
  if (!calibrateBias()) {
    Serial.println(F("{\"valid\":false,\"err\":\"bias_cal_failed_car_moving_or_bus_bad\"}"));
    return;
  }

  imu_ok = true;
  last_us = micros();
}

void loop() {
  if (!imu_ok) {
    // Say so, forever, rather than emitting plausible-looking zeros.
    Serial.println(F("{\"gyro_z\":0.00,\"valid\":false,\"err\":\"imu_init_failed\"}"));
    delay(200);
    return;
  }

  float gx, gy, gz;
  if (!readGyro(&gx, &gy, &gz)) {
    // A failed read is NOT "the car is not turning". Never pretend it is.
    Serial.println(F("{\"gyro_z\":0.00,\"valid\":false,\"err\":\"i2c\"}"));
    delay(20);
    return;
  }

  unsigned long now = micros();
  float dt = (now - last_us) * 1e-6f;
  last_us = now;
  if (dt > 0 && dt < 0.5f) {
    // Integrate locally too. The laptop mainly wants gyro_z, but a yaw integrated
    // here at 50 Hz is far better than one integrated from a 3 Hz sampled stream.
    if (fabs(gz) > 0.3f) yaw_deg += gz * dt;          // deadband kills bias creep
    while (yaw_deg < 0)     yaw_deg += 360.0f;
    while (yaw_deg >= 360)  yaw_deg -= 360.0f;
  }

  // gyro_z is RAW (bias-corrected, but no sign convention applied). The laptop
  // learns the sign from GPS — do not "help" by flipping it here.
  char line[128];
  snprintf(line, sizeof(line),
           "{\"gyro_x\":%.2f,\"gyro_y\":%.2f,\"gyro_z\":%.2f,\"yaw\":%.1f,\"valid\":true}",
           gx, gy, gz, yaw_deg);
  Serial.println(line);

  delay(20);      // ~50 Hz
}
