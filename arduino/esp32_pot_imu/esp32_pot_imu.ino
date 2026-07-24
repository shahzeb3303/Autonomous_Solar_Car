/*
 * ESP32 — steering pot + MPU-6500 gyro, on ONE board, ONE serial line.
 *
 * Flash THIS onto the pot board (the 1a86 one). It keeps the exact pot reading
 * you already have and adds the gyro. Output, 115200 baud, one line per cycle:
 *
 *     raw: 2014  voltage: 1.62V  {"gyro_z":-12.34,"yaw":137.4,"valid":true}
 *
 * The pot part is byte-for-byte what the board prints today, so the Pi's existing
 * pot parser keeps working untouched. The {...} JSON is appended for the gyro.
 *
 * Wiring:
 *   Steering pot wiper -> POT_PIN (GPIO34 — see note below; change if different)
 *   MPU: SDA=GPIO21, SCL=GPIO18, addr 0x68, VCC->3V3, AD0->GND, NCS->3V3
 *
 * gyro_z is RAW deg/s (no sign convention — the laptop learns the sign from GPS).
 * A failed I2C read reports "valid":false, NEVER a fake 0 (that lie is what made
 * the IMU look dead for weeks).
 */

#include <Wire.h>

// ---- Steering pot ----
// CLASSIC ESP32 (DevKit) wiring: pot wiper on GPIO34 (ADC1, input-only — ideal for
// an analog input). Outer pot legs to 3V3 and GND.
// (On the earlier ESP32-S3 board the pot was on GPIO1; this build targets the
// classic ESP32 where the MPU is known to work.)
#define POT_PIN 34

// ---- MPU-6500 ----  (classic ESP32 default I2C pins)
#define SDA_PIN 21
#define SCL_PIN 22
#define MPU_ADDR     0x68
#define WHO_AM_I     0x75
#define PWR_MGMT_1   0x6B
#define PWR_MGMT_2   0x6C
#define GYRO_CONFIG  0x1B
#define GYRO_XOUT_H  0x43
const float GYRO_LSB_PER_DPS = 131.0;

float bias_x = 0, bias_y = 0, bias_z = 0;
bool  imu_ok = false;
float yaw_deg = 0.0;
unsigned long last_us = 0;

bool wr(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg); Wire.write(val);
  return Wire.endTransmission() == 0;
}

// FALSE on failure — never invents a value.
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
  *gx = (int16_t)((b[0] << 8) | b[1]) / GYRO_LSB_PER_DPS - bias_x;
  *gy = (int16_t)((b[2] << 8) | b[3]) / GYRO_LSB_PER_DPS - bias_y;
  *gz = (int16_t)((b[4] << 8) | b[5]) / GYRO_LSB_PER_DPS - bias_z;
  return true;
}

// Bias only if genuinely still; reject if the spread is wide (car was moving).
bool calibrateBias() {
  const int N = 300;
  float sx = 0, sy = 0, sz = 0, minz = 1e9, maxz = -1e9;
  int got = 0;
  for (int i = 0; i < N; i++) {
    uint8_t b[6];
    if (!rd(GYRO_XOUT_H, 6, b)) { delay(5); continue; }
    float z = (int16_t)((b[4] << 8) | b[5]) / GYRO_LSB_PER_DPS;
    sx += (int16_t)((b[0] << 8) | b[1]) / GYRO_LSB_PER_DPS;
    sy += (int16_t)((b[2] << 8) | b[3]) / GYRO_LSB_PER_DPS;
    sz += z;
    if (z < minz) minz = z;
    if (z > maxz) maxz = z;
    got++; delay(5);
  }
  if (got < N / 2) return false;
  if ((maxz - minz) > 5.0) return false;
  bias_x = sx / got; bias_y = sy / got; bias_z = sz / got;
  return true;
}

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);            // 0..4095, matches the ~2014 counts today

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(50000);                // 50 kHz — robust
  delay(100);

  uint8_t who = 0;
  if (rd(WHO_AM_I, 1, &who) &&
      (who == 0x70 || who == 0x71 || who == 0x73 || who == 0x68)) {
    wr(PWR_MGMT_1, 0x80); delay(120);
    wr(PWR_MGMT_1, 0x01); delay(50);
    wr(PWR_MGMT_2, 0x00); delay(20);
    wr(GYRO_CONFIG, 0x00); delay(50);
    if (calibrateBias()) { imu_ok = true; last_us = micros(); }
  }
  // If the MPU is absent/flaky, imu_ok stays false and we emit valid:false —
  // the pot still works regardless.
}

void loop() {
  // ---- pot: exactly the existing format ----
  int raw = analogRead(POT_PIN);
  float voltage = raw * 3.3 / 4095.0;
  Serial.print("raw: ");     Serial.print(raw);
  Serial.print("  voltage: "); Serial.print(voltage, 2); Serial.print("V  ");

  // ---- gyro: appended JSON ----
  if (!imu_ok) {
    Serial.println("{\"gyro_z\":0.00,\"valid\":false}");
    delay(20);
    return;
  }
  float gx, gy, gz;
  if (!readGyro(&gx, &gy, &gz)) {
    Serial.println("{\"gyro_z\":0.00,\"valid\":false}");   // unknown, NOT zero
    delay(20);
    return;
  }
  unsigned long now = micros();
  float dt = (now - last_us) * 1e-6f;
  last_us = now;
  if (dt > 0 && dt < 0.5f && fabs(gz) > 0.3f) {
    yaw_deg += gz * dt;
    while (yaw_deg < 0)    yaw_deg += 360.0f;
    while (yaw_deg >= 360) yaw_deg -= 360.0f;
  }
  char j[96];
  snprintf(j, sizeof(j), "{\"gyro_z\":%.2f,\"yaw\":%.1f,\"valid\":true}", gz, yaw_deg);
  Serial.println(j);

  delay(20);   // ~50 Hz
}
