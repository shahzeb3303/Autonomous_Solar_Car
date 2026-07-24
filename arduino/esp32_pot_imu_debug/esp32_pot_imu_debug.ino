/*
 * DEBUG build of the merged pot+gyro firmware.
 *
 * Same wiring/pins as esp32_pot_imu (classic ESP32: pot GPIO34, MPU SDA21/SCL22).
 * The pot still works so steering is not lost. The difference: every line now
 * reports EXACTLY why the IMU is (not) reading, and it RE-TRIES the MPU init every
 * second, so a flaky/intermittent failure shows up as the reason flickering rather
 * than latching once at boot.
 *
 * Purpose: the board reads the MPU fine on a laptop but reports valid:false on the
 * Pi. This tells us which step fails on the Pi — bus/power, address (AD0), chip-id,
 * or bias calibration.
 *
 * Each line, 115200 baud:
 *   raw: 2694  voltage: 2.17V  {"gyro_z":-0.2,"valid":true,"dbg":"ok","scan":"0x68","whoami":"0x70"}
 * or on failure e.g.:
 *   raw: 2694  voltage: 2.17V  {"gyro_z":0.00,"valid":false,"dbg":"no_bus_device","scan":"none","whoami":"n/a"}
 *
 * dbg codes:
 *   ok                 - working
 *   no_bus_device      - I2C scan found NOTHING  -> power (3V3 sag on Pi USB?), GND,
 *                        SDA/SCL not connected, or NCS not high (chip in SPI mode)
 *   only_0x69          - device answers at 0x69, not 0x68 -> AD0 is HIGH, tie it to GND
 *   whoami_read_fail   - device seen on scan but WHO_AM_I read failed -> flaky bus / marginal power
 *   whoami_unexpected  - WHO_AM_I returned an unexpected byte (see "whoami" field)
 *   bias_fail_reads    - could not get enough clean reads to calibrate -> flaky bus
 *   bias_fail_moving   - readings too noisy/wide to be "still" -> vibration, or motor noise on the Pi
 */

#include <Wire.h>

#define POT_PIN 34
#define SDA_PIN 21
#define SCL_PIN 22
#define MPU_ADDR     0x68
#define MPU_ADDR_ALT 0x69
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
unsigned long last_retry = 0;

char dbg[24]   = "init";
char scanstr[40] = "?";
char whoamistr[8] = "n/a";

bool wr(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg); Wire.write(val);
  return Wire.endTransmission() == 0;
}
bool rd(uint8_t addr, uint8_t reg, uint8_t n, uint8_t *buf) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((int)addr, (int)n) != n) return false;
  for (uint8_t i = 0; i < n; i++) buf[i] = Wire.read();
  return true;
}

bool readGyro(float *gx, float *gy, float *gz) {
  uint8_t b[6];
  if (!rd(MPU_ADDR, GYRO_XOUT_H, 6, b)) return false;
  *gx = (int16_t)((b[0] << 8) | b[1]) / GYRO_LSB_PER_DPS - bias_x;
  *gy = (int16_t)((b[2] << 8) | b[3]) / GYRO_LSB_PER_DPS - bias_y;
  *gz = (int16_t)((b[4] << 8) | b[5]) / GYRO_LSB_PER_DPS - bias_z;
  return true;
}

bool calibrateBias() {
  const int N = 200;
  float sx = 0, sy = 0, sz = 0, minz = 1e9, maxz = -1e9;
  int got = 0;
  for (int i = 0; i < N; i++) {
    uint8_t b[6];
    if (!rd(MPU_ADDR, GYRO_XOUT_H, 6, b)) { delay(4); continue; }
    float z = (int16_t)((b[4] << 8) | b[5]) / GYRO_LSB_PER_DPS;
    sx += (int16_t)((b[0] << 8) | b[1]) / GYRO_LSB_PER_DPS;
    sy += (int16_t)((b[2] << 8) | b[3]) / GYRO_LSB_PER_DPS;
    sz += z;
    if (z < minz) minz = z;
    if (z > maxz) maxz = z;
    got++; delay(4);
  }
  if (got < N / 2) { strcpy(dbg, "bias_fail_reads"); return false; }
  if ((maxz - minz) > 6.0) { strcpy(dbg, "bias_fail_moving"); return false; }
  bias_x = sx / got; bias_y = sy / got; bias_z = sz / got;
  return true;
}

// Full IMU bring-up with diagnostics. Sets imu_ok and dbg/scan/whoami.
void tryInitIMU() {
  // 1. Scan the bus.
  int n68 = 0, n69 = 0, nother = 0;
  for (uint8_t a = 1; a < 127; a++) {
    Wire.beginTransmission(a);
    if (Wire.endTransmission() == 0) {
      if (a == MPU_ADDR) n68++; else if (a == MPU_ADDR_ALT) n69++; else nother++;
    }
  }
  if (n68) strcpy(scanstr, "0x68");
  else if (n69) strcpy(scanstr, "0x69");
  else if (nother) strcpy(scanstr, "other");
  else strcpy(scanstr, "none");

  if (!n68 && !n69 && !nother) { imu_ok = false; strcpy(dbg, "no_bus_device"); return; }
  if (!n68 && n69) { imu_ok = false; strcpy(dbg, "only_0x69"); return; }
  if (!n68) { imu_ok = false; strcpy(dbg, "wrong_addr"); return; }

  // 2. WHO_AM_I.
  uint8_t who = 0;
  if (!rd(MPU_ADDR, WHO_AM_I, 1, &who)) { imu_ok = false; strcpy(dbg, "whoami_read_fail"); strcpy(whoamistr, "fail"); return; }
  snprintf(whoamistr, sizeof(whoamistr), "0x%02X", who);
  if (who != 0x70 && who != 0x71 && who != 0x73 && who != 0x68) {
    imu_ok = false; strcpy(dbg, "whoami_unexpected"); return;
  }

  // 3. Wake + configure.
  wr(PWR_MGMT_1, 0x80); delay(120);
  wr(PWR_MGMT_1, 0x01); delay(50);
  wr(PWR_MGMT_2, 0x00); delay(20);
  wr(GYRO_CONFIG, 0x00); delay(50);

  // 4. Bias (sets its own dbg on failure).
  if (!calibrateBias()) { imu_ok = false; return; }

  imu_ok = true; strcpy(dbg, "ok"); last_us = micros();
}

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(50000);
  delay(100);
  tryInitIMU();
  last_retry = millis();
}

void loop() {
  // pot (unchanged, keeps steering alive)
  int raw = analogRead(POT_PIN);
  Serial.print("raw: "); Serial.print(raw);
  Serial.print("  voltage: "); Serial.print(raw * 3.3 / 4095.0, 2); Serial.print("V  ");

  // Re-try the IMU init every ~1.5s while it is down, so an intermittent fault
  // shows up as the reason changing, instead of latching once at boot.
  if (!imu_ok && millis() - last_retry > 1500) {
    last_retry = millis();
    tryInitIMU();
  }

  float gz = 0.0; bool ok = false;
  if (imu_ok) {
    float gx, gy, tz;
    if (readGyro(&gx, &gy, &tz)) {
      gz = tz; ok = true;
      unsigned long now = micros();
      float dt = (now - last_us) * 1e-6f; last_us = now;
      if (dt > 0 && dt < 0.5f && fabs(gz) > 0.3f) {
        yaw_deg += gz * dt;
        while (yaw_deg < 0) yaw_deg += 360.0f;
        while (yaw_deg >= 360) yaw_deg -= 360.0f;
      }
    } else {
      imu_ok = false; strcpy(dbg, "read_lost");   // was working, then a read failed
    }
  }

  char j[160];
  snprintf(j, sizeof(j),
    "{\"gyro_z\":%.2f,\"yaw\":%.1f,\"valid\":%s,\"dbg\":\"%s\",\"scan\":\"%s\",\"whoami\":\"%s\"}",
    gz, yaw_deg, ok ? "true" : "false", dbg, scanstr, whoamistr);
  Serial.println(j);

  delay(50);
}
