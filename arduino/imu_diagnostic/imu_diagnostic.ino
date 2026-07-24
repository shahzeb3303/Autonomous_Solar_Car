/*
 * IMU DIAGNOSTIC — run this INSTEAD of sensors_with_imu.ino, once.
 *
 * Why: the production firmware cannot distinguish "MPU working and still" from
 * "MPU not responding at all". mpuRead16() returns 0 on any I2C failure, the
 * bias calibration then averages those zeros to 0, and dps = (0-0)/131 = 0.
 * A dead chip and a stationary chip emit identical output. There is also no
 * WHO_AM_I check, so we have never actually confirmed the chip is present.
 *
 * This sketch removes every derived value and shows the ground truth:
 *   - WHO_AM_I               -> is the chip there, and WHICH chip is it?
 *   - I2C bus scan           -> is it at a different address (0x69)?
 *   - RAW LSB, all 3 axes    -> is it producing motion data at all, and on
 *                               WHICH axis is the car's yaw? (mounting check)
 *   - dps, all 3 axes        -> scaled, for the sign convention
 *
 * Upload, open Serial Monitor at 115200, then follow the printed instructions.
 * Nothing here touches the motors. USB power is enough.
 */

#include <Wire.h>

#define MPU_ADDR_A   0x68      // AD0 low
#define MPU_ADDR_B   0x69      // AD0 high
#define WHO_AM_I     0x75
#define PWR_MGMT_1   0x6B
#define PWR_MGMT_2   0x6C
#define GYRO_CONFIG  0x1B
#define GYRO_XOUT_H  0x43      // X, Y, Z are consecutive: 0x43, 0x45, 0x47

const float GYRO_LSB_PER_DPS = 131.0;   // ±250 dps full scale

uint8_t mpu_addr = 0;

bool wr(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(mpu_addr);
  Wire.write(reg); Wire.write(val);
  return Wire.endTransmission() == 0;
}

// Returns false if the I2C read FAILED (instead of silently pretending it's 0).
bool rd(uint8_t reg, uint8_t n, uint8_t *buf) {
  Wire.beginTransmission(mpu_addr);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((int)mpu_addr, (int)n) != n) return false;
  for (uint8_t i = 0; i < n; i++) buf[i] = Wire.read();
  return true;
}

void setup() {
  Serial.begin(115200);
  while (!Serial) {}
  delay(300);
  Wire.begin();
  Wire.setClock(100000);          // slower = more tolerant of long/noisy wiring
  delay(100);

  Serial.println(F("\n================ IMU DIAGNOSTIC ================"));

  // ---- 1. Scan the bus. Don't assume 0x68. ----
  Serial.println(F("\n[1] I2C bus scan:"));
  bool found_any = false;
  for (uint8_t a = 1; a < 127; a++) {
    Wire.beginTransmission(a);
    if (Wire.endTransmission() == 0) {
      Serial.print(F("    device found at 0x")); Serial.println(a, HEX);
      found_any = true;
      if (a == MPU_ADDR_A || a == MPU_ADDR_B) mpu_addr = a;
    }
  }
  if (!found_any) {
    Serial.println(F("    *** NOTHING ON THE I2C BUS. ***"));
    Serial.println(F("    The MPU is not electrically connected."));
    Serial.println(F("    Check: SDA->A4, SCL->A5, VCC(3.3V!), GND, AD0->GND."));
    Serial.println(F("    This alone explains gyro_z == 0. STOP HERE and fix wiring."));
    return;
  }
  if (mpu_addr == 0) {
    Serial.println(F("    Devices exist, but NONE at 0x68/0x69 -> that is not an MPU."));
    return;
  }

  // ---- 2. WHO_AM_I: which chip is this, really? ----
  uint8_t who = 0;
  Serial.print(F("\n[2] WHO_AM_I @0x")); Serial.print(mpu_addr, HEX); Serial.print(F(" = 0x"));
  if (!rd(WHO_AM_I, 1, &who)) { Serial.println(F("READ FAILED")); return; }
  Serial.print(who, HEX); Serial.print(F("  -> "));
  switch (who) {
    case 0x68: Serial.println(F("MPU-6050")); break;
    case 0x70: Serial.println(F("MPU-6500")); break;
    case 0x71:
    case 0x73: Serial.println(F("MPU-9250")); break;
    case 0x00:
    case 0xFF: Serial.println(F("*** BOGUS — chip is NOT really answering ***")); break;
    default:   Serial.println(F("*** UNKNOWN CHIP ***")); break;
  }

  // ---- 3. Wake it up PROPERLY (reset, then wake, then enable gyro). ----
  Serial.println(F("\n[3] Waking MPU (reset -> wake -> enable gyro -> +/-250dps)"));
  wr(PWR_MGMT_1, 0x80);           // device reset
  delay(120);
  wr(PWR_MGMT_1, 0x01);           // wake, clock = gyro X PLL (more stable than internal)
  delay(50);
  wr(PWR_MGMT_2, 0x00);           // make sure NO axis is left in standby
  delay(20);
  wr(GYRO_CONFIG, 0x00);          // +/-250 dps -> 131 LSB/dps
  delay(50);

  uint8_t chk = 0xAA;
  rd(PWR_MGMT_1, 1, &chk);
  Serial.print(F("    PWR_MGMT_1 readback = 0x")); Serial.println(chk, HEX);
  if (chk & 0x40) Serial.println(F("    *** STILL IN SLEEP MODE — gyro will read 0 ***"));

  // ---- 3b. THE BIG QUESTION: is there a MAGNETOMETER (= a real compass)? ----
  //
  // Boards are sold as "MPU9250/6500" because the seller will not commit to which
  // chip is on them, and a great many "MPU-9250" boards are really MPU-6500s.
  //   MPU-9250 -> has an AK8963 magnetometer -> A COMPASS -> absolute heading at
  //               any speed, even stopped. This solves the navigation problem.
  //   MPU-6500 -> gyro + accel only. NO compass.
  // The AK8963 sits behind the MPU's I2C bypass at address 0x0C. Open the bypass
  // and ask it who it is: it must answer 0x48.
  Serial.println(F("\n[3b] MAGNETOMETER (AK8963) — the thing that would fix navigation"));
  if (who == 0x71 || who == 0x73) {
    wr(0x37, 0x02);          // INT_PIN_CFG: BYPASS_EN = 1 -> expose the AK8963
    delay(20);
    Wire.beginTransmission(0x0C);
    if (Wire.endTransmission() == 0) {
      uint8_t sv = mpu_addr; mpu_addr = 0x0C;
      uint8_t wia = 0;
      bool got = rd(0x00, 1, &wia);        // AK8963 WIA register
      mpu_addr = sv;
      if (got && wia == 0x48) {
        Serial.println(F("     *** AK8963 FOUND (WIA=0x48). YOU HAVE A COMPASS. ***"));
        Serial.println(F("     -> absolute heading at any speed. This is what we need."));
      } else {
        Serial.print(F("     device at 0x0C but WIA=0x")); Serial.print(wia, HEX);
        Serial.println(F(" (expected 0x48) — not a working AK8963."));
      }
    } else {
      Serial.println(F("     NOTHING at 0x0C -> NO MAGNETOMETER."));
      Serial.println(F("     The chip claims to be a 9250 but the compass is absent/dead."));
    }
  } else {
    Serial.println(F("     WHO_AM_I is not a 9250 -> THIS BOARD HAS NO COMPASS."));
    Serial.println(F("     (MPU-6500 = gyro + accel only. Sold as '9250/6500'; it is a 6500.)"));
    Serial.println(F("     -> To get absolute heading you need a separate magnetometer"));
    Serial.println(F("        (HMC5883L / QMC5883L / LIS3MDL — a few dollars), or a"));
    Serial.println(F("        working gyro, or a dual-antenna GPS."));
  }

  Serial.println(F("\n[4] Streaming RAW gyro. Follow these steps:"));
  Serial.println(F("    (a) 3s  — hold PERFECTLY STILL   -> shows the noise floor + bias"));
  Serial.println(F("    (b) 5s  — YAW the car RIGHT, flat, clockwise seen from above"));
  Serial.println(F("    (c) 5s  — PITCH it (nose up/down), do NOT yaw"));
  Serial.println(F("    (d) 5s  — ROLL it (tilt side to side), do NOT yaw"));
  Serial.println(F("\n    The axis that spikes in (b) is the car's YAW axis."));
  Serial.println(F("    Its SIGN during (b) is the sign convention we need.\n"));
  Serial.println(F("      rawX   rawY   rawZ  |     dpsX     dpsY     dpsZ"));
  delay(1500);
}

void loop() {
  if (mpu_addr == 0) { delay(1000); return; }

  uint8_t b[6];
  if (!rd(GYRO_XOUT_H, 6, b)) {
    Serial.println(F("I2C READ FAILED  <-- this is the bug, not a still car"));
    delay(100);
    return;
  }
  int16_t gx = (int16_t)((b[0] << 8) | b[1]);
  int16_t gy = (int16_t)((b[2] << 8) | b[3]);
  int16_t gz = (int16_t)((b[4] << 8) | b[5]);

  char line[96];
  // Raw LSB first — if these do not move when you rotate, the chip is not
  // sensing, and no amount of scaling or sign-flipping downstream will help.
  snprintf(line, sizeof(line), "%6d %6d %6d  | %8.2f %8.2f %8.2f",
           gx, gy, gz,
           gx / GYRO_LSB_PER_DPS, gy / GYRO_LSB_PER_DPS, gz / GYRO_LSB_PER_DPS);
  Serial.println(line);
  delay(50);          // 20 Hz — fast enough to catch a hand rotation's peak
}
