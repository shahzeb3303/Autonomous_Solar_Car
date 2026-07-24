/*
 * IMU DIAGNOSTIC — ESP32 version.
 *
 * ONE QUESTION: does this IMU board actually have a MAGNETOMETER (a compass)?
 *
 * That is the whole ballgame. The car cannot drive straight because it does not
 * know which way it is pointing. It currently guesses by differencing GPS
 * positions, and with ~2-3 m of GPS noise over ~4 m of travel that guess can be
 * 30-40 degrees wrong — so it drives straight for 5-10 m, acquires a bad heading,
 * "corrects" toward it, and veers off.
 *
 * A compass fixes that outright: absolute heading, at any speed, even stopped.
 *
 * Boards are sold branded "MPU9250/6500" because the seller will not commit to
 * which chip is fitted. They are NOT the same:
 *     MPU-9250 -> has an AK8963 magnetometer  -> A COMPASS. Solves the problem.
 *     MPU-6500 -> gyro + accel only           -> NO compass. Useless for heading.
 * Many boards sold as "9250" are really 6500s. WHO_AM_I is the only truth.
 *
 * WIRING (ESP32 <-> MPU board):
 *     MPU VCC -> ESP32 3V3     <-- 3.3V, NOT 5V
 *     MPU GND -> ESP32 GND
 *     MPU SDA -> ESP32 GPIO21
 *     MPU SCL -> ESP32 GPIO22
 *     MPU AD0 -> GND  (sets I2C address 0x68)
 *     MPU NCS -> 3V3  (selects I2C mode, not SPI)
 *
 * Upload, open Serial Monitor at 115200, and send me the whole output.
 * Nothing here touches the motors. USB power is enough.
 */

#include <Wire.h>

#define SDA_PIN 21
#define SCL_PIN 22

#define WHO_AM_I     0x75
#define PWR_MGMT_1   0x6B
#define PWR_MGMT_2   0x6C
#define GYRO_CONFIG  0x1B
#define INT_PIN_CFG  0x37
#define GYRO_XOUT_H  0x43      // X, Y, Z are consecutive: 0x43, 0x45, 0x47

#define AK8963_ADDR  0x0C      // the magnetometer, behind the MPU's I2C bypass
#define AK8963_WIA   0x00      // must read 0x48
#define AK8963_CNTL1 0x0A
#define AK8963_HXL   0x03
#define AK8963_ST2   0x09

const float GYRO_LSB_PER_DPS = 131.0;   // +/-250 dps full scale

uint8_t mpu_addr = 0;
bool has_compass = false;

bool wr(uint8_t addr, uint8_t reg, uint8_t val) {
  Wire.beginTransmission(addr);
  Wire.write(reg); Wire.write(val);
  return Wire.endTransmission() == 0;
}

// Returns false if the read FAILED, instead of silently pretending the value is 0.
// (The production firmware returns 0 on I2C failure, which is why a DEAD imu and a
// STILL imu produced byte-identical output and nobody noticed for weeks.)
bool rd(uint8_t addr, uint8_t reg, uint8_t n, uint8_t *buf) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((int)addr, (int)n) != n) return false;
  for (uint8_t i = 0; i < n; i++) buf[i] = Wire.read();
  return true;
}

void setup() {
  Serial.begin(115200);
  delay(600);
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(100000);          // slow = tolerant of long/noisy jumper wires
  delay(100);

  Serial.println(F("\n=========== IMU DIAGNOSTIC (ESP32) ==========="));

  // ---- 1. Scan the bus. Assume nothing. ----
  Serial.println(F("\n[1] I2C scan (SDA=21, SCL=22):"));
  bool any = false;
  for (uint8_t a = 1; a < 127; a++) {
    Wire.beginTransmission(a);
    if (Wire.endTransmission() == 0) {
      Serial.print(F("    found 0x")); Serial.println(a, HEX);
      any = true;
      if (a == 0x68 || a == 0x69) mpu_addr = a;
    }
  }
  if (!any) {
    Serial.println(F("    *** NOTHING ON THE BUS. The IMU is not connected. ***"));
    Serial.println(F("    Check: SDA->21, SCL->22, VCC->3V3, GND->GND, AD0->GND."));
    return;
  }
  if (!mpu_addr) {
    Serial.println(F("    Devices exist but none at 0x68/0x69 -> that is not an MPU."));
    return;
  }

  // ---- 2. WHICH CHIP IS THIS, REALLY? ----
  uint8_t who = 0;
  Serial.print(F("\n[2] WHO_AM_I @0x")); Serial.print(mpu_addr, HEX); Serial.print(F(" = 0x"));
  if (!rd(mpu_addr, WHO_AM_I, 1, &who)) { Serial.println(F("READ FAILED")); return; }
  Serial.print(who, HEX); Serial.print(F("  -> "));
  switch (who) {
    case 0x68: Serial.println(F("MPU-6050  (no compass)")); break;
    case 0x70: Serial.println(F("MPU-6500  (NO COMPASS — it is a 6500, not a 9250)")); break;
    case 0x71:
    case 0x73: Serial.println(F("MPU-9250  (should have a compass — checking...)")); break;
    default:   Serial.println(F("*** UNKNOWN / not answering properly ***")); break;
  }

  // ---- 3. Wake it properly ----
  wr(mpu_addr, PWR_MGMT_1, 0x80); delay(120);   // reset
  wr(mpu_addr, PWR_MGMT_1, 0x01); delay(50);    // wake, PLL clock
  wr(mpu_addr, PWR_MGMT_2, 0x00); delay(20);    // no axis in standby
  wr(mpu_addr, GYRO_CONFIG, 0x00); delay(50);   // +/-250 dps
  uint8_t chk = 0xAA;
  rd(mpu_addr, PWR_MGMT_1, 1, &chk);
  Serial.print(F("    PWR_MGMT_1 readback = 0x")); Serial.println(chk, HEX);
  if (chk & 0x40) Serial.println(F("    *** STILL ASLEEP — gyro will read 0 ***"));

  // ---- 4. THE DECISIVE TEST: IS THERE A COMPASS? ----
  Serial.println(F("\n[3] MAGNETOMETER — this is the question that matters"));
  wr(mpu_addr, INT_PIN_CFG, 0x02);      // BYPASS_EN -> expose the AK8963 on the bus
  delay(20);
  Wire.beginTransmission(AK8963_ADDR);
  if (Wire.endTransmission() == 0) {
    uint8_t wia = 0;
    if (rd(AK8963_ADDR, AK8963_WIA, 1, &wia) && wia == 0x48) {
      has_compass = true;
      Serial.println(F("    *****************************************************"));
      Serial.println(F("    ***  AK8963 FOUND (WIA=0x48). YOU HAVE A COMPASS. ***"));
      Serial.println(F("    *****************************************************"));
      Serial.println(F("    -> Absolute heading at ANY speed, even stopped."));
      Serial.println(F("    -> This is exactly what the car is missing."));
      wr(AK8963_ADDR, AK8963_CNTL1, 0x00); delay(10);   // power down
      wr(AK8963_ADDR, AK8963_CNTL1, 0x16); delay(10);   // 16-bit, continuous 100Hz
    } else {
      Serial.print(F("    device at 0x0C but WIA=0x")); Serial.print(wia, HEX);
      Serial.println(F(" (expected 0x48) — not a working AK8963."));
    }
  } else {
    Serial.println(F("    NOTHING at 0x0C  ->  *** NO MAGNETOMETER ON THIS BOARD ***"));
    Serial.println(F("    It is an MPU-6500 sold as a '9250'. Gyro + accel only."));
    Serial.println(F("    -> Buy a QMC5883L or HMC5883L compass module (~$2) and hang"));
    Serial.println(F("       it on this same I2C bus (SDA 21 / SCL 22). That fixes it."));
  }

  Serial.println(F("\n[4] Live data. Now DO THIS:"));
  Serial.println(F("    (a) 3s  hold PERFECTLY STILL"));
  Serial.println(F("    (b) 5s  rotate the board FLAT, clockwise seen from above"));
  Serial.println(F("    (c) 5s  tilt it (pitch/roll), do NOT rotate flat"));
  Serial.println(F("\n    gyro: the axis that spikes in (b) is YAW; its sign is our convention."));
  if (has_compass)
    Serial.println(F("    mag : heading should sweep through 360 as you rotate in (b)."));
  Serial.println();
  delay(1500);
}

void loop() {
  if (!mpu_addr) { delay(1000); return; }

  uint8_t b[6];
  if (!rd(mpu_addr, GYRO_XOUT_H, 6, b)) {
    Serial.println(F("I2C READ FAILED  <-- the bug is here, not a still car"));
    delay(200);
    return;
  }
  int16_t gx = (int16_t)((b[0] << 8) | b[1]);
  int16_t gy = (int16_t)((b[2] << 8) | b[3]);
  int16_t gz = (int16_t)((b[4] << 8) | b[5]);

  char line[140];
  snprintf(line, sizeof(line), "gyro raw %6d %6d %6d | dps %7.1f %7.1f %7.1f",
           gx, gy, gz,
           gx / GYRO_LSB_PER_DPS, gy / GYRO_LSB_PER_DPS, gz / GYRO_LSB_PER_DPS);
  Serial.print(line);

  if (has_compass) {
    uint8_t m[7];
    // Reading ST2 (the 7th byte) is REQUIRED — it latches the measurement.
    if (rd(AK8963_ADDR, AK8963_HXL, 7, m) && !(m[6] & 0x08)) {   // bit3 = overflow
      int16_t mx = (int16_t)((m[1] << 8) | m[0]);   // AK8963 is little-endian
      int16_t my = (int16_t)((m[3] << 8) | m[2]);
      int16_t mz = (int16_t)((m[5] << 8) | m[4]);
      float hdg = atan2((float)my, (float)mx) * 57.29578f;
      if (hdg < 0) hdg += 360.0f;
      snprintf(line, sizeof(line), "  | mag %6d %6d %6d  HEADING %5.1f deg",
               mx, my, mz, hdg);
      Serial.print(line);
    } else {
      Serial.print(F("  | mag read failed/overflow"));
    }
  }
  Serial.println();
  delay(100);
}
