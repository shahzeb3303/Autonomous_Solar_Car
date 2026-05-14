/*
 * Ultrasonics + MPU-6500/6050/9250 — combined firmware for the autonomous car.
 *
 * Pin map (UNCHANGED from all_6_sensors.ino):
 *   FL: TRIG=5, ECHO=4
 *   FR: TRIG=3, ECHO=2
 *   FW: TRIG=13, ECHO=12
 *   BC: TRIG=11, ECHO=10
 *   RS: TRIG=9, ECHO=8
 *   LS: TRIG=7, ECHO=6
 *
 * MPU on hardware I2C (A4=SDA, A5=SCL), address 0x68, NCS pulled HIGH, AD0 LOW.
 *
 * Output: one JSON line per cycle on Serial @ 115200 baud, e.g.
 *   {"FL":50.1,"FR":-1,"FW":150.3,"BC":117.1,"LS":156.0,"RS":14.0,"heading":92.5,"gyro_z":0.2}
 * Values of -1 mean "no echo" (sensor saw nothing or timed out).
 */

#include <Wire.h>

// ============ Ultrasonic pins ============
#define TRIG_FL 5
#define ECHO_FL 4
#define TRIG_FR 3
#define ECHO_FR 2
#define TRIG_FW 13
#define ECHO_FW 12
#define TRIG_BC 11
#define ECHO_BC 10
#define TRIG_RS 9
#define ECHO_RS 8
#define TRIG_LS 7
#define ECHO_LS 6

// ============ MPU registers ============
#define MPU_ADDR     0x68
#define PWR_MGMT_1   0x6B
#define GYRO_CONFIG  0x1B
#define ACCEL_CONFIG 0x1C
#define GYRO_ZOUT_H  0x47

// MPU sensitivity for default ranges
// Gyro: ±250°/s -> 131 LSB per °/s
const float GYRO_LSB_PER_DPS = 131.0;

// ============ State ============
float yaw_deg = 0.0;
float gyro_z_bias = 0.0;
float gyro_z_dps_last = 0.0;
unsigned long last_imu_us = 0;

// ============ MPU helpers ============
bool mpuWrite(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  return Wire.endTransmission() == 0;
}

int16_t mpuRead16(uint8_t reg) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return 0;
  if (Wire.requestFrom((int)MPU_ADDR, 2) != 2) return 0;
  uint8_t hi = Wire.read();
  uint8_t lo = Wire.read();
  return (int16_t)((hi << 8) | lo);
}

void calibrateGyroBias() {
  long sum = 0;
  const int N = 200;
  for (int i = 0; i < N; i++) {
    sum += mpuRead16(GYRO_ZOUT_H);
    delay(5);
  }
  gyro_z_bias = sum / (float)N;
}

void updateIMU() {
  unsigned long now = micros();
  if (last_imu_us == 0) { last_imu_us = now; return; }
  float dt = (now - last_imu_us) * 1e-6f;
  last_imu_us = now;
  if (dt <= 0 || dt > 0.5f) return;

  int16_t raw = mpuRead16(GYRO_ZOUT_H);
  float dps = (raw - gyro_z_bias) / GYRO_LSB_PER_DPS;
  gyro_z_dps_last = dps;
  yaw_deg += dps * dt;
  while (yaw_deg < 0)     yaw_deg += 360.0;
  while (yaw_deg >= 360)  yaw_deg -= 360.0;
}

// ============ Ultrasonic ============
float readCm(int trig, int echo) {
  digitalWrite(trig, LOW);
  delayMicroseconds(3);
  digitalWrite(trig, HIGH);
  delayMicroseconds(10);
  digitalWrite(trig, LOW);
  long dur = pulseIn(echo, HIGH, 30000);
  if (dur == 0) return -1.0;
  float cm = dur * 0.01715;
  if (cm < 2 || cm > 400) return -1.0;
  return cm;
}

float readSensorWithIMU(int trig, int echo) {
  updateIMU();
  float v = readCm(trig, echo);
  updateIMU();
  return v;
}

// ============ Setup ============
void setup() {
  Serial.begin(115200);

  int trigs[] = {TRIG_FL, TRIG_FR, TRIG_FW, TRIG_BC, TRIG_RS, TRIG_LS};
  int echos[] = {ECHO_FL, ECHO_FR, ECHO_FW, ECHO_BC, ECHO_RS, ECHO_LS};
  for (int i = 0; i < 6; i++) {
    pinMode(trigs[i], OUTPUT);
    pinMode(echos[i], INPUT);
    digitalWrite(trigs[i], LOW);
  }

  Wire.begin();
  Wire.setClock(400000);
  delay(50);

  mpuWrite(PWR_MGMT_1, 0x00);
  delay(100);
  mpuWrite(GYRO_CONFIG, 0x00);
  mpuWrite(ACCEL_CONFIG, 0x00);
  delay(50);

  calibrateGyroBias();
  last_imu_us = micros();
}

// ============ Loop ============
void loop() {
  float fl = readSensorWithIMU(TRIG_FL, ECHO_FL); delay(20);
  float fr = readSensorWithIMU(TRIG_FR, ECHO_FR); delay(20);
  float fw = readSensorWithIMU(TRIG_FW, ECHO_FW); delay(20);
  float bc = readSensorWithIMU(TRIG_BC, ECHO_BC); delay(20);
  float ls = readSensorWithIMU(TRIG_LS, ECHO_LS); delay(20);
  float rs = readSensorWithIMU(TRIG_RS, ECHO_RS);
  updateIMU();

  Serial.print(F("{\"FL\":")); Serial.print(fl, 1);
  Serial.print(F(",\"FR\":"));  Serial.print(fr, 1);
  Serial.print(F(",\"FW\":"));  Serial.print(fw, 1);
  Serial.print(F(",\"BC\":"));  Serial.print(bc, 1);
  Serial.print(F(",\"LS\":"));  Serial.print(ls, 1);
  Serial.print(F(",\"RS\":"));  Serial.print(rs, 1);
  Serial.print(F(",\"heading\":")); Serial.print(yaw_deg, 1);
  Serial.print(F(",\"gyro_z\":"));  Serial.print(gyro_z_dps_last, 2);
  Serial.println(F("}"));
}
