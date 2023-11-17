#include <Adafruit_LSM6DSOX.h>

// Include the model after running tensorflow_porter.py
#include "wiggle_detector.h"

// QTPY has 2 I2C buses so you need to pass wire1 to 
#include <Wire.h>
extern TwoWire Wire1;

unsigned long start_time;
int count;
Adafruit_LSM6DSOX sox;
byte startByte = 0x55;
byte stopByte = 0xAA;

// Adjust window size of buffer to correspond with window size expected from sensorflow. 
#define W 23
float imuDataX[W];
float imuDataY[W];
float imuDataZ[W];
int sampleIndex = 0;


void setup(void) {
  Serial.begin(115200);
  while (!Serial)
    delay(10); // will pause Zero, Leonardo, etc until serial console opens
  Serial.println("Adafruit LSM6Dsox test!");

  while (!wiggle_detector.begin()) {
      Serial.print("Error in NN initialization: ");
      Serial.println(wiggle_detector.getErrorMessage());
  }

  // IF USING I2C:
  Wire1.begin();
  Wire1.setClock(5000000UL); // Before doing this it gives 400Hz but after setting to I2C fast mode you get 1200Hz. 
  if (!sox.begin_I2C(LSM6DS_I2CADDR_DEFAULT, &Wire1, 0)) {
    while (1) {
      delay(1000);
      Serial.println("Failed to find LSM6Dsox chip");
    }
  }

  Serial.println("LSM6Dsox Found!");
}

void loop() {
  // Get a new normalized sensor event
  sensors_event_t accel;
  sensors_event_t gyro;
  sensors_event_t temp;
  sox.getEvent(&accel, &gyro, &temp);

  /*
  // UNCOMMENT FOR READING TRAINING DATA OVER SERIAL WHILE RUNNING read_serial.py
  Serial.write(startByte);

  // Write X, Y, and Z as 4 byte floats read from I2C to the unified sensor events.
  byte x_bytes[4]; // Create a byte array to hold the float bytes
  memcpy(x_bytes, &accel.acceleration.x, sizeof(accel.acceleration.x));
  // Send each byte of the float
  for (int i = 0; i < sizeof(accel.acceleration.x); i++) {
      Serial.write(x_bytes[i]);
  }

  byte y_bytes[4]; // Create a byte array to hold the float bytes
  memcpy(y_bytes, &accel.acceleration.y, sizeof(accel.acceleration.y));
  // Send each byte of the float
  for (int i = 0; i < sizeof(accel.acceleration.y); i++) {
      Serial.write(y_bytes[i]);
  }
  
  byte z_bytes[4]; // Create a byte array to hold the float bytes
  memcpy(z_bytes, &accel.acceleration.z, sizeof(accel.acceleration.z));
  // Send each byte of the float
  for (int i = 0; i < sizeof(accel.acceleration.z); i++) {
      Serial.write(z_bytes[i]);
  }

  Serial.write(stopByte);
  Serial.flush();
  */
  

  // UNCOMMENT FOR RUNNING INFERENCE. 
  // Store data in respective arrays
  imuDataX[sampleIndex] = accel.acceleration.x;
  imuDataY[sampleIndex] = accel.acceleration.y;
  imuDataZ[sampleIndex] = accel.acceleration.z;

  sampleIndex++;


  // Naively run inference every window size with 0 overlap. 
  if (sampleIndex >= W) {
      // Reset index
      sampleIndex = 0;

      // Flatten data
      float input[3 * W];
      for (int i = 0; i < W; i++) {
          input[i] = imuDataX[i];
          input[i + W] = imuDataY[i];
          input[i + 2 * W] = imuDataZ[i];
      }

      // Run inference
      float y_pred = wiggle_detector.predict(input);
      Serial.print("\t predicted: ");
      Serial.println(y_pred);
      Serial.println();
  }

  // Modify in order to adjust sample rate of system. 
  delay(20);
}