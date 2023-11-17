# CSE475-tiny-ml-demo

This repo contains the arduino code for ESP32 + LSM IMU from CSE475 embedded ML systems tiny ML demo. 

To replicate demo:
1. Uncomment line 55-83 and comment out 83 to 113 of arduino code to write data to serial and run read_serial.py to recieve data locally. 
2. Repeat for all desired classes (current configuration is for 2 classes: a positive gesture and negative examples). 
3. Run tensorflow_porter.py to train the model using the data you collect and convert model to C.
4. Return arduino code to original state with inference mode uncommented. 
5. Test inference. 
