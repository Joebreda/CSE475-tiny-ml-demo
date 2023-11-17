import os
import json
import threading
import serial
import pyaudio
import time
import csv
from pynput import keyboard
import numpy as np
import struct
import serial.tools.list_ports
import tkinter as tk

def write_3_axis_IMU_data_to_file(gesture, buffers, metadata):
    x, y, z = buffers
    directory = f'data'
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    data_file = f'{gesture}_data.csv'
    metadata_file = f'{gesture}_metadata.json'
    data_path = os.path.join(directory, data_file)
    metadata_path = os.path.join(directory, metadata_file)

    min_len = min(len(x), len(y), len(z))
    with open(data_path, 'w', newline='\n') as f:  # newline='' to handle newlines correctly in CSVs
        writer = csv.writer(f)
        # Write headers if needed (you can remove this if headers are not required)
        writer.writerow(['x', 'y', 'z'])
        # Write the buffers to the CSV file
        for i in range(min_len):
            writer.writerow([x[i], y[i], z[i]])
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)
    print(f"IMU data saved to {data_path}")


def read_serial(stop_event, gesture):
    serial_port = '/dev/cu.usbmodem2101'
    accel_range = 2
    baud_rate = 115200 
    xbuffer = []
    ybuffer = []
    zbuffer = []
    cum_sum_latency = 0
    read_count = 0
    PRINT = True
    prev_time = time.time()
    start_time = time.time()

    ser = serial.Serial(serial_port, baud_rate)

    print("starting serial connection...")

    def i2c_bytes_to_float(byte_data):
        # Make sure we have exactly 4 bytes
        if len(byte_data) != 4:
            raise ValueError("Invalid data length, must be 4 bytes when reading float over I2C.")

        # Unpack 4 bytes into a float
        # The format '<f' means little-endian float. Use '>f' for big-endian.
        return struct.unpack('<f', byte_data)[0]
    
    def spi_bytes_to_float(bytes):
        if len(bytes) != 2:
            raise ValueError("Invalid data length, must be 2 bytes when reading float over SPI.")

        value = np.frombuffer(bytes, dtype=np.int16)[0]
        float_val = value * 0.061 * (accel_range >> 1) / 1000;
        return float_val

    # While the serial port is open
    # TODO to fix the sample rate it is probably easiest to wait a certain interval and then flush serial and read. but this makes it harder given start and stop bytes. 
    while not stop_event.is_set(): 
        start_byte = ser.read(1)
        if start_byte == b'\x55':
            # Read 6 bytes for the data containing x, y, and z low then high byte. 
            xdata = ser.read(4)
            ydata = ser.read(4)
            zdata = ser.read(4)
            stop_byte = ser.read(1)
            if stop_byte == b'\xAA':
                x_accel = i2c_bytes_to_float(xdata)
                y_accel = i2c_bytes_to_float(ydata)
                z_accel = i2c_bytes_to_float(zdata)
                xbuffer.append(x_accel)
                ybuffer.append(y_accel)
                zbuffer.append(z_accel)
                time_of_receive = time.time() - prev_time 
                if PRINT:
                    print(f"Recieved {x_accel}, {y_accel}, {z_accel}")
                else:
                    time.sleep(0.00005)
                cum_sum_latency += time_of_receive
                read_count += 1
                prev_time = time.time()
            else:
                print("Error: incorrect stop byte.")
        else:
            print("Error: incorrect start byte.")
    print("stopping reading serial...")
    
    print("Flushing and closing serial.")
    if ser.is_open:
        try:
            ser.flush()
            ser.close()
        except serial.serialutil.PortNotOpenError as e:
            print(f"An error occurred while trying to flush serial: {e}")
    finish_time = time.time()
    metadata = {
            'duration': finish_time - start_time,
            'samples': len(xbuffer),
            'avg_latency': cum_sum_latency / read_count,
            'avg_sample_rate': 1 / (cum_sum_latency / read_count)
        }
    write_3_axis_IMU_data_to_file(gesture, (xbuffer, ybuffer, zbuffer), metadata)

# Main function
def main():
    print('enter a gesture name: ')
    gesture = input()
    stop_event = threading.Event()
    serial_thread = threading.Thread(target=read_serial, args=(stop_event,gesture,))
    serial_thread.start()
    input("Press Enter to stop all threads...\n")
    stop_event.set()
    serial_thread.join()

if __name__ == "__main__":
    main()
