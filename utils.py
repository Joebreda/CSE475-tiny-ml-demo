import os
import json
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

def read_imu_data(name):
    # Get the regions which are "positive" and which are "negative"
    directory = f'data'
    metadata_file = f'{name}_metadata.json'
    data_file = f'{name}_data.csv'
    imu_data_path = os.path.join(directory, data_file)
    imu_metadata_path = os.path.join(directory, metadata_file)

    if os.path.exists(imu_data_path):
        imu_data = pd.read_csv(imu_data_path)
    else:
        imu_data = []

    f = open(imu_metadata_path)
    metadata = json.load(f)
    imu_empirical_sample_rate = metadata['samples'] // metadata['duration']
    print(f'Sample rate of the original signal is {imu_empirical_sample_rate}')

    pretty_json = json.dumps(metadata, indent=4)
    print(pretty_json)
    return imu_data, imu_empirical_sample_rate, metadata

def continuous_to_windows(sensor_signal, sensor_fs, win_len_sec=0.96, overlap_frac=0.75):
    """ A naive process for sampling a sliding window across the continuous signal ignoring timestamps. """
    timeseries_len = sensor_signal.shape[0]
    window_len = int(win_len_sec * sensor_fs)
    overlap = (overlap_frac * window_len)
    step = window_len - overlap
    start_indices = np.arange(0, timeseries_len - window_len + 1, step).astype(int)
    end_indices = start_indices + window_len
    indices = list(zip(start_indices, end_indices))
    x_windows = []
    for start, end in indices:
        x_windows.append(sensor_signal[start:end])
    return np.array(x_windows)

def resample_non_uniform_sample_rate(timeseries, new_fs):
    # Extract original timestamps and data
    original_times = timeseries[:, 0]
    data = timeseries[:, 1]
    # Generate new timestamps based on desired frequency
    start_time = original_times[0]
    end_time = original_times[-1]
    new_time_points = np.arange(start_time, end_time, 1/new_fs)
    # Create interpolation function
    interpolation_function = interp1d(original_times, data, bounds_error=False, fill_value="extrapolate")
    # Interpolate data at new time points
    new_data = interpolation_function(new_time_points)
    # Create new array
    new_array = np.column_stack((new_time_points, new_data))
    return new_array