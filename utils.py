import numpy as np
import torch
def convertCartesianToPolar(x, y, BaseX, BaseY):
    Radius = np.sqrt(np.power(x - BaseX, 2) + np.power(y - BaseY, 2))
    Theta = np.arctan2(y - BaseY, x - BaseX)
    return Radius, Theta
def get_mixed(mixed, noise, snr):
        # snr = 0
        mix_std = np.std(mixed)
        noise_std = np.std(noise)
        noise_gain = np.sqrt(10 ** (-snr / 10) * np.power(mix_std, 2) / np.power(noise_std, 2))
        noise = noise_gain * noise
        return noise
def get_starting_angles(window_size):
    divisor = int(round(2*np.pi/window_size))
    return np.array(list(range(-divisor+1,divisor+2)))*np.pi/divisor
def to_categorical(index,num_classes):
    data = np.zeros((num_classes))
    data[index] = 1
    return data

def shift_mixture(input_data,target_position,mic_radius,sr,inverse = False):
    num_channels = input_data.shape[0]

    mic_array = [[
        mic_radius * np.cos(2*np.pi / num_channels * i),
        mic_radius * np.sin(2*np.pi / num_channels * i),
    ] for i in range(num_channels)]

    distance_mic0 = np.linalg.norm(mic_array[0] - target_position)
    shifts = [0]

    if isinstance(input_data, np.ndarray):
        shift_fn = np.roll
    elif isinstance(input_data,torch.tensor):
        shift_fn = torch.roll
    else:
        raise TypeError("unknown type")
    
    for channel_idx in range(1,num_channels):
        distance = np.linalg.norm(mic_array[channel_idx] - target_position)
        distance_diff= distance - distance_mic0
        shift_time = distance_diff/340.0
        shift_samples = int(round(sr*shift_time))
        if inverse:
            input_data[channel_idx] = shift_fn(input_data[channel_idx],shift_samples)
        else:
            input_data[channel_idx] = shift_fn(input_data[channel_idx],-shift_samples)
        shifts.append(shift_samples)
    return input_data,shifts

        