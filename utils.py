import numpy as np
import torch
from scipy.signal import butter, filtfilt
import sounddevice as sd
import librosa
import os
import csv
import soundfile as sf
from scipy.io.wavfile import read,write
def convertCartesianToPolar(x, y, BaseX, BaseY):
    Radius = np.sqrt(np.power(x - BaseX, 2) + np.power(y - BaseY, 2))
    Theta = np.arctan2(y - BaseY, x - BaseX)
    return Radius, Theta
def write_topics_to_csv(topics, values, output_folder, filename='parameters.csv'):
    # Ensure the output folder exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Construct the full path for the CSV file
    file_path = os.path.join(output_folder, filename)

    # Open the file in write mode
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header
        writer.writerow(['Parameter', 'Value'])
        
        # Write the topics and values
        for topic, value in zip(topics, values):
            # Convert list values to a string representation
            if isinstance(value, list):
                value = ','.join(map(str, value))
            writer.writerow([topic, value])


def change_sampling_rate(folder_path, original_rate=16000, new_rate=44100):
    # List all WAV files in the folder
    wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    
    # Process each file
    for file in wav_files:
        file_path = os.path.join(folder_path, file)
        # Load the audio file
        audio, sr = librosa.load(file_path, sr=original_rate)  # Ensure it loads at 16000 Hz
        # Resample the audio
        audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=new_rate)
        # Save the resampled audio
        output_path = os.path.join(folder_path, file)
        sf.write(output_path, audio_resampled, new_rate)
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
def convolve_signals(signal,rir,ms50Idx,fs):
    rir[:ms50Idx] = 0
    filtered_rir = utils.highpass_filter(rir,50,fs)
    return fftconvolve(signal,filtered_rir,mode = "full")[:len(signal)]
def play_wave_file(file_path):
    sample_rate = 0
    try:
        sample_rate,data = read(file_path)
        plt.plot(data)
        plt.show()
        sd.play(data,sample_rate)
        sd.wait()
    except Exception as e:
        print(e)
    return sample_rate
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

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y
    