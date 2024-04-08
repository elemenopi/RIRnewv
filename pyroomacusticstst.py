from scipy.io import wavfile
import pyroomacoustics as pra
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavf
import scipy.io


def hear_signal(signal, out_f):

    signal = signal.astype(np.float32)

    fs = 16000

    wavf.write(out_f, fs, signal)
    return fs, out_f


rt60_tgt = 0.3  # seconds
room_dim = [10, 7.5, 3.5]  # meters

# import a mono wavfile as the source signal
# the sampling frequency should match that of the room
fs, audio = wavfile.read(r"LibriSpeech\Train\14\208\14-208-0000.wav")

# We invert Sabine's formula to obtain the parameters for the ISM simulator
e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)
print(max_order)

# Create the room
room = pra.ShoeBox(
    room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=10
)

# place the source in the room
room.add_source([2.5, 3.73, 1.76], signal=audio, delay=0.5)

# define the locations of the microphones
mic_locs = np.c_[
    [6.3, 4.87, 1.2], [6.3, 4.93, 1.2],  # mic 1  # mic 2
]

# finally place the array in the room
room.add_microphone_array(mic_locs)

# Run the simulation (this will also build the RIR automatically)
room.simulate()

room.mic_array.to_wav(
    f"pyroomacustics.wav",
    norm=True,
    bitdepth=np.int16,
)

# measure the reverberation time
rt60 = room.measure_rt60()
print("The desired RT60 was {}".format(rt60_tgt))
print("The measured RT60 is {}".format(rt60[1, 0]))

# Create a plot
plt.figure()

# plot one of the RIR. both can also be plotted using room.plot_rir()
rir_1_0 = room.rir[1][0]
plt.subplot(2, 1, 1)
plt.plot(np.arange(len(rir_1_0)) / room.fs, rir_1_0)
plt.title("The RIR from source 0 to mic 1")
plt.xlabel("Time [s]")

# plot signal at microphone 1
plt.subplot(2, 1, 2)
plt.plot(room.mic_array.signals[1, :])
plt.title("Microphone 1 signal")
plt.xlabel("Time [s]")

plt.tight_layout()
plt.show()


rir_1_0 = rir_1_0[:2000]
convolved_signal_rir = np.convolve(audio, rir_1_0, mode='full')
convolved_signal_rir = 0.9 * convolved_signal_rir / np.max(np.abs(convolved_signal_rir)) 

hear_signal(convolved_signal_rir, 'rir_convolved.wav')



rir_mat = scipy.io.loadmat('rir_example_0.4.mat')
array_data = rir_mat['RIR']
array_data = np.squeeze(array_data)

plt.plot(np.arange(len(array_data)) / fs, array_data)

convolved_signal = np.convolve(audio, array_data, mode='full')

convolved_signal = 0.9 * convolved_signal / np.max(np.abs(convolved_signal)) 

hear_signal(convolved_signal, 'convolved_signal_mat.wav')

audio = 0.9 * audio / np.max(np.abs(audio)) 

hear_signal(audio, 'original.wav')


