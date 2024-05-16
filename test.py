import numpy as np
import utils
from pathlib import Path
from scipy.io.wavfile import read,write
import json
import random
import torch
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve, oaconvolve
import time

import soundfile as sf

s1, fs1 = sf.read("C://Users//lipov//Documents//GitHub//project//RIRnewv//LibriSpeech//Train//1919//142785//1919-142785-0001.wav")
s2, fs2 = sf.read("C://Users//lipov//Documents//GitHub//project//RIRnewv//LibriSpeech//Train//3536//23268//3536-23268-0001.wav")
plt.plot(s2)
plt.show()
res = utils.get_mixed(s1, s2, 10)
plt.plot(res)
plt.show()
write("res_get_mixed.wav", 16000, res)
res = np.arctan2(2,2)

print(res)
