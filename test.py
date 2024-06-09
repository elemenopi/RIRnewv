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
import train.custom_dataset as custom_dataset
import torch.utils.data.dataloader as dataloader
import soundfile as sf
ds = custom_dataset()
dataloader = dataloader(ds)
for i in ds:
    print(i)
