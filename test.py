import numpy as np
import utils
from pathlib import Path
from scipy.io.wavfile import read,write
import json
import random
import torch
import matplotlib.pyplot as plt
from utils import get_starting_angles
from train.custom_dataset import customdataset
import main
#sound_dataset = customdataset("OUTPUTS")
#res = sound_dataset.__getitem__(8)

#print(type(res[0][0]))
#print(res[1].shape[0])
#write("res_mixed_data_getitem.wav",44100,np.array(res[0][0]))
#write("res_target_data_getitem.wav", 44100,np.array(res[1][0]))
g = main.generate_rirs()
g.generateRandomMixture(10)