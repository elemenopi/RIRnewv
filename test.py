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
images = np.array([[[ 10,  20,  30],  # Image 1
                    [ 40,  50,  60],
                    [ 70,  80,  90]],

                   [[100, 110, 120],  # Image 2
                    [130, 140, 150],
                    [160, 170, 180]],

                   [[190, 200, 210],  # Image 3
                    [220, 230, 240],
                    [250, 255, 255]]])
print(np.sum(images,axis = 2))
#import main
#sound_dataset = customdataset("OUTPUTS")
#res = sound_dataset.__getitem__(8)

#print(type(res[0][0]))
#print(res[1].shape[0])
#write("res_mixed_data_getitem.wav",44100,np.array(res[0][0]))
#write("res_target_data_getitem.wav", 44100,np.array(res[1][0]))
#g = main.generate_rirs()
#g.generateRandomMixture(10)