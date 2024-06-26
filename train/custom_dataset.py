import sys
import os

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath('..'))

import torch
import torchvision
import numpy as np
from pathlib import Path
import json
import random
import RIRnewv.utils as utils
from ..RIRnewv.data_augmentation import RandomAudioPerturbation
import librosa
#from RIRnewv.constants import FAR_FIELD_RADIUS,ALL_WINDOW_SIZES
import matplotlib.pyplot as plt
import yaml
yaml_file_path = Path(__file__).resolve().parent / 'constants.yaml'
with open(yaml_file_path, "r") as file:
    config = yaml.safe_load(file)
ALL_WINDOW_SIZES = config["ALL_WINDOW_SIZES"]
FAR_FIELD_RADIUS = config["FAR_FIELD_RADIUS"]


class customdataset(torch.utils.data.Dataset):
    def __init__(self,input_dir,n_mics = 6,sr = 44100,perturb_prob = 0.0,window_idx = -1,negatives = 0.2,mic_radius = 0.0725):
        super().__init__()
        self.dirs = sorted(list(Path(input_dir).glob('*[0-9]')))
        self.n_mics = n_mics
        self.sr = sr
        self.mic_radius = mic_radius

        self.perturb_prob = perturb_prob
        
        self.negatives = negatives
        self.window_idx = window_idx
    def __len__(self):
        return len(self.dirs)
    def __getitem__(self,idx:int):
        num_windows = len(ALL_WINDOW_SIZES)
        if self.window_idx == -1:
            curr_window_idx = np.random.randint(0,5)
        else:
            curr_window_idx = self.window_idx
        curr_window_size = ALL_WINDOW_SIZES[curr_window_idx]
        #get all angles of the window size around unit circle
        candidate_angles = utils.get_starting_angles(curr_window_size)
        #get directory name from index
        curr_dir = self.dirs[idx]
        #take the information about the data in the directory from metadata
        with open(Path(curr_dir)/'metadata.json') as json_file:
            metadata = json.load(json_file)
        
        if np.random.uniform()<self.negatives:
            #negative example target angle

            target_angle = self.get_negative_region(metadata,candidate_angles)
        else:
            #positive example target angle
            #takes a random voice and the candidate angles (possible windows)
            #returns the window angle closest to the candidate angle
            target_angle = self.get_positive_region(metadata,candidate_angles)
        all_sources, target_voice_data = self.get_mixture_and_gt(metadata,curr_dir,target_angle,curr_window_size)
        all_sources = torch.stack(all_sources,dim = 0)# [src1,src2,...],srci = [fval,....]
        mixed_data = torch.sum(all_sources,dim = 0)

        
        target_voice_data = torch.stack(target_voice_data,dim = 0)
        target_voice_data = torch.sum(target_voice_data,dim = 0)
        window_idx_one_hot = torch.tensor(utils.to_categorical(curr_window_idx,num_windows)).float()

        return (mixed_data,target_voice_data,window_idx_one_hot) [0,1,0,0,0]
    
    
    def get_mixture_and_gt(self,metadata,curr_dir,target_angle,curr_window_size):
        target_pos = np.array([FAR_FIELD_RADIUS * np.cos(target_angle),FAR_FIELD_RADIUS * np.sin(target_angle)])
        random_perturb = RandomAudioPerturbation()

        all_sources = []
        target_voice_data = []
        for key in metadata.keys():
            if "bg" in key:
                continue
            gt_audio_files = sorted(list(Path(curr_dir).rglob("*.wav")))

            gt_audio_files = [file for file in gt_audio_files if ('voice' in file.stem and key in file.stem)]
 
            assert len(gt_audio_files) > 0 , "no files found"
            gt_waveforms = []
            for _,gt_audio_file in enumerate(gt_audio_files):
                gt_waveform,_ = librosa.core.load(gt_audio_file,sr = self.sr,mono = True)
                gt_waveforms.append(torch.from_numpy(gt_waveform))
                shifted_gt,_ = utils.shift_mixture(np.stack(gt_waveforms),target_pos,self.mic_radius,self.sr)



            perturbed_source = torch.tensor(shifted_gt).float()
            all_sources.append(perturbed_source)
            if "bg" in key:
                continue
            locs_voice = metadata[key]["Position"]
            #voice_angle = np.arctan2(locs_voice[1],locs_voice[0])
            voice_angle = locs_voice[1]
            #todo : take into account front back confusion
            #print("the voice angle from metadata")
            #print(voice_angle)
            #print("for the key")
            #print(key)
            if abs(voice_angle - target_angle)<(curr_window_size/2):
                target_voice_data.append(perturbed_source.view(perturbed_source.shape[0],perturbed_source.shape[1]))
            else:
                target_voice_data.append(torch.zeros((perturbed_source.shape[0],perturbed_source.shape[1])))
        return all_sources,target_voice_data


    def get_positive_region(self,metadata,candidate_angles):
        voice_keys = [x for x in metadata if "voice" in x]
        random_key = random.choice(voice_keys)
        voice_pos = metadata[random_key]["Position"]
        voice_pos = np.array(voice_pos)
        voice_angle = np.arctan2(voice_pos[1],voice_pos[0])

        angle_idx = (np.abs(candidate_angles - voice_angle)).argmin()
        target_angle = candidate_angles[angle_idx]
        #("chosen positive, target angle")
        #print(target_angle)
        return target_angle
    
    def get_negative_region(self,metadata,candidate_angles):
        voice_keys = [x for x in metadata if "voice" in x]
        random_key = random.choice(voice_keys)
        voice_pos = np.array(metadata[random_key]["Position"])[:2]
        voice_angle = np.arctan2(voice_pos[1],voice_pos[0])
        angle_idx = (np.abs(candidate_angles-voice_angle)).argmin()

        p = np.zeros_like(candidate_angles)
        for i in range(p.shape[0]):
            if i == angle_idx:
                p[i] = 0
            else:
                dist = min(abs(i-angle_idx),(len(candidate_angles) - angle_idx + i))
                p[i] = 1/(dist)
        p/=p.sum()
        matching_shift = True
        _,true_shift = utils.shift_mixture(np.zeros((self.n_mics,10)),voice_pos,self.mic_radius,self.sr)
        while matching_shift:
            #choose a close but not target angle
            target_angle = np.random.choice(candidate_angles,p = p)
            #choose random position for shifting
            random_pos = np.array([FAR_FIELD_RADIUS*np.cos(target_angle),
                                   FAR_FIELD_RADIUS*np.sin(target_angle)])
            _,curr_shift = utils.shift_mixture(np.zeros((self.n_mics,10)),random_pos,self.mic_radius,self.sr)
            if true_shift!=curr_shift:
                matching_shift = False
        #print("chosen negative, target angle")
        #print(target_angle)
        return target_angle