import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from scipy.io.wavfile import read,write
from scipy.signal import convolve,fftconvolve
import sounddevice as sd
import itertools
import time
import constants
import soundfile as sf
import random
import math
import json
import pyroomacoustics as pra
import pickle
import utils
from pathlib import Path
import librosa
from concurrent.futures import ProcessPoolExecutor
from pyroomacoustics.directivities import (
    DirectivityPattern,
    DirectionVector,
    CardioidFamily,
)

class generate_rirs:
    #self.microphones
    #self.roomIdx
    #self.speakerPlacement
    #self.roomDims 
    #self.speaker_soundfile_map
    #self.PRA_pickle_file
    #self.Inner
    #self.generativeNoiseZ
    #self.Outer
    #self.numOfspeakers
    #self.sample_bg
    def __init__(self):
        self.min_room_x = constants.min_room_x 
        self.min_room_y = constants.min_room_y 
        self.min_room_z = constants.min_room_z
        self.max_room_x = constants.max_room_x 
        self.max_room_y = constants.max_room_y 
        self.max_room_z = constants.max_room_z
    def generateRoom(self,speakers,datafolder,trainORtest,allORpart,shape):
        self.setNumOfSpeakers(speakers)
        res = self.gather_wav_files(datafolder,trainORtest,allORpart)#this gets all the sound files possible
        
        self.get_random_sounds(res,speakers)#this chooses num_speekers speakers and takes a random sound from that speaker

        self.generate_room_dimensions()
        self.set_limits()
        self.generate_mic_array(shape)
        self.generate_speaker_placements()
        #g.showRoom()#generates speaking directions for the speakers, unused now
        print("generating the chanels - final sound")
        self.generate_channels_V2(isBackground = random.choice([True,False]))
    def generate_rooms_concurrently(self, num_rooms, speakers, datafolder, trainORtest, allORpart, shape):
        # Set the number of rooms to generate concurrently
        with ProcessPoolExecutor() as executor:
            # Create a future for each room generation task
            futures = [executor.submit(self.generateRoom, i, speakers, datafolder, trainORtest, allORpart, shape) for i in range(num_rooms)]
            # Wait for all futures to complete
            for future in futures:
                future.result()

    def set_limits(self):
        self.Outer = [[0.5,self.roomDims[0]-0.5],[0.5,self.roomDims[1]-0.5]]
        self.generativeNoiseZ = [1,1.2]
        self.Inner = [[1,self.roomDims[0]-1],[1,self.roomDims[1]-1]]

    def setOutoutDirectory(self,outputDirectory):
        self.outputDirectory = outputDirectory
    
    def setRoomIdx(self,roomIdx):
        self.roomIdx = roomIdx
    def setNumOfSpeakers(self,numOfSpeakers = 2):
        self.numOfSpeakers = numOfSpeakers
    def subsets(self,arr):
        res = []
        def helper(idx,curr):
            if idx == len(arr):
                if curr !=[]:
                    res.append(curr)
                return
            helper(idx+1,curr.copy())
            curr.append(arr[idx])
            helper(idx+1,curr.copy())

        helper(0,[])
        return res

    def gather_wav_files(self, root_folder: str, train_test: str, all: int) -> dict:
        """
        Gathers `.wav` files from a specified directory structure.

        Args:
            root_folder (str): The root directory to start the traversal.
            train_test (str): Specifies whether to use the "Test" or "Train" folder.
            all (int): Determines if all `.wav` files should be collected (1) or only one per speaker (0).

        Returns:
            dict: A dictionary mapping speakers to lists of `.wav` files or single files.
        """

        in_folder = os.path.join(root_folder, train_test)

        if not os.path.exists(in_folder):
            print(f"{train_test} folder not found.")
            return

        output_folders = {}  # speaker: [sound_files...]

        for speaker_folder in os.listdir(in_folder):
            speaker_path = os.path.join(in_folder, speaker_folder)

            if os.path.isdir(speaker_path):
                sound_files = []

                for sound_folder in os.listdir(speaker_path):
                    sound_path = os.path.join(speaker_path, sound_folder)
                    if os.path.isdir(sound_path):
                        for filename in os.listdir(sound_path):
                            if filename.endswith(".wav"):
                                src_filepath = os.path.join(sound_path, filename)
                                sound_files.append(src_filepath)
                if all == 0 and sound_files:
                    # If all == 0, randomly select one `.wav` file per speaker
                    output_folders[speaker_folder] = random.choice(sound_files)
                else:
                    # Otherwise, store all `.wav` files found
                    output_folders[speaker_folder] = sound_files

        return output_folders

    def get_random_sounds(self,speaker_sound_map, num_speakers):

        selected_speakers = random.sample(list(speaker_sound_map.keys()), num_speakers)

        random_sounds = {}
        for s in selected_speakers:
            random_sounds[s] = ""

        for speaker in selected_speakers:
            sounds_for_speaker = speaker_sound_map[speaker]

            if sounds_for_speaker:
                random_sound = random.choice(sounds_for_speaker)
                random_sounds[speaker] = random_sound
        self.speaker_soundfile_map =random_sounds
        return self.speaker_soundfile_map
    def generate_room_dimensions(self):
        room_x = round(random.uniform(self.min_room_x, self.max_room_x), 4)
        room_y = round(random.uniform(self.min_room_y, self.max_room_y), 4)
        room_z = round(random.uniform(self.min_room_z, self.max_room_z), 4)
        self.roomDims = [room_x, room_y, room_z]
        self.set_limits()#uesless for now
        return [room_x, room_y, room_z]

    def generate_mic_array(self,type,randomOrCenter = "center"):
    
        #check for not in the middle
        #let user get another option for array 
        #from drive
    
        if type == "circular":
            if randomOrCenter == "random":
                mic_x = round(random.uniform(self.Inner[0][0]-0.5,self.Inner[0][1]-0.5), 4)
                mic_y = round(random.uniform(self.Inner[1][0]-0.5,self.Inner[1][1]-0.5), 4)
            else:
                mic_x = round(self.roomDims[0]/2,4)
                mic_y = round(self.roomDims[1]/2,4)
            mic_z = round(random.uniform(1.4,1.5), 4)
            points = []
            center = [mic_x,mic_y]
            self.micArrayCenter = center
            num_points = 6
            radius = 0.075 #cm
            angular_difference_degrees = 60
            for i in range(num_points):
                angle = math.radians(i * angular_difference_degrees)
                x = center[0] + radius * math.cos(angle)
                y = center[1] + radius * math.sin(angle)
                points.append((x, y))
    
            res = []
            for p in points:
                res.append([round(p[0],4),round(p[1],4),mic_z])
            
            self.microphones = res
            return res

    def generate_speaker_placements(self):
        #plot scatter of the room and the characters
        num_of_speaker = self.numOfSpeakers
        x_mic,y_mic,_ = np.array(self.microphones).sum(axis = 0)/len(self.microphones)
        room_x ,room_y,room_z =self.roomDims
        speakers = []
        def valid(place):
            # check if at least 0.5 radius from mic array center
            if np.linalg.norm(np.array([place[0],place[1],0])- np.array([room_x,room_y,0]))<0.5:
                return False
            #speaker is at least 0.4 from other speaker
            for sp in speakers:
                if np.linalg.norm(np.array([place[0],place[1],0])-np.array([sp[0],sp[1],0]))<1.5:
                    return False
            return True
        def generatePlace(Inner):
            distance_from_wall = 0.3
            #parametric z
            #parametric everything, make a yaml
            #generate xyz for speaker
            
            #return [np.random.uniform(Inner[0][0],Inner[0][1])
            #        ,np.random.uniform(0+Inner[1][0],Inner[1][1]),np.random.uniform(1,1.9)] 
            radius = np.random.uniform(1.5,4)
            angle = np.random.uniform(0,2*np.pi)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            return [self.micArrayCenter[0] + x,self.micArrayCenter[1] + y,np.random.uniform(1.4,1.5)]

        generated = generatePlace(self.Inner) 
        for i in range(num_of_speaker):
            while not valid(generated):
                generated = generatePlace(self.Inner)
            speakers.append(generated)
        self.speaker_placements = np.round(speakers,4)
        return self.speaker_placements

    
    def showRoom(self):
        # Create a ShoeBox room model with given dimensions and properties
        room = pra.ShoeBox(self.roomDims, fs=16000, max_order=10)
        #room = pra.Room.(self.roomDims, fs=16000, max_order=10)

        # Add sources (speakers) at specified positions
        for speaker_position in self.speaker_placements:
            room.add_source(speaker_position)

        # Add a microphone array to the room
        room.add_microphone_array(np.array(self.microphones).transpose())

        # Create the plot
        if len(self.roomDims) == 3 and self.roomDims[2] != 0:  # Check if the room is truly 3D
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlim([0, self.roomDims[0]])
            ax.set_ylim([0, self.roomDims[1]])
            ax.set_zlim([0, self.roomDims[2]])
        else:  # Default to 2D if height is zero or missing
            fig, ax = plt.subplots()
            ax.set_xlim([0, self.roomDims[0]])
            ax.set_ylim([0, self.roomDims[1]])

        room.plot(fig=fig, ax=ax)

        # Plotting speaker positions
        speaker_positions = np.array(self.speaker_placements)
        if len(self.roomDims) == 3 and self.roomDims[2] != 0:
            ax.scatter(speaker_positions[:,0], speaker_positions[:,1], speaker_positions[:,2], marker='o', color='red', s=100, label='Speakers')
        else:
            ax.scatter(speaker_positions[:,0], speaker_positions[:,1], marker='o', color='red', s=100, label='Speakers')

        # Labeling speakers
        for i, pos in enumerate(speaker_positions):
            if len(self.roomDims) == 3 and self.roomDims[2] != 0:
                ax.text(pos[0], pos[1], pos[2], f'Speaker {i+1}', color='red')
            else:
                ax.text(pos[0], pos[1], f'Speaker {i+1}', color='red', ha='center')

        # Plotting microphone positions
        if hasattr(self, 'microphones') and self.microphones is not None:
            mic_positions = np.array(self.microphones).transpose()
            if len(self.roomDims) == 3 and self.roomDims[2] != 0:
                ax.scatter(mic_positions[0], mic_positions[1], mic_positions[2], marker='x', color='blue', s=100, label='Microphones')
            else:
                ax.scatter(mic_positions[0], mic_positions[1], marker='x', color='blue', s=100, label='Microphones')

        # Adding labels and legend
        ax.set_xlabel('Width [m]')
        ax.set_ylabel('Length [m]')
        if len(self.roomDims) == 3 and self.roomDims[2] != 0:
            ax.set_zlabel('Height [m]')
        ax.legend()

        # Show the plot
        plt.show()

    def generateRandomMixture(self,total_seconds):
        #to add: repeating sound
        allSoundsTrain = self.gather_wav_files(r"C:\Users\lipov\Documents\GitHub\project\RIRnewv\LibriSpeech","Train",1)
        allSoundsTest = self.gather_wav_files(r"C:\Users\lipov\Documents\GitHub\project\RIRnewv\LibriSpeech","Test",1)
        
        final_sound_buffer = 0
        for k in allSoundsTest.keys():
            s_path = random.choice(allSoundsTest[k])
            s,fs = sf.read(s_path)
            total_samples = int(total_seconds*fs)
            s = s/abs(s).max()
            while len(s)<total_seconds*fs:
                s = np.tile(s, 2)
            s = np.pad(s,(0,total_samples))[:total_samples]
            final_sound_buffer+=s
        
        return final_sound_buffer/abs(final_sound_buffer).max()

    
    def create_next_folder(self,base_folder, folder_prefix):
        # Get a list of existing folders in the base directory
        existing_folders = [folder for folder in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, folder))]

        # Find the latest folder with the specified prefix
        latest_folder = max((folder for folder in existing_folders if folder.startswith(folder_prefix)), default=None)

        if latest_folder is None:
            # If no folder found, create the first one
            new_folder_name = f"{folder_prefix}000"
        else:
            # Extract the numeric part and increment it
            folder_number = int(latest_folder[len(folder_prefix):])
            new_folder_number = folder_number + 1
            new_folder_name = f"{folder_prefix}{new_folder_number:03d}"

        # Create the new folder
        new_folder_path = os.path.join(base_folder, new_folder_name)
        os.makedirs(new_folder_path)

        return new_folder_path
    

    def Generate_bg_signals(self,room_dim,mic_locs,fs ,max_order ,rt60tgt ,type_of_speakers = "singles",total_seconds = 30):
        #make the room
        mixture = [] #change to generate mixture function if needed (if else....)
        
        e_absorption,max_order = pra.inverse_sabine(rt60tgt,room_dim)
        room = pra.ShoeBox(room_dim,fs = fs,max_order = max_order,materials = pra.Material(e_absorption))
        #room = pra.Room.from_corners(room_dim,fs = fs,max_order = max_order,materials = pra.Material(e_absorption))
        room.add_microphone_array(mic_locs)
        #add bg locations
        bg_loc1 = [0.2,0.2,1.4]
        bg_loc2 = [self.roomDims[0]-0.2,self.roomDims[1]-0.2,1.4]
        bg_loc3 = [self.roomDims[0]-0.2,0.2,1.4]
        bg_loc4 = [0.2,self.roomDims[1]-0.2,1.4]
        bg_locs = [bg_loc1,bg_loc2,bg_loc3,bg_loc4]
        signals = []
        #add speakers according to type
        if type_of_speakers == "mixture":
            signal1 =signal2 = signal3 = signal4 =mixture
        speakers = []
        if type_of_speakers == "singles":
            speakers = ["\\5338\\24640\\5338-24640-0000.wav",
                        "\\3853\\163249\\3853-163249-0002.wav",
                        "\\8297\\275154\\8297-275154-0026.wav",
                        "\\6295\\244435\\6295-244435-0000.wav"]
            signal1,fs = sf.read("C:\\Users\\lipov\\Documents\\GitHub\\project\\RIRnewv\\LibriSpeech\\Test"+speakers[0])
            signal2,fs = sf.read("C:\\Users\\lipov\\Documents\\GitHub\\project\\RIRnewv\\LibriSpeech\\Test"+speakers[1])
            signal3,fs = sf.read("C:\\Users\\lipov\\Documents\\GitHub\\project\\RIRnewv\\LibriSpeech\\Test"+speakers[2])
            signal4,fs = sf.read("C:\\Users\\lipov\\Documents\\GitHub\\project\\RIRnewv\\LibriSpeech\\Test"+speakers[3])
            signals = [signal1,signal2,signal3,signal4]
             
            for idx, s in enumerate(signals):
                total_samples = int(total_seconds * fs)
                s = s / np.abs(s).max()  # Normalization
                while len(s) < total_samples:
                    s = np.tile(s, 2)  # Double the length until it is long enough
                signals[idx] = s[:total_samples]  # Trim to the exact number of required samples
        for i in range(4):

            room.add_source(bg_locs[i], signals[i])
        #simulate room and get the rirs
        room.simulate()
        rirs = room.rir
        ms50Idx = int(0.05*fs)
        total_signal = np.zeros((len(rirs), len(signals[0])))

        for mic_idx in range(len(rirs)):
            mic_signal = np.zeros(len(signals[0]))
            for speaker_idx in range(len(rirs[1])):
                speaker_rir = rirs[mic_idx][speaker_idx][:150000]
                speaker_rir[:ms50Idx] = 0
                speaker_rir = utils.highpass_filter(speaker_rir,50,fs)
                convolved_signal_rir = fftconvolve(signals[speaker_idx],speaker_rir,mode = "full")
                mic_signal+=convolved_signal_rir[:len(signals[0])]
            
            total_signal[mic_idx] = mic_signal

        return bg_locs,total_signal,speakers


        #convolve each speaker through its microphones
        #add all up and return the sum
    def generate_channels_V2(self,rt60tgt = 0.7,snr = 8,sir = 1,isBackground = True):
        room_dim = self.roomDims
        rt60_tgt = rt60tgt
        #bg_recording = self.generateRandomMixture(30)
        bg_recording = isBackground
        duration = 5
        idx = 0
        all_fg_signals = []
        
        #pick axis sir for sources and change sir from 0 to 5 according to axis



        is_axis = False

        for s in self.speaker_soundfile_map.values():

            if is_axis == False:
                is_axis = True
                axisAudio,fs= sf.read(s)
                axisAudio = axisAudio/abs(axisAudio).max()
                audio = axisAudio
            else:
                audio,fs = sf.read(s)
                audio = audio/abs(audio).max()
                audio  = utils.get_mixed(axisAudio,audio,sir)
                
            e_absorption,max_order = pra.inverse_sabine(rt60_tgt,room_dim)
            room = pra.ShoeBox(room_dim,fs = fs,max_order = max_order,materials = pra.Material(e_absorption))

            #room = pra.Room.from_corners(room_dim,fs = fs,max_order = max_order,materials = pra.Material(e_absorption))
            total_samples = int(fs*duration)
            #source_dir = directions[idx]
            room.add_source(self.speaker_placements[idx], signal=audio)#,directivity=source_dir)
            idx+=1

            mic_locs = np.array(self.microphones).transpose()  # Assuming mic_array is defined somewhere
            room.add_microphone_array(mic_locs)
            room.simulate()#builds the rirs automatically
            fg_signals = room.mic_array.signals[:,:total_samples]
            #fg_target = np.random.uniform(FG_VOL_MIN, FG_VOL_MAX)
            
            fg_signals = fg_signals/abs(fg_signals).max()
            all_fg_signals.append(fg_signals)

        #generate bg signals for room
        if bg_recording is not False:
            mic_locs = np.array(self.microphones).transpose()  # Assuming mic_array is defined somewhere
            bg_locs, bg_signals,bg_speakers = self.Generate_bg_signals(room_dim,mic_locs,fs ,10 ,rt60_tgt ,"singles",total_seconds = 30)
            bg_signals = np.array(bg_signals)
            bg_target = 1#np.random.uniform(0.4, 0.7)
            bg_signals = bg_signals * bg_target / np.max(np.abs(bg_signals))
            #signals_for_testing = signals_for_testing * bg_target/ np.max(np.abs(signals_for_testing))
            
        #beta = utils.get_mixed
        #apply highpass filter to avoid waves
        for i in range(len(all_fg_signals)):
            for j in range(len(all_fg_signals[0])):
                all_fg_signals[i][j] = utils.highpass_filter(all_fg_signals[i][j],50,fs)


        #snr calculations
        if bg_recording is not False:
            if snr!=0:
                for i in range(6):# for each microphone 
                    fg = []
                    for voice_idx in range(self.numOfSpeakers):
                        fg.append(all_fg_signals[voice_idx][i])

                    mixed_signal = np.sum(fg,axis = 0)
                    bg = bg_signals[i]

                    bg_signals[i] = utils.get_mixed(mixed_signal,bg,snr)


        outputDirectory ='OUTPUTS'
        latestFolder = self.create_next_folder(outputDirectory,'output')
        for mic_idx in range(len(self.microphones)):
            output_prefix = str(Path(latestFolder) / "mic{:02d}_".format(mic_idx))
            all_fg_buffer = np.zeros((total_samples))
            for voice_idx in range(self.numOfSpeakers):
                curr_fg_buffer = np.pad(all_fg_signals[voice_idx][mic_idx],(0,total_samples))[:total_samples]
                
                write(output_prefix + "voice{:02d}.wav".format(voice_idx),  fs,curr_fg_buffer)#.astype(np.int16) )#32
                all_fg_buffer+=curr_fg_buffer
            if bg_recording is not False:
                bg_buffer = np.pad(bg_signals[mic_idx],(0,total_samples))[:total_samples]
                write(output_prefix + f"bg{mic_idx}.wav",fs,bg_buffer)
                write(output_prefix+"mixed.wav",fs,all_fg_buffer+bg_buffer)
            else:
                write(output_prefix + "mixed.wav", fs,all_fg_buffer)#.astype(np.int16))#32
        utils.change_sampling_rate(latestFolder)
        
        metadata = {}
        for voice_idx,speaker_id in enumerate(self.speaker_soundfile_map.keys()):
            
            r,theta = utils.convertCartesianToPolar(self.speaker_placements[voice_idx][0],self.speaker_placements[voice_idx][1],self.micArrayCenter[0],self.micArrayCenter[1])
            metadata['voice{:02d}'.format(voice_idx)] = {
                'Position': [r,theta,self.speaker_placements[voice_idx][2]],
                'speaker_id': speaker_id
            }
        if bg_recording is not False:
            for i in range(4):
                r,theta = utils.convertCartesianToPolar(bg_locs[i][0],bg_locs[i][1],self.micArrayCenter[0],self.micArrayCenter[1])
                metadata[f'bg{i}'] = {'position':[r,theta,bg_locs[i][2]]}
            
        metadata_file = str(Path(latestFolder)/"metadata.json")
        with open(metadata_file,"w") as f:
            json.dump(metadata,f,indent = 4)
        if bg_recording is not False:
            utils.write_topics_to_csv(["rt60","room dimentions","background speakers","snr"],[str(rt60_tgt),[str(i) for i in self.roomDims],bg_speakers,snr],latestFolder)
        else:
            utils.write_topics_to_csv(["rt60","room dimentions"],[str(rt60_tgt),[str(i) for i in self.roomDims]],latestFolder)
    
    def background_samples(self,order):
        res1 = g.gather_wav_files(r"C:\Users\lipov\Documents\GitHub\project\RIRnewv\LibriSpeech","Train",1)
        allsounds = []
        for s in res1.keys():
            for sound in res1[s]:
                allsounds.append(sound)
        selected_sounds = random.sample(allsounds, order)
        return selected_sounds



if __name__ == "__main__":
    g = generate_rirs()
#speakers,datafolder,trainORtest,1,"circular
    startTime = time.time()
    for i in range(constants.NUM_OF_ROOMS):
        g.generateRoom(3,r"C:\Users\lipov\Documents\GitHub\project\RIRnewv\LibriSpeech","Train",1,"circular")
    endTime = time.time()
    print(f"time: {endTime-startTime}")
#g.setNumOfSpeakers(3)
#res = g.gather_wav_files(r"C:\Users\lipov\Documents\GitHub\project\RIRnewv\LibriSpeech","Train",1)#this gets all the sound files possible
##g.background_samples(3)#unused for now
#g.get_random_sounds(res,num_speakers = 3)#this chooses num_speekers speakers and takes a random sound from that speaker
#
#g.generate_room_dimensions()
#g.set_limits()
#g.generate_mic_array("circular")
#g.generate_speaker_placements()
##g.showRoom()#generates speaking directions for the speakers, unused now
#print("generating the chanels - final sound")
#g.generate_channels_V2()

