import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from scipy.io.wavfile import read,write
from scipy.signal import convolve,fftconvolve
import sounddevice as sd
import itertools
import random
import math
import json
import pyroomacoustics as pra
import pickle
import utils
from pathlib import Path
import librosa
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

    def play_wave_file(self,file_path):
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
    def generate_many_sounds_for_a_room(self,rirs_info,sound_files,roomNumber):
        df = pd.read_csv(rirs_info)
        rir_files = df['WAV_Path'].values.tolist()
        index_subsets_soundfiles = self.subsets(range(len(sound_files)))
        index_subsets_rirfiles = self.subsets(range(len(rir_files)))
        for i in index_subsets_soundfiles:
            for k in index_subsets_rirfiles:
                if len(i) == len(k):
                    convolved_audios = []
                    permutation = np.random.permutation(len(i))
                    for j in range(len(i)):
                        sample_rate,data_rir_file = read(rir_files[permutation[j]])
                        sample_rate,data_sound_file = read(sound_files[permutation[j]])
                        sound_data = data_sound_file.astype(np.float32)/np.max(np.abs(data_sound_file))
                        response_data = data_rir_file.astype(np.float32)/np.max(np.abs(data_rir_file))
                        convolved_audios.append(fftconvolve(sound_data,response_data,mode = 'full'))
                    numpy_array = np.array(convolved_audios)
                    result = np.sum(numpy_array,axis = 0)

                    write(f'outFiles/output_{len(i)}_num_of_speakers_room_number_{roomNumber}.wav',sample_rate,result.astype(np.int16))

    def gather_wav_files(self,root_folder,train_test,all):
        #train_test = "Test"\"Train"
        
        in_folder = os.path.join(root_folder, train_test)

        if not os.path.exists(in_folder):
            print(f"{train_test} folder not found.")
            return

        output_folders = {}#speaker:[soundsfiles...]

        for speaker_folder in os.listdir(in_folder):
            speaker_path = os.path.join(in_folder, speaker_folder)

            if os.path.isdir(speaker_path):
                output_folders[speaker_folder] = []

                for sound_folder in os.listdir(speaker_path):
                    sound_path = os.path.join(speaker_path, sound_folder)
                    if os.path.isdir(sound_path):
                        for filename in os.listdir(sound_path):
                            if filename.endswith(".wav"):
                                src_filepath = os.path.join(sound_path, filename)
                                output_folders[speaker_folder].append(src_filepath)
                        if all == 0:
                            #if all is 0 then take one random and continue
                            output_folders[speaker_folder] = random.choice(output_folders[speaker_folder])

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
        room_x = round(random.uniform(min_room_x, max_room_x), 4)
        room_y = round(random.uniform(min_room_y, max_room_y), 4)
        room_z = round(random.uniform(min_room_z, max_room_z), 4)
        self.roomDims = [room_x, room_y, room_z]
        print(self.roomDims)
        self.set_limits()
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
            mic_z = round(random.uniform(1,1.5), 4)
            #mic_x = round(mic_x/2,4)
            #mic_y = round(mic_y/2,4)
            #mic_z = round(mic_z/2,4)
    
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
            #res = rotate_points_numpy(np.array(res),theta = np.random.uniform(0, 2*np.pi))
            self.microphones = res
            return res
    #self.Inner
    #self.generativeNoiseZ
    #self.Outer
    def generate_speaker_placements(self,num_of_speaker):
        #plot scatter of the room and the characters

        x_mic,y_mic,_ = np.array(self.microphones).sum(axis = 0)/len(self.microphones)
        room_x ,room_y,room_z =self.roomDims
        speakers = []
        def valid(place):
            # check if at least 0.5 radius from mic array center
            if np.linalg.norm(np.array([place[0],place[1],0])- np.array([room_x,room_y,0]))<0.5:
                return False
            #speaker is at least 0.4 from other speaker
            for sp in speakers:
                if np.linalg.norm(np.array([place[0],place[1],0])-np.array([sp[0],sp[1],0]))<1:
                    return False
            return True
        def generatePlace(Inner):
            distance_from_wall = 0.3
            #parametric z
            #parametric everything, make a yaml
            #generate xyz for speaker
            
            #return [np.random.uniform(Inner[0][0],Inner[0][1])
            #        ,np.random.uniform(0+Inner[1][0],Inner[1][1]),np.random.uniform(1,1.9)] 
            radius = np.random.uniform(1,3)
            angle = np.random.uniform(0,2*np.pi)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            return [self.micArrayCenter[0] + x,self.micArrayCenter[1] + y,np.random.uniform(1,1.5)]

        generated = generatePlace(self.Inner) 
        for i in range(num_of_speaker):
            while not valid(generated):
                generated = generatePlace(self.Inner)
            speakers.append(generated)
        self.speaker_placements = np.round(speakers,4)
        return self.speaker_placements

    def showRoom(self):
        room = pra.ShoeBox(self.roomDims, fs=16000, max_order=10)
        source_pattern = DirectivityPattern.HYPERCARDIOID
        directions = []

        
        for s in self.speaker_placements:
            source_dir = CardioidFamily(
                orientation=DirectionVector(azimuth=random.uniform(0,360), colatitude=90+random.uniform(-20,20), degrees=True),
                pattern_enum=source_pattern,
            )   
            directions.append(source_dir)
            room.add_source(s,directivity=source_dir)

        room.add_microphone_array(np.array(self.microphones).transpose())

        room.plot()
        plt.show()
        return directions
    def results_to_csv(sounds):
        
        df = pd.DataFrame()
    #self.Inner
    #self.generativeNoiseZ
    #self.Outer

    def generateRandomMixture(self,total_seconds):
        #to add: repeating sound
        allSoundsTrain = self.gather_wav_files(r"C:\Users\lipov\Documents\GitHub\project\RIRnewv\LibriSpeech","Train",1)
        allSoundsTest = self.gather_wav_files(r"C:\Users\lipov\Documents\GitHub\project\RIRnewv\LibriSpeech","Test",1)
        
        final_sound_buffer = 0
        for k in allSoundsTest.keys():
            s_path = random.choice(allSoundsTest[k])
            fs,s = read(s_path)
            total_samples = int(total_seconds*fs)
            s = s/abs(s).max()
            while len(s)<total_seconds*fs:
                s = np.tile(s, 2)
            s = np.pad(s,(0,total_samples))[:total_samples]
            final_sound_buffer+=s
        #for k in allSoundsTrain.keys():
        #    s_path = random.choice(allSoundsTrain[k])
        #    fs,s = read(s_path)
        #    total_samples = int(total_seconds*fs)
        #    s = s/abs(s).max()
        #    while len(s)<total_seconds*fs:
        #
        #        s = np.tile(s, 2)
        #    s = np.pad(s,(0,total_samples))[:total_samples]
        #    final_sound_buffer+=s
        return final_sound_buffer/abs(final_sound_buffer).max()
        #write("final_background.wav",fs,final_sound_buffer)


    def diffusionNoiseV2(self,diffOrder):
        allSoundsTrain = self.gather_wav_files(r"C:\Users\lipov\Documents\GitHub\project\RIRnewv\LibriSpeech","Train",1)
        allSoundsTest = self.gather_wav_files(r"C:\Users\lipov\Documents\GitHub\project\RIRnewv\LibriSpeech","Test",1)
        allSoundsTrain.update(allSoundsTest)
        #get a sample sound
        audios = []
        for s in allSoundsTrain.values():
            print(s)
            sampleSound = s[0]
            fs, audio = read(sampleSound)
            audio = audio*0.1
            audios.append(audio)
        x_axis = np.linspace(self.Outer[0][0],self.Outer[0][1],diffOrder)
        y_axis = np.linspace(self.Outer[1][0],self.Outer[1][1],diffOrder)
        z_axis = np.random.uniform(1,1.2,3)

        wall0 = np.vstack([x_axis,self.Outer[1][0]*np.ones(diffOrder),z_axis]).T#may be a bug later in this section
        wall1 = np.vstack([self.Outer[0][0]*np.ones(diffOrder),y_axis,z_axis]).T
        wall2 = np.vstack([x_axis,self.Outer[1][1]*np.ones(diffOrder),z_axis]).T
        wall3 = np.vstack([self.Outer[0][1]*np.ones(diffOrder),y_axis,z_axis]).T
        room = pra.ShoeBox(self.roomDims,fs = fs,max_order=20)
        pattern = DirectivityPattern.HYPERCARDIOID
        orientations = [DirectionVector(azimuth=90*i, colatitude=90, degrees=True) for i in [0,1,2,3]]
        dir_obj_walls = [CardioidFamily(orientation=o, pattern_enum=pattern) for o in orientations] 
        k = 0
        mod = len(audios)
        for i in range(diffOrder):
            room.add_source(wall0[i],directivity=dir_obj_walls[3],signal=audios[k%mod], delay=np.random.uniform(0.2,1))
            k+=1
            room.add_source(wall1[i],directivity=dir_obj_walls[2],signal=audios[k%mod], delay=np.random.uniform(0.2,1))
            k+=1
            room.add_source(wall2[i],directivity=dir_obj_walls[1],signal=audios[k%mod], delay=np.random.uniform(0.2,1))
            k+=1
            room.add_source(wall3[i],directivity=dir_obj_walls[0],signal=audios[k%mod], delay=np.random.uniform(0.2,1))
            k+=1
        room.add_microphone_array(np.array(self.microphones).transpose())
        fig,ax =room.plot()
        ax.set_xlim([0,6])
        ax.set_zlim([0,6])
        ax.set_ylim([0,6])
        plt.show()
        room.simulate()
        room.mic_array.to_wav(
            f"diffnoisev2.wav",
            norm=True,
            bitdepth=np.int16,
        )


    
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
    
    def generate_channels_V2(self,directions,rt60tgt = 0.7):
        room_dim = self.roomDims
        rt60_tgt = rt60tgt
        bg_recording = self.generateRandomMixture(30)
        duration = 5

        
        #for s in self.speaker_soundfile_map.values():
        #    fs, audio = read(s)
        #    e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)
        #    room = pra.ShoeBox(
        #        room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order
        #    )
        #    break
        #total_samples = int(fs*3)
        idx = 0
        all_fg_signals = []
        fs = 44100
        for s in self.speaker_soundfile_map.values():
            #audio, _ = librosa.core.load(s,sr = fs,mono= True)
            fs,audio= read(s)
            audio = audio/abs(audio).max()
            #print("theaudio")
            #print(audio)
            #44100
            e_absorption,max_order = pra.inverse_sabine(rt60_tgt,room_dim)
            room = pra.ShoeBox(room_dim,fs = fs,max_order = max_order,materials = pra.Material(e_absorption))
            total_samples = int(fs*duration)
            source_dir = directions[idx]
            room.add_source(self.speaker_placements[idx], signal=audio)#,directivity=source_dir)
            idx+=1

            mic_locs = np.array(self.microphones).transpose()  # Assuming mic_array is defined somewhere
            room.add_microphone_array(mic_locs)
            room.simulate()#builds the rirs automatically
            fg_signals = room.mic_array.signals[:,:total_samples]
            #fg_target = np.random.uniform(FG_VOL_MIN, FG_VOL_MAX)
            fg_signals = fg_signals#/abs(fg_signals).max()
            all_fg_signals.append(fg_signals)
            #room.mic_array.to_wav(
            #    f"result_post_pyroom.wav",
            #    bitdepth=np.int16,
            #)
        if bg_recording is not None:
            bg_length = len(bg_recording)
            bg_start_idx = np.random.randint(bg_length - total_samples)
            sample_bg = bg_recording[bg_start_idx:bg_start_idx + total_samples]
            sample_bg = sample_bg/abs(sample_bg).max()
        if bg_recording is not None:
            bg_radius = np.random.uniform(3.8,4.2)
            bg_theta = np.random.uniform(0,2*np.pi)
            bg_loc = [0.1,0.1,1.2]#[bg_radius*np.cos(bg_theta),bg_radius*np.sin(bg_theta),np.random.uniform(1.2,1.5)]
            room = pra.ShoeBox(room_dim,fs = fs,max_order = max_order,materials = pra.Material(e_absorption))
            room.add_source(bg_loc,signal = sample_bg)
            mic_locs = np.array(self.microphones).transpose()  # Assuming mic_array is defined somewhere
            room.add_microphone_array(mic_locs)
            room.simulate()
            bg_signals = room.mic_array.signals[:,:total_samples]
            bg_target = np.random.uniform(0.4, 0.7)
            bg_signals = bg_signals * bg_target / abs(bg_signals).max() 
            

        outputDirectory ='OUTPUTS'
        latestFolder = self.create_next_folder(outputDirectory,'output')
        for mic_idx in range(len(self.microphones)):
            output_prefix = str(Path(latestFolder) / "mic{:02d}_".format(mic_idx))
            all_fg_buffer = np.zeros((total_samples))
            for voice_idx in range(self.numOfSpeakers):
                curr_fg_buffer = np.pad(all_fg_signals[voice_idx][mic_idx],(0,total_samples))[:total_samples]
                
                write(output_prefix + "voice{:02d}.wav".format(voice_idx),  fs,curr_fg_buffer)#.astype(np.int16) )#32
                all_fg_buffer+=curr_fg_buffer
            if bg_recording is not None:
                bg_buffer = np.pad(bg_signals[mic_idx],(0,total_samples))[:total_samples]
                write(output_prefix + "bg.wav",fs,bg_buffer)
                write(output_prefix+"mixed.wav",fs,all_fg_buffer+bg_buffer)
            else:
                write(output_prefix + "mixed.wav", fs,all_fg_buffer)#.astype(np.int16))#32
        
        metadata = {}
        for voice_idx,speaker_id in enumerate(self.speaker_soundfile_map.keys()):
            print(self.speaker_placements[voice_idx])
            r,theta = utils.convertCartesianToPolar(self.speaker_placements[voice_idx][0],self.speaker_placements[voice_idx][1],self.micArrayCenter[0],self.micArrayCenter[1])
            metadata['voice{:02d}'.format(voice_idx)] = {
                'Position': [r,theta,self.speaker_placements[voice_idx][2]],
                'speaker_id': speaker_id
            }
        if bg_recording is not None:
            r,theta = utils.convertCartesianToPolar(bg_loc[0],bg_loc[1],self.micArrayCenter[0],self.micArrayCenter[1])
            metadata['bg'] = {'position':[r,theta,bg_loc[2]]}
        metadata_file = str(Path(latestFolder)/"metadata.json")
        with open(metadata_file,"w") as f:
            json.dump(metadata,f,indent = 4)
        
        
        #self.mixPaths = ["result_post_pyroom_mic0.wav","result_post_pyroom_mic1.wav","result_post_pyroom_mic2.wav","result_post_pyroom_mic3.wav","result_post_pyroom_mic4.wav","result_post_pyroom_mic5.wav"]
        #self.mixPaths = [latestFolder +'\\'+ mP  for mP in self.mixPaths]
        
        #for i in range(6):
        #    write(self.mixPaths[i],16000,results[i].astype(np.int16))    
    
    def background_samples(self,order):
        res1 = g.gather_wav_files(r"C:\Users\lipov\Documents\GitHub\project\RIRnewv\LibriSpeech","Train",1)
        allsounds = []
        for s in res1.keys():
            for sound in res1[s]:
                allsounds.append(sound)
        selected_sounds = random.sample(allsounds, order)
        return selected_sounds

    def generate_channels(self,directions,rt60tgt = 0.7):
        room_dim = self.roomDims
        rt60_tgt = rt60tgt

        for s in self.speaker_soundfile_map.values():
            fs, audio = read(s)
            e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)
            room = pra.ShoeBox(
                room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order
            )
            break
        
        idx = 0
        for s in self.speaker_soundfile_map.values():
            fs, audio = read(s)
            source_dir = directions[idx]
            room.add_source(self.speaker_placements[idx], signal=audio, delay=0.5,directivity=source_dir)
            idx+=1

        mic_locs = np.array(self.microphones).transpose()  # Assuming mic_array is defined somewhere
        room.add_microphone_array(mic_locs)
        room.simulate()#builds the rirs automatically
        room.mic_array.to_wav(
            f"result_post_pyroom.wav",
            norm=True,
            bitdepth=np.int16,
        )
        # measure the reverberation time
        self.rt60 = room.measure_rt60()
        print("The desired RT60 was {}".format(rt60_tgt))
        print("The measured RT60 is {}".format(self.rt60[1, 0]))


        results = [room.mic_array.signals[i, :] for i in range(6)]
        #result_audio_mic0 = room.mic_array.signals[0, :]
        #result_audio_mic2 = room.mic_array.signals[2, :]
        #result_audio_mic3 = room.mic_array.signals[3, :]
        #result_audio_mic4 = room.mic_array.signals[4, :]
        #result_audio_mic5 = room.mic_array.signals[5, :]

        # Save the audio to a WAV file
        outputDirectory ='OUTPUTS'
        latestFolder = self.create_next_folder(outputDirectory,'output')
        
        self.mixPaths = ["result_post_pyroom_mic0.wav","result_post_pyroom_mic1.wav","result_post_pyroom_mic2.wav","result_post_pyroom_mic3.wav","result_post_pyroom_mic4.wav","result_post_pyroom_mic5.wav"]
        self.mixPaths = [latestFolder +'\\'+ mP  for mP in self.mixPaths]
        
        for i in range(6):
            write(self.mixPaths[i],16000,results[i].astype(np.int16))
        
        #write("result_post_pyroom_mic0.wav", 16000, result_audio_mic0.astype(np.int16))
        #write("result_post_pyroom_mic1.wav", 16000, result_audio_mic1.astype(np.int16))
        #write("result_post_pyroom_mic2.wav", 16000, result_audio_mic2.astype(np.int16))
        #write("result_post_pyroom_mic3.wav", 16000, result_audio_mic3.astype(np.int16))
        #write("result_post_pyroom_mic4.wav", 16000, result_audio_mic4.astype(np.int16))
        #write("result_post_pyroom_mic5.wav", 16000, result_audio_mic5.astype(np.int16))
        #self.VartoPickle(room)
    def CalculateArrayAngle(self):
        pass
    def toJsonAndSave(self):
        pass
    #def generate_CSV(self):
    #    #Scenario : index
    #    #PathToMix : str [] 
    #    #SpeakersPaths : str []
    #    #Reverb : int
    #    #SNR : int
    #    #SpeakerPlacement : int [][]
    #    #angleOfArray : int
    #    Scenario = self.roomIdx
    #    PathToMix = self.mixPaths
    #    angleFromCenterOfMicrophones = self.CalculateArrayAngle()
    #    SpeakersPaths = []
    #    for p in self.speaker_soundfile_map.values():
    #        SpeakersPaths.append(s)
    #    data ={'Scenario':Scenario,
    #           'PathToMix':PathToMix,
    #           'SpeakersPaths':SpeakersPaths,
    #           'Reverb':self.rt60,
    #           'roomDimentions':self.roomDims,
    #           'SpeakerPlacement':self.speaker_placements,
    #           'angleFromCenterOfMicrophones':angleFromCenterOfMicrophones,
    #           'SNR':1}
#
    #    df = pd.DataFrame(data)
    #    df.to_csv('ALL_SCENARIOS_METADATA')
#
min_room_x = 12
min_room_y = 12
min_room_z = 3

max_room_x = 15
max_room_y = 15
max_room_z = 4




def convolve_audio_with_rir(audio_file):
    pass
def add_info_to_csv():
    pass


def rotate_points_numpy(points, theta):
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

    rotated_points = np.dot(np.array(points), rotation_matrix.T)

    return rotated_points






#get all sounds from train
g = generate_rirs()
g.setRoomIdx(1)
g.setNumOfSpeakers(2)
res = g.gather_wav_files(r"C:\Users\lipov\Documents\GitHub\project\RIRnewv\LibriSpeech","Train",1)
 
g.background_samples(3)
 

print(res)
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
#sample 1-2 random soun
# ds from train
g.get_random_sounds(res,num_speakers = 2)
g.generate_room_dimensions()
g.set_limits()
g.generate_mic_array("circular")
g.generate_speaker_placements(2)
directions = g.showRoom()
#g.generate_channels(directions)
g.generate_channels_V2(directions)

#g.diffusionNoiseV2(3)




#rirs = generate_rirs(speaker_placements,mic_array,sounds,room)
#print("therirs:")
#print(rirs)

#make a function of gather_wav_files with less data for efficiency 

#get random sounds
#generate speakers
#generate the 6 microphones
# for each [microphones]:speaker using place and associated wav file, generate #microphones rirs using pyrirgen
# convolve and add all
#get sounds

