def create_bg(self):
        bg_radius = np.random.uniform(low = 10.0,high = 20.0)
        bg_theta = np.random.uniform(0,np.pi*2)
        z = np.random(1.5,1.9)
        bg_loc = [bg_radius*np.cos(bg_theta),bg_radius*np.sin(bg_theta),z]
        for s in self.speaker_soundfile_map.values():
            audio, _ = librosa.core.load(s,sr = fs,mono= True)
            #_,audio= read(s,fs)
            #print("theaudio")
            print(audio)
            #44100
            e_absorption,max_order = pra.inverse_sabine(rt60_tgt,room_dim)
            room = pra.ShoeBox(self.room_dim,fs = fs,max_order = max_order,materials = pra.Material(e_absorption))
    def diffusionNoise(self):
        allSoundsTrain = self.gather_wav_files(r"C:\Users\lipov\Documents\GitHub\project\RIRnewv\LibriSpeech","Train",1)
        allSoundsTest = self.gather_wav_files(r"C:\Users\lipov\Documents\GitHub\project\RIRnewv\LibriSpeech","Test",0)
        allSounds = allSoundsTest+allSoundsTrain
        source_pattern = DirectivityPattern.HYPERCARDIOID
        
        for s in sounds.values():
            fs, audio = read(s)
            e_absorption, max_order = pra.inverse_sabine(0.7, self.roomDims)
            room = pra.ShoeBox(
                self.roomDims, fs=fs, materials=pra.Material(e_absorption), max_order=max_order
            )
            break
            
        for a in allSounds.values():
            for s in a:
                freeze_axis = random.randint(0, 2)
                if freeze_axis == 0:    
                    wall_normal = np.array([1, 0, 0])
                    y = random.uniform(0,self.roomDims[1])
                    z = random.uniform(0,self.roomDims[2])
                    fs, audio = read(s)
                    audio = (audio * 0.02).astype(np.float64)
                    source_dir = CardioidFamily(
                        orientation=DirectionVector(azimuth=random.uniform(0,360), colatitude=0, degrees=True),
                        pattern_enum=source_pattern,
                    ) 
                    room.add_source([0.2,y,z], signal=audio, delay=0.5,directivity=source_dir)
                if freeze_axis == 1:
                    wall_normal = np.array([0, 1, 0])
                    x = random.uniform(0,self.roomDims[0])
                    z = random.uniform(0,self.roomDims[2])
                    fs, audio = read(s)
                    audio = (audio * 0.02).astype(np.float64)
                    source_dir = CardioidFamily(
                        orientation=DirectionVector(azimuth=0, colatitude=90, degrees=True),
                        pattern_enum=source_pattern,
                    ) 
                    room.add_source([x,0.2,z], signal=audio, delay=0.5,directivity=source_dir)
                if freeze_axis == 2:
                    wall_normal = np.array([0, 0, 1])
                    x = random.uniform(0,self.roomDims[0])
                    y = random.uniform(0,self.roomDims[1])
                    fs, audio = read(s)
                    audio = (audio * 0.02).astype(np.float64)
                    source_dir = CardioidFamily(
                        orientation=DirectionVector(azimuth=0, colatitude=90, degrees=True),
                        pattern_enum=source_pattern,
                    ) 
                    room.add_source([x,y,0.2], signal=audio, delay=0.5,directivity=source_dir)
                
        mic_locs = np.array(self.microphones).transpose()  # Assuming mic_array is defined somewhere
        room.add_microphone_array(mic_locs)
        #room.simulate()#builds the rirs automatically
        #room.mic_array.to_wav(
        #    f"result_noise_pyroom.wav",
        #    norm=True,
        #    bitdepth=np.int16,
        #)
        room.plot()
        plt.show()
    def VartoPickle(self,variable):
        #make to pickle
        #save to filePath
        filePath = 'room_'+ str(self.roomIdx) + '.pk1'
        with open(filePath,'wb') as file:
            pickle.dump(variable,file)

        return filePath
    def PickletoVar(self,filePath):
        with open(filePath, 'rb') as file:
            loaded_variable = pickle.load(file)
        return loaded_variable
    
    def RoomToDF(self):
        pass
    def generateRoom(self):
        room = pra.ShoeBox(self.roomDims,fs = 16000,max_order = 10)
        