from __future__ import division
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import numpy as np
import librosa
import os
import glob
from plot_results import plot_hist
from scipy import signal
from keras import backend as K

class AudioDataSet:
    def __init__(self,path):
        # Folder names for mixed audio file and separate bass/drums files
        self.mix_path=path+'seg_mix/'
        self.bass_path=path+'seg_bass/'
        self.drums_path=path+'seg_drums/'
        self.test_path=path+'Test/'
        # Number of audio files to load
        self.number_files=99
        # audio length
        self.length_audio=12
        # audio offset
        self.offset_audio=0
        # STFT window size
        self.sampleperframe=8192
        # Scalers for input normalisation
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.scaler_b = MinMaxScaler(feature_range=(-1, 1))
        self.scaler_d = MinMaxScaler(feature_range=(-1, 1))
        self.scaler_hps = MinMaxScaler(feature_range=(-1, 1))
        # Sample rate of the audio files
        self.fs = 44100
        # Max number of STFT windows
        self.maxframesong=0
        self.maxframesong_hps=0
        # Temp lists to store loaded audio
        self.input_file = list()
        self.output_file_bass = list()
        self.output_file_drums = list()
        # Validation song info
        self.audio_filenames_val = list()
        self.input_file_val_mag = list()
        self.input_file_val_mag_hps = list()
        self.output_file_bass_val_mag = list()
        self.output_file_drums_val_mag = list()
        self.input_file_val_ph = list()
        self.output_file_bass_val_ph = list()
        self.output_file_drums_val_ph = list()
        # Original audio files
        self.bass_org_audio = list()
        self.drums_org_audio = list()
        # Ratio of songs to use for testing in the available dataset
        self.test_songs = 0
        # Flag to say if songs a band pass filter should be applied on the audio
        self.freq_filter = False
        # Flag to say if harmonic percussive separation should be used as input
        self.hps_input = False
        # Flag to say if dataset should be doubled by extracting 2 fragments of audio per file
        self.double_set = False
        # Float that says if the audio files should be normalised after loading and to what level [0.,1.]
        self.normvolume =0.0
        # Temp list of the hps data
        self.hps = list()

   
    
    ###############################################################################
    # FUNCTION: _complexarray(r, theta)
    # DESCRIPTION: Merges the frequency and phase matrices into a complex matrix before iSTFT
    # PARAMETERS:
    # - r: frequency magnitude matrix
    # - theta: phase information matrix
    # RETURNS:
    # - complexarray_temp: new complex matrix.
    ###############################################################################
    def _complexarray(self, r, theta):
        X=np.zeros(shape=(r.shape[0],r.shape[1]))
        Y=np.zeros(shape=(r.shape[0],r.shape[1]), dtype=complex)
        for j in range(r.shape[0]):
    
            for i in range(r.shape[1]):
    
                X[j,i]=r[j,i]*np.cos(theta[j,i])
    
                Y[j,i]=complex(0,r[j,i]*np.sin(theta[j,i]))
    
        complexarray_temp=X+Y
    
    
        return complexarray_temp
    

    ###############################################################################
    # FUNCTION: _maxlength_song(inputs)
    # DESCRIPTION: Calculates the max length among all songs in the dataset to later shape all arrays the same size.
    # PARAMETERS:
    # - inputs: list with the loaded wavs
    # RETURNS:
    # - max_length: max length among all songs as integer
    ###############################################################################
    def _maxlength_song(self, inputs):
        max_length=0
        for i in range(len(inputs)):
            if inputs[i].shape[0]>max_length:
                max_length=inputs[i].shape[0]
        return max_length
    

    ###############################################################################
    # FUNCTION: _bandpass_filter(my_audio, low_cut, high_cut)
    # DESCRIPTION: Filter an audio file with an order 5 band pass filter.
    # PARAMETERS:
    # - my_audio: array with loaded wav file
    # - low_cut: low frequency of the band pass filter
    # - high_cut: high frequency of the band pass filter
    # RETURNS:
    # - filtered_audio: new audio array filtered in frequency
    ###############################################################################   
    def _bandpass_filter(self, my_audio, low_cut, high_cut):
        
        # High-pass filter
        nyquist_rate = self.fs/2.
        low = low_cut/nyquist_rate
        high = high_cut/nyquist_rate
        # order 5
        b,a = signal.butter(5,[low, high], btype='band', analog=False)
        filtered_audio = signal.filtfilt(b,a,my_audio)
        
        return filtered_audio
    
    ###############################################################################
    # FUNCTION: wav_to_stft(audio, start, length)
    # DESCRIPTION: Loads a wav file and transforms it from time to frequency domain, ready to process
    # PARAMETERS:
    # - audio: loaded wav file
    # - start: offset to apply to the audio
    # - length: length of the audio file
    # RETURNS:
    # - mag_stft: frequency magnitud in an array of shape (nsteps, magnitude)
    # - ph_stft: phase information for later reconstruction of the audio file
    # - hps: frequency information split in harmonic and percussive; 0 if using standard frequency info.
    #        Matrix double the size of the frequency output, harmonic first, percussive last.
    ###############################################################################      
    def wav_to_stft(self, audio, start, length):
        print("Loading file %s" %audio)
        # Load the file
        audioFile, fs = librosa.load(audio, duration=length, offset=start, sr=self.fs, dtype=np.float64)
        # Apply the filter if required - filter fixed to 80Hz - 15KHz
        if self.freq_filter:
            audioFile = self._bandpass_filter(audioFile, 80.0, 15000.0)
        # Normalise the volume of the audio file if required
        if self.normvolume > 0.0:
            audioFile = audioFile*(self.normvolume/np.max(audioFile))
            audioFile = librosa.util.normalize(audioFile)
        # Get the STFT
        stft = (librosa.stft(audioFile, n_fft=self.sampleperframe))
        # Split STFT in magnitued and frequency matrices  
        mag_stft, ph_stft= librosa.magphase(stft)
        # Get the phase angle in radians
        ph_stft=np.angle(ph_stft)
        # Normalise all angles to values between 0 and pi
        ph_stft=np.unwrap(ph_stft)
        
        # Split between harmonic and percusive information if required      
        if self.hps_input:
            D_harmonic, D_percussive = librosa.decompose.hpss(stft)
            D_harmonic,_ = librosa.magphase(D_harmonic)
            D_percussive,_ = librosa.magphase(D_percussive)
        # Concatenate harmonic and percussive into a single matrix.
            hps = np.concatenate((D_harmonic,D_percussive), axis=0)
        # Transpose matrix to have as first element the stft window index/nstep
            return mag_stft.T, ph_stft, hps.T
        else:
        # Transpose matrix to have as first element the stft window index/nstep
            return mag_stft.T, ph_stft, 0

    
    ###############################################################################
    #returns two lists (inputs and outputs) and a int value for the longest song frame
    #we'll need it later
    ###############################################################################
    # FUNCTION: _load_dataset()
    # DESCRIPTION: 
    # PARAMETERS:
    # -  NONE
    # RETURNS:
    # - 
    ###############################################################################
    def _load_dataset(self):
        mix_files=list()
        bass_files=list()
        drums_files=list()
        # Get audio files from training data
        my_path = os.getcwd()
        os.chdir(self.mix_path)
        for file in glob.glob("*.wav"):
            mix_files.append(file)
        
        os.chdir(self.bass_path)   
        for file in glob.glob("*.wav"):
            bass_files.append(file)
        
        os.chdir(self.drums_path)
        for file in glob.glob("*.wav"):
            drums_files.append(file)
        # Back to experiment folder to leave the results there.
        os.chdir(my_path)
        # Variables to analyse the input data ranges.
        max_o = 0.0
        max_b= 0.0
        max_d= 0.0
        min_o= 0.0
        min_b= 0.0
        min_d= 0.0
        
        mix_files.sort()
        bass_files.sort()
        drums_files.sort()
    
    
        all_file=list()
        all_file_b=list()
        all_file_d=list()
        all_file_hps=list()
        

        # Make sure that the specified number of files to load is not greater
        # than the available files in the training data set
        if self.number_files > len(mix_files):
            self.number_files = len(mix_files)-1
        
        # Check if dataset should be double extracting 2 audio sequencies per file
        if self.double_set:
            self.test_songs = int(self.number_files*2*self.validation_size)
            self.train_songs = self.number_files*2 - self.test_songs
        else:
            self.test_songs = int(self.number_files*self.validation_size)
            self.train_songs = self.number_files - self.test_songs
            
        
        print('Loading %s training files and %s validation files.' %(self.train_songs,self.test_songs))
        songCounter = 0
        for i in range(self.number_files):
            if songCounter<self.train_songs:
                # Load training audio file - we only want the frequency info
                input_mix,_,hps=self.wav_to_stft('%s%s' % (self.mix_path,mix_files[i]), self.offset_audio, self.length_audio)
                output_bass,_,_=self.wav_to_stft('%s%s' % (self.bass_path,bass_files[i]), self.offset_audio, self.length_audio)
                output_drums,_,_=self.wav_to_stft('%s%s' % (self.drums_path,drums_files[i]), self.offset_audio, self.length_audio)
                if self.double_set:   
                    input_mix2,_,hps2=self.wav_to_stft('%s%s' % (self.mix_path,mix_files[i]), self.offset_audio + self.length_audio, self.length_audio)
                    output_bass2,_,_=self.wav_to_stft('%s%s' % (self.bass_path,bass_files[i]), self.offset_audio + self.length_audio, self.length_audio)
                    output_drums2,_,_=self.wav_to_stft('%s%s' % (self.drums_path,drums_files[i]), self.offset_audio + self.length_audio, self.length_audio)
            else:
                # Load test data file
                # For validation we now need the phase to reconstruct the audio files
                # Train and test data are split to save memory by discarding phase info in train dataset
                input_mix, input_mix_ph,hps=self.wav_to_stft('%s%s' % (self.mix_path,mix_files[i]), self.offset_audio, self.length_audio)
                output_bass, output_bass_ph,_=self.wav_to_stft('%s%s' % (self.bass_path,bass_files[i]), self.offset_audio, self.length_audio)
                output_drums, output_drums_ph,_=self.wav_to_stft('%s%s' % (self.drums_path,drums_files[i]), self.offset_audio, self.length_audio)
                if self.double_set:
                    input_mix2, input_mix_ph2,hps2=self.wav_to_stft('%s%s' % (self.mix_path,mix_files[i]), self.offset_audio+self.length_audio, self.length_audio)
                    output_bass2, output_bass_ph2,_=self.wav_to_stft('%s%s' % (self.bass_path,bass_files[i]), self.offset_audio+self.length_audio, self.length_audio)
                    output_drums2, output_drums_ph2,_=self.wav_to_stft('%s%s' % (self.drums_path,drums_files[i]), self.offset_audio+self.length_audio, self.length_audio)
        
            # Get max and min data values in the input and expected output
            if max_o < (input_mix.max()):
                max_o = (input_mix.max())
            if max_b < (output_bass.max()):
                max_b = (output_bass.max())
            if max_d < (output_drums.max()):
                max_d = (output_drums.max())            
            if min_o > (input_mix.min()):
                min_o = (input_mix.min())
            if min_b > (output_bass.min()):
                min_b = (output_bass.min())
            if min_d > (output_drums.min()):
                min_d = (output_drums.min())
               
            # Add all audio files (inputs and outputs) to a list for scaler definition
            all_file.append(np.vstack(input_mix))
            all_file_b.append(np.vstack(output_bass))
            all_file_d.append(np.vstack(output_drums))
            if self.double_set:
                all_file.append(np.vstack((input_mix2, output_bass2, output_drums2)))
            if self.hps_input:
                all_file_hps.append(np.vstack(hps))
                if self.double_set:
                    all_file_hps.append(np.vstack(hps2))
            
            # Add audio input/output information to the training data lists.
            if songCounter<self.train_songs:
                self.input_file.append(input_mix)
                self.output_file_bass.append(output_bass)
                self.output_file_drums.append(output_drums)
                if self.hps_input:
                    self.hps.append(hps)
                if self.double_set:
                    self.input_file.append(input_mix2)
                    self.output_file_bass.append(output_bass2)
                    self.output_file_drums.append(output_drums2)
                    if self.hps_input:
                        self.hps.append(hps2)
            # Add validation data info to the lists:
            # - audio_filenames_val: file name
            # - input_file_val_mag: input frequency magnitude 
            # - output_file_bass_val_mag: expected bass frequency magnitude 
            # - output_file_drums_val_mag: expected drums frequency magnitude
            # - input_file_val_ph: input file phase info to reconstruct the audio file from masked data.
            # - output_file_bass_val_ph: bass phase info to reconstruct the audio file from masked data.
            # - output_file_drums_val_ph: drums phase info to reconstruct the audio file from masked data.
            else:
                self.audio_filenames_val.append(mix_files[i])
                self.input_file_val_mag.append(input_mix)
                self.output_file_bass_val_mag.append(output_bass)
                self.output_file_drums_val_mag.append(output_drums)
                
                self.input_file_val_ph.append(input_mix_ph)
                self.output_file_bass_val_ph.append(output_bass_ph)
                self.output_file_drums_val_ph.append(output_drums_ph)
                if self.hps_input:
                    self.input_file_val_mag_hps.append(hps)
                if self.double_set:
                    self.audio_filenames_val.append('2'+mix_files[i])
                    self.input_file_val_mag.append(input_mix2)
                    self.output_file_bass_val_mag.append(output_bass2)
                    self.output_file_drums_val_mag.append(output_drums2)                    
                    self.input_file_val_ph.append(input_mix_ph2)
                    self.output_file_bass_val_ph.append(output_bass_ph2)
                    self.output_file_drums_val_ph.append(output_drums_ph2)
                    if self.hps_input:
                        self.input_file_val_mag_hps.append(hps2)
                
                # Load original bass/drum files for result evaluation
                bassFile, fs = librosa.load('%s%s' % (self.bass_path,bass_files[i]), duration=self.length_audio, offset=self.offset_audio, sr=self.fs)
                drumsFile, fs = librosa.load('%s%s' % (self.drums_path,drums_files[i]), duration=self.length_audio, offset=self.offset_audio, sr=self.fs)
                if self.double_set:
                    bassFile2, fs = librosa.load('%s%s' % (self.bass_path,bass_files[i]), duration=self.length_audio, offset=self.offset_audio+self.length_audio, sr=self.fs)
                    drumsFile2, fs = librosa.load('%s%s' % (self.drums_path,drums_files[i]), duration=self.length_audio, offset=self.offset_audio+self.length_audio, sr=self.fs)
                if self.freq_filter:
                    bassFile = self._bandpass_filter(bassFile, 80.0, 15000.0)
                    drumsFile = self._bandpass_filter(drumsFile, 80.0, 15000.0)
                    
                self.bass_org_audio.append(bassFile)
                self.drums_org_audio.append(drumsFile)
                if self.double_set:
                    self.bass_org_audio.append(bassFile2)
                    self.drums_org_audio.append(drumsFile2)
            if self.double_set:
                songCounter+=2
            else:
                songCounter+=1
        
                
        # Get the maximum number of frames among all songs
        self.maxframesong=self._maxlength_song(self.input_file)
        
        # Compute the min and max values among all songs to use later in the scaler
        print('All files size: ', len(all_file))
        print('Max files stft value o %2f, b %2f, d %2f : ' %(max_o, max_b, max_d))
        print('Min files stft value: o %2f, b %2f, d %2f : ' %(min_o, min_b, min_d))
#        print(np.histogram(np.array(self.input_file).ravel().astype(int), bins=5))
#        plot_hist(np.array(self.input_file).ravel(), 'MIX')
#        plot_hist(np.array(self.output_file_bass).ravel(), 'BASS')
#        plot_hist(np.array(self.output_file_drums).ravel(), 'DRUMS')
        # 3 different scalers, (1 input, 2 outputs)
        self.myscaler=self.scaler.fit(np.concatenate(all_file))
        self.myscaler_b=self.scaler_b.fit(np.concatenate(all_file_b))
        self.myscaler_d=self.scaler_d.fit(np.concatenate(all_file_d))
        # 4th scaler for hps input
        if self.hps_input:
            self.maxframesong_hps=self._maxlength_song(self.hps)
            self.myscaler_hps=self.scaler_hps.fit(np.concatenate(all_file_hps))
    

    ###############################################################################
    # FUNCTION: _populate_training_data()
    # DESCRIPTION: Populate X and y data from STFT matrices
    # PARAMETERS:
    # - NONE 
    # RETURNS:
    # - NONE
    ###############################################################################
    def _populate_training_data(self):
        # Initialise X and y to matrices of dimensions I tracks x J STFT frames x K STFT samples per frame
        if self.hps_input:
            print('%s-%s-%s'%(len(self.hps),self.maxframesong_hps,self.hps[0].shape[1]))
            self.X_train=np.zeros(shape=(len(self.hps),self.maxframesong_hps,self.hps[0].shape[1]))
        else:
            self.X_train=np.zeros(shape=(len(self.input_file),self.maxframesong,self.input_file[0].shape[1]))
        self.y_train_bass=np.zeros(shape=(len(self.output_file_bass),self.maxframesong, self.output_file_bass[0].shape[1]))
        self.y_train_drums=np.zeros(shape=(len(self.output_file_drums),self.maxframesong, self.output_file_drums[0].shape[1]))

    
        print('input shape:', self.X_train.shape)
        for i in range(len(self.input_file)):
            # Normalise input using the right scaler
            if self.hps_input:
                tempx=self.myscaler_hps.transform(self.hps[i])
            else:
                tempx=self.myscaler.transform(self.input_file[i])
            # Scales bass and drums expected output
            tempy_bass=self.myscaler_b.transform(self.output_file_bass[i])
            tempy_drums=self.myscaler_d.transform(self.output_file_drums[i])
            
            # Add audio info to final X_train and y_train_[] matrices ready for training
            if self.hps_input:
                for j in range(self.hps[i].shape[0]):
                    for kx in range(self.hps[i].shape[1]):
                        self.X_train[i,j,kx]=tempx[j,kx]
        
                    for ky in range(self.output_file_bass[i].shape[1]):
                        self.y_train_bass[i,j,ky]=tempy_bass[j,ky]
                    
                    for ky in range(self.output_file_drums[i].shape[1]):
                        self.y_train_drums[i,j,ky]=tempy_drums[j,ky]
            else:
                for j in range(self.input_file[i].shape[0]):
                    for kx in range(self.input_file[i].shape[1]):
                        self.X_train[i,j,kx]=tempx[j,kx]
        
                    for ky in range(self.output_file_bass[i].shape[1]):
                        self.y_train_bass[i,j,ky]=tempy_bass[j,ky]
                    
                    for ky in range(self.output_file_drums[i].shape[1]):
                        self.y_train_drums[i,j,ky]=tempy_drums[j,ky]
        
        # Force the dtype set in the main file. Either float32 or float64
        K.cast_to_floatx(self.X_train)
        K.cast_to_floatx(self.y_train_bass)
        K.cast_to_floatx(self.y_train_drums)
        # Sanity check of the generated data.
        print ('X training shape: ' + str(self.X_train.shape))
        print ('y bass training shape: ' + str(self.y_train_bass.shape))
        print ('y drums training shape: ' + str(self.y_train_drums.shape))
        print ('Max value after scaling: %2f' %(self.X_train.max()))
        print ('Max value after scaling: %2f' %(self.y_train_bass.max()))
        print ('Max value after scaling: %2f' %(self.y_train_drums.max()))
        print ('Min value after scaling: %2f' %(self.X_train.min()))
        print ('Min value after scaling: %2f' %(self.y_train_bass.min()))
        print ('Min value after scaling: %2f' %(self.y_train_drums.min()))
        print ('X train type : ', self.X_train.dtype)
        print ('y_bass train type: ', self.y_train_bass.dtype)
        print ('y_drums train type: ', self.y_train_drums.dtype)


    ###############################################################################
    # FUNCTION: _populate_validation_data()
    # DESCRIPTION: Populate X_val variable with audio validation dataset info
    # PARAMETERS:
    # -  NONE
    # RETURNS:
    # - NONE
    ###############################################################################
    def _populate_validation_data(self):
        # Initialise X to a matrix of dimensions I tracks x J STFT frames x K STFT samples per frame
        print('Validation songs: %s' %len(self.input_file_val_mag))
        self.X_val=np.zeros(shape=(len(self.input_file_val_mag),self.maxframesong,self.input_file_val_mag[0].shape[1]))
        
        # Build the final X_val matrix
        for i in range(len(self.input_file_val_mag)):
            # JM: Normalise input using scaler
            tempx=self.myscaler.transform(self.input_file_val_mag[i])    
    
            for j in range(self.input_file_val_mag[i].shape[0]):
                for kx in range(self.input_file_val_mag[i].shape[1]):
                    self.X_val[i,j,kx]=tempx[j,kx]
        # Force the dtype set in the main file. Either float32 or float64
        K.cast_to_floatx(self.X_val)
        # Sanity check of the generated data.
        print ('X validation shape: ' + str(self.X_val.shape))
        print ('X validation type : ', self.X_val.dtype)
    
    ###############################################################################
    # FUNCTION: _populate_validation_data_hps
    # DESCRIPTION: Populate X_val variable with audio hps validation dataset info 
    # PARAMETERS:
    # -  
    # RETURNS:
    # - 
    ###############################################################################
    def _populate_validation_data_hps(self):
        # Initialise X to a matrix of dimensions I tracks x J STFT frames x K STFT samples per frame
        print('Validation songs: %s' %len(self.input_file_val_mag_hps))
        self.X_val=np.zeros(shape=(len(self.input_file_val_mag_hps),self.maxframesong_hps,self.input_file_val_mag_hps[0].shape[1]))
    
        # Build the final X_val matrix    
        for i in range(len(self.input_file_val_mag_hps)):
            # JM: Normalise input using scaler
            tempx=self.myscaler_hps.transform(self.input_file_val_mag_hps[i])      
            for j in range(self.input_file_val_mag[i].shape[0]):
                for kx in range(self.input_file_val_mag_hps[i].shape[1]):
                    self.X_val[i,j,kx]=tempx[j,kx]     
        K.cast_to_floatx(self.X_val)
        print ('X validation shape: ' + str(self.X_val.shape))


   
    ###############################################################################
    # FUNCTION: create_dataset()
    # DESCRIPTION: Main function called to populate all training and validation data
    # PARAMETERS:
    # -  
    # RETURNS:
    # - 
    ###############################################################################
    def create_dataset(self):
        self._load_dataset()
        self._populate_training_data()
        self.input_file.clear()
        self.output_file_bass.clear()
        self.output_file_drums.clear()
        if self.hps_input:
            self._populate_validation_data_hps()
        else:
            self._populate_validation_data()
        
    ###############################################################################
    # FUNCTION: _stft_to_wav(save_path, filename, recovered_mag, stft_phase, track_type)
    # DESCRIPTION: Save predicted STFT info to a wav file
    # PARAMETERS:
    # - save_path: path to save the file
    # - filename: name of the audio file
    # - recovered_mag: frequency magnitude of the predicted audio
    # - stft_phase: phase of the original audio file to reconstruct audio file
    # - track_type: [b] bass, [d] drums or [] mix track type to chose the scaler
    # RETURNS:
    # - stft_mag: unscaled frequency magnitude of the audio file in the original shape of STFT
    # - recovered_wav: ISTFT output (time domain data)
    ###############################################################################    
    def _stft_to_wav(self, save_path, filename, recovered_mag, stft_phase, track_type):
        # Scale back both tracks using the same scaler
        if track_type == 'b':
            unscaled_mag= self.myscaler_b.inverse_transform(recovered_mag)
        elif track_type == 'd':
            unscaled_mag= self.myscaler_d.inverse_transform(recovered_mag)
        else:
            unscaled_mag= self.myscaler.inverse_transform(recovered_mag)

        
        # Trim and transpose back to original STFT shape
        if self.maxframesong > 0:
            stft_mag=unscaled_mag[0:self.maxframesong,0:int(self.sampleperframe/2)+1].T
        else:
            stft_mag=unscaled_mag[:,0:int(self.sampleperframe/2)+1].T
        # Merge back frequency and phase information
        stft_array=self._complexarray(stft_mag,stft_phase)
        # Apply the inverse STFT
        recovered_wav=librosa.istft(stft_array)
        
        # Re-apply filter on the predicted output if required 
        if self.freq_filter:
            recovered_wav = self._bandpass_filter(recovered_wav, 80.0, 15000.0)


        print('Saving file %s' %(save_path+filename))
        # Save file to wav with normalised volume.
        librosa.output.write_wav(save_path+filename, recovered_wav, self.fs, norm=True)

        return stft_mag, recovered_wav

    ###############################################################################
    # FUNCTION: samples_to_wav(save_path, filename, wav_samples)
    # DESCRIPTION: Simplified function to save samples to wav file
    # PARAMETERS:
    # - save_path: path to save the file
    # - filename: name of the audio file
    # - wav_samples: audio data in time domain
    # RETURNS:
    # - NONE
    ###############################################################################    
    def samples_to_wav(self, save_path, filename, wav_samples):
        librosa.output.write_wav(save_path+filename, wav_samples, self.fs, norm=True)
        print('Saved file %s' %(save_path+filename))

    ###############################################################################
    # FUNCTION: generate_tests(experimentPath, i, recoveredbass_mag, recovereddrums_mag)
    # DESCRIPTION: Generate audio data to evaluate from predicted output and save results to wav
    # PARAMETERS:
    # - experimentPath: path to save the file
    # - i: index of the processed file to recover info from lists
    # - recoveredbass_mag: predicted bass frequency magnitude
    # - recovereddrums_mag: predicted drums frequency magnitude
    # RETURNS:
    # - r_bass_wav_raw: bass audio file in time domain
    # - r_drums_wav_raw: drums audio file in time domain
    # - bass_mask: masked bass audio file in time domain
    # - drums_mask: masked drums audio file in time domain
    ###############################################################################    
    def generate_tests(self, experimentPath, i, recoveredbass_mag, recovereddrums_mag):
        # Save the tracks without masking first (raw prediction) and get back the audio info in time domain
        # Bass:
        r_bass_stft, r_bass_wav_raw = self._stft_to_wav(experimentPath, self.audio_filenames_val[i][:-5]+'_r_bass_raw.wav', \
                                             recoveredbass_mag, self.output_file_bass_val_ph[i],'b')
        # Drums:
        r_drums_stft, r_drums_wav_raw = self._stft_to_wav(experimentPath, self.audio_filenames_val[i][:-5]+'_r_drums_raw.wav', \
                                              recovereddrums_mag, self.output_file_drums_val_ph[i],'d')
 
        # Create the bass and drums masks to apply to the original input file
        # using the unscaled stft frequency magnitude information
        maskone=r_bass_stft/(r_bass_stft+r_drums_stft)
        masktwo=r_drums_stft/(r_bass_stft+r_drums_stft)
        # Workaround to get rid of Nans, which make saving the file fail
        maskone[np.isnan(maskone)]=0
        masktwo[np.isnan(masktwo)]=0

        # Apply the masks to the original input file (on the frequency magnitude matrix)
        # mask is on the shape of the original STFT, input is transposed - shapes are adjusted
        recoveredbass_mag=maskone*(self.input_file_val_mag[i].T)
        recovereddrums_mag=masktwo*(self.input_file_val_mag[i].T)
        
        # Using the original phase information from the input, reconstruct the STFT complex matrix
        # for bass and drums
        drums_stft_mask=self._complexarray(recovereddrums_mag,self.input_file_val_ph[i])
        bass_stft_mask=self._complexarray(recoveredbass_mag,self.input_file_val_ph[i])
    
        # Transform data back to time domain
        bass_mask=librosa.istft(bass_stft_mask)
        drums_mask=librosa.istft(drums_stft_mask)
        
        # Save the resulting files - predicted bass/drums used as a mask on the original audio file
        librosa.output.write_wav(experimentPath+self.audio_filenames_val[i][:-5]+'_r_bass_masked.wav', bass_mask, self.fs, norm=True)
        librosa.output.write_wav(experimentPath+self.audio_filenames_val[i][:-5]+'_r_drums_masked.wav', drums_mask,self.fs, norm=True)
        return r_bass_wav_raw, r_drums_wav_raw, bass_mask, drums_mask
     
    ###############################################################################
    # FUNCTION: generate_audio_from_test_predictions(experimentPath, filename, recoveredbass_mag, \
    #                                                recovereddrums_mag, stft_mag, stft_phase)
    # DESCRIPTION: Save wav files of the predicted test data
    # PARAMETERS:
    # - experimentPath: path to save the file
    # - filename: name of the audio file to save
    # - recoveredbass_mag: predicted bass frequency magnitude
    # - recovereddrums_mag: predicted drums frequency magnitude
    # - stft_mag: frequency mangitude of the audio original test audio file
    # - stft_phase: phase information of the audio original test audio file 
    # RETURNS (NOT USED):
    # - NONE
    ###############################################################################    
    def generate_audio_from_test_predictions(self, experimentPath, filename, recoveredbass_mag, recovereddrums_mag, stft_mag, stft_phase):
        # Save the tracks without masking first (raw prediction) and get back the audio info in time domain
        # Bass:
        r_bass_stft, r_bass_wav_raw = self._stft_to_wav(experimentPath, filename[:-5]+'_r_bass_raw.wav', \
                                             recoveredbass_mag, stft_phase,' ')
        # Drums:
        r_drums_stft, r_drums_wav_raw = self._stft_to_wav(experimentPath, filename[:-5]+'_r_drums_raw.wav', \
                                              recovereddrums_mag, stft_phase,' ')
        
        # Create the bass and drums masks to apply to the original input file
        # using the unscaled stft frequency magnitude information
        maskone=r_bass_stft/(r_bass_stft+r_drums_stft)
        masktwo=r_drums_stft/(r_bass_stft+r_drums_stft)
        # Workaround to get rid of Nans, which make saving file fail
        maskone[np.isnan(maskone)]=0
        masktwo[np.isnan(masktwo)]=0

        # Apply the masks to the original input file (on the frequency magnitude matrix)
        # mask is on the shape of the original STFT, input is transposed - shapes are adjusted       
        recoveredbass_mag=maskone*stft_mag.T
        recovereddrums_mag=masktwo*stft_mag.T

        # Using the original phase information from the input, reconstruct the STFT complex matrix
        # for bass and drums
        drums_stft_mask=self._complexarray(recovereddrums_mag,stft_phase)
        bass_stft_mask=self._complexarray(recoveredbass_mag,stft_phase)

        # Transform data back to time domain   
        bass_mask=librosa.istft(bass_stft_mask)
        drums_mask=librosa.istft(drums_stft_mask)
  
        # Save the resulting files - predicted bass/drums used as a mask on the original audio file        
        librosa.output.write_wav(experimentPath+filename[:-5]+'_r_bass_masked.wav', bass_mask, self.fs, norm=True)
        librosa.output.write_wav(experimentPath+filename[:-5]+'_r_drums_masked.wav', drums_mask,self.fs, norm=True)



    
   
