from __future__ import division
import importlib


from AudioDataSet import AudioDataSet

from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.externals import joblib
import plot_results
import argparse

import numpy as np
from prettytable import PrettyTable


import os, errno, time,sys

from mir_eval.separation import bss_eval_sources as evaluation
#from pylab import plot, show, title, xlabel, ylabel, subplot
#import seaborn
from keras.models import load_model
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau
from keras.callbacks import TensorBoard as dtb
from keras.callbacks import TensorBoard as btb
from keras import backend as K




##################################################################################################
#    ARGUMENT PROCESSING AND INITIALISATION
##################################################################################################


my_path = os.getcwd()
# data type used in Keras by default.
# Most tests were done with float64
# CNN in Keras only supports float32
K.set_floatx('float64')


parser = argparse.ArgumentParser(description='SCSS')
parser.add_argument('-dspath', type=str, help='Dataset path [default: <code folder>/SCSS_DataSet/]', default= my_path +'/SCSS_DataSet/')
parser.add_argument('-exppath', type=str, help='Path where the experiment results is stored [default: <code folder>/Experiment-[nnconfig]-[time]/]', default= './')
parser.add_argument('-nsongs', type=int, help='Number of songs to load [default: 99]', default=99)
parser.add_argument('--doubleset', help='Double the dataset by taking a second extract from the audio files [default: Disabled]', action='store_true')
parser.add_argument('-ntests', type=int, help='Number of external songs to test [default: 0]', default=0)
parser.add_argument('-nval', type=float, help='%% of training songs to use for validation after training [default: 5%%]', default=0.05)
parser.add_argument('-normvol', type=float, help='Normalise audio file volumes before STFT to given absolute value (between -1 and 1)  [default: Disabled]', default=0.0)
parser.add_argument('-fs', type=int, help='Sample rate of audio files in Hz [default: 44100]', default=44100)
parser.add_argument('--ffilter', help='Apply band pass filter between 20 and 16,000 Hz', action='store_true')
parser.add_argument('-alength', type=int, help='Length of audio in seconds [default: 12]', default=8)
parser.add_argument('-aoffset', type=int, help='Offset of audio in seconds [default: 5]', default=5)
parser.add_argument('-stftwindow', type=int, help='Number of samples per STFT window [default: 8192]', default=8192)
parser.add_argument('-scaler', type=int, help='Scale to normalise NN input; 0=[-1,1], 1=[0,1] [default: 0]', default=0)
parser.add_argument('-nnconfig', type=int, help='Configuration of the DRNN to train [default: 1]', default=1)
parser.add_argument('--load', help='Load data from pretrained NN for a specific configuration', action='store_true')
parser.add_argument('--trial', help='Short trial of the DRNN', action='store_true')
parser.add_argument('--eval', help='Only evaluate a DRNN, no training', action='store_true')
parser.add_argument('-nLSTM', type=int, help='Number of LSTM layers [default: 3]', default=3)
parser.add_argument('-dcells', type=int, help='Number of hidden units for drums NN [default: 200]', default=200)
parser.add_argument('-bcells', type=int, help='Number of hidden units for bass NN [default: 200]', default=100)
parser.add_argument('-diters', type=int, help='Number of training iterations for drums NN [default: 200]', default=200)
parser.add_argument('-biters', type=int, help='Number of training iterations for for bass NN [default: 200]', default=200)
parser.add_argument('-batchsize', type=int, help='Batch size when training [default: 8]', default=8)
parser.add_argument('-earlystop', type=int, help='Patience value for early stop [default: No Early stop]', default=0)
parser.add_argument('-afunction', type=str, help='Activation function to use [default: tanh]', default='relu')
parser.add_argument('-l1', type=float, help='L1 recurrent regularises [default: 0.0]', default=0.0)
parser.add_argument('-l2', type=float, help='L2 recurrent regulariser [default: 0.0]', default=0.0)
parser.add_argument('-reducelr', type=int, help='Patience value for reduce LR on plateau [default: Disabled]', default=100)
parser.add_argument('-lrate', type=float, help='Learning rate of the optimiser [default: 0.001]', default=0.001)
parser.add_argument('--bnorm', help='Apply batch normalitation between LSTM layers [default: Disabled]', action='store_true')
parser.add_argument('--hps', help='Use hps as input [default: Disabled]', action='store_true')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

ds = AudioDataSet(args.dspath) # Initilise the dataset class with path to dataset files
ds.fs=args.fs # set audio sample rate
ds.length_audio=args.alength  # number of seconds to load for each song
ds.offset_audio=args.aoffset # seconds of offset when starting the song
ds.sampleperframe=args.stftwindow # number of samples per stft frame
ds.freq_filter = args.ffilter # set whether applying band pass filter on audio or not

# Set the value to normalise the audio volume after loading the data file (0.0 = no normalisation)
if args.normvol>0.0:
    ds.normvolume = args.normvol

# Initialise the scalers for data input normalisation.
# If using negative values, use tanh activation function; otherwise use ReLu
if args.scaler==0:
    ds.scaler = MinMaxScaler(feature_range=(-1, 1)) # initialise scaler to normalise input
    ds.scaler_b = MinMaxScaler(feature_range=(-1, 1)) # initialise scaler to normalise input
    ds.scaler_d = MinMaxScaler(feature_range=(-1, 1)) # initialise scaler to normalise input
    ds.scaler_hps = MinMaxScaler(feature_range=(-1, 1))
    activation_function='tanh'
elif args.scaler==1:
    ds.scaler = MinMaxScaler(feature_range=(0, 1)) # initialise scaler to normalise input
    ds.scaler_b = MinMaxScaler(feature_range=(0, 1)) # initialise scaler to normalise input
    ds.scaler_d = MinMaxScaler(feature_range=(0, 1)) # initialise scaler to normalise input
    ds.scaler_hps = MinMaxScaler(feature_range=(0, 1))
    activation_function='relu'
elif args.scaler==2:
    ds.scaler = MinMaxScaler(feature_range=(0, 2)) # initialise scaler to normalise input
    ds.scaler_b = MinMaxScaler(feature_range=(0, 2)) # initialise scaler to normalise input
    ds.scaler_d = MinMaxScaler(feature_range=(0, 2)) # initialise scaler to normalise input
    ds.scaler_hps = MinMaxScaler(feature_range=(0, 2))
    activation_function='relu'
elif args.scaler==3:
    ds.scaler = MinMaxScaler(feature_range=(-0.5, 0.5)) # initialise scaler to normalise input
    ds.scaler_b = MinMaxScaler(feature_range=(-0.5, 0.5)) # initialise scaler to normalise input
    ds.scaler_d = MinMaxScaler(feature_range=(-0.5, 0.5)) # initialise scaler to normalise input
    ds.scaler_hps = MinMaxScaler(feature_range=(-0.5, 0.5))
    activation_function='tanh'
elif args.scaler==4:
    ds.scaler = RobustScaler() # initialise scaler to normalise input
    ds.scaler_hps = RobustScaler()
    activation_function='relu'
else:
    print('Wrong scaler type - using default [-1,1]')
    

# Whether to use trial mode.
# Trial mode sets a small number of audio files to load and 10 iterations of training
# Useful to make sure a new configuration works without wasting time.
trial = args.trial

# Whether to double the dataset size by extracting 2 audio samples from each audio file
if args.doubleset:
    ds.double_set = True
else:
    ds.double_set = False

# Number of audio files to load
ds.number_files = args.nsongs

# Whether to use harmonic percusive separation as input for the neural networks
# instead using a plain STFT. The training will be much slower since the input data is doubled.
if args.hps:
    ds.hps_input = True
else:
    ds.hps_input = False

# Number of neural network configuration to run (refer for drnn.py for further details)
experimentN = args.nnconfig

# Size of the training batch
n_batch=args.batchsize 
# Number of test songs; the results will not be evaluated - audio and bass tracks are not
# available.  The script will generate the raw and masked predicted tracks   
songs_to_test = args.ntests
# Number of songs to use from the training dataset for testing.  Since we have  the original
# bass and drums audio files, with this tracks we can evaluate the results using SAR, SDR and SIR
# The downside is that the training dataset will be smaller
ds.validation_size = args.nval
num_iters_bass = args.biters #Number of iterations for bass training
num_iters_drums = args.diters #Number of iterations for drums training

# path to the folder where the experiment files will be stored.
experimentPath = my_path+'/Experiment-'+str(experimentN)+'-'+str(time.time())+'/'

# Trial mode values
if trial:
    num_iters_bass = 10
    num_iters_drums = 10
    ds.number_files = 4
    ds.validation_size = 0.5
    songs_to_test = 1

# Whether to train the NN or just evaluate the results of a previous experiment
train = not args.eval # if we want to train or just evaluate

if train:
    ds.create_dataset()
# Whether to to load previously trained data (weights) for evaluation or further training
loadData = args.load
# Number of neurons in hidden layers (Drums)
dunits = args.dcells
# Number of neurons in hidden layers (Bass)
bunits =args.bcells
# Number of LSTM layers
nLSTM = args.nLSTM
# Patience (number of iterations) for the early stop
earlystop = args.earlystop
# L2 regularisation factor
l2 = args.l2
# L1 regularisation factor
l1 = args.l1

plr=args.reducelr
if plr==100:
    # Learning rate of the optimisation
    lrate=args.lrate 
    ilrate=args.lrate
else:
    # Learning rate of the optimisation
    lrate=args.lrate
    # Initial learning rate of the NN is 0.1 - it will decrease when loss function not improving
    ilrate=0.1
# Whether to use batch normalisation between the LSTM layers
if args.bnorm:
    bnorm = True
else:
    bnorm = False
# If loading data, a path to the experiment folder is needed.
if loadData:
    if args.exppath == './':
        print('To load an experiment you need to give an experiment path with -exppath')
        sys.exit(0)
    else:
        experimentPath=my_path+args.exppath
        
##################################################################################################
#   MAIN
##################################################################################################

# Create experiment folder
if not os.path.exists(experimentPath):
    try:
        os.makedirs(experimentPath)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
with open(experimentPath+'config-'+str(experimentN)+'.txt','w') as fh:
    print(args, file=fh)
               
            
if train:
    print('#################################################')
    print('#               TRAINING NETWORK')
    print ('# Step,Features to train : ' + str(ds.X_train[0].shape))

    if loadData:
        print('Loading previous NN trained weights')
        # Load neural network from files
        network_bass = load_model(experimentPath+'bass'+str(experimentN)+'.h5')
        network_drums = load_model(experimentPath+'drums'+str(experimentN)+'.h5')
        # Load scalers to avoid having to load the whole dataset
        ds.myscaler = joblib.load(experimentPath+'scaler.sklearn')

    else:
        print('Creating new NN model for bass and drums')
        # create new NNs using the configrations in file drnn.py
        create_lstm_network = getattr(importlib.import_module('drnn'), 'create_lstm_network'+str(experimentN))
          
        network_bass=create_lstm_network(ds.X_train[0].shape, bunits, nLSTM, activation_function,l2, bnorm, ds.hps_input,ilrate)
        # Plot a graph and save is as png - NOT WORKING IN NEWER VERSION OF Tensorflow/Keras
        #plot_model(network_bass, to_file=experimentPath+'network_bass-'+str(experimentN)+'.png', show_shapes=True)

            
        network_drums=create_lstm_network(ds.X_train[0].shape, dunits, nLSTM, activation_function,l2, bnorm, ds.hps_input,ilrate)
        # Plot a graph and save is as png - NOT WORKING IN NEWER VERSION OF Tensorflow/Keras
        #plot_model(network_drums, to_file=experimentPath+'network_drums-'+str(experimentN)+'.png', show_shapes=True)


    # Interval to register train/test loss information
    epochs_per_iter = 10
    cur_iter_drums = 0
    cur_iter_bass = 0
    history_size = 0

    train_err_plot_bass = np.zeros(num_iters_bass)
    test_err_plot_bass = np.zeros(num_iters_bass)
    
    train_err_plot_drums = np.zeros(num_iters_drums)
    test_err_plot_drums = np.zeros(num_iters_drums)

    ###############################################################################
    print ('Training drums NN...')
    #Configure the callback ReduceLROnPlateau to reduce learning rate after plr iterations of no improvement
    reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=plr, min_lr=lrate, verbose=1)
    # Initialise callbacks for tensorboard - it only works when using a single tensorbard, so unused.
    dtb = dtb(histogram_freq=1, write_grads = True, log_dir=experimentPath+'/logs/drums/')
    btb = btb(histogram_freq=1, write_grads = True, log_dir=experimentPath+'/logs/bass/')


    while cur_iter_drums < num_iters_drums:

        print('Iteration: ' + str(cur_iter_drums))
        if earlystop == 0:
            history_drums=network_drums.fit(ds.X_train, ds.y_train_drums, batch_size=n_batch, epochs=epochs_per_iter, verbose=1, \
                                callbacks=[reducelr],validation_split=0.05,shuffle=True)
        else:
            history_drums=network_drums.fit(ds.X_train, ds.y_train_drums, batch_size=n_batch, epochs=epochs_per_iter, verbose=1, \
                                callbacks=[reducelr,EarlyStopping(monitor='val_loss', patience=earlystop)],validation_split=0.05,shuffle=True)
        # Store traububg and validation loss values
        history_size = len(history_drums.history.get('loss'))
        train_err_plot_drums[cur_iter_drums:history_size+cur_iter_drums]=history_drums.history.get('loss')
        test_err_plot_drums[cur_iter_drums:history_size+cur_iter_drums]=history_drums.history.get('val_loss')
        # Save the NN weights for debugging 
        #with open(experimentPath+'weights-drums'+str(experimentN)+'.txt','a') as wf:
        #    for i in range(len(network_bass.layers)):          
        #        print('layer weights: %s - %s' %(i,np.histogram(network_bass.layers[i].get_weights()[0].ravel(), bins=4)), file=wf)
        #        print('layer bias: %s - %s' %(i,np.histogram(network_bass.layers[i].get_weights()[1].ravel(), bins=4)), file=wf)

        cur_iter_drums+= history_size
        # If early stop triggered - adjust the size of the arrays to plot
        if history_size !=epochs_per_iter:
            print('Early stopping triggered')
            train_err_plot_drums = train_err_plot_drums[:cur_iter_drums-num_iters_drums]
            test_err_plot_drums = test_err_plot_drums[:cur_iter_drums-num_iters_drums]
            num_iters_drums = train_err_plot_drums.size
            
    # Save the drums model to file 
    network_drums.save(experimentPath+'drums'+str(experimentN)+'.h5') 
    # Save drums train/test error to a png file
    plot_results.plot_loss('loss_chart_drums'+str(experimentN), experimentPath, num_iters_drums, train_err_plot_drums, test_err_plot_drums)
    
    print ('Training bass NN...')    
    while cur_iter_bass < num_iters_bass:
        print('Iteration: ' + str(cur_iter_bass))
        if earlystop == 0:
            history_bass=network_bass.fit(ds.X_train, ds.y_train_bass,batch_size=n_batch, epochs=epochs_per_iter, verbose=1,\
                              callbacks=[reducelr],validation_split=0.05,shuffle=True)
        else: 
            history_bass=network_bass.fit(ds.X_train, ds.y_train_bass,batch_size=n_batch, epochs=epochs_per_iter, verbose=1, \
                              callbacks=[reducelr,EarlyStopping(monitor='val_loss', patience=earlystop)],validation_split=0.05,shuffle=True)
        # Store traububg and validation loss values
        history_size = len(history_bass.history.get('loss'))
        train_err_plot_bass[cur_iter_bass:history_size+cur_iter_bass]=history_bass.history.get('loss')
        test_err_plot_bass[cur_iter_bass:history_size+cur_iter_bass]=history_bass.history.get('val_loss')
        # Save the NN weights for debugging 
        #with open(experimentPath+'weights-bass'+str(experimentN)+'.txt','a') as wf:
        #    for i in range(len(network_bass.layers)):          
        #        print('layer weights: %s - %s' %(i,np.histogram(network_bass.layers[i].get_weights()[0].ravel(), bins=4)), file=wf)
        #        print('layer bias: %s - %s' %(i,np.histogram(network_bass.layers[i].get_weights()[1].ravel(), bins=4)), file=wf)
        cur_iter_bass+= history_size
        
        # If early stop triggered - adjust the size of the arrays to plot
        if history_size !=epochs_per_iter:
            print('Early stopping triggered')
            train_err_plot_bass = train_err_plot_bass[:cur_iter_bass-num_iters_bass]
            test_err_plot_bass = test_err_plot_bass[:cur_iter_bass-num_iters_bass]
            num_iters_bass = train_err_plot_bass.size
            
    # Save the bass model to file
    network_bass.save(experimentPath+'bass'+str(experimentN)+'.h5')
    
    # Save bass train/test error to a png file
    plot_results.plot_loss('loss_chart_bass'+str(experimentN),experimentPath, num_iters_bass, train_err_plot_bass, test_err_plot_bass)

    #Save sklearn scalers
    joblib.dump(ds.myscaler,experimentPath+'scaler.sklearn')
#    joblib.dump(ds.myscaler_b,experimentPath+'scaler_b.sklearn')
#    joblib.dump(ds.myscaler_d,experimentPath+'scaler_d.sklearn') 
    ################################################################################
    #                   TESTING
    ################################################################################
    # Test on songs with audio/bass available
        
    # Predict bass/drums for a batch of songs X_val
    septest_bass = network_bass.predict_on_batch(ds.X_val)
    septest_drums = network_drums.predict_on_batch(ds.X_val)
    
    print('Number of predictions: ' + str(septest_bass.shape))
    # Initialise evaluation result files
    
    # Evaluation measures SIR, SDR and SAR using both bass and drums prediction
    fh2 = open(experimentPath+'Validation2-'+str(experimentN)+'.txt', 'w')
    
    # Process each prediction
    for i in range(septest_bass.shape[0]):
        # Sanity check of the range in predicted (normalised) values
        print('Max/Min value in bass prediction: %s/%s' %(septest_bass[i].max(),septest_bass[i].min()))
        print('Max/Min value in drums prediction: %s/%s' %(septest_drums[i].max(),septest_drums[i].min()))
        
        # Generate the audio files to evaluate the results of the predictions
        bass_pred_samples_raw, drums_pred_samples_raw, bass_pred_samples_masked, drums_pred_samples_masked = \
                ds.generate_tests(experimentPath, i, septest_bass[i], septest_drums[i])
    
        # Get te minimum array length among all files
        min_size = min(bass_pred_samples_raw.shape[0], drums_pred_samples_raw.shape[0],\
                       bass_pred_samples_masked.shape[0], drums_pred_samples_masked.shape[0], \
                       ds.bass_org_audio[i].shape[0], ds.drums_org_audio[i].shape[0])
        
        # make original bass/drums and predicted audios the same size (min) and save them to a file:
        ds.samples_to_wav(experimentPath, ds.audio_filenames_val[i][:-4]+'_org_bass.wav', ds.bass_org_audio[i][0:min_size])
        ds.samples_to_wav(experimentPath, ds.audio_filenames_val[i][:-4]+'_org_drums.wav', ds.drums_org_audio[i][0:min_size])
    
        # Prepare the arrays to compare using the minimum size
        reference_array=np.zeros(shape=(2,min_size))
        recovered_array=np.zeros(shape=(2,min_size))
    
        # Evaluate predictions and save the results in a text file   
        with open(experimentPath+'Validation2-'+str(experimentN)+'.txt','a') as fh2:
            
            print('#########################################################', file=fh2)
            print('Evaluation of file: '+ ds.audio_filenames_val[i], file=fh2)
            print('#########################################################', file=fh2)
            t = PrettyTable(['Audio Output', 'SDR','SIR','SAR']) 
            
            # Evaluate RAW prediction
            # Trim all arrays to the same length
            reference_array[0]=ds.bass_org_audio[i][0:min_size]
            recovered_array[0]=bass_pred_samples_raw[0:min_size]
            reference_array[1]=ds.drums_org_audio[i][0:min_size]
            recovered_array[1]=drums_pred_samples_raw[0:min_size]
            # Evaluate
            sdr, sir, sar, popt=evaluation(reference_array,recovered_array)
            t.add_row(['Bass raw',"%.2f" % sdr[0],"%.2f" % sir[0],"%.2f" % sar[0]])
            t.add_row(['Drums raw',"%.2f" % sdr[1],"%.2f" % sir[1],"%.2f" % sar[1]])
            
            # Evaluate MASKED prediction
            # Trim all arrays to the same length
            recovered_array[0]=bass_pred_samples_masked[0:min_size]
            recovered_array[1]=drums_pred_samples_masked[0:min_size]
            # Evaluate
            sdr, sir, sar, popt=evaluation(reference_array,recovered_array)        
            t.add_row(['Bass masked',"%.2f" % sdr[0],"%.2f" % sir[0],"%.2f" % sar[0]])
            t.add_row(['Drums masked',"%.2f" % sdr[1],"%.2f" % sir[1],"%.2f" % sar[1]])
        
               
            print(t, file=fh2)



else:
    # Load the model and evaluate it
    network_bass = load_model(experimentPath+'bass'+str(experimentN)+'.h5')
    network_drums = load_model(experimentPath+'drums'+str(experimentN)+'.h5')
#    ds.myscaler_b = joblib.load(experimentPath+'scaler_b.sklearn')
#    ds.myscaler_d = joblib.load(experimentPath+'scaler_d.sklearn')
    ds.myscaler = joblib.load(experimentPath+'scaler.sklearn')


# Save NN information in a file experiment-[x].txt
with open(experimentPath+'experiment-'+str(experimentN)+'.txt','w') as fh:
    # Pass the file handle in as a lambda function to make it callable
    print('###### BASS '+str(experimentN)+ ' ######', file=fh)
    network_bass.summary(print_fn=lambda x: fh.write(x + '\n'))
    print('###### DRUMS '+str(experimentN)+' ######', file=fh)
    network_drums.summary(print_fn=lambda x: fh.write(x + '\n'))
    print('###################', file=fh)


###############################################################################  
# Test on other songs without evaluation

if songs_to_test>0:
    # Load the list of songs to test
    import glob
    test_songs=list()
    os.chdir(ds.test_path)
    for i in range(songs_to_test):
        test_songs.append(glob.glob("*.wav")[i])
    for song in test_songs:
        print('Testing file: ', song)
        # Predict using harmonic percussive separation input
        if ds.hps_input:
            stft_test, ph_stft_test, hps= ds.wav_to_stft(ds.test_path+song, ds.offset_audio, ds.length_audio)
            ds.myscaler_hps=ds.scaler_hps.fit(hps)
            stft_test_scaled = ds.myscaler_hps.transform(hps)
        # Predict using standard frequency input
        else:
            stft_test, ph_stft_test,_= ds.wav_to_stft(ds.test_path+song, ds.offset_audio, ds.length_audio)
            lenghtsong=stft_test.shape[0]
            ds.myscaler=ds.scaler.fit(stft_test)
            stft_test_scaled = ds.myscaler.transform(stft_test)
            
        stft_test_scaled=np.reshape(stft_test_scaled, newshape=(1,stft_test_scaled.shape[0],stft_test_scaled.shape[1]))
    
        
        # Run bass DRNN
        septest_bass = network_bass.predict_on_batch(stft_test_scaled)
        septest_bass=septest_bass[0]
        print('Max/Min value in bass prediction: %s/%s' %(septest_bass.max(),septest_bass.min()))
        # Run drums DRNN
        septest_drums = network_drums.predict_on_batch(stft_test_scaled)
        septest_drums=septest_drums[0]
        print('Max/Min value in drums prediction: %s/%s' %(septest_drums.max(),septest_drums.min()))

        # Save results as wav files (raw and masked)
        ds.generate_audio_from_test_predictions(experimentPath, song, septest_bass, septest_drums, stft_test, ph_stft_test)

  
