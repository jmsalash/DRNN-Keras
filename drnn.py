#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 12:24:07 2018

@author: user
"""
from keras.models import Sequential
from keras.layers import TimeDistributed, Dense, Dropout, Conv2D, Activation, Reshape, InputLayer 
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM
#from keras import optimizers
from keras.optimizers import RMSprop,Adam
from keras.regularizers import l2,l1


    
###############################################################################
# EXPERIMENT 1: [Original research paper configuration]
# - 3 time distributed layers (no activation) 
# - n LSTM layers (default 3) 
# - 1 time distributed layer with activation
# - RMSProp optimiser with default values (gamma=0.9, default learning rate=0.001)
# Optional: batch normalisation between LSTM layers, L2 recurrent regularisation
###############################################################################
def create_lstm_network1(num_frequency_dimensions, num_hidden_dimensions, \
                         num_recurrent_units=3, activation_function='relu', fl2=0.0, bn=False, hps=False, lrate=0.001):
    print('Creating DRNN for experiment 1')
    model = Sequential()
	#This layer converts frequency space to hidden space
    model.add(TimeDistributed(Dense(num_hidden_dimensions),input_shape=num_frequency_dimensions))
    model.add(TimeDistributed(Dense(num_hidden_dimensions)))
    model.add(TimeDistributed(Dense(num_hidden_dimensions)))
    

    for cur_unit in range(num_recurrent_units):
        print('Creating LSTM with %s neurons' %(num_hidden_dimensions))
        model.add(LSTM(num_hidden_dimensions, return_sequences=True, recurrent_regularizer=l2(fl2)))
#        model.add(LSTM(num_hidden_dimensions, return_sequences=True))
        if bn:
            model.add(BatchNormalization())
#        model.add(LSTM(num_hidden_dimensions, return_sequences=True))

	#This layer converts hidden space back to frequency space
#    model.add(TimeDistributed(Dense(num_frequency_dimensions[1],activation=activation_function)))
    if hps:
        myoutput = int(num_frequency_dimensions[1]/2)
    else:
        myoutput = num_frequency_dimensions[1]
    print('Samples per step: ', myoutput)
    model.add(TimeDistributed(Dense(myoutput,activation=activation_function)))
    rms = RMSprop(lr=lrate)
    model.compile(loss='mean_squared_error', optimizer=rms)

    return model

###############################################################################
# EXPERIMENT 2: 
# - 1 time distributed layer (no activation) 
# - n LSTM layers
# - 2 time distributed layer (no activation)
# - 1 time distributed layer with activation
# - RMSProp optimiser with default values (gamma=0.9, default learning rate=0.001)
# Optional: batch normalisation between LSTM layers, L2 recurrent regularisation
###############################################################################
def create_lstm_network2(num_frequency_dimensions, num_hidden_dimensions, \
                         num_recurrent_units=1, activation_function='relu', fl2=0.0, bn=False, hps=False, lrate=0.001):
    print('Creating DRNN for experiment 2')
    model = Sequential()
	#This layer converts frequency space to hidden space
    print('Initialising dense time distributed layer with dimensions ', num_frequency_dimensions)
    model.add(TimeDistributed(Dense(num_hidden_dimensions),input_shape=num_frequency_dimensions))
    

    for cur_unit in range(num_recurrent_units):
        print('Creating LSTM with %s neurons' %(num_hidden_dimensions))
        model.add(LSTM(num_hidden_dimensions, return_sequences=True, recurrent_regularizer=l2(fl2)))
        if bn:
            model.add(BatchNormalization())
        
    model.add(TimeDistributed(Dense(num_hidden_dimensions)))
    model.add(TimeDistributed(Dense(num_hidden_dimensions)))
	#This layer converts hidden space back to frequency space
    model.add(TimeDistributed(Dense(num_frequency_dimensions[1],activation=activation_function)))
    rms = RMSprop(lr=lrate)
    model.compile(loss='mean_squared_error', optimizer=rms)

    return model

###############################################################################
# EXPERIMENT 3: 
# - n LSTM layers
# - 3 time distributed layer (no activation)
# - 1 time distributed layer with activation
# - RMSProp optimiser with default values (gamma=0.9, default learning rate=0.001)
# Optional: batch normalisation between LSTM layers, L2 recurrent regularisation
###############################################################################
def create_lstm_network3(num_frequency_dimensions, num_hidden_dimensions, \
                         num_recurrent_units=1, activation_function='relu', fl2=0.0, bn=False, hps=False, lrate=0.001):
    print('Creating DRNN for experiment 3')
    model = Sequential()
	#This layer converts frequency space to hidden space
    print('Initialising dense time distributed layer with dimensions ', num_frequency_dimensions[0:2])

    for cur_unit in range(num_recurrent_units):
        print('Creating LSTM with %s neurons' %(num_hidden_dimensions))
        model.add(LSTM(num_hidden_dimensions, return_sequences=True, input_shape=num_frequency_dimensions[0:2], recurrent_regularizer=l2(fl2)))
        if bn:
            model.add(BatchNormalization())
            
    model.add(TimeDistributed(Dense(num_hidden_dimensions,activation=activation_function)))    
    model.add(TimeDistributed(Dense(num_hidden_dimensions,activation=activation_function)))
    model.add(TimeDistributed(Dense(num_hidden_dimensions,activation=activation_function)))
	#This layer converts hidden space back to frequency space
    model.add(TimeDistributed(Dense(num_frequency_dimensions[1],activation=activation_function)))
    rms = RMSprop(lr=lrate)
    model.compile(loss='mean_squared_error', optimizer=rms)

    return model

###############################################################################
# EXPERIMENT 4: 
# - 1 Dropout layer with a probability of 0.2 dropout at input
# - 3 time distributed layer (no activation) 
# - n LSTM layers
# - 1 time distributed layer with activation
# - RMSProp optimiser with default values (gamma=0.9, default learning rate=0.001)
# Optional: batch normalisation between LSTM layers, L2 recurrent regularisation
###############################################################################
def create_lstm_network4(num_frequency_dimensions, num_hidden_dimensions, \
                         num_recurrent_units=1, activation_function='relu', fl2=0.0, bn=False, hps=False, lrate=0.001):
    print('Creating DRNN for experiment 4')
    model = Sequential()
	#This layer converts frequency space to hidden space
    model.add(TimeDistributed(Dropout(0.2),input_shape=num_frequency_dimensions))
    model.add(TimeDistributed(Dense(num_hidden_dimensions),input_shape=num_frequency_dimensions))
    model.add(TimeDistributed(Dense(num_hidden_dimensions)))
    model.add(TimeDistributed(Dense(num_hidden_dimensions)))

    for cur_unit in range(num_recurrent_units):
        print('Creating LSTM with %s neurons' %(num_hidden_dimensions))
        model.add(LSTM(num_hidden_dimensions, return_sequences=True, recurrent_regularizer=l2(fl2)))
        if bn:
            model.add(BatchNormalization())

	#This layer converts hidden space back to frequency space
    model.add(TimeDistributed(Dense(num_frequency_dimensions[1],activation=activation_function)))
    rms = RMSprop(lr=lrate)
    model.compile(loss='mean_squared_error', optimizer=rms)

    return model

###############################################################################
# EXPERIMENT 5: 
# - 3 time distributed layer with activation 
# - n LSTM layers
# - 1 time distributed layer with activation
# - RMSProp optimiser with default values (gamma=0.9, default learning rate=0.001)
# Optional: batch normalisation between LSTM layers, L2 recurrent regularisation
###############################################################################
def create_lstm_network5(num_frequency_dimensions, num_hidden_dimensions, \
                         num_recurrent_units=1, activation_function='relu', fl2=0.0, bn=False, hps=False, lrate=0.001):
    print('Creating DRNN for experiment 5')
    model = Sequential()
	#This layer converts frequency space to hidden space

    model.add(TimeDistributed(Dense(num_hidden_dimensions,activation=activation_function),input_shape=num_frequency_dimensions))
    model.add(TimeDistributed(Dense(num_hidden_dimensions,activation=activation_function)))
    model.add(TimeDistributed(Dense(num_hidden_dimensions,activation=activation_function)))

    for cur_unit in range(num_recurrent_units):
        print('Creating LSTM with %s neurons' %(num_hidden_dimensions))
        model.add(LSTM(num_hidden_dimensions, return_sequences=True, recurrent_regularizer=l2(fl2)))
        if bn:
            model.add(BatchNormalization())


	#This layer converts hidden space back to frequency space
    model.add(TimeDistributed(Dense(num_frequency_dimensions[1],activation=activation_function)))
    rms = RMSprop(lr=lrate)
    model.compile(loss='mean_squared_error', optimizer=rms)

    return model

###############################################################################
# EXPERIMENT 6: 
# - 3 time distributed layer (no activation) 
# - n LSTM layers
# - 1 time distributed layer with activation
# - 1 Dropout layer with a probability of 0.2 dropout at output
# - RMSProp optimiser with default values (gamma=0.9, default learning rate=0.001)
# Optional: batch normalisation between LSTM layers, L2 recurrent regularisation
###############################################################################
def create_lstm_network6(num_frequency_dimensions, num_hidden_dimensions, \
                         num_recurrent_units=1, activation_function='relu', fl2=0.0, bn=False, hps=False, lrate=0.001):
    print('Creating DRNN for experiment 6')
    model = Sequential()
	#This layer converts frequency space to hidden space
    model.add(TimeDistributed(Dense(num_hidden_dimensions,activation=activation_function),input_shape=num_frequency_dimensions))
    model.add(TimeDistributed(Dense(num_hidden_dimensions,activation=activation_function)))
    model.add(TimeDistributed(Dense(num_hidden_dimensions,activation=activation_function)))
    

    for cur_unit in range(num_recurrent_units):
        print('Creating LSTM with %s neurons' %(num_hidden_dimensions))
        model.add(LSTM(num_hidden_dimensions, return_sequences=True, recurrent_regularizer=l2(fl2)))
        if bn:
            model.add(BatchNormalization())


	#This layer converts hidden space back to frequency space
    model.add(TimeDistributed(Dense(num_frequency_dimensions[1],activation=activation_function)))
    model.add(TimeDistributed(Dropout(0.2)))
    rms = RMSprop(lr=lrate)
    model.compile(loss='mean_squared_error', optimizer=rms)

    return model

###############################################################################
# EXPERIMENT 7: 
# - 3 time distributed layer with activation 
# - n LSTM layers with dropout in the recurrent cells of 0.2 probability
# - 1 time distributed layer with activation
# - RMSProp optimiser with default values (gamma=0.9, default learning rate=0.001)
# Optional: batch normalisation between LSTM layers, L2 recurrent regularisation
###############################################################################
def create_lstm_network7(num_frequency_dimensions, num_hidden_dimensions, \
                         num_recurrent_units=1, activation_function='relu', fl2=0.0, bn=False, hps=False, lrate=0.001):
    print('Creating DRNN for experiment 7')
    model = Sequential()
	#This layer converts frequency space to hidden space

    model.add(TimeDistributed(Dense(num_hidden_dimensions,activation=activation_function),input_shape=num_frequency_dimensions))
    model.add(TimeDistributed(Dense(num_hidden_dimensions,activation=activation_function)))
    model.add(TimeDistributed(Dense(num_hidden_dimensions,activation=activation_function)))
    
    for cur_unit in range(num_recurrent_units):
        print('Creating LSTM with %s neurons' %(num_hidden_dimensions))
        model.add(LSTM(num_hidden_dimensions, return_sequences=True, dropout=0.2, recurrent_regularizer=l2(fl2)))
        if bn:
            model.add(BatchNormalization())

	#This layer converts hidden space back to frequency space
    model.add(TimeDistributed(Dense(num_frequency_dimensions[1],activation=activation_function)))
    rms = RMSprop(lr=lrate)
    model.compile(loss='mean_squared_error', optimizer=rms)

    return model

###############################################################################
# EXPERIMENT 8: 
# - 3 time distributed layer (no activation) 
# - n LSTM layers with batch normalisation between LSTM and before the activation function
# - 1 time distributed layer with activation
# - RMSProp optimiser with default values (gamma=0.9, default learning rate=0.001)
# Optional: batch normalisation between LSTM layers, L2 recurrent regularisation
###############################################################################
def create_lstm_network8(num_frequency_dimensions, num_hidden_dimensions, \
                         num_recurrent_units=1, activation_function='relu', fl2=0.0, bn=False, hps=False, lrate=0.001):
    print('Creating DRNN for experiment 8')
    model = Sequential()
	#This layer converts frequency space to hidden space

    model.add(TimeDistributed(Dense(num_hidden_dimensions),input_shape=num_frequency_dimensions))
    model.add(TimeDistributed(Dense(num_hidden_dimensions)))
    model.add(TimeDistributed(Dense(num_hidden_dimensions)))
    

    for cur_unit in range(num_recurrent_units):
        print('Creating LSTM with %s neurons' %(num_hidden_dimensions))
        if bn:
            model.add(LSTM(num_hidden_dimensions, activation=None, return_sequences=True, recurrent_regularizer=l2(fl2)))       
            model.add(BatchNormalization())
            model.add(Activation('tanh'))
        else:
            model.add(LSTM(num_hidden_dimensions, return_sequences=True, recurrent_regularizer=l2(fl2)))

	#This layer converts hidden space back to frequency space
    model.add(TimeDistributed(Dense(num_frequency_dimensions[1],activation=activation_function)))
    rms = RMSprop(lr=lrate)
    model.compile(loss='mean_squared_error', optimizer=rms)

    return model

###############################################################################
# EXPERIMENT 9: 
# - 3 time distributed layer (no activation) 
# - n LSTM layers
# - 1 time distributed layer with activation
# - Adam optimiser with default values learning rate=0.001
# Optional: batch normalisation between LSTM layers, L2 recurrent regularisation
###############################################################################
def create_lstm_network9(num_frequency_dimensions, num_hidden_dimensions, \
                         num_recurrent_units=1, activation_function='relu', fl2=0.0, bn=False, hps=False, lrate=0.001):
    print('Creating DRNN for experiment 9')
    model = Sequential()
	#This layer converts frequency space to hidden space

    model.add(TimeDistributed(Dense(num_hidden_dimensions),input_shape=num_frequency_dimensions))
    model.add(TimeDistributed(Dense(num_hidden_dimensions)))
    model.add(TimeDistributed(Dense(num_hidden_dimensions)))
    

    for cur_unit in range(num_recurrent_units):
        print('Creating LSTM with %s neurons' %(num_hidden_dimensions))
        model.add(LSTM(num_hidden_dimensions, return_sequences=True, recurrent_regularizer=l2(fl2)))
        if bn:
            model.add(BatchNormalization())

	#This layer converts hidden space back to frequency space
    model.add(TimeDistributed(Dense(num_frequency_dimensions[1],activation=activation_function)))
    rms = Adam(lr=lrate)
    model.compile(loss='mean_squared_error', optimizer=rms)

    return model

###############################################################################
# EXPERIMENT 10:
# - 1 2D CNN with 1 filter, kernel size (1,No frequencies/16), strides (1,16) 
# - 1 time distributed layer with activation 
# - n LSTM layers
# - 1 time distributed layer with activation
# - RMSProp optimiser with default values (gamma=0.9, default learning rate=0.001)
# Optional: batch normalisation between LSTM layers, L2 recurrent regularisation
###############################################################################   
def create_lstm_network10(num_frequency_dimensions, num_hidden_dimensions, \
                         num_recurrent_units=1, activation_function='relu', fl2=0.0, bn=False, hps=False, lrate=0.001):
    print('Creating DRNN for experiment 10')
    print("Original size: ", num_frequency_dimensions)
    model = Sequential()
    model.add(InputLayer(input_shape=(num_frequency_dimensions)))
    model.add(Reshape(num_frequency_dimensions +(1,)))
    kernelSize = ((num_frequency_dimensions[1]-1)//16)
    expectedSize = ((num_frequency_dimensions[1]-kernelSize)//16)+1
    model.add(Conv2D(1, (1,kernelSize), strides=(1,16), data_format="channels_last", activation='relu', padding='valid', \
                                     input_shape=num_frequency_dimensions))
    model.add(Reshape((num_frequency_dimensions[0],expectedSize)))
    model.add(TimeDistributed(Dense(expectedSize,activation=activation_function)))


    for cur_unit in range(num_recurrent_units):
        print('Creating LSTM with %s neurons' %(expectedSize))
        model.add(LSTM(expectedSize, return_sequences=True, recurrent_regularizer=l2(fl2)))
        if bn:
            model.add(BatchNormalization())


	#This layer converts hidden space back to frequency space
    model.add(TimeDistributed(Dense(num_frequency_dimensions[1],activation=activation_function)))
    model.compile(loss='mean_squared_error', optimizer='RMSprop')

    return model

###############################################################################
# EXPERIMENT 11:
# - 1 2D CNN with 1 filter, kernel size (1,No frequencies/256), strides (1,1) 
# - 3 time distributed layer with activation 
# - n LSTM layers
# - 1 time distributed layer with activation
# - RMSProp optimiser with default values (gamma=0.9, default learning rate=0.001)
# Optional: batch normalisation between LSTM layers, L2 recurrent regularisation
###############################################################################   
def create_lstm_network11(num_frequency_dimensions, num_hidden_dimensions, \
                         num_recurrent_units=1, activation_function='relu', fl2=0.0, bn=False, hps=False, lrate=0.001):
    print('Creating DRNN for experiment 11')
    print("Original size: ", num_frequency_dimensions)
    model = Sequential()
    model.add(InputLayer(input_shape=(num_frequency_dimensions)))
    model.add(Reshape(num_frequency_dimensions +(1,)))
    kernelSize = ((num_frequency_dimensions[1]-1)//256)
    expectedSize = ((num_frequency_dimensions[1]-kernelSize)//stride)+1
    model.add(Conv2D(1, (1,kernelSize), strides=(1,1), data_format="channels_last", activation='relu', padding='valid', \
                                     input_shape=num_frequency_dimensions))
    model.add(Reshape((num_frequency_dimensions[0],expectedSize)))
    model.add(TimeDistributed(Dense(num_hidden_dimensions,activation=activation_function)))
    model.add(TimeDistributed(Dense(num_hidden_dimensions,activation=activation_function)))
    model.add(TimeDistributed(Dense(num_hidden_dimensions,activation=activation_function)))

    for cur_unit in range(num_recurrent_units):
        print('Creating LSTM with %s neurons' %(num_hidden_dimensions))
        model.add(LSTM(num_hidden_dimensions, return_sequences=True, recurrent_regularizer=l2(fl2)))
        if bn:
            model.add(BatchNormalization())


	#This layer converts hidden space back to frequency space
    model.add(TimeDistributed(Dense(num_frequency_dimensions[1],activation=activation_function)))
    model.compile(loss='mean_squared_error', optimizer='RMSprop')

    return model

###############################################################################
# EXPERIMENT 12:
# - 1 time distributed dense layer 
# - Batch normalisation 
# - Activation function
# - 1 time distributed dense layer 
# - Batch normalisation
# - Activation function
# - 1 time distributed dense layer 
# - Batch normalisation
# - Activation function
# - n LSTM layers
# - 1 time distributed layer with activation
# - RMSProp optimiser with default values (gamma=0.9, default learning rate=0.001)
# Optional: batch normalisation between LSTM layers, L2 recurrent regularisation
############################################################################### 
def create_lstm_network12(num_frequency_dimensions, num_hidden_dimensions, \
                         num_recurrent_units=3, activation_function='relu', fl2=0.0, bn=False, hps=False, lrate=0.001):
    print('Creating DRNN for experiment 12')
    model = Sequential()
	#This layer converts frequency space to hidden space
    model.add(TimeDistributed(Dense(num_hidden_dimensions),input_shape=num_frequency_dimensions))
    model.add(BatchNormalization())
    model.add(Activation(activation_function))
    model.add(TimeDistributed(Dense(num_hidden_dimensions)))
    model.add(BatchNormalization())
    model.add(Activation(activation_function))
    model.add(TimeDistributed(Dense(num_hidden_dimensions)))
    model.add(BatchNormalization())
    model.add(Activation(activation_function))
    

    for cur_unit in range(num_recurrent_units):
        print('Creating LSTM with %s neurons' %(num_hidden_dimensions))
        model.add(LSTM(num_hidden_dimensions, return_sequences=True, recurrent_regularizer=l2(fl2)))
        if bn:
            model.add(BatchNormalization())

	#This layer converts hidden space back to frequency space
    if hps:
        myoutput = int(num_frequency_dimensions[1]/2)
    else:
        myoutput = num_frequency_dimensions[1]
    print('Samples per step: ', myoutput)
    model.add(TimeDistributed(Dense(myoutput,activation=activation_function)))

    rms = RMSprop(lr=lrate)
    model.compile(loss='mean_squared_error', optimizer=rms)

    return model