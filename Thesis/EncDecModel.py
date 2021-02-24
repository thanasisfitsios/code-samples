# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 16:03:17 2019

@author: AthanasiosFitsios
"""

import numpy as np
import math
import keras
from keras.layers import Input, Conv2D, LSTM, Conv2DTranspose
from keras.layers import Concatenate, Lambda, Reshape
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K

#model architecture. I tried to make it easily expandable to a variable number of layers. 


##define list of CNN layers
def CONVs(filter_nums, strides, kernels):
    layers = []
    for filters, stride, kernel in zip(filter_nums, strides, kernels):
        layer = Conv2D(filters=filters, strides=stride, kernel_size=kernel, padding='same',
                                       activation='selu')
        layers.append(layer)
    return layers

#define list of  RNN layers
def RNNs(unit_nums):
    layers = []
    for units in unit_nums:
        layer = LSTM(units=units, return_sequences=True, activation='sigmoid')
        layers.append(layer)
    return layers

#deifne list of CNN decoder layers
def DeCONVs(filter_nums, strides, kernels):
    layers = []
    for filters, stride, kernel in zip(filter_nums, strides, kernels):
        layer = Conv2DTranspose(filters=filters, strides=stride, kernel_size=kernel, padding='same',
                                                  activation='selu')
        layers.append(layer)
    return layers

##### Model Architecture #####
# Model can be adjusted for variable num of layers
class EncDec:

    def __init__(self, w, m, units, strides, kernels):
        self.window = w
        self.features = m
        self.units = units
        self.dec_units = self.units[-2::-1] + [2]
        self.strides = strides
        self.kernels = kernels
        self.rnn_units = []
        #compute the units for the rnn layers 
        
        for i, (u, s) in enumerate(zip(self.units, self.strides)):
            if i==0: prev = self.features / s
            else: prev = prev / s
            flat_dim = u * math.ceil(prev)
            self.rnn_units.append(flat_dim)
           
    def make_model(self):
        inputs = Input(shape=(self.window, self.features, 2))
#        c_inputs = Reshape((self.window, self.features, 1))(inputs)     # In case your input has only one channel.
        
        self.enc_layers = CONVs(self.units, self.strides, self.kernels) # define encoder conv layers
        #implements pipeline of encoding layers
        def encoder(inputs):

            outs = []
            prev = inputs
            for i, layer in enumerate(self.enc_layers):
                outs.append(layer(prev))
                prev = outs[-1]
            return outs

        self.rnn_layers = RNNs(self.rnn_units)
        # RNN layers
        def rnn_outs(encodeds):
            # encoded input is None, time, features, filters
            rnn_outs = []
            for encoded, layer in zip(encodeds, self.rnn_layers):
                in_shape = K.int_shape(encoded)[1:] #without batch size
                rnn_input = Reshape((in_shape[0], -1)) (encoded) #We need flattened input for the lstm
                rnn_out = layer(rnn_input)
                rnn_out = Reshape((in_shape[0], in_shape[1], in_shape[2])) (rnn_out)
                rnn_outs.append(rnn_out)
            return rnn_outs

        self.dec_layers = DeCONVs(self.dec_units, self.strides[::-1], self.kernels[::-1])
        
        
        #pipeline of decoding layers
        
        def decoder(rnn_outs):
            x = None
            for layer, h in zip(self.dec_layers, rnn_outs[::-1]):
                if x is None:
                    x = layer(h)
                else:
                    x = Lambda(lambda k: k[:, :K.int_shape(h)[1], :K.int_shape(h)[2], :]) (x) # trim conv2DTranspose if necessary
                    x = layer(x)
            return x

        self.encoder = encoder
        self.rnn_outs = rnn_outs
        self.decoder = decoder
        
#        Here we have essentially built 3 main blocks, CNN encoder, RNN, and CNN decoder
        #so we just use the functional model to define flow between these blocks
        
        encodeds = self.encoder(inputs) #Input passes in the CNN encoder
        rnnouts = self.rnn_outs(encodeds) #CNN encoder output feeds through RNN
        reconstructed = self.decoder(rnnouts) #CNN decoder reconstructs the input
        #reconstructed = self.decoder(encodeds)  #replace above line with this, if you want to ignore RNN and use a purely convolutional model
        
        
        model = Model(inputs, reconstructed)
        model.summary()
        return model

