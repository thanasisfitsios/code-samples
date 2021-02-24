# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 17:57:59 2019

@author: AthanasiosFitsios
"""
import pickle
import numpy as np
import argparse
import matplotlib.pyplot as plt
from numpy.random import seed
from tensorflow import set_random_seed
from EncDecModel import EncDec
from keras.callbacks import EarlyStopping


#necessary arguments for building and training the model. The --data argument currently is a location of a pickle file
parser = argparse.ArgumentParser(description='CNN/RNN Encoder-Decoder')
parser.add_argument('--data', type=str, required=True,
                    help='location of the data file')

parser.add_argument('--CNN_units', type=int, nargs="*", default=[256,128,64,32],
                    help='list of CNN units per conv layer')

parser.add_argument('--window', type=int, default=218,
                    help='time window size')
parser.add_argument('--CNN_kernels', type=int, nargs="*", default=[3,3,2,2],
                    help='the kernel size of the CNN layers')
parser.add_argument('--strides', type=int, nargs="*", default=[1,2,2,2],
                    help='the strides of the CNN layers')
parser.add_argument('--clip', type=float, default=10.,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=4, metavar='N',
                    help='batch size')
parser.add_argument('--seed', type=int, default=54321,
                    help='random seed')
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--loss', type=str, default='mse')

args = parser.parse_args()
seed(args.seed)
set_random_seed(args.seed)

#load data dictionary from .pkl files. Can be changed if you use input of different type.
with open(args.data, 'rb') as f:
    data_dict = pickle.load(f)
data = data_dict['Data']

#define model
model = EncDec(args.window, data.shape[1], args.CNN_units, args.strides, args.CNN_kernels).make_model()
es = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True) #early stopping callback
from keras.optimizers import Adam
optim = Adam(lr=args.lr, clipnorm=args.clip)
model.compile(optimizer=args.optim, loss=args.loss) #compile model

## data preprocessing
#To apply scaling, we reshape to one channel, scale, and then reshape again to two channels. If you have single-channel input, slightly tweak the code
data2d = np.reshape(data, (data.shape[0], data.shape[1]*2))
for i in range(data2d.shape[1]):
    data2d[:,i] = np.log1p(data2d[:,i]) #log-normalization
from sklearn.preprocessing import RobustScaler, MinMaxScaler
scaler = RobustScaler().fit(data2d)
dat = scaler.transform(data2d)  #scaling
dat = np.reshape(dat, data.shape)
dat = np.reshape(dat, (-1,args.window,data.shape[1],2))

 
#train model 
history = model.fit(x = dat, y = dat, epochs=args.epochs, batch_size=args.batch_size, validation_split=0.2, callbacks=[])

# below I get training plots and also save the output in a pickle file

plt.plot(history.history['loss'])
plt.savefig('train_loss.png')
plt.figure()
plt.plot(history.history['val_loss'])
plt.savefig('val_loss.png')
plt.figure()
pred = model.predict(x=dat) #reconstructed output

#plot output and input for an example KPI
plt.plot(pred.reshape((-1,data.shape[1],2))[:,2,1], '--')    
plt.plot(dat.reshape((-1,data.shape[1],2))[:,2,1])
plt.savefig('pred.png')


#save input and output tensors in a pickle file
test_in = data
test_out = pred.reshape(data.shape)
test_out = scaler.inverse_transform(test_out.reshape((data.shape[0],data.shape[1]*2)))
test_out = test_out.reshape(data.shape)
my_dict = {'out': test_out, 'in': test_in}
with open("out.pkl", "wb") as f:
    pickle.dump(my_dict, f)
    
#save trained model weights. 
model.save_weights('my_bn_weights.h5')
