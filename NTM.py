import numpy as np
np.random.seed(124)
import cPickle
import logging
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape,Lambda,LSTM,merge,normalization,\
                        BatchNormalization,ZeroPadding2D,Merge,Dropout,Input,Embedding,\
                        RepeatVector,Permute
from keras.layers.convolutional import Convolution2D,Convolution3D
#from keras.layers.recurrent_convolutional import LSTMConv2D

from keras import backend as K

from keras.models import Model
from keras.engine import Layer
from keras import initializations
from keras.utils.visualize_util import plot
from keras.optimizers import Adam,RMSprop,SGD
import h5py

from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
import cv2
import theano
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping
from seya.layers.ntm import NeuralTuringMachine as NTM
from keras.utils import np_utils
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
from theano import tensor, function

from keras.datasets import mnist
from keras.models import  Sequential
from keras.layers.core import TimeDistributedDense, Dropout, Activation, Flatten, Masking
from keras.layers import TimeDistributed,Dense,Embedding,Input,Merge,merge
from keras.layers.recurrent import LSTM
from keras.utils import np_utils, generic_utils
from keras.optimizers import Adam, SGD
from keras import backend as K

from seya.layers.ntm import NeuralTuringMachine
#### text data

from keras.utils import np_utils
import sys
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
hyperopt=reload(sys.modules['hyperopt'])
hyperas=reload(sys.modules['hyperas'])
hyperas.distributions=reload(sys.modules['hyperas.distributions'])
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
theano.config.exception_verbosity='high'


def data():
	
	import questions
	questions = reload(questions)
	from questions import get_textdata
	
	h5f = h5py.File('dequar/training_DEQUAR_vgg19_net.h5','r')
	image_data = h5f['data'][:]
	h5f.close()
	h5f = h5py.File('dequar/testing_DEQUAR_vgg19_net.h5','r')
	testimage_data = h5f['data'][:]
	h5f.close()
	
	train_x,train_y,test_x,test_y= get_textdata()
	return train_x,train_y,test_x,test_y, image_data, testimage_data

def model(train_x,train_y,test_x,test_y, image_data, testimage_data):
	reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2, min_lr=0,verbose=0)
	early_stopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=3, verbose=0, mode='auto')
	h_dim = {{choice([32])}}
	n_slots = {{choice([5])}}
	m_length = {{choice([16])}}
	input_dim = 512
	shift_range = {{choice([3])}}
	ntm = NeuralTuringMachine(h_dim, n_slots=n_slots, m_length=m_length,\
		 shift_range=shift_range,inner_rnn='lstm', return_sequences=False, input_dim=input_dim)
	Qmodel = Sequential()
	Qmodel.add(Embedding(output_dim=512, input_dim=880, input_length=30))
	
	Imodel = Input(shape=(49,512))

	
	model = merge([Imodel,Qmodel.layers[-1].output],mode='concat',concat_axis=1)
	model = ntm(model)
	model = Dense(503,activation='softmax')(model)
	model = Model([Imodel,Qmodel.layers[0].input],model)
	if conditional({{choice(['Adam', 'rmsporp'])}}) == 'Adam':
		sgd = Adam(lr={{uniform(.001,.1)}}, clipnorm={{uniform(4,10)}})
	elif conditional({{choice(['Adam', 'rmsporp'])}}) == 'rmsporp':
		sgd = Adam(lr={{uniform(.001,.1)}}, clipnorm={{uniform(4,10)}})
	model.compile(optimizer=sgd,\
	 loss={{choice(['categorical_crossentropy','hinge'])}},metrics=['acc'])
	history = model.fit([image_data,train_x],train_y,
						nb_epoch=1,
						batch_size={{choice([8])}},
						verbose=1,
						validation_data = ([testimage_data,test_x],test_y),
						callbacks=[reduce_lr,early_stopping])
	
	return {'loss':  history.history['val_loss'][-1], 'status': STATUS_OK,'model':model}



if __name__ == '__main__':
	best_run, best_model = optim.minimize(model=model,
										  data=data,
										  algo=tpe.suggest,
										  max_evals=1,
										  trials=Trials())


	print best_run
	best_model.save('my_model.h5')
	print '######################'