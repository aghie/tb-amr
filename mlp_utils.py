import numpy as np
import keras
import sys
from numpy import argmax

from keras import backend as K
from keras.engine.topology import Layer
from keras.callbacks import Callback
import tensorflow as tf
import numpy as np
import random
from keras.callbacks import EarlyStopping





class ConceptDataGenerator(object):



    def __init__(self, batch_size, X, y_c, n_c, training, indexes_to_randomize,
                 not_to_randomize, randomized_prob, randomized_value):
        
        self.batch_size = batch_size
        
        self.X = X
        self.y_c = y_c
        self.n_c = n_c
        
        self.training = training
        self.indexes_to_randomize = indexes_to_randomize
        self.randomize_prob = randomized_prob
        self.not_to_randomize = not_to_randomize
        self.randomized_value = randomized_value
        
        
        
    def generate_inputs(self):
        i=0
        while 1: #and (stop_condition is None or not stop_condition):
            
            X =    np.array(self.X[i*self.batch_size:(i+1)*self.batch_size])
                      
            i+=1
    
            if i >= len(self.X)/ float(self.batch_size):
                i=0

            newX = []
            
            for x in range(X.shape[1]):
                input = []
                for sample in range(X.shape[0]):
                    input.append(np.array([X[sample][x]]))
                newX.append(np.array(input))
            
            yield newX
            



    def generate(self,  stop_condition=None):
        
        i=0
        while 1: #and (stop_condition is None or not stop_condition):
            
            X =    np.array(self.X[i*self.batch_size:(i+1)*self.batch_size])
            y_c =  keras.utils.to_categorical(self.y_c[i*self.batch_size:(i+1)*self.batch_size],  self.n_c)
                      
            i+=1

            if i >= len(self.y_c)/ float(self.batch_size):
                i=0

            newX = []
            
            indexes_to_unkown = []
            for nx in range(X.shape[1]):
                                            
                input = []
                for sample in range(X.shape[0]):
                    
                    if self.training and nx in self.indexes_to_randomize and random.random() < self.randomize_prob and X[sample][nx] not in self.not_to_randomize:
                        input.append(np.array([self.randomized_value]))
                    else:
                        input.append(np.array([X[sample][nx]]))
                newX.append(np.array(input))            

            yield (newX,[y_c])



class SGDLearningRateTracker(Callback):
     
    def on_epoch_end(self, epoch, logs={}):
         
        optimizer = self.model.optimizer
        _lr = tf.to_float(optimizer.lr, name='ToFloat')
        _decay = tf.to_float(optimizer.decay, name='ToFloat')
        _iter = tf.to_float(optimizer.iterations, name='ToFloat')
         
        lr = K.eval(_lr * (1. / (1. + _decay * _iter)))
        print(' - LR: {:.6f}\n'.format(lr))

