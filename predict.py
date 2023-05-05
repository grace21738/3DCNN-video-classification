# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 08:48:11 2022

@author: TUF
"""
from keras.models import model_from_json
import os
from tensorflow import keras
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import (Input,Activation, Conv3D, Dense, Dropout, Flatten,
                          MaxPooling3D, BatchNormalization, average)
from keras.layers import ELU, PReLU, LeakyReLU 
from keras.losses import categorical_crossentropy
from keras.models import Model
from keras.optimizers import Adam,SGD
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
import pandas as pd
import videoto3d
import tensorflow as tf
from numba import cuda
import math

def load_test_data( vid3d, df, root_dir,nclass, color, skip=True  ):
    test_data_X = []
    n = -1
    for name in df['video_name'].values:
        test_data_X.append(vid3d.video3d(os.path.join(root_dir, name), n,color=color, skip=skip))
        
    if color:
        return np.array(test_data_X).transpose((0, 2, 3, 4, 1))
    else:
        return np.array(test_data_X).transpose((0, 2, 3, 1))


def create_3dcnn(input_shape, nb_classes):
    # Define model
    model = tf.keras.models.Sequential([
       Conv3D(32, kernel_size=(3, 3, 3), activation = LeakyReLU(), input_shape=(
       input_shape), padding="same"),
       BatchNormalization(),
       Conv3D(32, padding="same",activation = LeakyReLU(), kernel_size=(3, 3, 3)),
       MaxPooling3D(pool_size=(3, 3, 3), padding="same"),
       BatchNormalization(),
       Dropout(0.25),
       Conv3D(64, padding="same",activation = LeakyReLU(), kernel_size=(3, 3, 3)),
       BatchNormalization(),
       Conv3D(64, padding="same",activation = LeakyReLU(), kernel_size=(3, 3, 3)),
       BatchNormalization(),
       MaxPooling3D(pool_size=(3, 3, 3), padding="same"),
       Dropout(0.25),
       
       Conv3D(64, padding="same",activation = LeakyReLU(), kernel_size=(3, 3, 3)),
       BatchNormalization(),
       Conv3D(64, padding="same",activation = LeakyReLU(), kernel_size=(3, 3, 3)),
       BatchNormalization(),
       MaxPooling3D(pool_size=(3, 3, 3), padding="same"),
       Dropout(0.5),
       
       Flatten(),
       Dense(512, activation='relu'),
       BatchNormalization(),
       Dropout(0.5),
       Dense(nb_classes, activation='softmax')
    ])
    return model

root_dir = 'final\\'
BATCH_SIZE = 32
IMG_SIZE = 50
EPOCH = 130
NCLASS = 39
COLOR = True
DEPTH = 4
NMODLE = 7
ORIGIN = True
CHANNEL = 3

def load_new_model( nb_classes ):
    
    # load json and create model
    '''
    json_file = open(json_name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.summary()
    '''
    models = []
    all_models = []
    optimize = 'rmsprop'#SGD( rate_scheduler)
    for i in range(NMODLE):
        weight_name = 'nmodel_ucf101_3dcnnmodel_{}.h5'.format(i)
        weight_path = os.path.join(root_dir, weight_name)
        models.append(create_3dcnn((IMG_SIZE, IMG_SIZE, DEPTH, CHANNEL), nb_classes))
        models[-1].compile(loss='categorical_crossentropy',
                          optimizer=optimize, metrics=['accuracy'])
        models[-1].load_weights(weight_path)
    print(models) 
    model_inputs = [Input(shape=(IMG_SIZE,IMG_SIZE, DEPTH, CHANNEL)) for _ in range (NMODLE)]
    model_outputs = [models[i](model_inputs[i]) for i in range (NMODLE)]
    model_outputs = average(inputs=model_outputs)
    model = Model(inputs=model_inputs, outputs=model_outputs)
    model.compile(loss=categorical_crossentropy, optimizer=optimize, metrics=['accuracy'])
    print("Loaded model from disk")
   
    return model
    

test_df = pd.read_csv('test.csv')
root_test_dir = '.\\data\\test\\'
nclass = 39
def main():
    
    test_fname_npz = '50_color_test_dataset_{}_{}_{}.npz'.format(
        NCLASS, DEPTH, True)
    
    if os.path.exists(test_fname_npz):
        loadeddata = np.load(test_fname_npz)
        X = loadeddata["X"]
    else:
        img_rows, img_cols, frames = IMG_SIZE, IMG_SIZE, DEPTH
        channel = 3 if COLOR else 1
        vid3d = videoto3d.Videoto3D(img_rows, img_cols, frames)
        x= load_test_data(vid3d, test_df, root_test_dir,nclass,COLOR, skip=True)
        X = x.reshape((x.shape[0], img_rows, img_cols, frames, channel))
        X = X.astype('float32')
        np.savez(test_fname_npz, X=X)
        print('Saved test dataset to dataset.npz.')
        print('test data X_shape:{}'.format(X.shape))
    
    
    model = load_new_model( NCLASS )
    model.summary()
    prediction=model.predict( [X]*NMODLE )
    prediction_class = np.argmax(prediction,axis=1)
    print( "prediction shape:", prediction_class.shape )
    with open("nmodel_mean_prediction.csv", 'w') as f:
        f.write("name,label\n")
        for i in range(10000):
            f.write(f"{i + 1:05d}.mp4,{prediction_class[i]}\n")
    
    
if __name__ == '__main__':
    main()