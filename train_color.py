import argparse
import os
from tensorflow import keras
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10
from keras.layers import (Input,Activation, Conv3D, Dense, Dropout, Flatten,
                          MaxPooling3D, BatchNormalization, average)
from keras.layers import ELU, PReLU, LeakyReLU 
from keras.losses import categorical_crossentropy
from keras.models import Model
from keras.optimizers import Adam,SGD
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
import pandas as pd
import videoto3d
import tensorflow as tf
from keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
import random
from clr_callback import *

#tf.debugging.set_log_device_placement(True)

# Create some tensors
#a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
#b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
#c = tf.matmul(a, b)

#print(c)

import math



root_dir = '.\\data\\train\\'
root_test_dir = '.\\data\\test\\'



BATCH_SIZE = 32
IMG_SIZE = 50
EPOCH = 80
NCLASS = 39
COLOR = True
DEPTH = 4
NMODLE = 8
ORIGIN = True



def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.5
    if epoch < 50:
        return initial_lrate
    else:
        epochs_drop = 10.0
        lrate = initial_lrate * math.pow(drop,  
            math.floor((1+epoch)/epochs_drop))
    return lrate

def plot_history(history, result_dir,name):
    plt.plot(history.history['accuracy'], marker='.')
    plt.plot(history.history['val_accuracy'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig(os.path.join(result_dir,'{}_accuracy.png'.format(name)))
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, '{}_loss.png'.format(name)))
    plt.close()


def save_history(history, result_dir, name):
    loss = history.history['loss']
    acc = history.history['accuracy']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_accuracy']
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, '{}_result.txt'.format(name)), 'w') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i]))

#load data
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print(train_df.sample(10))
print(f"Total videos for training: {len(train_df)}")
print(f"Total videos for testing: {len(test_df)}")

def load_test_data(vid3d, root_dir,nclass, color, skip=True  ):
    test_data_X = []
    for name in test_df['video_name'].values:
        test_data_X.append(vid3d.video3d(os.path.join(root_dir, name), color=color, skip=skip))
        
    if color:
        return np.array(test_data_X).transpose((0, 2, 3, 4, 1))
    else:
        return np.array(test_data_X).transpose((0, 2, 3, 1))


def loaddata( vid3d, root_dir,nclass, color, skip=True):
    X = []
    labels = [] #training的label
    labellist = [] #所有的label
    labellist = keras.layers.IntegerLookup(
        num_oov_indices=0,vocabulary_dtype="int64",vocabulary=np.unique(train_df["label"])
    )
    print(labellist.get_vocabulary())
    for name in train_df['video_name'].values:
        n = random.randint(0,7)
        X.append(vid3d.video3d(os.path.join(root_dir, name), n, color=color, skip=skip))
    for label in train_df['label'].values:
        labels.append(label)
    
    if color:
        return np.array(X).transpose((0, 2, 3, 4, 1)), labels
    else:
        return np.array(X).transpose((0, 2, 3, 1)), labels


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


def main():

    img_rows, img_cols, frames = IMG_SIZE, IMG_SIZE, DEPTH
    channel = 3 if COLOR else 1
    if COLOR:
        fname_npz = 'mean_50_color_dataset_{}_{}_{}.npz'.format(
            NCLASS, DEPTH, True)
    else:
        fname_npz = '50_dataset_{}_{}_{}.npz'.format(
            NCLASS, DEPTH, True)
    vid3d = videoto3d.Videoto3D(img_rows, img_cols, frames)
    nb_classes = NCLASS
    if os.path.exists(fname_npz):
        loadeddata = np.load(fname_npz)
        X, Y = loadeddata["X"], loadeddata["Y"]
    else:
        x, y = loaddata(vid3d, root_dir,NCLASS,COLOR, skip=True)
        if ORIGIN:
            X = x.reshape((x.shape[0], img_rows, img_cols, frames, channel))
        else:
            X = x.reshape((x.shape[0], img_rows, img_cols, channel))
        Y = np_utils.to_categorical(y, nb_classes)
        X = X.astype('float32')
        np.savez(fname_npz, X=X, Y=Y)
        print('Saved dataset to dataset.npz.')
    print('X_shape:{}\nY_shape:{}'.format(X.shape, Y.shape))
    
    
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=43)

    models=[]
    optimize = 'rmsprop'#SGD( rate_scheduler)
    callback = LearningRateScheduler(step_decay)
    if not os.path.isdir('./train_nmodel'):
        os.makedirs('./train_nmodel')   
    
    
    clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                    step_size=20., mode='triangular2')
    for i in range(6,NMODLE):
        print('model{}:'.format(i))
        model = create_3dcnn(X.shape[1:], nb_classes)
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimize, metrics=['accuracy'])
        history = model.fit(X_train, Y_train, validation_data=(
            X_test, Y_test), batch_size=BATCH_SIZE, epochs=EPOCH, verbose=1, shuffle=True)
        plot_history(history , './train_nmodel', i)
        save_history(history, './train_nmodel', i)
        model.save_weights(os.path.join('./train_nmodel', 'nmodel_ucf101_3dcnnmodel_{}.h5'.format(i)))
    

if __name__ == '__main__':
    K.clear_session()
    main()