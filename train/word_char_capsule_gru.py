# coding=utf-8

import os

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import sys
sys.path.append("..")
import math
from my_utils.data_preprocess import  word_char_cnn_train_batch_generator
from model.deepzoo import get_word_char_capsule_gru
import pandas as pd
import numpy as np
import gc

import keras.backend as K
from keras.utils.np_utils import to_categorical
from my_utils.metrics import score

print("Load train @ test")
maxlen = 2000

print('Loading word data...')
def load_data_and_embedding():
    """Prepare data and embedding lookup table for training procedure."""

    # Load data
    df_data = pd.read_csv('../new_data/train_ids_and_labels_2000.txt',nrows=10000)
    y = df_data['class'] - 1  # class (0 ~ 18)
    X = df_data.drop(['class'], axis=1).values

    # Transform to binary class matrix
    y = to_categorical(y.values)

    # Randomly shuffle data
    np.random.seed(10)

    shuffle_indices = np.random.permutation(range(len(y)))
    X_shuffled = X[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split to train/test set
    # TODO: This is very crude, should use cross validation
    val_sample_index = -1 * int(0.2 * len(y))
    X_train, X_val = X_shuffled[:val_sample_index], X_shuffled[val_sample_index:]
    y_train, y_val = y_shuffled[:val_sample_index], y_shuffled[val_sample_index:]

    del df_data, X, y, X_shuffled, y_shuffled

    embedding_matrix = np.load("../embedding/word-embedding-min.npy")

    return X_train, y_train, X_val, y_val,embedding_matrix


print('Loading char data...')
def load_char_data_and_embedding():
    """Prepare data and embedding lookup table for training procedure."""

    # Load data
    df_data = pd.read_csv('../new_data/char_train_ids_and_labels_2000.txt',nrows=10000)
    y = df_data['class'] - 1  # class (0 ~ 18)
    X = df_data.drop(['class'], axis=1).values

    # Transform to binary class matrix
    y = to_categorical(y.values)

    # Randomly shuffle data
    np.random.seed(10)

    shuffle_indices = np.random.permutation(range(len(y)))
    X_shuffled = X[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split to train/test set
    # TODO: This is very crude, should use cross validation
    val_sample_index = -1 * int(0.2 * len(y))
    X_train, X_val = X_shuffled[:val_sample_index], X_shuffled[val_sample_index:]
    y_train, y_val = y_shuffled[:val_sample_index], y_shuffled[val_sample_index:]

    del df_data, X, y, X_shuffled, y_shuffled

    embedding_matrix = np.load("../embedding/char-embedding-min.npy")

    return X_train, y_train, X_val, y_val,embedding_matrix

word_x_train,word_y_train, word_x_val, word_y_val, word_embedding_matrix= load_data_and_embedding()

char_x_train,char_y_train, char_x_val, char_y_val, char_embedding_matrix= load_char_data_and_embedding()


x_train = [word_x_train,char_x_train]
y_train = [word_y_train,char_y_train]
x_val = [word_x_val,char_x_val]
y_val = word_y_val

del word_x_train,word_y_train, word_x_val, word_y_val,char_x_train,char_y_train, char_x_val, char_y_val
gc.collect()



load_val = True
batch_size = 64
model_name = "word_char_capsule_gru"
trainable_layer = ["word_embedding", "char_embedding"]
train_batch_generator = word_char_cnn_train_batch_generator



model = get_word_char_capsule_gru(maxlen, maxlen, word_embedding_matrix, char_embedding_matrix)

def cw_softmax(cw):
    z_exp = [math.exp(i) for i in cw]
    sum_z_exp = sum(z_exp)
    softmax = [round(i / sum_z_exp, 3) for i in z_exp]
    return softmax

cw = [3,1,1,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1,3]
cw = cw_softmax(cw)
print (cw)

best_f1 = 0
for i in range(26):
    print('---------------------EPOCH------------------------')
    print(i)
    print ('best_f1 ' + ' >>> ' + str(best_f1))
    if best_f1 > 0.68 :
        K.set_value(model.optimizer.lr, 0.0001)
    if best_f1 > 0.69 :
        for l in trainable_layer:
            model.get_layer(l).trainable = True

    model.fit_generator(
        train_batch_generator(x_train, y_train, batch_size=batch_size),
        epochs=1,
        steps_per_epoch=int(x_train[0].shape[0]/ batch_size),
        validation_data=(x_val, y_val),
        class_weight=cw,

    )
    if i <30 :
        pred = np.squeeze(model.predict(x_val))
        pre, rec, f1 = score(pred, y_val)
        # print (myAcc(pred,y_val))
        print("precision", pre)
        # print("recall", rec)
        print("f1_score", f1)

        if (f1 > 0.70 and float(f1) > best_f1):
            print('saving model (｡・`ω´･) ')
            best_f1 = f1

            # model.save(Config.cache_dir + '/rcnn/dp_embed_%s_epoch_%s_%s.h5'%(model_name, i, f1))
