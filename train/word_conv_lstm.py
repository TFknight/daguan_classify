# encoding=utf-8

import os
import gc
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import sys
sys.path.append("..")
import pandas as pd
import numpy as np
import math
import keras.backend as K
from my_utils.data_preprocess import  word_cnn_train_batch_generator
from model.deepzoo import get_cgru,get_conv_add_lstm
from my_utils.metrics import score
from keras.utils.np_utils import to_categorical

print("Load train @ test")
maxlen = 2000



print('Loading data...')

def load_data_and_embedding():
    """Prepare data and embedding lookup table for training procedure."""

    # Load data
    df_data = pd.read_csv('../new_data/train_ids_and_labels_2000.txt',nrows=10000)
    y = df_data['class'] - 1
    X = df_data.drop(['class'], axis=1).values

    # Transform to binary class matrix
    y = to_categorical(y)

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

    embed_weight= np.load("../embedding/word-embedding-min.npy")

    return X_train, y_train, X_val, y_val,embed_weight


print('Pad sequences done')
x_train, y_train, x_val, y_val,embed_weight = load_data_and_embedding()

# x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
# x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_val shape:', x_val.shape)


load_val = True
batch_size = 64
model_name = "c_gru"
trainable_layer = ["embedding"]
train_batch_generator = word_cnn_train_batch_generator


print("Load Word")

def cw_softmax(cw):
    z_exp = [math.exp(i) for i in cw]
    sum_z_exp = sum(z_exp)
    softmax = [round(i / sum_z_exp, 3) for i in z_exp]
    return softmax

model = get_conv_add_lstm(maxlen, embed_weight)


cw = [3,1,1,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1,3]
cw = cw_softmax(cw)
print (cw)

best_f1 = 0
for i in range(20):
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
        steps_per_epoch=int(x_train.shape[0]/ batch_size),
        validation_data=(x_val, y_val),
        class_weight=cw,

    )
    if i <30 :
        pred = np.squeeze(model.predict(x_val))
        pre, rec, f1 = score(pred, y_val)

        print("precision", pre)
        # print("recall", rec)
        print("f1_score", f1)

        if (f1 > 0.70 and float(f1) > best_f1):
            print('saving model (｡・`ω´･) ')
            best_f1 = f1

            # model.save(Config.cache_dir + '/rcnn/dp_embed_%s_epoch_%s_%s.h5'%(model_name, i, f1))