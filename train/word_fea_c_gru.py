# coding=utf-8

import os

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import sys
sys.path.append("..")
import math
import pandas as pd
import numpy as np
import pickle

from my_utils.data_preprocess import word_char_cnn_train_batch_generator
from model.deepzoo import get_word_char_capsule_gru,get_cgru_word_fea
import gc
import keras.backend as K
from keras.utils import np_utils
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

    file = open('../new_data/data_tfidf_select_lsa.pkl', 'rb')
    x_train_sparse, y_train_sparse, x_test_sparse = pickle.load(file)
    y_train_sparse = list(y_train_sparse)
    y_train_sparse = np.array(y_train_sparse)
    y_train_sparse = np_utils.to_categorical(y_train_sparse, 19)

    # Randomly shuffle data
    np.random.seed(10)

    # ------------------------------------ load content ------------------------
    shuffle_indices = np.random.permutation(range(len(y)))
    X_shuffled = X[shuffle_indices]
    y_shuffled = y[shuffle_indices]
    # Split to train/test set
    # TODO: This is very crude, should use cross validation
    val_sample_index = -1 * int(0.2 * len(y))
    X_train, X_val = X_shuffled[:val_sample_index], X_shuffled[val_sample_index:]
    y_train, y_val = y_shuffled[:val_sample_index], y_shuffled[val_sample_index:]

    # ------------------------------------ load sparse vec ------------------------
    x_train_sparse_shuffled = x_train_sparse[shuffle_indices]
    y_train_sparse_shuffled = y_train_sparse[shuffle_indices]
    # Split to train/test set
    # TODO: This is very crude, should use cross validation
    val_sample_index = -1 * int(0.2 * len(y))
    x_train_sparse, x_val_sparse = x_train_sparse_shuffled[:val_sample_index], x_train_sparse_shuffled[val_sample_index:]
    y_train_sparse, y_val_sparse = y_train_sparse_shuffled[:val_sample_index], y_train_sparse_shuffled[
                                                                               val_sample_index:]

    # ------------------------------------ load sparse chi2 vec ------------------------
    file = open('../new_data/chi2_data_lsa.pkl', 'rb')
    x_train_sparse_chi, x_test_sparse_chi = pickle.load(file)
    # y_train_sparse_chi = list( y_train_sparse_chi)
    # y_train_sparse_chi = np.array( y_train_sparse_chi)
    # y_train_sparse_chi = np_utils.to_categorical( y_train_sparse_chi, 19)

    x_train_sparse_shuffled_chi = x_train_sparse_chi[shuffle_indices]
    # y_train_sparse_shuffled_chi = y_train_sparse_chi[shuffle_indices]
    # Split to train/test set
    # TODO: This is very crude, should use cross validation
    val_sample_index = -1 * int(0.2 * len(y))
    x_train_sparse_chi, x_val_sparse_chi = x_train_sparse_shuffled_chi[:val_sample_index], x_train_sparse_shuffled_chi[
                                                                               val_sample_index:]
    # y_train_sparse_chi, y_val_sparse_chi = y_train_sparse_shuffled_chi[:val_sample_index], y_train_sparse_shuffled_chi[
    #                                                                            val_sample_index:]


    del df_data, X, y, X_shuffled, y_shuffled, x_train_sparse_shuffled, y_train_sparse_shuffled, x_train_sparse_shuffled_chi

    embedding_matrix = np.load("../embedding/word-embedding-min.npy")

    return X_train, y_train, X_val, y_val, embedding_matrix, x_train_sparse, y_train_sparse, x_val_sparse, y_val_sparse,x_train_sparse_chi,  y_train_sparse,x_val_sparse_chi,y_val_sparse


word_x_train,word_y_train, word_x_val, word_y_val, word_embedding_matrix,\
x_train_sparse, y_train_sparse,x_val_sparse,y_val_sparse,\
x_train_sparse_chi, y_train_sparse_chi,x_val_sparse_chi,y_val_sparse_chi= load_data_and_embedding()


print('x_train shape:', word_x_train.shape)
print('x_train_sparse shape:', x_train_sparse.shape)

x_train = [word_x_train,x_train_sparse_chi]
y_train = [word_y_train,y_train_sparse_chi]
x_val = [word_x_val,x_val_sparse_chi]
y_val = word_y_val

del word_x_train,word_y_train, word_x_val, word_y_val,x_train_sparse, y_train_sparse,y_val_sparse
gc.collect()


load_val = True
batch_size = 64
model_name = "word_sparse_cnn_chi2"
trainable_layer = ["word_embedding"]
train_batch_generator = word_char_cnn_train_batch_generator


def cw_softmax(cw):
    z_exp = [math.exp(i) for i in cw]
    sum_z_exp = sum(z_exp)
    softmax = [round(i / sum_z_exp, 3) for i in z_exp]
    return softmax


model = get_cgru_word_fea(maxlen, x_val_sparse_chi.shape[1],word_embedding_matrix)

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