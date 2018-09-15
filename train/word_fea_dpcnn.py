import os

import gc

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import sys

sys.path.append("..")
import pandas as pd
import numpy as np
import keras.backend as K
from my_utils.data_preprocess import word_cnn_train_batch_generator
from keras.utils.np_utils import to_categorical
from model.deepzoo import deepcnn
from my_utils.metrics import score

print("Load train @ test")
maxlen = 1400

print('Loading data...')
def load_data_and_embedding():
    """Prepare data and embedding lookup table for training procedure."""

    # Load data
    df_data = pd.read_csv('../new_data/train_ids_and_labels_1400.txt',nrows=10000)
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

    embedding_matrix = np.load("../embedding/word-embedding-200d-mc5.npy")

    return X_train, y_train, X_val, y_val,embedding_matrix



x_train,y_train, x_val,  y_val, embedding_matrix= load_data_and_embedding()


# x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
# x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_val shape:', x_val.shape)
load_val = True
batch_size = 128
model_name = "word_deepcnn"
trainable_layer = ["embedding"]
train_batch_generator = word_cnn_train_batch_generator

print("Load Word")

model = deepcnn(maxlen, embedding_matrix)

# from keras.utils.vis_utils import plot_model
# plot_model(model, to_file= model_name+ '.png',show_shapes=True)

best_f1 = 0
for i in range(50):
    print('---------------------EPOCH------------------------')
    print(i)
    print ('best_f1 ' + ' >>> ' + str(best_f1))
    if best_f1 > 0.71 :
        K.set_value(model.optimizer.lr, 0.0001)
    if best_f1 > 0.72 :
        for l in trainable_layer:
            model.get_layer(l).trainable = True

    model.fit_generator(
        train_batch_generator(x_train, y_train, batch_size=batch_size),
        epochs=1,
        steps_per_epoch=int(x_train[0].shape[0]/ batch_size),
        validation_data=(x_val, y_val)
    )
    if i > 1 :
        pred = np.squeeze(model.predict(x_val))
        pre, rec, f1 = score(pred, y_val)
        # print (myAcc(pred,y_val))
        print("precision", pre)
        # print("recall", rec)
        print("f1_score", f1)

        if (f1 > 0.70 and float(f1) > best_f1):
            print('saving model (｡・`ω´･) ')
            best_f1 = f1

            # model.save(Config.cacheave(Config.cache_dir + '/rcnn/dp_embed_%s_epoch_%s_%s.h5'%(model_name, i, f1))