# encoding=utf-8

'''
This model concat lsi + rnn
'''
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import sys
sys.path.append("..")
from my_utils.data_preprocess import  word_char_cnn_train_batch_generator
from model.deepzoo import get_rcnn_word_feature
import pandas as pd
import gc
from keras.utils.np_utils import to_categorical
from my_utils.metrics import score
import pickle
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.layers import *


def myAcc(y_true, y_pred):
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_true, axis=1)

    return np.mean(y_true == y_pred)


maxlen = 1400

print('Loading word data...')
def load_data_and_embedding():
    """Prepare data and embedding lookup table for training procedure."""

    # Load data
    df_data = pd.read_csv('../new_data/train_ids_and_labels_2000.txt',nrows=1000)
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
    x_train_sparseshuffled = x_train_sparse[shuffle_indices]
    y_train_sparse_shuffled = y_train_sparse[shuffle_indices]
    # Split to train/test set
    # TODO: This is very crude, should use cross validation
    val_sample_index = -1 * int(0.2 * len(y))
    x_train_sparse, x_val_sparse = x_train_sparseshuffled[:val_sample_index], x_train_sparseshuffled[val_sample_index:]
    y_train_sparse, y_val_sparse = y_train_sparse_shuffled[:val_sample_index], y_train_sparse_shuffled[val_sample_index:]

    del df_data, X, y, X_shuffled, y_shuffled,x_train_sparseshuffled,y_train_sparse_shuffled

    embedding_matrix = np.load("../embedding/word-embedding-min.npy")

    return X_train, y_train, X_val, y_val,embedding_matrix,x_train_sparse, y_train_sparse,x_val_sparse,y_val_sparse

word_x_train,word_y_train, word_x_val, word_y_val, word_embedding_matrix,x_train_sparse, y_train_sparse,x_val_sparse,y_val_sparse= load_data_and_embedding()


print('x_train shape:', word_x_train.shape)
print('x_train_sparse shape:', x_train_sparse.shape)

x_train = [word_x_train,x_train_sparse]
y_train = [word_y_train,y_train_sparse]
x_val = [word_x_val,x_val_sparse]
y_val = word_y_val

del word_x_train,word_y_train, word_x_val, word_y_val,x_train_sparse, y_train_sparse,y_val_sparse
gc.collect()


load_val = True
batch_size = 128
model_name = "word_sparse_rcnn"
trainable_layer = ["word_embedding"]
train_batch_generator = word_char_cnn_train_batch_generator



model = get_rcnn_word_feature(maxlen, x_val_sparse.shape[1], word_embedding_matrix)

best_f1 = 0
for i in range(50):
    print('---------------------EPOCH------------------------')
    print(i)
    print ('best_f1 ' + ' >>> ' + str(best_f1))
    if i > 7 :
        K.set_value(model.optimizer.lr, 0.0001)
    if i > 8 :
        for l in trainable_layer:
            model.get_layer(l).trainable = True

    model.fit_generator(
        train_batch_generator(x_train, y_train, batch_size=batch_size),
        epochs=1,
        steps_per_epoch=int(x_train[0].shape[0]/ batch_size),
        validation_data=(x_val, y_val)
    )
    if i < 7 :
        pred = np.squeeze(model.predict(x_val))
        pre, rec, f1 = score(pred, y_val)
        # print (myAcc(pred,y_val))
        print("precision", pre)
        # print("recall", rec)
        print("f1_score", f1)

        if (f1 > 0.73 and float(f1) > best_f1):
            print('saving model (｡・`ω´･) ')
            best_f1 = f1

            # model.save(Config.cache_dir + '/rcnn/dp_embed_%s_epoch_%s_%s.h5'%(model_name, i, f1))
