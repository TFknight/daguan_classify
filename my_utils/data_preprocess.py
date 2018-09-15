# coding=utf-8
import pickle
import re
import time
import numpy as np
from keras.preprocessing import sequence


def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size/float(batch_size)))
    return [(i*batch_size, min(size, (i+1)* batch_size)) for i in range(0, nb_batch)]

def batch_generator(contents, labels, batch_size=128, shuffle=True, keep=False, preprocessfunc=None):

    assert preprocessfunc != None


    if(type(contents) != list):
        print ('word')
        sample_size = contents.shape[0]
        index_array = np.arange(sample_size)

        while 1:
            if shuffle:
                np.random.shuffle(index_array)
            batches = make_batches(sample_size, batch_size)
            for batch_index, (batch_start, batch_end) in enumerate(batches):
                batch_ids = index_array[batch_start: batch_end]
                batch_contents = contents[batch_ids]
                # batch_contents = preprocessfunc(batch_contents, keep=keep)
                batch_labels = labels[batch_ids]
                # print (batch_contents,batch_labels[0])

                yield (batch_contents, batch_labels)

    #  word char embedding ----------------------------------------
    else:
        print ('word char')
        sample_size = contents[0].shape[0]
        index_array = np.arange(sample_size)

        while 1:
            if shuffle:
                np.random.shuffle(index_array)
            batches = make_batches(sample_size, batch_size)
            for batch_index, (batch_start, batch_end) in enumerate(batches):
                batch_ids = index_array[batch_start: batch_end]
                word_batch_contents = contents[0][batch_ids]
                char_batch_contents = contents[1][batch_ids]

                word_batch_labels = labels[0][batch_ids]
                char_batch_labels = labels[1][batch_ids]

                batch_labels = labels[0][batch_ids]

                yield ([word_batch_contents,char_batch_contents], word_batch_labels)



def word_cnn_preprocess(contents):
    word_seq = contents
    return word_seq


def word_cnn_train_batch_generator(train_content, train_label, batch_size=128, keep=False):
    return batch_generator(contents=train_content, labels=train_label,batch_size=batch_size, keep=keep, preprocessfunc=word_cnn_preprocess)


def char_cnn_preprocess(contents):
    char_seq = contents
    return char_seq

def char_cnn_train_batch_generator(train_content, train_label, batch_size=128, keep=False):
    return batch_generator(contents=train_content, labels=train_label, batch_size=batch_size, keep=keep, preprocessfunc=char_cnn_preprocess)

def word_char_cnn_preprocess(contents):
    
    word_seq = contents[0]
    char_seq = contents[1]
    return [word_seq, char_seq]


def word_char_cnn_train_batch_generator(train_content, train_label, batch_size=128, keep=False):
    return batch_generator(contents=train_content, labels=train_label, batch_size=batch_size, keep=keep, preprocessfunc=word_char_cnn_preprocess)



# train_data = get_train_all_data()
# vali_data = get_validation_data()
# train_content = train_data["content"]
# vali_content = vali_data["content"]
#
# get_word_seq(train_content)
# get_word_seq(vali_content)
# get_char_seq(train_content)
# get_char_seq(vali_content)

