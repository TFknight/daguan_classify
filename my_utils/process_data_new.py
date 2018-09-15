# _*_ coding: utf-8 _*_


import pickle
import numpy as np
import pandas as pd

# Global variable
PAD_STR = '<PAD>'
SEQUENCE_LENGTH = 2000  # documents with the number of words less than 2000 is 95.3147%


def load_word_samples_and_labels(data, header=True, train=True):
    """Load words and labels of each sample (document)."""
    if header:
        start_index = 1
    else:
        start_index = 0

    column = 'word_mf2'

    lines = data[column]
    try:
        labels = data['class']
    except:
        pass

    word_samples = [line for line in lines]
    word_samples = [word_sample.split() for word_sample in word_samples]

    if train:
        labels = [int(label) for label in labels]
    else:
        labels = []

    return word_samples, labels


def preprocess(data, sequence_length=3000):
    """Process the words of each sample to a fixed length."""
    res = []
    for sample in data:
        if len(sample) > sequence_length:
            sample = sample[:sequence_length]
            res.append(sample)
        else:
            str_added = [PAD_STR] * (sequence_length - len(sample))
            sample += str_added
            res.append(sample)
    return res


def transform_to_ids(data, word_to_id_map):
    """Transform the words (characters) of a sample to its ids."""
    res = list()
    for words in data:
        ids = list()
        for word in words:
            if word in word_to_id_map:
                ids.append(word_to_id_map[word])
            else:
                ids.append(1)  # 1 is the id of '<UNK>'
        res.append(ids)
    return res


# Load the mapping from words to its corresponding ids
# ======================================================================================

print("Load the mapping from words to its corresponding ids...")
word2id_file = "../new_data/word200_word2id.pkl"
with open(word2id_file, 'rb') as fin:
    word_to_id_map = pickle.load(fin)

# Load data, truncate to fixed length and transform to ids
# ======================================================================================

print("Load data...")
train = pd.read_csv('../new_data/New.train.csv')
test = pd.read_csv('../new_data/New.test.csv')
words_train, labels_train = load_word_samples_and_labels(train, header=True, train=True)
words_test, _ = load_word_samples_and_labels(test, header=True, train=False)

print("Truncate to fixed length...")
words_train = preprocess(words_train, sequence_length=SEQUENCE_LENGTH)
words_test = preprocess(words_test, sequence_length=SEQUENCE_LENGTH)

print("Transform to ids...")
ids_train = transform_to_ids(words_train, word_to_id_map)
ids_test = transform_to_ids(words_test, word_to_id_map)

# Save to file
# ======================================================================================

ids_train = pd.DataFrame(ids_train, dtype=np.int32)
ids_train['class'] = pd.Series(labels_train, dtype=np.int32)
ids_test = pd.DataFrame(ids_test, dtype=np.int32)

print("Save to file...")
ids_train.to_csv("../new_data/train_ids_and_labels_2000.txt", index=False)
ids_test.to_csv("../new_data/test_ids_2000.txt", index=False)
print("Finished! ( ^ _ ^ ) V")
