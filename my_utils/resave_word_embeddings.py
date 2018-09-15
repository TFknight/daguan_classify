# _*_ coding: utf-8 _*_


import pickle
import numpy as np

EMBEDDING_SIZE = 200
SPECIAL_SYMBOLS = ['<PAD>', '<UNK>']

# Load words and its corresponding embeddings
# ===========================================================================================

print("Load words and its corresponding embeddings...")
np.random.seed(42)
word_embedding_file = "../new_data/datagrand-word-select_min.txt"
with open(word_embedding_file, 'r', encoding='utf-8') as f:
    lines = f.read().splitlines()[1:]

    word_to_id_map = dict()
    id_to_word_map = dict()
    for i, symbol in enumerate(SPECIAL_SYMBOLS):
        id_to_word_map[i] = symbol
        word_to_id_map[symbol] = i

    num_total_symbols = len(lines) + len(SPECIAL_SYMBOLS)
    word_embeddings = np.zeros((num_total_symbols, EMBEDDING_SIZE), dtype=np.float32)
    word_embeddings[1] = np.random.randn(EMBEDDING_SIZE)  # the values of 'UNK' satisfy the normal distribution

    index = 2
    for line in lines:
        cols = line.split()
        id_to_word_map[index] = cols[0]
        word_to_id_map[cols[0]] = index
        word_embeddings[index] = np.array(cols[1:], dtype=np.float32)
        index += 1

# Save to file
# ===========================================================================================

print("Save to file...")
id2word_file = "../new_data/word200_id2word.pkl"
word2id_file = "../new_data/word200_word2id.pkl"
word_embeddings_file = "../embedding/word-embedding-min.npy"
with open(id2word_file, 'wb') as fout:
    pickle.dump(id_to_word_map, fout)
with open(word2id_file, 'wb') as fout:
    pickle.dump(word_to_id_map, fout)
np.save(word_embeddings_file, word_embeddings)
print("Finished! ( ^ _ ^ ) V")
