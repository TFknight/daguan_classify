# _*_ coding: utf-8 _*_


import gc
import time
from time import time
from gensim.models.word2vec import Word2Vec
import pandas as pd

def load_char_samples():
    """Load training and testing data, get the characters of each sample in two dataset and return."""
    # train_lines = open(train_data_file, 'r').read().splitlines()[1:]
    # test_lines = open(test_data_file, 'r').read().splitlines()[1:]


    column = 'char_mf2'
    train = pd.read_csv('../new_data/New.train.csv')
    test = pd.read_csv('../new_data/New.test.csv')

    train_lines = train[column]
    test_lines = test[column]

    train_char_samples = [line for line in train_lines]
    test_char_samples = [line for line in test_lines]
    char_samples = train_char_samples + test_char_samples

    char_samples = [char_sample.split() for char_sample in char_samples]

    del train_lines, test_lines, train_char_samples, test_char_samples
    gc.collect()

    return char_samples


def batch_iter(data, batch_size=5000):
    """Generate batch iterator."""
    data_size = len(data)
    num_batches = ((data_size - 1) // batch_size) + 1
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield data[start_index:end_index]


def main():
    # Load data
    # =========================================================================

    print("[INFO] Loading data...")

    sentences = load_char_samples()
    print("[INFO] The total number of samples is: %d" % len(sentences))

    # Calculate the size of vocabulary
    # =========================================================================

    chars = []
    for sentence in sentences:
        chars.extend(sentence)
    print("[INFO] The total number of characters is: %d" % len(set(chars)))

    del chars
    gc.collect()

    # Train and save word2vec model
    # =========================================================================

    print("[INFO] Initialize word2vec model...")
    model = Word2Vec(size=200, min_count=2, sg=0, iter=30, workers=16, seed=42)
    model.build_vocab(sentences)
    print("[INFO] ", end='')
    print(model)

    print("[INFO] Start training...")
    t0 = time()
    batches = batch_iter(sentences, batch_size=1000)
    for batch in batches:
        model.train(batch, total_examples=len(batch), epochs=model.epochs)
    print("[INFO] Done in %.3f seconds!" % (time() - t0))
    print("[INFO] Training finished! ( ^ _ ^ ) V")

    print("[INFO] Save to file...")
    # model.wv.save("./new_data/datagrand-word-200d.bin")
    model.wv.save_word2vec_format("../new_data/datagrand-char-select_min.txt", binary=False)
    print("[INFO] Finished!")


if __name__ == '__main__':
    main()
