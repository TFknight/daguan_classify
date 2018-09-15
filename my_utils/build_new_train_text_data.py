import os
import re
import pickle
import pandas as pd
from tqdm import tqdm



train_df = pd.read_csv('../new_data/train_set.csv')
test_df = pd.read_csv('../new_data/test_set.csv')


# 处理低频词
def construct_dict(df, d_type='word'):
    word_dict = {}
    corput = df.word_seg if d_type == 'word' else df.article
    for line in tqdm(corput):
        for e in line.strip().split():
            word_dict[e] = word_dict.get(e, 0) + 1
    return word_dict
word_dict = construct_dict(train_df, d_type='word')

word_dict_test = construct_dict(test_df,d_type='word')
# char_dict = construct_dict(train_df, d_type='char')

word_stop_word = [e for e in word_dict if word_dict[e] <=2]
word_stop_word_test = [e for e in word_dict_test if word_dict_test[e] <=2]

word_stop_word = word_stop_word + word_stop_word_test
# char_stop_word = [e for e in char_dict if char_dict[e] <=2]
pickle.dump(set(word_stop_word), open('../new_data/word_stopword.pkl', 'wb'))
# pickle.dump(set(char_stop_word), open('./save/char_stopword.pkl', 'wb'))

# 过滤低频词
def filter_low_freq(df,dict_word):
    min_freq = 2
    word_seg_mf2 = []
    char_mf2 = []
    for w in tqdm(df.word_seg):
        word_seg_mf2.append(' '.join([e for e in w.split() if dict_word[e] > min_freq]))
    # for w in tqdm(df.article):
    #     char_mf2.append(' '.join([e for e in w.split() if char_dict[e] > min_freq]))
    df['word_mf2'] = word_seg_mf2
    # df['char_mf2'] = char_mf2

    return df

train_df = filter_low_freq(train_df,word_dict)

test_df = filter_low_freq(test_df,word_dict_test)



# 保存训练数据
train_df.to_csv('../new_data/New.train.csv')
test_df.to_csv('../new_data/New.test.csv')
