# coding=utf-8

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def score(pred, label):
    p = np.argmax(pred, axis=1)
    l = np.argmax(label, axis=1)

    pre_score = precision_score(l, p, average=None)
    rec_score = recall_score(l, p, average='weighted')
    f_score = f1_score(l,p,average='weighted')
    return pre_score, rec_score, f_score
