from keras.models import load_model
from keras.utils.np_utils import to_categorical
import numpy as np
import pandas as pd
import pickle

maxlen = 2000

# ------------------------------word------------------------------------------

print('Loading data...')
def load_data_and_embedding():
    """Prepare data and embedding lookup table for training procedure."""

    # Load data
    df_data = pd.read_csv('../new_data/test_ids_2000.txt')
    # y = df_data['class'] - 1  # class (0 ~ 18)
    X = df_data.values

    embedding_matrix = np.load("../embedding/word-embedding-200d-mc5.npy")

    return X,embedding_matrix

x_test,embedding_matrix = load_data_and_embedding()

print('x_test shape:', x_test.shape)

# ------------------------------feature------------------------------------------
file = open('../new_data/chi2_data_lsa.pkl', 'rb')
x_train_sparse, x_test_sparse = pickle.load(file)
# del x_train_sparse, y_train_sparse
#
x_pre = [x_test,x_test_sparse]
model_name = "word_sparse_cgru_chi2"
model = load_model('../save_model/' + "dp_embed_%s.h5" % (model_name),compile=False)

print ('load model done')
preds = model.predict(x_pre,batch_size=64)

preds = np.argmax(preds,axis=1)

fid0=open('baseline_word_sparse_cgru_chi2.csv','w')
i=0
fid0.write("id,class"+"\n")
for item in preds:
    fid0.write(str(i)+","+str(item+1)+"\n")
    i=i+1
fid0.close()
