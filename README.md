# 达观比赛代码总结

</br> 大二暑假，和实验室的几个小伙伴一起参加2018“达观杯”文本智能处理挑战赛

# Task description
Create a model to predict the category of text by long text data
More details see [达观 2018](http://www.dcjingsai.com/common/cmpt/%E2%80%9C%E8%BE%BE%E8%A7%82%E6%9D%AF%E2%80%9D%E6%96%87%E6%9C%AC%E6%99%BA%E8%83%BD%E5%A4%84%E7%90%86%E6%8C%91%E6%88%98%E8%B5%9B_%E7%AB%9E%E8%B5%9B%E4%BF%A1%E6%81%AF.html?slxydc=2881bc)

## 1.Requirements
* python
* keras
* gensim
* scikit-learn
* pickle
* numpy
* pandas
* tensorflow-gpu

## 2.Data preprocessing
* Word/char vector training uses the word2vec package. Training word vectors and char vectors are used for all word text and char text in the training set and test set, respectively, and the vector dimension is set to 200 dimensions.
* The word text has a truncation length of 2000 , the article text truncation length is also 2000.
*  More details see **./my_utils**

* you can run it: 
> sh clean_data.sh


## 3.Model
* We mainly used 12 models in this competition.Due to the time limit of this competition, we have not done enough for the model combination of words and char.In test dataset, we only adopt a simple but efficient voting mechanism for ensembling.

 >The best model is   **word_fea_c_gru.py**

|  | model |Description  |
| ------------ | ------------ | ------------ |
|1|word_char_cnn| ./train/word_char_cnn.py |
|2|word_char_rcnn| ./train/word_char_rcnn.py |
|3|word_char_capsule_gru| ./train/word_char_capsule_gru.py |
|4|word_char_c_gru| ./train/word_char_c_gru.py |
|5|word_dpcnn_gru| ./train/word_dpcnn_gru.py |
|6|word_conv_lstm| ./train/word_conv_lstm.py |
|7|word_rcnn_Triples| ./train/word_rcnn_Triples.py |
|8|word_rcnn| ./train/word_rcnn.py |
|9|word_rnn_att| ./train/word_rnn_att.py |
|10|word_fea_c_gru| ./train/word_fea_c_gru.py |
|11|word_fea_cnn| ./train/word_fea_cnn.py |
|12|word_fea_dpcnn| ./train/word_fea_dpcnn.py |

More models' details see: [xuxuanbo's keras model ](https://github.com/xuxuanbo/keras_learning)

## 4.Feature engineering
* In this competition, we tried to use the method of lexical clustering to extract the features of the article. Its principle is that related words constitute a potential topic.We tried several dimensionality reduction methods，such as lsi,lda,pca,nmf..We put the extracted text features into the **nn** to compensate for its shortcomings. 
It seems useful  _(:△」∠)_
* More details see: [Terence's feature engineering](https://github.com/TerenceLiu2/MLpack)

# Acknowledgment
</br> Thanks for all the efforts of my teammates 
</br> If you like this blog, welcome to click on the **star** and **fork** , thank you!

