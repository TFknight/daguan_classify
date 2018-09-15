#  coding=utf-8

import keras
from keras.models import *
from keras.layers import *
from keras.optimizers import SGD
from keras.preprocessing import sequence
from keras.regularizers import l2
from keras import backend as K
from keras.engine.topology import Layer
from keras.backend.tensorflow_backend import set_session
# from recurrentshop import *
import tensorflow as tf
import keras.backend as K
from keras.optimizers import Adam
from model.keras_util import *
from model.Attention import *
from model.Capsule import *


def att_max_avg_pooling(x):
    x_att = AttentionWithContext()(x)
    x_avg = GlobalAvgPool1D()(x)
    x_max = GlobalMaxPool1D()(x)
    return concatenate([x_att, x_avg, x_max])


def mycrossentropy(y_true, y_pred, e=0.1):
    loss1 = K.categorical_crossentropy(y_true, y_pred)
    loss2 = K.categorical_crossentropy(K.ones_like(y_pred)/19, y_pred)
    return (1-e)*loss1 + e*loss2

def convs_block(data, convs = [3,4,5], f = 256, name = "conv_feat"):
    pools = []
    for c in convs:
        conv = Activation(activation="relu")(BatchNormalization()(Conv1D(filters=f, kernel_size=c, padding="valid")(data)))
        pool = GlobalMaxPool1D()(conv)
        pools.append(pool)
    return concatenate(pools, name=name)

# def inception_convs(data, convs=[3,4,5], f = )

def deepcnn(maxlen,embed_weight):
    filter_nr = 64
    filter_size = 3
    max_pool_size = 3
    max_pool_strides = 2
    dense_nr = 256
    spatial_dropout = 0.2
    dense_dropout = 0.5
    train_embed = False
    conv_kern_reg = regularizers.l2(0.00001)
    conv_bias_reg = regularizers.l2(0.00001)
    
    content = Input(shape=(maxlen,))
    embedding = Embedding(
        name="embedding",
        input_dim=embed_weight.shape[0],
        weights=[embed_weight],
        output_dim=embed_weight.shape[1],
        trainable=False)
    emb_comment = embedding(content)
    
    # emb_comment = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=train_embed)(comment)
    emb_comment = SpatialDropout1D(spatial_dropout)(emb_comment)

    block1 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                    kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(emb_comment)
    block1 = BatchNormalization()(block1)
    block1 = PReLU()(block1)
    block1 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                    kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block1)
    block1 = BatchNormalization()(block1)
    block1 = PReLU()(block1)

    # we pass embedded comment through conv1d with filter size 1 because it needs to have the same shape as block output
    # if you choose filter_nr = embed_size (300 in this case) you don't have to do this part and can add emb_comment directly to block1_output
    resize_emb = Conv1D(filter_nr, kernel_size=1, padding='same', activation='linear',
                        kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(emb_comment)
    resize_emb = PReLU()(resize_emb)

    block1_output = add([block1, resize_emb])
    block1_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block1_output)

    block2 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                    kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block1_output)
    block2 = BatchNormalization()(block2)
    block2 = PReLU()(block2)
    block2 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                    kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block2)
    block2 = BatchNormalization()(block2)
    block2 = PReLU()(block2)

    block2_output = add([block2, block1_output])
    block2_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block2_output)

    block3 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                    kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block2_output)
    block3 = BatchNormalization()(block3)
    block3 = PReLU()(block3)
    block3 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                    kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block3)
    block3 = BatchNormalization()(block3)
    block3 = PReLU()(block3)

    block3_output = add([block3, block2_output])
    block3_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block3_output)

    block4 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                    kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block3_output)
    block4 = BatchNormalization()(block4)
    block4 = PReLU()(block4)
    block4 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                    kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block4)
    block4 = BatchNormalization()(block4)
    block4 = PReLU()(block4)

    block4_output = add([block4, block3_output])
    block4_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block4_output)

    block5 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                    kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block4_output)
    block5 = BatchNormalization()(block5)
    block5 = PReLU()(block5)
    block5 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                    kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block5)
    block5 = BatchNormalization()(block5)
    block5 = PReLU()(block5)

    block5_output = add([block5, block4_output])
    block5_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block5_output)

    block6 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                    kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block5_output)
    block6 = BatchNormalization()(block6)
    block6 = PReLU()(block6)
    block6 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                    kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block6)
    block6 = BatchNormalization()(block6)
    block6 = PReLU()(block6)

    block6_output = add([block6, block5_output])
    block6_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block6_output)

    block7 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                    kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block6_output)
    block7 = BatchNormalization()(block7)
    block7 = PReLU()(block7)
    block7 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                    kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block7)
    block7 = BatchNormalization()(block7)
    block7 = PReLU()(block7)

    block7_output = add([block7, block6_output])
    output = GlobalMaxPooling1D()(block7_output)

    output = Dense(dense_nr, activation='linear')(output)
    output = BatchNormalization()(output)
    output = PReLU()(output)
    output = Dropout(dense_dropout)(output)
    output = Dense(19, activation='sigmoid')(output)

    model = Model(content, output)

    model.compile(loss=mycrossentropy,
                  optimizer=optimizers.Adam(),
                  metrics=['accuracy'])
    return model

def get_rcnn_high(word_max_len, embed_weight):
    dropout_p = 0.2
    hidden_dim_1 = 200
    hidden_dim_2 = 256


    document = Input(shape=(word_max_len,), name='word')
    Bi_context = Input(shape=(word_max_len,), name='word_Bi')

    doc_mask = Masking(mask_value=0.)(document)
    Bi_mask = Masking(mask_value=0.)(Bi_context)

    word_embedding = Embedding(
        name="embedding",
        input_dim=embed_weight.shape[0],
        weights=[embed_weight],
        output_dim=embed_weight.shape[1],
        trainable=False)

    doc_embed = word_embedding(doc_mask)
    doc_embed = BatchNormalization()(doc_embed)
    doc_embed = Bidirectional(CuDNNGRU(hidden_dim_2, return_sequences=True))(doc_embed)

    Bi_embed = word_embedding(Bi_mask)

    Bi_embed = BatchNormalization()(Bi_embed)
    forward = Bidirectional(CuDNNGRU(hidden_dim_2, return_sequences=True))(Bi_embed)

    # r_embed = Dropout(dropout_p)(r_embed)
    backward = Bidirectional(CuDNNGRU(hidden_dim_2, return_sequences=True, go_backwards=True))(Bi_embed)

    # reverse backward
    backward = Lambda(lambda x: K.reverse(x, axes=1))(backward)

    together = concatenate([forward, doc_embed, backward], axis=2)

    x = Conv1D(hidden_dim_2, kernel_size=1, activation='relu')(together)
    # pool_rnn = Lambda(lambda x: K.max(x, axis=1), output_shape=(hidden_dim_2,))(semantic)

    # pool_rnn = Dropout(dropout_p)(pool_rnn)

    maxpool = GlobalMaxPooling1D()(x)
    attn = AttentionWithContext()(x)
    average = GlobalAveragePooling1D()(x)
    all_views = concatenate([maxpool, attn, average], axis=1)
    x = Dropout(0.5)(all_views)

    output = Dense(19, input_dim=hidden_dim_2, activation='softmax')(x)
    model = Model(inputs=[document,Bi_context], output=output)
    model.compile(loss=mycrossentropy, optimizer='adam', metrics=['accuracy'])
    # plot_model(model, to_file="{}.png".format(self.name), show_shapes=True)
    return model

def get_textcnn2(word_len, embed_weight):
    filter_sizes = [3, 4, 5]
    num_filters = 300

    def train_model(embed_layer):
        embed_layer = BatchNormalization()(embed_layer)

        embedding_dim = embed_weight.shape[1]
        reshape = Reshape((word_len, embedding_dim, 1))(embed_layer)
        # ===========================================================================================

        # embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length)(inputs)
        # reshape = Reshape((sequence_length,embedding_dim,1))(embedding)

        conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid',
                        kernel_initializer='normal', activation='relu')(reshape)
        conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid',
                        kernel_initializer='normal', activation='relu')(reshape)
        conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid',
                        kernel_initializer='normal', activation='relu')(reshape)

        maxpool_0 = MaxPool2D(pool_size=(word_len - filter_sizes[0] + 1, 1), strides=(1, 1), padding='valid')(
            conv_0)
        maxpool_1 = MaxPool2D(pool_size=(word_len - filter_sizes[1] + 1, 1), strides=(1, 1), padding='valid')(
            conv_1)
        maxpool_2 = MaxPool2D(pool_size=(word_len - filter_sizes[2] + 1, 1), strides=(1, 1), padding='valid')(
            conv_2)

        concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
        conc = Flatten()(concatenated_tensor)

        return conc



    # one input
    content = Input(shape=(word_len,), dtype="int32")

    # embedding = Embedding(embed_weight.shape[0],
    #                       300,
    #                       input_length=word_len, )
    # trans_word = embedding(content)
    embedding = Embedding(
        name="embedding",
        input_dim=embed_weight.shape[0],
        weights=[embed_weight],
        output_dim=embed_weight.shape[1],
        trainable=False)
    trans_word = embedding(content)

    word_embed_layer = SpatialDropout1D(0.2)(trans_word)

    word_dence = train_model(word_embed_layer)

    # word_capsule = Flatten()(word_dence)


    dropfeat = Dropout(0.5)( word_dence)

    fc = Activation(activation="relu")(BatchNormalization()(Dense(64)(dropfeat)))
    output = Dense(11, activation='softmax')(fc)
    model = Model(inputs=[content], outputs=output)
    model.compile(loss=mycrossentropy, optimizer='adam', metrics=['accuracy'])
    return model

# 简单的cnn
def get_textcnn(seq_length, embed_weight):
    content = Input(shape=(seq_length,), dtype="int32")

    embedding = Embedding(embed_weight.shape[0],
                            300,
                            input_length=seq_length, )

    # embedding = Embedding(
    #     name="embedding",
    #     input_dim=embed_weight.shape[0],
    #     weights=[embed_weight],
    #     output_dim=embed_weight.shape[1],
    #     trainable=False)
    embedding_vec = embedding(content)
    trans_content = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(embedding_vec))))
    feat = convs_block(trans_content)
    dropfeat = Dropout(0.5)(feat)
    fc = Activation(activation="relu")(BatchNormalization()(Dense(256)(dropfeat)))
    output = Dense(11, activation="softmax")(fc)
    model = Model(inputs=content, outputs=output)
    model.compile(loss=mycrossentropy, optimizer="adam", metrics=['accuracy'])
    return model


def get_hcnn(sent_num, sent_length, embed_weight, mask_zero=False):
    sentence_input = Input(shape=(sent_length, ), dtype="int32")
    embedding = Embedding(
        input_dim = embed_weight.shape[0],
        weights = [embed_weight],
        output_dim=embed_weight.shape[1],
        mask_zero = mask_zero,
        trainable = False
    )
    sent_embed = embedding(sentence_input)
    word_bigru = Bidirectional(GRU(128, return_sequences=True))(sent_embed)
    word_attention = Attention(sent_length)(word_bigru)
    sent_encode = Model(sentence_input, word_attention)

    review_input = Input(shape=(sent_num, sent_length), dtype="int32")
    review_encode = TimeDistributed(sent_encode)(review_input)
    feat = convs_block(review_encode)
    dropfeat = Dropout(0.2)(feat)
    fc = Activation(activation="relu")(BatchNormalization()(Dense(256)(dropfeat)))
    output = Dense(2, activation="softmax")(fc)
    model = Model(review_input, output)
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    return model


def get_han2(sent_num, sent_length, embed_weight, mask_zero=False):
    input = Input(shape=(sent_num, sent_length,), dtype="int32")
    embedding = Embedding(
        name= "embeeding",
        input_dim = embed_weight.shape[0],
        weights = [embed_weight],
        output_dim = embed_weight.shape[1],
        mask_zero=mask_zero,
        trainable=False
    )
    sent_embed = embedding(input)
    # print(np.shape(sent_embed))
    sent_embed = Reshape((1, sent_length, embed_weight.shape[1]))(sent_embed)
    print(np.shape(sent_embed))
    word_bigru = Bidirectional(GRU(128, return_sequences=True))(sent_embed)
    word_bigru = Reshape((sent_length, 256))(word_bigru)
    # print(np.shape(word_bigru))
    word_attention = Attention(sent_length)(word_bigru)
    sent_encode = Reshape((-1, sent_num))(word_attention)
    # sent_encode = Model(sentence_input, word_attention)
    #
    # doc_input = Input(shape=(sent_num, sent_length), dtype="int32")
    # doc_encode = TimeDistributed(sent_encode)(doc_input)
    sent_bigru = Bidirectional(GRU(128, return_sequences=True))(sent_encode)
    doc_attention = Attention(sent_num)(sent_bigru)
    fc = Activation(activation="relu")(BatchNormalization()(Dense(256)(doc_attention)))
    output = Dense(4, activation='softmax')(fc)
    model = Model(input, output)
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    return model



def get_han(sent_num, sent_length, embed_weight, mask_zero=False):
    sentence_input = Input(shape=(sent_length,), dtype="int32",name='word_input')
    embedding = Embedding(
        name= "embeding",
        input_dim = embed_weight.shape[0],
        weights = [embed_weight],
        output_dim = embed_weight.shape[1],
        mask_zero=mask_zero,
        trainable=False
    )
    sent_embed = embedding(sentence_input)
    word_bigru = Bidirectional(GRU(128, return_sequences=True))(sent_embed)
    word_attention = Attention(sent_length)(word_bigru)
    sent_encode = Model(sentence_input, word_attention,name='sent_encode')

    doc_input = Input(shape=(sent_num, sent_length), dtype="int32",name="sent_input")
    doc_encode = TimeDistributed(sent_encode)(doc_input)
    sent_bigru = Bidirectional(GRU(128, return_sequences=True))(doc_encode)
    sent_attention = Attention(sent_num)(sent_bigru)
    fc = Activation(activation="relu")(BatchNormalization()(Dense(256)(sent_attention)))
    output = Dense(4, activation='softmax')(fc)
    model = Model(doc_input, output, name='doc_encode')
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    return model

def get_word_char_cnn(word_len, char_len, word_embed_weight, char_embed_weight):
    word_input = Input(shape=(word_len,), dtype="int32")
    char_input = Input(shape=(char_len,), dtype="int32")
    word_embedding = Embedding(
        name="word_embedding",
        input_dim=word_embed_weight.shape[0],
        weights = [word_embed_weight],
        output_dim=word_embed_weight.shape[1],
        trainable=False
    )
    char_embedding = Embedding(
        name="char_embedding",
        input_dim =char_embed_weight.shape[0],
        weights =[char_embed_weight],
        output_dim=char_embed_weight.shape[1],
        trainable=False
    )
    trans_word = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(word_embedding(word_input)))))
    trans_char = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(char_embedding(char_input)))))
    word_feat = convs_block(trans_word, convs=[1,2,3,4,5], f=256, name="word_conv")
    char_feat = convs_block(trans_char, convs=[1,2,3,4,5], f=256, name="char_conv")
    feat = concatenate([word_feat, char_feat])
    dropfeat = Dropout(0.4)(feat) # 0.4
    fc = Activation(activation="relu")(BatchNormalization()(Dense(128)(dropfeat))) # 256
    output = Dense(4, activation="softmax")(fc)
    model = Model(inputs=[word_input, char_input], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    return model

def get_word_char_hcnn(sent_num, sent_word_length, sent_char_length, word_embed_weight, char_embed_weight, mask_zero=False):
    sentence_word_input = Input(shape=(sent_word_length,), dtype="int32")
    word_embedding = Embedding(
        name = "word_embedding",
        input_dim=word_embed_weight.shape[0],
        weights=[word_embed_weight],
        output_dim= word_embed_weight.shape[1],
        mask_zero=mask_zero,
        trainable=False
    )
    sent_word_embed = word_embedding(sentence_word_input)
    word_bigru = Bidirectional(GRU(128, return_sequences=True))(sent_word_embed)
    word_attention = Attention(sent_word_length)(word_bigru)
    sent_word_encode = Model(sentence_word_input, word_attention)

    sentence_char_input = Input(shape=(sent_char_length,), dtype="int32")
    char_embedding = Embedding(
        name="char_embedding",
        input_dim=char_embed_weight.shape[0],
        weights=[char_embed_weight],
        output_dim=char_embed_weight.shape[1],
        mask_zero = mask_zero,
    )
    sent_char_embed = char_embedding(sentence_char_input)
    char_bigru = Bidirectional(GRU(64, return_sequences=True))(sent_char_embed)
    char_attention = Attention(sent_char_length)(char_bigru)
    sent_char_encode = Model(sentence_char_input, char_attention)

    review_word_input = Input(shape=(sent_num, sent_word_length), dtype="int32")
    review_word_encode = TimeDistributed(sent_word_encode)(review_word_input)
    review_char_input = Input(shape=(sent_num, sent_char_length),dtype="int32")
    review_char_encode = TimeDistributed(sent_char_encode)(review_char_input)
    review_encode = concatenate([review_word_encode, review_char_encode])
    unvec = convs_block(review_encode, convs=[1,2,3,4,5], f=256)
    dropfeat = Dropout(0.2)(unvec)
    fc = Activation(activation='relu')(BatchNormalization()(Dense(256)(dropfeat)))
    output = Dense(4, activation="softmax")(fc)
    model = Model([review_word_input, review_char_input], output)
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accracy'])
    return model


def convs_block_v2(data, convs = [3,4,5], f=256, name="conv2_feat"):
    pools = []
    for c in convs:
        conv = Conv1D(f, c, activation='elu', padding='same')(data)
        conv = MaxPooling1D(c)(conv)
        conv = Conv1D(f, c-1, activation='elu', padding='same')(conv)
        conv = MaxPooling1D(c-1)(conv)
        conv = Conv1D(int(f/2), 2, activation='elu', padding='same')(conv)
        conv = MaxPooling1D(2)(conv)
        conv = Flatten()(conv)
        pools.append(conv)
    return concatenate(pools, name=name)


def resnet_convs_block(data, convs =[3,4,5], f=256, name="deep_conv_feat"):

    pools = []
    x_short = data
    for c in convs:
        conv = Conv1D(int(f/2), c, activation='relu', padding='same')(data)
        conv = MaxPooling1D(c)(conv)
        conv = Conv1D(f, c-1, activation='relu', padding='same')(conv)
        conv = MaxPooling1D(c-1)(conv)
        conv = Conv1D(f*2, 2, activation='relu', padding='same')(conv)
        conv = MaxPooling1D(2)(conv)
        conv = Flatten()(conv)
        pools.append(conv)
    conv_shocut = BatchNormalization()(Conv1D(filters=f, kernel_size=3, padding="valid")(x_short))
    # conv_shocut = GlobalMaxPooling1D()(conv_shocut)
    conv_shocut = MaxPooling1D(3)(conv_shocut)
    conv_shocut = Flatten()(conv_shocut)
    pools.append(conv_shocut)
    return Activation('relu')(concatenate(pools, name=name))


    # conv_shortcut = Conv1D(f,c=1, activation='relu')

def get_textcnn_v2(seq_length, embed_weight):
    content = Input(shape=(seq_length,), dtype='int32')
    embedding = Embedding(
        input_dim=embed_weight.shape[0],
        weights=[embed_weight],
        output_dim=embed_weight.shape[1],
        name="embedding",
        trainable=False)
    embed = embedding(content)
    embed = SpatialDropout1D(0.2)(embed)
    trans_content = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(embed))))
    # unvec = convs_block_v2(trans_content)
    unvec = resnet_convs_block(trans_content)
    dropfeat = Dropout(0.4)(unvec)
    fc = Activation(activation='relu')(BatchNormalization()(Dense(128)(dropfeat)))
    output = Dense(19, activation="softmax")(fc)
    # output = Dense(4, activation="sigmoid")(fc)
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model = Model(inputs=content, outputs=output)
    model.compile(loss=mycrossentropy, optimizer=adam, metrics=['accuracy'])
    # model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
    return model

def get_word_char_cnn_v2(word_len, char_len, word_embed_weight, char_embed_weight):
    word_input = Input(shape=(word_len,), dtype="int32")
    char_input = Input(shape=(char_len,), dtype="int32")
    word_embedding = Embedding(
        name="word_embedding",
        input_dim = word_embed_weight.shape[0],
        weights=[word_embed_weight],
        output_dim=word_embed_weight.shape[1],
        trainable=False
    )
    char_embedding = Embedding(
        name="char_embedding",
        input_dim=char_embed_weight.shape[0],
        weights = [char_embed_weight],
        output_dim = char_embed_weight.shape[1],
        trainable=False
    )

    word_embed = SpatialDropout1D(0.2)(word_embedding(word_input))
    char_embed = SpatialDropout1D(0.2)(char_embedding(char_input))

    trans_word = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(word_embed))))
    trans_char = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(char_embed))))

    word_feat = convs_block_v2(trans_word, name='word_conv')
    char_feat = convs_block_v2(trans_char, name='char_conv')

    unvec = concatenate([word_feat, char_feat])
    dropfeat = Dropout(0.4)(unvec)
    fc = Activation(activation="relu")(BatchNormalization()(Dropout(0.2)(Dense(512)(dropfeat))))
    output = Dense(4, activation="softmax")(fc)
    model = Model(inputs=[word_input, char_input], outputs = output)
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    return model

def get_wordp_char_cnn_v2(word_len, char_len, word_embed_weight, char_embed_weight):
    word_input = Input(shape=(word_len,), dtype="int32")
    wordp_input = Input(shape=(word_len,), dtype='int32')
    char_input = Input(shape=(char_len,), dtype="int32")
    word_embedding = Embedding(
        name = "word_embeding",
        input_dim = word_embed_weight.shape[0],
        weights=[word_embed_weight],
        output_dim = word_embed_weight.shape[1],
        trainable=False
    )
    wordp_embedding = Embedding(
        name="wordp_embedding",
        input_dim=57,
        output_dim=64
    )
    char_embedding = Embedding(
        name="char_embedding",
        input_dim=char_embed_weight.shape[0],
        weights=[char_embed_weight],
        output_dim=char_embed_weight.shape[1],
        trainable=False
    )
    word_union = concatenate([word_embedding(word_input), wordp_embedding(wordp_input)])
    trans_word = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(320))(word_union))))
    trans_char = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(char_embedding(char_input)))))
    word_feat = convs_block_v2(trans_word)
    char_feat = convs_block_v2(trans_char)
    unvec = concatenate([word_feat, char_feat])
    dropfeat = Dropout(0.4)(unvec)
    fc = Activation(activation="relu")(BatchNormalization()(Dense(512)(dropfeat)))
    output = Dense(4, activation="softmax")(Dropout(0.2)(fc))
    model = Model(inputs=[word_input, wordp_input, char_input], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    return model


def get_capsule_gru(maxlen, embed_weight):

    Num_capsule = 10
    Dim_capsule = 20
    Routings = 5

    input = Input(shape=(maxlen,))
    embedding = Embedding(
        name = "embedding",
        input_dim = embed_weight.shape[0],
        weights = [embed_weight],
        output_dim = embed_weight.shape[1],
        trainable=False
    )
    embed_layer = SpatialDropout1D(0.2)(embedding(input))

    # bigru = Bidirectional(GRU(128, return_sequences=True))(embed_layer)
    capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings, share_weights=True)(embed_layer)
    capsule = Bidirectional(GRU(128, return_sequences=True))(capsule)
    # capsule = Flatten()(capsule)
    avg_pool = GlobalAveragePooling1D()(capsule)
    max_pool = GlobalMaxPooling1D()(capsule)
    conc = concatenate([avg_pool, max_pool])
    dropfeat = Dropout(0.4)(conc)
    fc = Activation(activation="relu")(BatchNormalization()(Dense(256)(dropfeat)))
    output = Dense(4, activation='softmax')(fc)
    model = Model(inputs=input, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def get_rnn_att(maxlen, embed_weight):
    
    input = Input(shape=(maxlen,))
    # 自己构建一个
    # embed_layer = Embedding(max_features,
    #                         300,
    #                         input_length=maxlen,
    #                         
    #                         )(input)

    w_mask = Masking(mask_value=0.)(input)

    embedding = Embedding(
        name="embedding",
        input_dim=embed_weight.shape[0],
        weights=[embed_weight],
        output_dim=embed_weight.shape[1],
        trainable=False
    )

    embed_layer = SpatialDropout1D(0.2)(embedding(w_mask))
    
    embed_layer = BatchNormalization()(embed_layer)

    bigru = Bidirectional(CuDNNGRU(60, return_sequences=True))(embed_layer)

    bigru = BatchNormalization()(bigru)

    # # 第二层的 RNN
    # model.add(Bidirectional(GRU(units=50,activation='selu',kernel_initializer='orthogonal', recurrent_initializer='orthogonal',
    #               bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),
    #               bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None,
    #               bias_constraint=None, dropout=0.5, recurrent_dropout=0.0, implementation=1, return_sequences=True,
    #               return_state=False, go_backwards=False, stateful=False, unroll=False),merge_mode='concat'))   #input_shape=(max_lenth, max_features),
    #

    bigru = PReLU()(bigru)
    bigru = Dropout(0.5)(bigru)
    bigru = AttentionWithContext()(bigru)

    bigru = BatchNormalization()(bigru)
    output = Dense(19, activation='softmax')(bigru)
    model = Model(inputs=input, outputs=output)
    model.compile(loss=mycrossentropy, optimizer='adam', metrics=['accuracy'])
    return model


def get_cnn_word_feature(word_len, fea_len, word_embed_weight):
    #  --------------------------- the first input ------------------------------
    filter_sizes = [3, 4, 5]
    num_filters = 300

    def train_model(embed_layer):
        embed_layer = BatchNormalization()(embed_layer)

        embedding_dim = word_embed_weight.shape[1]


        reshape = Reshape((word_len, embedding_dim, 1))(embed_layer)

        # ===========================================================================================

        # embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length)(inputs)
        # reshape = Reshape((sequence_length,embedding_dim,1))(embedding)

        conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid',
                        kernel_initializer='normal', activation='relu')(reshape)
        conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid',
                        kernel_initializer='normal', activation='relu')(reshape)
        conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid',
                        kernel_initializer='normal', activation='relu')(reshape)

        maxpool_0 = MaxPool2D(pool_size=(word_len - filter_sizes[0] + 1, 1), strides=(1, 1), padding='valid')(
            conv_0)
        maxpool_1 = MaxPool2D(pool_size=(word_len - filter_sizes[1] + 1, 1), strides=(1, 1), padding='valid')(
            conv_1)
        maxpool_2 = MaxPool2D(pool_size=(word_len - filter_sizes[2] + 1, 1), strides=(1, 1), padding='valid')(
            conv_2)


        concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
        conc = Flatten()(concatenated_tensor)

        return conc

    word_input = Input(shape=(word_len,), dtype="int32")

    word_embedding = Embedding(
        name="word_embedding",
        input_dim=word_embed_weight.shape[0],
        weights=[word_embed_weight],
        output_dim=word_embed_weight.shape[1],
        trainable=False
    )
    trans_word = word_embedding(word_input)
    word_embed_layer = SpatialDropout1D(0.2)(trans_word)
    word_dence = train_model(word_embed_layer)

    #  --------------------------- the second input ------------------------------
    fea_input1 = Input(shape=(fea_len,), dtype="float32")

    fea_input = Dense(128)(fea_input1)
    fea_input = Dropout(0.4)(fea_input)
    fea_dence = BatchNormalization()(fea_input)

    # #  --------------------------- the third input ------------------------------
    # chi2_input = Input(shape=(chi2_len,), dtype="float32")
    # 
    # chi2_input = Dense(128)(chi2_input)
    # chi2_input = Dropout(0.4)(chi2_input)
    # fea_dence = BatchNormalization()(chi2_input)

    conc = concatenate([word_dence, fea_dence])
    dropfeat = Dropout(0.5)(conc)

    fc = Activation(activation="relu")(BatchNormalization()(Dense(64)(dropfeat)))
    output = Dense(19, activation='softmax')(fc)

    model = Model(inputs=[word_input, fea_input1], outputs=output)
    

    model.compile(loss=mycrossentropy, optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def get_rcnn_word_feature(word_len,fea_len,word_embed_weight):

    #  --------------------------- the first input ------------------------------
    def train_model(embed_layer):
        embed_layer = BatchNormalization()(embed_layer)

        bigru = Bidirectional(
            GRU(units=128, activation='selu', kernel_initializer='orthogonal', recurrent_initializer='orthogonal',
                bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01),
                recurrent_regularizer=regularizers.l2(0.01),
                bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None,
                bias_constraint=None, dropout=0.5, recurrent_dropout=0.0, implementation=1, return_sequences=True,
                # 多层时需设置为true
                return_state=False, go_backwards=False, stateful=False, unroll=False), merge_mode='concat')(embed_layer)

        bigru = BatchNormalization()(bigru)
        bigru = PReLU()(bigru)
        bigru = Dropout(0.5)(bigru)

        conv = Conv1D(128, kernel_size=3, padding='valid', kernel_initializer='glorot_uniform',
                      kernel_regularizer=regularizers.l2(0.01))(bigru)
        avg_pool = GlobalAveragePooling1D()(conv)
        max_pool = GlobalMaxPooling1D()(conv)
        conc = concatenate([avg_pool, max_pool])

        # conc = Dense(128)(conc)
        # conc = Dropout(0.4)(conc)
        # conc = BatchNormalization()(conc)
        return conc
    
    
    
    word_input = Input(shape=(word_len,), dtype="int32")

    word_embedding = Embedding(
        name="word_embedding",
        input_dim=word_embed_weight.shape[0],
        weights=[word_embed_weight],
        output_dim=word_embed_weight.shape[1],
        trainable=False
    )
    trans_word = word_embedding(word_input)
    word_embed_layer = SpatialDropout1D(0.2)(trans_word)
    word_dence = train_model(word_embed_layer)


    #  --------------------------- the second input ------------------------------
    fea_input1 = Input(shape=(fea_len,), dtype="float32")

    fea_input = Dense(128)(fea_input1)
    fea_input = Dropout(0.4)(fea_input)
    fea_dence = BatchNormalization()(fea_input)

    conc = concatenate([word_dence, fea_dence])
    dropfeat = Dropout(0.5)(conc)

    fc = Activation(activation="relu")(BatchNormalization()(Dense(64)(dropfeat)))
    output = Dense(19, activation='softmax')(fc)
    
    
    model = Model(inputs=[word_input, fea_input1], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='AdaDelta', metrics=['accuracy'])
    model.summary()
    return model



def get_cnn_word_char(word_len, char_len, word_embed_weight, char_embed_weight):

    filter_sizes = [3, 4, 5]
    num_filters = 300
    def train_model(embed_layer):
        
        embed_layer = BatchNormalization()(embed_layer)
    
        embedding_dim =  word_embed_weight.shape[1]
        reshape = Reshape((word_len, embedding_dim, 1))(embed_layer)
        # ===========================================================================================

        # embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length)(inputs)
        # reshape = Reshape((sequence_length,embedding_dim,1))(embedding)

        conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid',
                        kernel_initializer='normal', activation='relu')(reshape)
        conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid',
                        kernel_initializer='normal', activation='relu')(reshape)
        conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid',
                        kernel_initializer='normal', activation='relu')(reshape)

        maxpool_0 = MaxPool2D(pool_size=(word_len - filter_sizes[0] + 1, 1), strides=(1, 1), padding='valid')(
            conv_0)
        maxpool_1 = MaxPool2D(pool_size=(word_len - filter_sizes[1] + 1, 1), strides=(1, 1), padding='valid')(
            conv_1)
        maxpool_2 = MaxPool2D(pool_size=(word_len - filter_sizes[2] + 1, 1), strides=(1, 1), padding='valid')(
            conv_2)

        concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
        conc = Flatten()(concatenated_tensor)

        return conc

    Num_capsule = 10
    Dim_capsule = 20
    Routings = 5

    # one input
    word_input = Input(shape=(word_len,), dtype="int32")
    char_input = Input(shape=(char_len,), dtype="int32")

    word_embedding = Embedding(
        name="word_embedding",
        input_dim=word_embed_weight.shape[0],
        weights=[word_embed_weight],
        output_dim=word_embed_weight.shape[1],
        trainable=False
    )
    char_embedding = Embedding(
        name="char_embedding",
        input_dim=char_embed_weight.shape[0],
        weights=[char_embed_weight],
        output_dim=char_embed_weight.shape[1],
        trainable=False
    )

    trans_word = word_embedding(word_input)
    trans_char = char_embedding(char_input)

    word_embed_layer = SpatialDropout1D(0.2)(trans_word)
    char_embed_layer = SpatialDropout1D(0.2)(trans_char)

    word_dence = train_model(word_embed_layer)
    char_dence = train_model(char_embed_layer)
    # word_capsule = Flatten()(word_dence)
    # char_capsule = Flatten()(char_dence)

    conc = concatenate([word_dence, char_dence])

    dropfeat = Dropout(0.5)(conc)

    fc = Activation(activation="relu")(BatchNormalization()(Dense(64)(dropfeat)))
    output = Dense(19, activation='softmax')(fc)
    model = Model(inputs=[word_input, char_input], outputs=output)
    model.compile(loss=mycrossentropy, optimizer='adam', metrics=['accuracy'])
    return model


def get_c_gru_word_char(word_len, char_len, word_embed_weight, char_embed_weight):
    def train_model(embed_layer):
        embed_layer = BatchNormalization()(embed_layer)

        conv = Conv1D(filters=300, kernel_size=3, activation='relu', padding='valid',
                      kernel_regularizer=regularizers.l2(0.01))(embed_layer)

        bigru = Bidirectional(
            CuDNNGRU(128, kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),
                     return_sequences=True))(conv)
        conc = att_max_avg_pooling(bigru)
        return conc

    # one input
    word_input = Input(shape=(word_len,), dtype="int32")
    char_input = Input(shape=(char_len,), dtype="int32")

    word_embedding = Embedding(
        name="word_embedding",
        input_dim=word_embed_weight.shape[0],
        weights=[word_embed_weight],
        output_dim=word_embed_weight.shape[1],
        trainable=False
    )
    char_embedding = Embedding(
        name="char_embedding",
        input_dim=char_embed_weight.shape[0],
        weights=[char_embed_weight],
        output_dim=char_embed_weight.shape[1],
        trainable=False
    )

    trans_word = word_embedding(word_input)
    trans_char = char_embedding(char_input)

    word_embed_layer = SpatialDropout1D(0.2)(trans_word)
    char_embed_layer = SpatialDropout1D(0.2)(trans_char)

    word_dence = train_model(word_embed_layer)
    char_dence = train_model(char_embed_layer)
    # word_capsule = Flatten()(word_dence)
    # char_capsule = Flatten()(char_dence)

    conc = concatenate([word_dence, char_dence])

    dropfeat = Dropout(0.5)(conc)

    fc = Activation(activation="relu")(BatchNormalization()(Dense(64)(dropfeat)))
    output = Dense(19, activation='softmax')(fc)
    model = Model(inputs=[word_input, char_input], outputs=output)
    model.compile(loss=mycrossentropy, optimizer='adam', metrics=['accuracy'])
    return model



def get_rcnn_word_char(word_len, char_len, word_embed_weight, char_embed_weight):
    
    def train_model(embed_layer):
        embed_layer = BatchNormalization()(embed_layer)

        bigru = Bidirectional(
            GRU(units=128, activation='selu', kernel_initializer='orthogonal', recurrent_initializer='orthogonal',
                bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01),
                recurrent_regularizer=regularizers.l2(0.01),
                bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None,
                bias_constraint=None, dropout=0.5, recurrent_dropout=0.0, implementation=1, return_sequences=True,
                # 多层时需设置为true
                return_state=False, go_backwards=False, stateful=False, unroll=False), merge_mode='concat')(embed_layer)

        bigru = BatchNormalization()(bigru)
        bigru = PReLU()(bigru)
        bigru = Dropout(0.5)(bigru)

        conv = Conv1D(128, kernel_size=3, padding='valid', kernel_initializer='glorot_uniform',
                      kernel_regularizer=regularizers.l2(0.01))(bigru)
        avg_pool = GlobalAveragePooling1D()(conv)
        max_pool = GlobalMaxPooling1D()(conv)
        conc = concatenate([avg_pool, max_pool])
        return conc


    # one input
    word_input = Input(shape=(word_len,), dtype="int32")
    char_input = Input(shape=(char_len,), dtype="int32")

    word_embedding = Embedding(
        name="word_embedding",
        input_dim=word_embed_weight.shape[0],
        weights=[word_embed_weight],
        output_dim=word_embed_weight.shape[1],
        trainable=False
    )
    char_embedding = Embedding(
        name="char_embedding",
        input_dim=char_embed_weight.shape[0],
        weights=[char_embed_weight],
        output_dim=char_embed_weight.shape[1],
        trainable=False
    )

    trans_word = word_embedding(word_input)
    trans_char = char_embedding(char_input)
    
    word_embed_layer = SpatialDropout1D(0.2)(trans_word)
    char_embed_layer = SpatialDropout1D(0.2)(trans_char)

    word_dence = train_model(word_embed_layer)
    char_dence = train_model(char_embed_layer)
    # word_capsule = Flatten()(word_dence)
    # char_capsule = Flatten()(char_dence)
    
    conc = concatenate([word_dence, char_dence])
    
    dropfeat = Dropout(0.5)(conc)

    fc = Activation(activation="relu")(BatchNormalization()(Dense(64)(dropfeat)))
    output = Dense(19, activation='softmax')(fc)
    model = Model(inputs=[word_input, char_input], outputs=output)
    model.compile(loss=mycrossentropy, optimizer='adam', metrics=['accuracy'])
    return model

def get_rcnn(maxlen, embed_weight):
    
    def mycrossentropy(y_true, y_pred, e=0.1):
        loss1 = K.categorical_crossentropy(y_true, y_pred)
        loss2 = K.categorical_crossentropy(K.ones_like(y_pred)/19, y_pred)
        return (1-e)*loss1 + e*loss2
    
    input = Input(shape=(maxlen,))

    w_mask = Masking(mask_value=0.)(input)

    embedding = Embedding(
        name = "embedding",
        input_dim = embed_weight.shape[0],
        weights = [embed_weight],
        output_dim = embed_weight.shape[1],
        trainable = False
    )

    embed_layer = SpatialDropout1D(0.2)(embedding(w_mask))
    
    embed_layer = BatchNormalization()(embed_layer)

    bigru = Bidirectional(CuDNNLSTM(128 // 2, return_sequences=True))(embed_layer)

    bigru = BatchNormalization()(bigru)
    bigru = PReLU()(bigru)
    bigru = Dropout(0.5)(bigru)

    conv = Conv1D(300, kernel_size=3, padding='valid', kernel_initializer='glorot_uniform',kernel_regularizer=regularizers.l2(0.01))(bigru)
    # avg_pool = GlobalAveragePooling1D()(conv)
    # max_pool = GlobalMaxPooling1D()(conv)
    # conc = concatenate([avg_pool, max_pool])
    conc = att_max_avg_pooling(conv)
    dropfeat = Dropout(0.5)(conc)

    fc = Activation(activation="relu")(BatchNormalization()(Dense(64)(dropfeat)))
    output = Dense(19, activation='softmax')(fc)
    model = Model(inputs=input, outputs=output)
    model.compile(loss=mycrossentropy, optimizer='adam', metrics=['accuracy'])
    return model


def get_cgru_word_fea(maxlen,fea_len,embed_weight):
    
    word_input = Input(shape=(maxlen,))

    w_mask = Masking(mask_value=0.)(word_input)
    embedding = Embedding(
        name="word_embedding",
        input_dim=embed_weight.shape[0],
        weights=[embed_weight],
        output_dim=embed_weight.shape[1],
        trainable=False
    )
    # embed_layer = SpatialDropout1D(0.2)(embedding(input))
    # embed_layer = (embedding(input))

    trans_word = embedding(w_mask)

    conv = Conv1D(filters=300, kernel_size=3, activation='relu', padding='valid',
                  kernel_regularizer=regularizers.l2(0.01))(trans_word)
    bigru = Bidirectional(CuDNNGRU(128 // 2, return_sequences=True))(conv)
    avg_pool = GlobalAveragePooling1D()(bigru)
    max_pool = GlobalMaxPooling1D()(bigru)
    conc = concatenate([avg_pool, max_pool])
    dropfeat = Dropout(0.4)(conc)
    word_dence = BatchNormalization()(dropfeat)

    #  --------------------------- the second input ------------------------------
    fea_input1 = Input(shape=(fea_len,), dtype="float32")
    fea_input = Dense(128)(fea_input1)
    fea_input = Dropout(0.4)(fea_input)
    fea_dence = BatchNormalization()(fea_input)

    conc = concatenate([word_dence, fea_dence])
    dropfeat = Dropout(0.5)(conc)

    fc = Activation(activation="relu")(BatchNormalization()(Dense(64)(dropfeat)))
    output = Dense(19, activation='softmax')(fc)

    model = Model(inputs=[word_input, fea_input1], outputs=output)
    model.compile(loss=mycrossentropy, optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def get_conv_add_lstm(maxlen,word_embed_weight):
    dropout_p = 0.2
    word_embedding = Embedding(
        name="embedding",
        input_dim=word_embed_weight.shape[0],
        weights=[word_embed_weight],
        output_dim=word_embed_weight.shape[1],
        trainable=False
    )
    # char_embedding = Embedding(
    #     name="char_embedding",
    #     input_dim=char_embed_weight.shape[0],
    #     weights=[char_embed_weight],
    #     output_dim=char_embed_weight.shape[1],
    #     trainable=False
    # )

    # one input
    word_input = Input(shape=(maxlen,), name='word')
    # char_input = Input(shape=(maxlen,), dtype="int32")

    # c_mask = Masking(mask_value=0.)(char_input)
    w_mask = Masking(mask_value=0.)(word_input)

    trans_word = word_embedding(w_mask)
    # trans_char = char_embedding(c_mask)

    def get_conv_layer(x):
        x = BatchNormalization()(x)
        x = SpatialDropout1D(dropout_p)(x)
        x = Conv1D(256, kernel_size=3, padding="valid", kernel_initializer="glorot_uniform")(x)
        x = Bidirectional(CuDNNLSTM(60, return_sequences=True))(x)
        x = SpatialDropout1D(dropout_p)(x)
        x = Bidirectional(CuDNNGRU(60, return_sequences=True))(x)

        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        attn = AttentionWithContext()(x)
        return avg_pool,max_pool,attn

    def get_lstm_layer(embedding_layer):
        word_embedding_layer = SpatialDropout1D(0.2)(embedding_layer)

        recurrent_units = 60
        word_rnn_1 = Bidirectional(CuDNNGRU(recurrent_units // 2, return_sequences=True))(word_embedding_layer)
        word_rnn_1 = SpatialDropout1D(0.1)(word_rnn_1)
        word_rnn_2 = Bidirectional(CuDNNGRU(recurrent_units // 2, return_sequences=True))(word_rnn_1)

        word_maxpool = GlobalMaxPooling1D()(word_rnn_2)
        word_average = GlobalAveragePooling1D()(word_rnn_2)
        return word_maxpool, word_average

    avg_pool, max_pool, attn = get_conv_layer(trans_word)
    word_maxpool, word_average = get_lstm_layer(trans_word)

    concat2 = concatenate([avg_pool, max_pool, word_maxpool, word_average], axis=-1)
    concat2 = Dropout(0.5)(concat2)
    dense2 = Dense(19, activation="softmax")(concat2)
    # res_model = Model(inputs=[char_input, word_input], outputs=dense2)
    model = Model(inputs=word_input, outputs=dense2)
    model.compile(loss=mycrossentropy, optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def get_dpcnn_add_gru(maxlen,word_embed_weight):
    dp = 7
    filter_nr = 64
    filter_size = 3
    max_pool_size = 3
    max_pool_strides = 2
    dense_nr = 256
    spatial_dropout = 0.2
    dense_dropout = 0.5
    conv_kern_reg = regularizers.l2(0.00001)
    conv_bias_reg = regularizers.l2(0.00001)

    dropout_p = 0.2
    word_embedding = Embedding(
        name="embedding",
        input_dim=word_embed_weight.shape[0],
        weights=[word_embed_weight],
        output_dim=word_embed_weight.shape[1],
        trainable=False
    )
    # char_embedding = Embedding(
    #     name="char_embedding",
    #     input_dim=char_embed_weight.shape[0],
    #     weights=[char_embed_weight],
    #     output_dim=char_embed_weight.shape[1],
    #     trainable=False
    # )

    # one input
    word_input = Input(shape=(maxlen,), name='word')
    # char_input = Input(shape=(maxlen,), dtype="int32")

    # c_mask = Masking(mask_value=0.)(char_input)

    trans_word = word_embedding(word_input)

    w_mask = Masking(mask_value=0.)(word_input)
    trans_word2 = word_embedding(w_mask)
    # trans_char = char_embedding(c_mask)

    def get_dpcnn_layer(x):
        x = BatchNormalization()(x)
        block1 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                        kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(x)
        block1 = BatchNormalization()(block1)
        block1 = PReLU()(block1)
        block1 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                        kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block1)
        block1 = BatchNormalization()(block1)
        block1 = PReLU()(block1)

        resize_emb = Conv1D(filter_nr, kernel_size=1, padding='same', activation='linear',
                            kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(x)
        resize_emb = PReLU()(resize_emb)

        block1_output = add([block1, resize_emb])
        x = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block1_output)

        for i in range(dp):
            block1 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                            kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(x)
            block1 = BatchNormalization()(block1)
            block1 = PReLU()(block1)
            block1 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                            kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block1)
            block1 = BatchNormalization()(block1)
            block1 = PReLU()(block1)

            block_output = add([block1, x])
            # print(i)
            if i + 1 != dp:
                x = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block_output)

        x = GlobalMaxPooling1D()(block_output)
        output = Dense(dense_nr, activation='linear')(x)
        output = BatchNormalization()(output)
        output = PReLU()(output)

        return output

    def get_lstm_layer(embedding_layer):
        recurrent_units = 60
        word_embedding_layer = SpatialDropout1D(0.2)(embedding_layer)

        word_rnn_1 = Bidirectional(CuDNNGRU(recurrent_units // 2, return_sequences=True))(word_embedding_layer)
        word_rnn_1 = SpatialDropout1D(0.5)(word_rnn_1)
        word_rnn_2 = Bidirectional(CuDNNGRU(recurrent_units // 2, return_sequences=True))(word_rnn_1)
        word_maxpool = GlobalMaxPooling1D()(word_rnn_2)
        word_average = GlobalAveragePooling1D()(word_rnn_2)

        return word_maxpool, word_average


    dp_vec = get_dpcnn_layer(trans_word)
    word_maxpool, word_average = get_lstm_layer(trans_word2)

    concat2 = concatenate([dp_vec, word_maxpool, word_average], axis=-1)
    concat2 = Dropout(0.5)(concat2)
    dense2 = Dense(19, activation="softmax")(concat2)
    # res_model = Model(inputs=[char_input, word_input], outputs=dense2)
    model = Model(inputs=word_input, outputs=dense2)
    model.compile(loss=mycrossentropy, optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def get_cgru(maxlen, embed_weight):

    input = Input(shape=(maxlen,))
    embedding = Embedding(
        name = "embedding",
        input_dim = embed_weight.shape[0],
        weights = [embed_weight],
        output_dim= embed_weight.shape[1],
        trainable = False
    )
    # input_mask = Masking(mask_value=0.)(input)
    # embed_layer = SpatialDropout1D(0.2)(embedding(input))
    embed_layer = (embedding(input))

    conv = Conv1D(filters=300, kernel_size=3, activation='relu', padding='valid',kernel_regularizer=regularizers.l2(0.01))(embed_layer)

    bigru = Bidirectional(CuDNNGRU(128, kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),return_sequences=True))(conv)
    conc = att_max_avg_pooling(bigru)
    dropfeat = Dropout(0.4)(conc)
    fc = Activation(activation="relu")(BatchNormalization()(Dense(128)(dropfeat)))
    output = Dense(19, activation='softmax')(fc)
    model = Model(inputs=input, outputs=output)
    model.compile(loss=mycrossentropy, optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


def get_word_char_rnn(word_len, char_len, word_embed_weight, char_embed_weight):
    word_input = Input(shape=(word_len,), dtype="int32")
    char_input = Input(shape=(char_len,), dtype="int32")
    word_embedding = Embedding(
        name="word_embedding",
        input_dim=word_embed_weight.shape[0],
        weights = [word_embed_weight],
        output_dim=word_embed_weight.shape[1],
        trainable=False
    )
    char_embedding = Embedding(
        name="char_embedding",
        input_dim =char_embed_weight.shape[0],
        weights =[char_embed_weight],
        output_dim=char_embed_weight.shape[1],
        trainable=False
    )
    trans_word = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(word_embedding(word_input)))))
    trans_char = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(char_embedding(char_input)))))
    word_bigru = Bidirectional(GRU(128, return_sequences=True))(trans_word)
    # word_bigru2 = Bidirectional(GRU(128, return_sequences=True))(word_bigru)
    avg_pool = GlobalAveragePooling1D()(word_bigru)
    max_pool = GlobalMaxPooling1D()(word_bigru)
    word_feat = concatenate([avg_pool, max_pool], axis=1)

    char_bigru = Bidirectional(GRU(128, return_sequences=True))(trans_char)
    # char_bigru2 = Bidirectional(GRU(128, return_sequences=True))(char_bigru)
    avg_pool = GlobalAveragePooling1D()(char_bigru)
    max_pool = GlobalMaxPooling1D()(char_bigru)
    char_feat = concatenate([avg_pool, max_pool], axis=1)
    feat = concatenate([word_feat, char_feat], axis=1)
    dropfeat = Dropout(0.3)(feat) # 0.4
    fc = Activation(activation="relu")(BatchNormalization()(Dense(128)(dropfeat))) # 256
    output = Dense(4, activation="softmax")(fc)
    model = Model(inputs=[word_input, char_input], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    return model

def get_word_char_rcnn(word_len, char_len, word_embed_weight, char_embed_weight):

    word_input = Input(shape=(word_len,), dtype="int32")
    char_input = Input(shape=(char_len,), dtype="int32")
    word_embedding = Embedding(
        name = "word_embedding",
        input_dim = word_embed_weight.shape[0],
        weights = [word_embed_weight],
        output_dim = word_embed_weight.shape[1],
        trainable = False
    )
    char_embedding = Embedding(
        name="char_embedding",
        input_dim=char_embed_weight.shape[0],
        weights=[char_embed_weight],
        output_dim=char_embed_weight.shape[1],
        trainable=False
    )
    trans_word = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(word_embedding(word_input)))))
    trans_char = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(char_embedding(char_input)))))
    # trans_word = word_embedding(word_input)
    # trans_char = char_embedding(char_input)

    bigru_word = Bidirectional(GRU(128, return_sequences=True))(trans_word)
    conv_word = convs_block(bigru_word,convs=[1,2,3], name='conv_word')

    bigru_char = Bidirectional(GRU(128, return_sequences=True))(trans_char)
    conv_char = convs_block(bigru_char, convs=[1,2,3], name='conv_char')

    conc = concatenate([conv_word, conv_char])
    dropfeat = Dropout(0.4)(conc)
    fc = Activation(activation="relu")(BatchNormalization()(Dense(128)(dropfeat)))
    output = Dense(4, activation='softmax')(fc)
    model = Model(inputs=[word_input, char_input], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def get_word_char_capsule_gru(word_len, char_len, word_embed_weight, char_embed_weight):
# def get_word_char_capsule_gru(word_len, word_embed_weight):
    Num_capsule = 10
    Dim_capsule = 20
    Routings = 5

    # one input
    word_input = Input(shape=(word_len,), dtype="int32")
    char_input = Input(shape=(char_len,), dtype="int32")

    word_embedding = Embedding(
        name = "word_embedding",
        input_dim = word_embed_weight.shape[0],
        weights = [word_embed_weight],
        output_dim = word_embed_weight.shape[1],
        trainable=False
    )
    char_embedding = Embedding(
        name="char_embedding",
        input_dim=char_embed_weight.shape[0],
        weights=[char_embed_weight],
        output_dim=char_embed_weight.shape[1],
        trainable=False
    )

    trans_word = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(word_embedding(word_input)))))
    trans_char = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(char_embedding(char_input)))))

    word_bigru = Bidirectional(GRU(128, return_sequences=True))(trans_word)
    char_bigru = Bidirectional(GRU(128, return_sequences=True))(trans_char)

    word_capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings, share_weights=True)(word_bigru)
    char_capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings, share_weights=True)(char_bigru)

    word_capsule = Flatten()(word_capsule)
    char_capsule = Flatten()(char_capsule)
    # conc = Flatten()(word_capsule)

    conc = concatenate([word_capsule, char_capsule])
    dropfeat = Dropout(0.4)(conc)
    fc = Activation(activation="relu")(BatchNormalization()(Dense(128)(dropfeat)))
    output = Dense(19, activation='softmax')(fc)

    model = Model(inputs=[word_input, char_input], outputs=output)
    # model = Model(inputs=word_input, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def get_word_rcnn_char_rnn(word_len, char_len, word_embed_weight, char_embed_weight):

    word_input = Input(shape=(word_len, ), dtype="int32")
    char_input = Input(shape=(char_len, ), dtype="int32")
    word_embedding = Embedding(
        name="word_embedding",
        input_dim = word_embed_weight.shape[0],
        weights = [word_embed_weight],
        output_dim = word_embed_weight.shape[1],
        trainable = False
    )
    char_embedding = Embedding(
        name="char_embedding",
        input_dim = char_embed_weight.shape[0],
        weights = [char_embed_weight],
        output_dim=char_embed_weight.shape[1],
        trainable = False
    )
    trans_word = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(word_embedding(word_input)))))
    trans_char = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(char_embedding(char_input)))))

    bigru_word = Bidirectional(GRU(128, return_sequences=True))(trans_word)
    word_feat = convs_block(bigru_word, convs=[1,2,3], name='conv_word')

    char_bigru = Bidirectional(GRU(128, return_sequences=True))(trans_char)
    avg_pool = GlobalAveragePooling1D()(char_bigru)
    max_pool = GlobalMaxPooling1D()(char_bigru)
    char_feat = concatenate([avg_pool, max_pool], axis=1)

    conc = concatenate([word_feat, char_feat])
    dropfeat =  Dropout(0.4)(conc)
    fc = Activation(activation="relu")(BatchNormalization()(Dense(128)(dropfeat)))
    output = Dense(4, activation='softmax')(fc)
    model = Model(inputs=[word_input, char_input], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def get_word_rnn_char_rcnn(word_len, char_len, word_embed_weight, char_embed_weight):

    word_input = Input(shape=(word_len, ), dtype="int32")
    char_input = Input(shape=(char_len, ), dtype="int32")
    word_embedding = Embedding(
        name="word_embedding",
        input_dim = word_embed_weight.shape[0],
        weights = [word_embed_weight],
        output_dim = word_embed_weight.shape[1],
        trainable = False
    )
    char_embedding = Embedding(
        name="char_embedding",
        input_dim = char_embed_weight.shape[0],
        weights = [char_embed_weight],
        output_dim=char_embed_weight.shape[1],
        trainable = False
    )
    trans_word = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(word_embedding(word_input)))))
    trans_char = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(char_embedding(char_input)))))

    # rnn
    bigru_word = Bidirectional(GRU(128, return_sequences=True))(trans_word)
    avg_pool = GlobalAveragePooling1D()(bigru_word)
    max_pool = GlobalMaxPooling1D()(bigru_word)
    word_feat = concatenate([avg_pool, max_pool], axis=1)

    # rcnn
    char_bigru = Bidirectional(GRU(128, return_sequences=True))(trans_char)
    char_feat = convs_block(char_bigru, convs=[1, 2, 3], name='conv_char')


    conc = concatenate([word_feat, char_feat])
    dropfeat =  Dropout(0.4)(conc)
    fc = Activation(activation="relu")(BatchNormalization()(Dense(128)(dropfeat)))
    output = Dense(4, activation='softmax')(fc)
    model = Model(inputs=[word_input, char_input], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def get_word_rcnn_char_cgru(word_len, char_len, word_embed_weight, char_embed_weight):
    word_input = Input(shape=(word_len, ), dtype="int32")
    char_input = Input(shape=(char_len, ), dtype="int32")
    word_embedding = Embedding(
        name = "word_embedding",
        input_dim = word_embed_weight.shape[0],
        weights = [word_embed_weight],
        output_dim = word_embed_weight.shape[1],
        trainable = False
    )
    char_embedding = Embedding(
        name = "char_embedding",
        input_dim = char_embed_weight.shape[0],
        weights = [char_embed_weight],
        output_dim = char_embed_weight.shape[1],
        trainable = False
    )

    trans_word = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(word_embedding(word_input)))))
    trans_char = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(char_embedding(char_input)))))

    # word rcnn
    bigru_word = Bidirectional(GRU(128, return_sequences=True))(trans_word)
    word_feat = convs_block(bigru_word, convs=[1,2,3], name='conv_word')

    # char cgru
    conv_char = (BatchNormalization()(Conv1D(filters=256, kernel_size=3, padding="valid")(trans_char)))
    bigru_char = Bidirectional(GRU(128, return_sequences=True))(conv_char)
    avg_pool = GlobalAveragePooling1D()(bigru_char)
    max_pool = GlobalMaxPooling1D()(bigru_char)
    char_feat = concatenate([avg_pool, max_pool], axis=1)

    feat = concatenate([word_feat, char_feat], axis=1)
    dropfeat = Dropout(0.3)(feat)
    fc = Activation(activation="relu")(BatchNormalization()(Dense(128)(dropfeat)))
    output = Dense(4, activation="softmax")(fc)
    model = Model(inputs=[word_input, char_input], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def get_word_cgru_char_rcnn(word_len, char_len, word_embed_weight, char_embed_weight):

    word_input = Input(shape=(word_len, ), dtype="int32")
    char_input = Input(shape=(char_len, ), dtype="int32")
    word_embedding = Embedding(
        name = "word_embedding",
        input_dim = word_embed_weight.shape[0],
        weights = [word_embed_weight],
        output_dim = word_embed_weight.shape[1],
        trainable = False
    )
    char_embedding = Embedding(
        name = "char_embedding",
        input_dim = char_embed_weight.shape[0],
        weights = [char_embed_weight],
        output_dim = char_embed_weight.shape[1],
        trainable = False
    )
    trans_word = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(word_embedding(word_input)))))
    trans_char = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(char_embedding(char_input)))))\

    # word cgru
    conv_word = (BatchNormalization()(Conv1D(filters=256, kernel_size=3, padding="valid")(trans_word)))
    bigru_word = Bidirectional(GRU(128, return_sequences=True))(conv_word)
    avg_pool = GlobalAveragePooling1D()(bigru_word)
    max_pool = GlobalMaxPooling1D()(bigru_word)
    word_feat = concatenate([avg_pool, max_pool], axis=1)

    # char rcnn
    bigru_char = Bidirectional(GRU(128, return_sequences=True))(trans_char)
    char_feat = convs_block(bigru_char, convs=[1,2,3], name="conv_char")

    feat = concatenate([word_feat, char_feat], axis=1)
    dropfeat = Dropout(0.4)(feat)
    fc = Activation(activation="relu")(BatchNormalization()(Dense(128)(dropfeat)))
    output = Dense(4, activation='softmax')(fc)
    model = Model(inputs=[word_input, char_input], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def get_word_char_cnn_fe(word_len, char_len, fe_len, word_embed_weight, char_embed_weight):

    word_input = Input(shape=(word_len,), dtype="int32")
    char_input = Input(shape=(char_len,), dtype="int32")
    feature_input = Input(shape=(fe_len, ), dtype="int32")

    word_embedding = Embedding(
        name="word_embedding",
        input_dim = word_embed_weight.shape[0],
        weights = [word_embed_weight],
        output_dim = word_embed_weight.shape[1],
        trainable = False
    )
    char_embedding = Embedding(
        name = "char_embedding",
        input_dim = char_embed_weight.shape[0],
        weights = [char_embed_weight],
        output_dim = char_embed_weight.shape[1],
        trainable = False
    )
    trans_word = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(word_embedding(word_input)))))
    trans_char = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(char_embedding(char_input)))))

    word_feat = convs_block(trans_word, convs=[1,2,3,4,5], f=256, name="word_conv")
    char_feat = convs_block(trans_char, convs=[1,2,3,4,5], f=256, name="char_conv")
    feat = concatenate([word_feat, char_feat])
    dropfeat = Dropout(0.4)(feat)
    fc = Activation(activation="relu")(BatchNormalization()(Dense(128)(dropfeat)))
    fc = concatenate([fc, feature_input])
    output = Dense(4, activation="softmax")(fc)
    model = Model(inputs=[word_input, char_input, feature_input], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accruracy'])
    return model