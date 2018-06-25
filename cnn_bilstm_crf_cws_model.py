# coding:utf-8
# py3.5+tensorflow-1.0.1+keras-2.0.6
# seq2seq bilstm+cnn+crf
import os
import codecs
import pickle
import numpy as np

import gensim

import keras
from keras.layers import *
from keras.models import *
from keras_contrib.layers import CRF
from keras import optimizers

from keras import backend as K

from keras.utils import plot_model
from keras.utils import np_utils

from keras.callbacks import ModelCheckpoint

import score
import visualization
import utils
import embedding_model
global model_type

model_type = 6      #7
# input:
# maxlen  char_value_dict_len  class_label_count
def CNN_Bilstm_Crf(maxlen, char_value_dict_len, class_label_count, embedding_weights=None,embedding_size=100, model_type=6, is_train=True):
    DROPOUT_RATE = 0.2
    half_window_size = 2
    word_input = Input(shape=(maxlen,), dtype='int32', name='word_input')
    if is_train:
        word_emb = Embedding(char_value_dict_len + 2, output_dim=embedding_size, \
                             input_length=maxlen, weights=[embedding_weights], \
                             name='word_emb')(word_input)
    else:
        word_emb = Embedding(char_value_dict_len + 2, output_dim=embedding_size, \
                             input_length=maxlen, \
                             name='word_emb')(word_input)
    if model_type == 1: # 思路1
        print("model one:")
        # cnn
        padding_layer = ZeroPadding1D(padding=half_window_size)(word_emb)  # 对1D输入的首尾端（如时域序列）填充0，以控制卷积以后向量的长度
        # 卷积核的数目（即输出的维度）
        conv = Conv1D(nb_filter=50, filter_length=2 * half_window_size + 1, \
                      padding='valid', use_bias=True, bias_initializer='zeros')(padding_layer)
        conv_d = Dropout(DROPOUT_RATE)(conv)
        dense_conv = TimeDistributed(Dense(50))(conv_d)
        char_max_pooling = MaxPooling1D(pool_length=1)(dense_conv)
        # merge
        total_emb = merge([word_emb, char_max_pooling], mode='concat', concat_axis=2, name='total_emb')
        #emb_droput = Dropout(DROPOUT_RATE)(total_emb)

        # bilstm
        bilstm = Bidirectional(LSTM(64, return_sequences=True, activation='tanh'))(total_emb)  # 隐藏元个数
        bilstm_d = Dropout(DROPOUT_RATE)(bilstm)  # DROPOUT_RATE控制输入线性变换的神经元断开比例
        # TimeDistributedDense将同样的密集(全连接)操作应用到3D张量的每一个时间间隔上。
        #dense = TimeDistributed(Dense(class_label_count))(bilstm_d)

        # crf
        crf = CRF(class_label_count, sparse_target=False)
        crf_output = crf(bilstm_d)

        # build model
        model = Model(input=[word_input], output=[crf_output])
        model.compile('rmsprop', loss=crf.loss_function, metrics=[crf.accuracy])  # number:1,78%

        model.summary()
        print("Test:Bilstm_CNN_Crf()")

    elif model_type == 2: #思路2 复杂点的
        print("model two:")
        # cnn
        padding_layer = ZeroPadding1D(padding=half_window_size)(word_emb)  # 对1D输入的首尾端（如时域序列）填充0，以控制卷积以后向量的长度
        # 卷积核的数目（即输出的维度）
        conv = Conv1D(nb_filter=50, filter_length=2 * half_window_size + 1, \
                      padding='valid', use_bias=True, bias_initializer='zeros')(padding_layer)
        conv_d = Dropout(DROPOUT_RATE)(conv)
        dense_conv = TimeDistributed(Dense(50))(conv_d)
        char_max_pooling = MaxPooling1D(pool_length=1)(dense_conv)

        # merge
        total_emb = merge([word_emb, char_max_pooling], mode='concat', concat_axis=2, name='total_emb')
        emb_droput = Dropout(0.1)(total_emb)

        # bilstm
        bilstm = Bidirectional(LSTM(64, return_sequences=True, activation='tanh'))(emb_droput)  # 隐藏元个数
        bilstm_d = Dropout(DROPOUT_RATE)(bilstm)  # DROPOUT_RATE控制输入线性变换的神经元断开比例
        # TimeDistributedDense将同样的密集(全连接)操作应用到3D张量的每一个时间间隔上。
        #dense = TimeDistributed(Dense(class_label_count))(bilstm_d)

        rnn_cnn_merge = merge([bilstm_d, char_max_pooling], mode='concat', concat_axis=2)
        dense = TimeDistributed(Dense(class_label_count))(rnn_cnn_merge)

        # crf
        crf = CRF(class_label_count, sparse_target=False)
        crf_output = crf(dense)

        # build model
        model = Model(input=[word_input], output=[crf_output])
        # sgd = SGD(lr=0.001, decay=0.05, momentum=0.9, nesterov=False, clipvalue=5)
        # optmr = optimizers.Adam(lr=0.001, beta_1=0.5)
        # model.compile(loss=crf.loss_function,optimizer=sgd,metrics=[crf.accuracy]) # number:2 78%
        # model.compile(loss=crf.loss_function, optimizer='adam', metrics=[crf.accuracy]) # number:3 56%
        model.compile('rmsprop', loss=crf.loss_function, metrics=[crf.accuracy])  # number:1,78%

        # model.summary()
        print("Test:Bilstm_CNN_Crf()")
    elif model_type == 3: #思路3 复杂
        print("model three:")
        # cnn
        padding_layer = ZeroPadding1D(padding=half_window_size)(word_emb)  # 对1D输入的首尾端（如时域序列）填充0，以控制卷积以后向量的长度
        # 卷积核的数目（即输出的维度）
        conv = Conv1D(nb_filter=50, filter_length=2 * half_window_size + 1, \
                      padding='valid', use_bias=True, bias_initializer='zeros')(padding_layer)
        conv_d = Dropout(DROPOUT_RATE)(conv)
        dense_conv = TimeDistributed(Dense(50))(conv_d)
        char_max_pooling = MaxPooling1D(pool_length=1)(dense_conv)

        # bilstm
        bilstm1 = Bidirectional(LSTM(64, return_sequences=True, activation='tanh'))(word_emb)  # 隐藏元个数
        bilstm_d1 = Dropout(DROPOUT_RATE)(bilstm1)  # DROPOUT_RATE控制输入线性变换的神经元断开比例
        # TimeDistributedDense将同样的密集(全连接)操作应用到3D张量的每一个时间间隔上。
        #dense1 = TimeDistributed(Dense(class_label_count, activation='softmax'))(bilstm_d1)

        # merge
        total_emb = merge([word_emb, char_max_pooling], mode='concat', concat_axis=2, name='total_emb')
        #emb_droput = Dropout(0.1)(total_emb)

        # bilstm
        bilstm2 = Bidirectional(LSTM(64, return_sequences=True, activation='tanh'))(total_emb)  # 隐藏元个数
        bilstm_d2 = Dropout(DROPOUT_RATE)(bilstm2)  # DROPOUT_RATE控制输入线性变换的神经元断开比例
        #TimeDistributedDense将同样的密集(全连接)操作应用到3D张量的每一个时间间隔上。

        rnn_cnn_merge = merge([char_max_pooling, bilstm_d1, bilstm_d2], mode='concat', concat_axis=2)
        #dense = TimeDistributed(Dense(class_label_count))(rnn_cnn_merge)

        # crf
        crf = CRF(class_label_count, sparse_target=False)
        crf_output = crf(rnn_cnn_merge)

        # build model
        model = Model(input=[word_input], output=[crf_output])
        # sgd = optimizers.SGD(lr=0.001, decay=0.05, momentum=0.9, nesterov=False, clipvalue=5)
        # optmr = optimizers.Adam(lr=0.001, beta_1=0.5)
        # model.compile(loss=crf.loss_function,optimizer=sgd,metrics=[crf.accuracy]) # number:2 78%
        # model.compile(loss=crf.loss_function, optimizer='adam', metrics=[crf.accuracy]) # number:3 56%
        model.compile('rmsprop', loss=crf.loss_function, metrics=[crf.accuracy])  # number:1,78%
    elif model_type == 6: #思路3 复杂
        print("model six zuihao:")
        # cnn
        padding_layer = ZeroPadding1D(padding=half_window_size)(word_emb)  # 对1D输入的首尾端（如时域序列）填充0，以控制卷积以后向量的长度
        # 卷积核的数目（即输出的维度）
        conv = Conv1D(nb_filter=50, filter_length=2 * half_window_size + 1, \
                      padding='valid', use_bias=True, bias_initializer='zeros')(padding_layer)
        conv_d = Dropout(DROPOUT_RATE)(conv)
        dense_conv = TimeDistributed(Dense(50))(conv_d)
        char_max_pooling = MaxPooling1D(pool_length=1)(dense_conv)

        # bilstm
        bilstm1 = Bidirectional(LSTM(64, return_sequences=True, activation='tanh'))(word_emb)  # 隐藏元个数
        bilstm_d1 = Dropout(DROPOUT_RATE)(bilstm1)  # DROPOUT_RATE控制输入线性变换的神经元断开比例
        # merge
        total_emb = merge([word_emb, char_max_pooling], mode='concat', concat_axis=2, name='total_emb')
        #emb_droput = Dropout(0.1)(total_emb)

        # bilstm
        bilstm2 = Bidirectional(LSTM(64, return_sequences=True, activation='tanh'))(total_emb)  # 隐藏元个数
        bilstm_d2 = Dropout(DROPOUT_RATE)(bilstm2)  # DROPOUT_RATE控制输入线性变换的神经元断开比例

        rnn_cnn_merge = merge([bilstm_d1, bilstm_d2], mode='concat', concat_axis=2)
        dense = TimeDistributed(Dense(class_label_count))(rnn_cnn_merge)

        # crf
        crf = CRF(class_label_count, sparse_target=False)
        crf_output = crf(dense)

        # build model
        model = Model(input=[word_input], output=[crf_output])
        # sgd = SGD(lr=0.001, decay=0.05, momentum=0.9, nesterov=False, clipvalue=5)
        # optmr = optimizers.Adam(lr=0.001, beta_1=0.5)
        # model.compile(loss=crf.loss_function,optimizer=sgd,metrics=[crf.accuracy]) # number:2 78%
        # model.compile(loss=crf.loss_function, optimizer='adam', metrics=[crf.accuracy]) # number:3 56%
        #model.compile(loss=crf.loss_function, optimizer='adaDelta', metrics=[crf.accuracy]) # number:3 56%
        model.compile('rmsprop', loss=crf.loss_function, metrics=[crf.accuracy])  # number:1,78%
    elif model_type == 4:# 他人思路
        print("model four:")
        # bilstm
        bilstm = Bidirectional(LSTM(128, return_sequences=True))(word_emb)
        bilstm_d = Dropout(0.1)(bilstm)

        # cnn
        padding_layer = ZeroPadding1D(padding=half_window_size)(word_emb)
        conv = Conv1D(nb_filter=100, filter_length=2 * half_window_size + 1, \
                      padding='valid')(padding_layer)
        conv_d = Dropout(0.1)(conv)
        dense_conv = TimeDistributed(Dense(100))(conv_d)

        # merge
        rnn_cnn_merge = merge([bilstm_d, dense_conv], mode='concat', concat_axis=2)
        dense = TimeDistributed(Dense(class_label_count))(word_emb)

        # crf
        crf = CRF(class_label_count, sparse_target=False)
        crf_output = crf(dense)

        # build model
        model = Model(input=[word_input], output=[crf_output])
        model.compile('rmsprop', loss=crf.loss_function, metrics=[crf.accuracy])  # number:1,78%
        #model.compile(loss=crf.loss_function, optimizer='adam', metrics=[crf.accuracy])
    else: #      CNN+Bi-LSTM+CNN+CRF
        
        padding_layer1 = ZeroPadding1D(padding=half_window_size)(word_emb)  # 对1D输入的首尾端（如时域序列）填充0，以控制卷积以后向量的长度
        # 卷积核的数目（即输出的维度）
        conv1 = Conv1D(nb_filter=50, filter_length=2 * half_window_size + 1, \
                      padding='valid', use_bias=True, bias_initializer='zeros')(padding_layer1)
        conv_d1 = Dropout(DROPOUT_RATE)(conv1)
        dense_conv1 = TimeDistributed(Dense(50))(conv_d1)
        char_max_pooling1 = MaxPooling1D(pool_length=1)(dense_conv1)
        # merge
        total_emb = merge([word_emb, char_max_pooling1], mode='concat', concat_axis=2, name='total_emb')

        # bilstm
        bilstm=Bidirectional(LSTM(64,return_sequences=True))(total_emb)
        bilstm_d=Dropout(0.2)(bilstm)

        # cnn
        padding_layer2=ZeroPadding1D(padding=half_window_size)(bilstm_d)
        conv2=Conv1D(nb_filter=50,filter_length=2*half_window_size+1,\
                padding='valid')(padding_layer2)
        conv_d2=Dropout(0.2)(conv2)
        dense_conv2=TimeDistributed(Dense(50))(conv_d2)
        char_max_pooling2 = MaxPooling1D(pool_length=1)(dense_conv2)
        # merge
        #rnn_cnn_merge=merge([bilstm_d,dense_conv],mode='concat',concat_axis=2)
        dense=TimeDistributed(Dense(class_label_count))(char_max_pooling2)

        # crf
        crf=CRF(class_label_count,sparse_target=False)
        crf_output=crf(dense)

        # build model
        model=Model(input=[word_input],output=[crf_output])

        #model.compile(loss=crf.loss_function,optimizer='adam',metrics=[crf.accuracy])
        model.compile('rmsprop', loss=crf.loss_function, metrics=[crf.accuracy])  # number:1,78%

    return model

# 训练模型 保存weights
def process_train(corpus_train_path,corpus_test_path,prf_file,base_model_weight=None,flag=6):

    # 训练语料
    raw_train_file = [corpus_train_path + os.sep + type_path + os.sep + type_file \
                      for type_path in os.listdir(corpus_train_path) \
                      for type_file in os.listdir(corpus_train_path + os.sep + type_path)]

    raw_test_file = [corpus_test_path + os.sep + type_path + os.sep + type_file \
                      for type_path in os.listdir(corpus_test_path) \
                      for type_file in os.listdir(corpus_test_path + os.sep + type_path)]

    if flag == 4:# 0 为padding的label 4tag
        label_2_index = {'Pad': 0, 'B': 1, 'M': 2, 'E': 3, 'S': 4, 'Unk': 5}
        index_2_label = {0: 'Pad', 1: 'B', 2: 'M', 3: 'E', 4: 'S', 5: 'Unk'}
        utils.process_data(raw_train_file, 'train.data')
        utils.process_data(raw_test_file, 'test.data')
    else: # 6tag
        label_2_index = {'Pad': 0, 'B': 1, 'B2': 2, 'B3': 3, 'M': 4, 'E': 5, 'S': 6, 'Unk': 7}
        index_2_label = {0: 'Pad', 1: 'B', 2: 'B2', 3: 'B3', 4: 'M', 5: 'E', 6: 'S', 7: 'Unk'}
        utils.process_dataB(raw_train_file, 'train.data')
        utils.process_dataB(raw_test_file, 'test.data')
    
    class_label_count = len(label_2_index)

    train_documents = utils.create_documents('train.data')
    test_documents = utils.create_documents('test.data')
    # 生成词典
    lexicon, lexicon_reverse = utils.get_lexicon(train_documents+test_documents)
    # 词典内字符个数
    print(len(lexicon), len(lexicon_reverse))

    print(len(test_documents))  # 测试语料划分句子个数
    print(len(train_documents)) # 训练语料划分句子个数

    #embedding_model = gensim.models.Word2Vec.load(r'model_embedding_bakeoff2005-50.m') #size = 50
    #embedding_model = gensim.models.Word2Vec.load(r'model_embedding_bakeoff2005.m') #size = 100
    #embedding_model = gensim.models.Word2Vec.load(r'model_embedding_bakeoff2005-150.m') #size = 150

    embedding_model = gensim.models.Word2Vec.load(r'model_embedding_pku_100.m') #size = 200
    embedding_size = embedding_model.vector_size
    print(embedding_size)

    # 预训练词向量
    embedding_weights = utils.create_embedding(embedding_model, embedding_size, lexicon_reverse)
    print(embedding_weights.shape)
    
    train_data_list, train_label_list, train_index_list=utils.create_matrix(train_documents,lexicon,label_2_index)
    test_data_list, test_label_list, test_index_list=utils.create_matrix(test_documents,lexicon,label_2_index)
    

    print(len(train_data_list), len(train_label_list), len(train_index_list))
    print(len(test_data_list), len(test_label_list), len(test_index_list))
    # print(train_data_list[0])
    # print(train_label_list[0])
    #查看句子长度分布
    #print("查看句子长度分布")
    #visualization.plot_sentence_length(train_data_list+test_data_list,train_label_list+test_label_list)

    max_len = max(map(len, train_data_list))
    print('maxlen:', max_len)
    #if max_len > 64:
    #    max_len = 64
    print('maxlen:', max_len)

    train_data_array, train_label_list_padding = utils.padding_sentences(train_data_list, train_label_list, max_len) #定长 都是二维数据
    test_data_array, test_label_list_padding = utils.padding_sentences(test_data_list, test_label_list, max_len)

    print(train_data_array.shape)
    print(test_data_array.shape)
    #print(train_data_array[0])

    train_label_array = np_utils.to_categorical(train_label_list_padding, class_label_count). \
        reshape((len(train_label_list_padding), len(train_label_list_padding[0]), -1))

    test_label_array = np_utils.to_categorical(test_label_list_padding, class_label_count). \
        reshape((len(test_label_list_padding), len(test_label_list_padding[0]), -1))  # 实现多分类问题  变成三维数据
    # 测试用的句子个数 * 句子长度 * 6
    print(train_label_array.shape)
    print(test_label_array.shape)

    # model
    model = CNN_Bilstm_Crf(max_len, len(lexicon), class_label_count, embedding_weights, embedding_size, model_type)
    print(model.input_shape)
    print(model.output_shape)
    model.summary()
    model_name = 'model_%d.png'%model_type
    #plot_model(model, to_file=model_name, show_shapes=True, show_layer_names=True)

    train_nums = len(train_data_array)  # 对应的train_data_list填充0后就是 train_data_array  填充0后的字在字典中的索引

    train_array, val_array = train_data_array[:int(train_nums * 0.9)], train_data_array[int(train_nums * 0.9):]  # 0.9的行用于训练 0.1的行用于防止过拟合
    train_label, val_label = train_label_array[:int(train_nums * 0.9)], train_label_array[int(train_nums * 0.9):]

    checkpointer = ModelCheckpoint(filepath='train_model_pku_100_m6.hdf5', verbose=1, \
                                   save_best_only=True, monitor='val_loss', mode='auto')

    hist = model.fit(train_array, train_label, batch_size=256, epochs=4, verbose=1,validation_data=(val_array,val_label),callbacks=[checkpointer])

    # save model
    model.save_weights('train_model_pku_100_m6.hdf5')

    print(hist.history['val_loss'])  # 记录下每次的平均损失大小
    best_model_epoch = np.argmin(hist.history['val_loss'])
    print('best_model_epoch:', best_model_epoch)

    # 可视化loss acc
    #visualization.plot_acc_loss(hist)
    #visualization.plot_acc(hist)
    #visualization.plot_loss(hist)

    print(hist.history)

    model.load_weights('train_model_pku_100_m6.hdf5')
    # test_data_array 是测试句子个数 * 句子索引中各字在字典中（填充0后）的长度（填充0后）
    test_y_pred = model.predict(test_data_array,batch_size=256,verbose=1) # 本函数按batch获得输入数据对应的输出，函数的返回值是预测值的numpy array
    print("test_y_pred.shape:")  # 测试句子个数 * 测试句子长度 * 5
    print(test_y_pred.shape) #句子个数 * 句子长度 * 5
	# pred_label是预测出的标签 [0,0,....,1,2,3,1]  句子个数 * 句子长度
    pred_label = np.argmax(test_y_pred,axis=2)  # 预测出的测试句子个数 * 句子长度

    # save lexicon
    pickle.dump([lexicon, lexicon_reverse, max_len, index_2_label], open('lexicon_pku_100_m6.pkl', 'wb'))

    K.clear_session()  # 清除session中的缓存数据
    # 生成输出文档
    # 字典大小 lexicon_reverse: {index:char}
    real_text_list, pred_text_list, real_label_list, pred_label_list = utils.create_pred_text( \
        lexicon_reverse, test_data_array, pred_label, test_label_list_padding, test_index_list, class_label_count)
    # {index：char}, 测试句子个数 * 句子长度（填充0后），# 预测出的测试句子个数 * 句子长度，test_label_list_padding对应标签填充0后的数据，每一行的索引
    # 写进文件
    utils.write_2_file(real_text_list, pred_text_list)
    # score
    F = score.prf_score('real_text.txt', 'pred_text.txt', prf_file,model_type, best_model_epoch,class_label_count)  # 返回平均值
    #print('ave_f:', F)

if __name__ == '__main__':
    corpus_train_path = 'corpus_train'
    corpus_test_path = 'corpus_test'
    prf_file = 'prf_result_max_epoch_10_pku.txt'
    spath = 'corpus_train/train'  # 训练语料
    test_path = 'corpus_test/test'  # 测试语料
    #embedding_model.get_word2vec(spath, test_path)  # 预训练vector
    process_train(corpus_train_path,corpus_test_path,prf_file)