#coding:utf-8
import os
import codecs
import codecs
from keras.preprocessing import sequence
import numpy as np
import random

class Documents():
    def __init__(self, chars, labels, index):
        self.chars = chars
        self.labels = labels
        self.index = index

# 读取数据
def create_documents(file_name):
    documents = []
    chars, labels = [], []

    with codecs.open(file_name, 'r', 'utf-8') as f:
        index = 0
        for line in f:

            line = line.strip()

            if len(line) == 0:
                if len(chars) != 0:
                    documents.append(Documents(chars, labels, index))
                    chars = []
                    labels = []
                index += 1

            else:
                pieces = line.strip().split()
                chars.append(pieces[0])
                labels.append(pieces[1])

                if pieces[0] in ['。','，','；','?','!']:#'，','。','！','；','？'
                    documents.append(Documents(chars, labels, index))
                    chars = []
                    labels = []

        if len(chars) != 0:
            documents.append(Documents(chars, labels, index))
            chars, labels = [], []
    return documents #documents[i].chars = ['河'，'北'，'工'，'业','大'，'学','!']
                     #documents[i].labless = ['B'，'E'，'B'，'E','B'，'E','S']
                     #documents[i].index = 句子行数 i即是句子的索引数目
# 生成词典 lexicon{char:index} {'我'，1} {'是'，2} lexicon_reverse{index:char} {1，'我'} {2，'是'}
def get_lexicon(all_documents):
    chars = {}
    for doc in all_documents:
        for char in doc.chars:
            chars[char] = chars.get(char, 0) + 1

    sorted_chars = sorted(chars.items(), key=lambda x: x[1], reverse=True)

    # 下标从1开始 0用来补长
    lexicon = dict([(item[0], index + 1) for index, item in enumerate(sorted_chars)])
    lexicon_reverse = dict([(index + 1, item[0]) for index, item in enumerate(sorted_chars)])
    print("lexicon：cws",len(lexicon))
    return lexicon, lexicon_reverse


def create_embedding(embedding_model, embedding_size, lexicon_reverse):  #{index,char}
    embedding_weights = np.zeros((len(lexicon_reverse) + 2, embedding_size))

    for i in range(len(lexicon_reverse)):
        embedding_weights[i + 1] = embedding_model[lexicon_reverse[i + 1]] #根据字的索引获取当前字 根据当前字去预训练向量中获取该向量 赋给新权重向量

    embedding_weights[-1] = np.random.uniform(-1, 1, embedding_size)

    return embedding_weights


def create_matrix(documents, lexicon, label_2_index):
    data_list = []
    label_list = []
    index_list = []
    for doc in documents:
        data_tmp = []
        label_tmp = []

        for char, label in zip(doc.chars, doc.labels):
            data_tmp.append(lexicon[char])
            label_tmp.append(label_2_index[label])

        data_list.append(data_tmp)
        label_list.append(label_tmp)
        index_list.append(doc.index)

    return data_list, label_list, index_list #data_list二维数组m*n m是句子个数 内容是字在字典中的索引 n是句子长度 是个不定值 lable_list也是二维数组其是对应BMES的索引（12345）index_list是一维数组 是句子个数

def padding_sentences(data_list, label_list, max_len):       #向前填充0 为定长度max_len
    padding_data_list = sequence.pad_sequences(data_list, maxlen=max_len, dtype='int32',
                                               padding='pre', truncating='post', value=0.)
    padding_label_list = sequence.pad_sequences(label_list, maxlen=max_len, dtype='int32',
                                                padding='pre', truncating='post', value=0.)
    print("Test:padding_sentences()")

    return padding_data_list, np.array(padding_label_list)

def process_data(s_file_list, t_file):  # 产生4tag标签文件BMES
    ft = codecs.open(t_file, 'w', 'utf-8')
    k = 0
    for s_file in s_file_list:
        with codecs.open(s_file, 'r', 'utf-8') as fs:
            lines = fs.readlines()
            # print(len(lines))
            for line in lines:
                word_list = line.strip().split()
                for word in word_list:
                    if len(word) == 1:
                        ft.write(word + '\tS\n')
                    else:
                        ft.write(word[0] + '\tB\n')
                        for w in word[1:-1]:
                            ft.write(w + '\tM\n')
                        ft.write(word[-1] + '\tE\n')
                ft.write('\n')
    ft.close()

def process_dataM(s_file_list, t_file):  # 产生6tag标签文件BMM1M2MMMES
    ft = codecs.open(t_file, 'w', 'utf-8')
    k = 0
    for s_file in s_file_list:
        with codecs.open(s_file, 'r', 'utf-8') as fs:
            lines = fs.readlines()
            # print(len(lines))
            for line in lines:
                word_list = line.strip().split()
                for word in word_list:
                    if len(word) == 1:
                        ft.write(word + '\tS\n')
                    elif len(word) == 2:
                        ft.write(word[0] + '\tB\n')
                        ft.write(word[1] + '\tE\n')
                    elif len(word) == 3:
                        ft.write(word[0] + '\tB\n')
                        ft.write(word[1] + '\tM\n')
                        ft.write(word[2] + '\tE\n')
                    elif len(word) == 4:
                        ft.write(word[0] + '\tB\n')
                        ft.write(word[1] + '\tM1\n')
                        ft.write(word[2] + '\tM\n')
                        ft.write(word[3] + '\tE\n')
                    elif len(word) == 5:
                        ft.write(word[0] + '\tB\n')
                        ft.write(word[1] + '\tM1\n')
                        ft.write(word[2] + '\tM2\n')
                        ft.write(word[3] + '\tM\n')
                        ft.write(word[4] + '\tE\n')
                    else: #>6
                        ft.write(word[0] + '\tB\n')
                        ft.write(word[1] + '\tM1\n')
                        ft.write(word[2] + '\tM2\n')
                        for w in word[3:-1]:
                            ft.write(w + '\tM\n')
                        ft.write(word[-1] + '\tE\n')
                ft.write('\n')
    ft.close()

def process_dataB(s_file_list, t_file):  # 产生6tag标签文件BB2B3MMMMMES
    ft = codecs.open(t_file, 'w', 'utf-8')
    k = 0
    for s_file in s_file_list:
        with codecs.open(s_file, 'r', 'utf-8') as fs:
            lines = fs.readlines()
            # print(len(lines))
            for line in lines:
                word_list = line.strip().split()
                for word in word_list:
                    if len(word) == 1:
                        ft.write(word + '\tS\n')
                    elif len(word) == 2:
                        ft.write(word[0] + '\tB\n')
                        ft.write(word[1] + '\tE\n')
                    elif len(word) == 3:
                        ft.write(word[0] + '\tB\n')
                        ft.write(word[1] + '\tB2\n')
                        ft.write(word[2] + '\tE\n')
                    elif len(word) == 4:
                        ft.write(word[0] + '\tB\n')
                        ft.write(word[1] + '\tB2\n')
                        ft.write(word[2] + '\tB3\n')
                        ft.write(word[3] + '\tE\n')
                    elif len(word) == 5:
                        ft.write(word[0] + '\tB\n')
                        ft.write(word[1] + '\tB2\n')
                        ft.write(word[2] + '\tB3\n')
                        ft.write(word[3] + '\tM\n')
                        ft.write(word[4] + '\tE\n')
                    else: #>6
                        ft.write(word[0] + '\tB\n')
                        ft.write(word[1] + '\tB2\n')
                        ft.write(word[2] + '\tB3\n')
                        for w in word[3:-1]:
                            ft.write(w + '\tM\n')
                        ft.write(word[-1] + '\tE\n')
                ft.write('\n')
    ft.close()
# {index：char}, 测试句子个数 * 句子长度（填充0后），# 预测出的测试句子个数 * 句子中字标签索引的长度，test_label_list_padding对应标签填充0后的数据，每一行的索引
def create_pred_text(lexicon_reverse,test_data_array,pred_label,test_label_list_padding,test_index_list,class_label_count):
	real_text_list = []
	pred_text_list = []
	real_label_list = []
	pred_label_list = []

	real_text = ''
	pred_text = ''

	non_pad_real = []
	non_pad_pred = []
	non_pad_text = []

	sindex = 0

	for pred, real, index, text in zip(pred_label, test_label_list_padding, test_index_list,
									   test_data_array):  # zip()打包为元组的列表
		start_index = np.argwhere(real > 0)[0][0]  # 条件查找，返回满足条件的数组元素的索引值：np.where(条件)

		# 条件查找，返回下标：np.argwhere(条件)
		# print(start_index)

		if index != sindex:
			real_text_list.append(real_text)
			pred_text_list.append(pred_text)

			real_label_list.append(non_pad_real)
			pred_label_list.append(non_pad_pred)

			real_text = ''
			pred_text = ''

			non_pad_real = []
			non_pad_pred = []
			non_pad_text = []

		for r, p, t in zip(real[start_index:], pred[start_index:], text[start_index:]):
			if class_label_count == 6:#4-tag
				if r in [0, 3, 4, 5]:
					real_text += (lexicon_reverse[t] + ' ')
				else:
					real_text += lexicon_reverse[t]
				if p in [0, 3, 4, 5]:
					pred_text += (lexicon_reverse[t] + ' ')
				else:
					pred_text += lexicon_reverse[t]
			else: #6-tag
				if r in [0, 5, 6, 7]:
					real_text += (lexicon_reverse[t] + ' ')
				else:
					real_text += lexicon_reverse[t]
				if p in [0, 5, 6, 7]:
					pred_text += (lexicon_reverse[t] + ' ')
				else:
					pred_text += lexicon_reverse[t]

		non_pad_real += list(real[start_index:])
		non_pad_pred += list(pred[start_index:])
		non_pad_text += list(text[start_index:])

		sindex = index

	if pred_text != '':
		real_text_list.append(real_text)
		pred_text_list.append(pred_text)

		real_label_list.append(non_pad_real)
		pred_label_list.append(non_pad_pred)

		real_text = ''
		pred_text = ''

		non_pad_real = []
		non_pad_pred = []
		non_pad_text = []

	return real_text_list, pred_text_list, real_label_list, pred_label_list

def write_2_file(real_text_list,pred_text_list):
	f = codecs.open('real_text.txt', 'w', 'utf-8')
	f.write('\n'.join(real_text_list))
	f.close()
	f=codecs.open('pred_text.txt','w','utf-8')
	f.write('\n'.join(pred_text_list))
	f.close()
	print("Test:write_2_file()")

def file_to_utf8(spath,filenames,save_name):
    contents=[]
    for fname in filenames:
        f=codecs.open(spath+os.sep+fname,'r','utf-8')
        lines=f.readlines()
        f.close()
        contents+=lines

    f=codecs.open(save_name,'w','utf-8')
    f.write(''.join(contents))
    f.close()