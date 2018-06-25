#coding:utf-8
# 可视化
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 可视化句子长度分布
def plot_sentence_length(datas,labels):

	df_data = pd.DataFrame({'words': datas, 'tags': labels}, index=range(len(datas)))
	# 　句子长度
	df_data['sentence_len'] = df_data['words'].apply(lambda words: len(words))


	df_data['sentence_len'].hist(bins=100)
	plt.xlim(0, 100)
	plt.xlabel('sentence_length')
	plt.ylabel('sentence_num')
	plt.title('Distribution of the Length of Sentence')
	plt.show()

# 可视化acc lost
def plot_acc_loss(hist):
	plt.plot(range(len(hist.history['acc'])),hist.history['acc'],marker='o',label='acc')
	plt.plot(range(len(hist.history['val_acc'])),hist.history['val_acc'],marker='*',label='val_acc')
	plt.plot(range(len(hist.history['loss'])), hist.history['loss'], marker='x', label='loss')
	plt.plot(range(len(hist.history['val_loss'])), hist.history['val_loss'], marker='+', label='val_loss')
	plt.legend()
	plt.xlabel('iters')
	plt.ylabel('acc-loss')
	plt.title('Acc/Loss & val_Acc/Loss')
	plt.show()
# 可视化acc
def plot_acc(hist):
        plt.plot(range(len(hist.history['acc'])),hist.history['acc'],marker='o',label='acc')
        plt.plot(range(len(hist.history['val_acc'])),hist.history['val_acc'],marker='*',label='val_acc')
        plt.legend()
        plt.xlabel('iters')
        plt.ylabel('acc')
        plt.title('Acc & val_Acc')
        plt.show()
# 可视化lost
def plot_loss(hist):
        plt.plot(range(len(hist.history['loss'])), hist.history['loss'], marker='x', label='loss')
        plt.plot(range(len(hist.history['val_loss'])), hist.history['val_loss'], marker='+', label='val_loss')
        plt.legend()
        plt.xlabel('iters')
        plt.ylabel('loss')
        plt.title('Loss & val_Loss')
        plt.show()

'''
**Markers**

        =============    ===============================
        character        description
        =============    ===============================
        ``'.'``          point marker
        ``','``          pixel marker
        ``'o'``          circle marker
        ``'v'``          triangle_down marker
        ``'^'``          triangle_up marker
        ``'<'``          triangle_left marker
        ``'>'``          triangle_right marker
        ``'1'``          tri_down marker
        ``'2'``          tri_up marker
        ``'3'``          tri_left marker
        ``'4'``          tri_right marker
        ``'s'``          square marker
        ``'p'``          pentagon marker
        ``'*'``          star marker
        ``'h'``          hexagon1 marker
        ``'H'``          hexagon2 marker
        ``'+'``          plus marker
        ``'x'``          x marker
        ``'D'``          diamond marker
        ``'d'``          thin_diamond marker
        ``'|'``          vline marker
        ``'_'``          hline marker
        =============    ===============================
        
        **Line Styles**

        =============    ===============================
        character        description
        =============    ===============================
        ``'-'``          solid line style
        ``'--'``         dashed line style
        ``'-.'``         dash-dot line style
        ``':'``          dotted line style
        =============    ===============================
        
        **Colors**

        The following color abbreviations are supported:

        =============    ===============================
        character        color
        =============    ===============================
        ``'b'``          blue
        ``'g'``          green
        ``'r'``          red
        ``'c'``          cyan
        ``'m'``          magenta
        ``'y'``          yellow
        ``'k'``          black
        ``'w'``          white
        =============    ===============================
        
'''