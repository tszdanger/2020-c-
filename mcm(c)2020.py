# 加载必要的程序包
# PyTorch的程序包
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 数值运算和绘图的程序包
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


# 加载机器学习的软件包
from sklearn.decomposition import PCA

#加载Word2Vec的软件包
import gensim as gensim
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.word2vec import LineSentence

#加载‘结巴’中文分词软件包

import jieba

#加载正则表达式处理的包
import re
import csv

words = []
lines = []
with open('mcm/hair_dryer.csv','r',encoding='ISO-8859-1') as f:

    reader = csv.reader(f)

    for i in reader:

        review_headline = (i[12])
        review_body = i[13]
        # temp = jieba.lcut(review_headline+' '+review_body)
        # words = []
        # for i in temp:
            # 过滤掉所有的标点符号
            # i = re.sub("[\s+\.\!\/_,$%^*(+\"\'””《》]+|[+——！，。？、~@#￥%……&*（）：；‘]+", "", i)
        if len(review_headline) > 1:
            temp = jieba.lcut(review_headline)
            temp1 = []
            for i in temp:
                i = re.sub("[\s+\.\!\/_,$%^*(+\"\'””《》]+|[+——！，。？、~@#￥%……&*（）：；‘]+", "", i)
                if(len(i)>1):
                    temp1.append(i)
            words.append(temp1)
        if len(review_body) > 1:
            temp = jieba.lcut(review_body)
            temp1 = []
            for i in temp:
                i = re.sub("[\s+\.\!\/_,$%^*(+\"\'””《》]+|[+——！，。？、~@#￥%……&*（）：；‘]+", "", i)
                if (len(i) > 1):
                    temp1.append(i)
            words.append(temp1)


model = Word2Vec(words, size = 25, window = 3 , min_count = 3)


res1 = model.wv.most_similar('easy', topn = 20)
print(res1)
rawWordVec = []
word2ind = {}
for i, w in enumerate(model.wv.vocab):
    rawWordVec.append(model[w])
    word2ind[w] = i
rawWordVec = np.array(rawWordVec)
X_reduced = PCA(n_components=2).fit_transform(rawWordVec)


fig = plt.figure(figsize = (15, 10))
ax = fig.gca()
ax.set_facecolor('white')
ax.plot(X_reduced[:, 0], X_reduced[:, 1], '.', markersize = 1, alpha = 0.3, color = 'black')


words1 = ['good','bad','like']
for w in words1:
    if w in word2ind:
        ind = word2ind[w]
        xy = X_reduced[ind]
        plt.plot(xy[0], xy[1], '.', alpha =1, color = 'green')
        plt.text(xy[0], xy[1], w, alpha = 1, color = 'blue')

fig = plt.figure(figsize = (30, 15))
ax = fig.gca()
ax.set_facecolor('black')
ax.plot(X_reduced[:, 0], X_reduced[:, 1], '.', markersize = 1, alpha = 0.1, color = 'white')

ax.set_xlim([-12,12])
ax.set_ylim([-10,20])


words = {'good', 'great', 'use', 'works', 'bad', 'same'}
all_words = []
for w in words:
    lst = model.wv.most_similar(w)
    wds = [i[0] for i in lst]
    metrics = [i[1] for i in lst]
    wds = np.append(wds, w)
    all_words.append(wds)


zhfont1 = matplotlib.font_manager.FontProperties(size=16)
colors = ['red', 'yellow', 'orange', 'green', 'cyan', 'cyan']
for num, wds in enumerate(all_words):
    for w in wds:
        if w in word2ind:
            ind = word2ind[w]
            xy = X_reduced[ind]
            plt.plot(xy[0], xy[1], '.', alpha =1, color = colors[num])
            plt.text(xy[0], xy[1], w, fontproperties = zhfont1, alpha = 1, color = colors[num])
plt.savefig('88.png',dpi =600)



plt.show()



