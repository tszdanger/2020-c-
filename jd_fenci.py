#抓取网页内容用的程序包
import json
import requests

#PyTorch用的包
import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Variable

# 自然语言处理相关的包
import re #正则表达式的包
import jieba #结巴分词包
from collections import Counter #搜集器，可以让统计词频更简单

#绘图、计算用的程序包
import matplotlib.pyplot as plt
import numpy as np


# def get_comments(url):
#     comments = []
#     # 打开指定页面
#     resp = requests.get(url)
#     resp.encoding = 'gbk'
#
#     # 如果200秒没有打开则失败
#     if resp.status_code != 200:
#         return []
#
#     # 获得内容
#     content = resp.text
#     if content:
#         # 获得（）括号中的内容
#         ind = content.find('(')
#         s1 = content[ind + 1:-2]
#         try:
#             # 尝试利用jason接口来读取内容，并做jason的解析
#             js = json.loads(s1)
#             # 提取出comments字段的内容
#             comment_infos = js['comments']
#         except:
#             print('error')
#             return ([])
#
#         # 对每一条评论进行内容部分的抽取
#         for comment_info in comment_infos:
#             comment_content = comment_info['content']
#             str1 = comment_content + '\n'
#             comments.append(str1)
#     return comments
#
#
# good_comments = []
#
# good_comment_url_templates = [
#     'https://club.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98vv8914&productId=10359162198&score=3&sortType=5&page={}&pageSize=10&isShadowSku=0',
#     'https://club.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98vv73&productId=10968941641&score=3&sortType=5&page={}&pageSize=10&isShadowSku=0',
#     'https://club.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98vv4653&productId=10335204102&score=3&sortType=5&page={}&pageSize=10&isShadowSku=0',
#     'https://club.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98vv1&productId=1269194114&score=3&sortType=5&page={}&pageSize=10&isShadowSku=0',
#     'https://club.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98vv2777&productId=1409704820&score=3&sortType=5&page={}&pageSize=10&isShadowSku=0',
#     'https://club.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98vv174&productId=10103790891&score=3&sortType=5&page={}&pageSize=10&isShadowSku=0',
#     'https://club.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98vv9447&productId=1708318938&score=3&sortType=5&page={}&pageSize=10&isShadowSku=0',
#     'https://club.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98vv111&productId=10849803616&score=3&sortType=5&page={}&pageSize=10&isShadowSku=0'
# ]
#
# # 对上述网址进行循环，并模拟翻页100次
# j=0
# for good_comment_url_template in good_comment_url_templates:
#     for i in range(100):
#         url = good_comment_url_template.format(i)
#         good_comments += get_comments(url)
#         print('第{}条纪录，总文本长度{}'.format(j, len(good_comments)))
#         j += 1
# #将结果存储到good.txt文件中
# fw = open('data/good.txt', 'w')
# fw.writelines(good_comments)




# since the web has already been cancelled, we use the data directly


good_file = 'data/good.txt'
bad_file = 'data/bad.txt'

# lets get rid of the comma
def filter_punc(sentence):
    sentence = re.sub("[\s+\.\!\/_,$%^*(+\"\'“”《》?“]+|[+——！，。？、~@#￥%……&*（）：]+", "", sentence)
    return sentence

def Prepare_data(good_file,bad_file,is_filter = True):
    all_words = []
    pos_sentences = []
    neg_sentences = []
    with open(good_file,'r',encoding='utf-8') as fr:
        for idx,line in enumerate(fr):
#这种情况在你遍历文件时想在错误消息中使用行号定位时候非常有用：

            if is_filter:
                line = filter_punc(line)
            words = jieba.lcut(line)
            if len(words) > 0:
                all_words += words
                pos_sentences.append(words)


    print('{0} 包含 {1} 行, {2} 个词.'.format(good_file, idx + 1, len(all_words)))
    count = len(all_words)

    with open(bad_file, 'r',encoding='UTF-8') as fr:
        for idx, line in enumerate(fr):
            if is_filter:
                line = filter_punc(line)
            words = jieba.lcut(line)
            if len(words) > 0:
                all_words += words
                neg_sentences.append(words)
    print('{0} 包含 {1} 行, {2} 个词.'.format(bad_file, idx + 1, len(all_words) - count))
    # 建立词典，diction的每一项为{w:[id, 单词出现次数]}
    diction = {}
    cnt = Counter(all_words)
    #Counter自带的统计
    # 转为(elem, cnt)格式的列表
    for word, freq in cnt.items():
        diction[word] = [len(diction),freq]
    print('字典大小:{}'.format(len(diction)))
    return (pos_sentences,neg_sentences,diction)


pos_sentences, neg_sentences, diction = Prepare_data(good_file, bad_file, True)

st = sorted([(v[1], w) for w, v in diction.items()])

def word2index(word,diction):
    if word in diction:
        value = diction[word][0]
    else:
        value = -1
    return value

def index2word(index,diction):
    for w,v in diction.items():
        if v[0]==index:
            return w

    return None



# by doing all the things above, we get the diction perfectly
# 输入一个句子和相应的词典，得到这个句子的向量化表示
# 向量的尺寸为词典中词汇的个数，i位置上面的数值为第i个单词在sentence中出现的频率
# 例如字典有a,b,c,d,  sen = a d c a
# 传入 1,4,3,1 vector = [0.5,0,0.25,0.25]结束

def sentence2vec(sentence,dictionary):
    vector = np.zeros(len(dictionary))
    for l in sentence:
        vector[l]+=1
    return (1.0*vector/len(sentence))


dataset = []
labels = []   #标签
sentences =[]  # 原始的句子，分过词的
for sentence in pos_sentences:
    new_sentence = []
    for l in sentence:
        if l in diction:
            new_sentence.append(word2index(l,diction))
    dataset.append(sentence2vec(new_sentence,diction))
    labels.append(0)#正标签为0
    sentences.append(sentence)



for sentence in neg_sentences:
    new_sentence = []
    for l in sentence:
        if l in diction:
            new_sentence.append(word2index(l, diction))
    dataset.append(sentence2vec(new_sentence, diction))
    labels.append(1) #负标签为1
    sentences.append(sentence)

#打乱顺序重新生成数据集
indices = np.random.permutation(len(dataset))
#返回一个随机全排列
dataset = [dataset[i] for i in indices]
labels = [labels[i] for i in indices]
sentences = [sentences[i] for i in indices]

test_size = len(dataset) // 10
train_data = dataset[2 * test_size :]
train_label = labels[2 * test_size :]

valid_data = dataset[: test_size]
valid_label = labels[: test_size]

test_data = dataset[test_size : 2 * test_size]
test_label = labels[test_size : 2 * test_size]

# ok lets build the neu

# 输入层7133 / 隐含层10/输出层2

model = torch.nn.Sequential(
    nn.Linear(len(diction),10),
    nn.ReLU(),
    nn.Linear(10,2),
    nn.LogSoftmax(dim=1),
)


# 为什么要对列求？因为一行是一个数据有7k维，选里面最大的

def rightness(predictions,labels):
    pred = torch.max(predictions.data,dim=1)[1]
    # this equals to     pred = torch.max(predictions.data,dim=1).indices
    rights= pred.eq(labels.data.view_as(pred)).sum()
    return rights,len(labels)

cost = torch.nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)
records =[]

losses = []
for epoch in range(10):
    for i,data in enumerate(zip(train_data,train_label)):
        x,y =data
        x = Variable(torch.FloatTensor(x).view(1,-1))
        #给train_data增加1维
        # x -> (batch_size=1,len_dictionary)
        # y = Variable(torch.LongTensor(np.array([[y]])))
        y = Variable(torch.tensor(np.array([y]), dtype=torch.long))

        # y -> (1,len(y))
        # y = Variable(torch.LongTensor(y).view(1,-1))

        optimizer.zero_grad()
        predict = model(x)
        # predict的维度和内容？ (batch_size,num_classes)
        loss = cost(predict,y)
        losses.append(loss.data.numpy())
        loss.backward()
        optimizer.step()

        if i%3000 == 0 :
            val_losses = []
            rights = []
            # test them in the val
            for j,val in  enumerate(zip(valid_data,valid_label)):
                x,y = val
                x = Variable(torch.FloatTensor(x).view(1,-1))
                # y = Variable(torch.LongTensor(np.array([y])))
                y = Variable(torch.tensor(np.array([y]), dtype=torch.long))
                # y = Variable(torch.LongTensor(y).view(1, -1))

                predict = model(x)
                right = rightness(predict,y)
                rights.append(right)
                loss = cost(predict, y)

                val_losses.append(loss.data.numpy())

            right_ratio = 1.0 * np.sum([i[0] for i in rights]) / np.sum([i[1] for i in rights])
            print('第{}轮，训练损失：{:.2f}, 校验损失：{:.2f}, 校验准确率: {:.2f}'.format(epoch, np.mean(losses),
                                                                        np.mean(val_losses), right_ratio))
            records.append([np.mean(losses), np.mean(val_losses), right_ratio])

# a = [i[0] for i in records]
# b = [i[1] for i in records]
# c = [i[2] for i in records]
# plt.plot(a, label = 'Train Loss')
# plt.plot(b, label = 'Valid Loss')
# plt.plot(c, label = 'Valid Accuracy')
# plt.xlabel('Steps')
# plt.ylabel('Loss & Accuracy')
# plt.legend()
# plt.show()






vals = [] #记录准确率所用列表

#对测试数据集进行循环
for data, target in zip(test_data, test_label):
    data, target = torch.tensor(data, dtype = torch.float).view(1,-1), torch.tensor(np.array([target]), dtype = torch.long)
    output = model(data) #将特征数据喂入网络，得到分类的输出
    val = rightness(output, target) #获得正确样本数以及总样本数
    vals.append(val) #记录结果

#计算准确率
rights = (sum([tup[0] for tup in vals]), sum([tup[1] for tup in vals]))
right_rate = 1.0 * rights[0].data.numpy() / rights[1]
right_rate



plt.figure(figsize = (10, 7))
for i in range(model[2].weight.size()[0]):
    #if i == 1:
        weights = model[2].weight[i].data.numpy()
        plt.plot(weights, 'o-', label = i)
plt.legend()
plt.xlabel('Neuron in Hidden Layer')
plt.ylabel('Weights')

plt.figure(figsize = (10, 7))
for i in range(model[0].weight.size()[0]):
    #if i == 1:
        weights = model[0].weight[i].data.numpy()
        plt.plot(weights, alpha = 0.5, label = i)
plt.legend()
plt.xlabel('Neuron in Input Layer')
plt.ylabel('Weights')


plt.show()