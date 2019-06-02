
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-

import numpy as np

class LogisticRegression(object):

    def __init__(self, learning_rate=0.1, max_iter=100, seed=None):
        self.seed = seed        #随机数种子
        self.lr = learning_rate # 学习率
        self.max_iter = max_iter  #最大迭代次数

    def fit(self, x, y):
        np.random.seed(self.seed)
        self.w = np.random.normal(loc=0.0, scale=1.0, size=x.shape[1])  # 生成标准正态分布
        self.b = np.random.normal(loc=0.0, scale=1.0)
        self.x = x
        self.y = y
        for i in range(self.max_iter):
            self._update_step()
            print('loss: \t{}'.format(self.loss()))
            print('score: \t{}'.format(self.score()))
            print('w: \t{}'.format(self.w))
            print('b: \t{}'.format(self.b))

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z)) #  sigmod函数

    def _f(self, x, w, b):
        z = x.dot(w) + b        # 激活函数输出
        return self._sigmoid(z)

    def predict_proba(self, x=None):
        #   预测概率值
        if x is None:
            x = self.x
        y_pred = self._f(x, self.w, self.b) # 仅是概率数值
        return y_pred

    def predict(self, x=None):
        #预测类别
        if x is None:
            x = self.x
        y_pred_proba = self._f(x, self.w, self.b) # 仅是概率数值
        y_pred = np.array([0 if y_pred_proba[i] < 0.5 else 1 for i in range(len(y_pred_proba))])    # 阈值为0.5，预测 0 or 1
        return y_pred

    def score(self, y_true=None, y_pred=None):
        if y_true is None or y_pred is None:
            y_true = self.y #真实值
            y_pred = self.predict() #预测值
        acc = np.mean([1 if y_true[i] == y_pred[i] else 0 for i in range(len(y_true))]) #精度
        return acc

    def loss(self, y_true=None, y_pred_proba=None):
        #计算损失函数
        if y_true is None or y_pred_proba is None:
            y_true = self.y #真实值
            y_pred_proba = self.predict_proba() #概率值
        return np.mean(-1.0 * (y_true * np.log(y_pred_proba) + (1.0 - y_true) * np.log(1.0 - y_pred_proba)))    #交叉熵损失函数

    def _calc_gradient(self):
        # 计算梯度，批量梯度下降
        y_pred = self.predict()
        d_w = (y_pred - self.y).dot(self.x) / len(self.y)
        d_b = np.mean(y_pred - self.y)
        return d_w, d_b

    def _update_step(self):
        #更新参数
        d_w, d_b = self._calc_gradient()
        self.w = self.w - self.lr * d_w
        self.b = self.b - self.lr * d_b
        return self.w, self.b


# In[2]:


# -*- coding: utf-8 -*-

import numpy as np

def generate_data(seed):
    np.random.seed(seed)
    data_size_1 = 300   # 300个正样本
    x1_1 = np.random.normal(loc=5.0, scale=1.0, size=data_size_1)   # 两个特征
    x2_1 = np.random.normal(loc=4.0, scale=1.0, size=data_size_1)
    y_1 = [0 for _ in range(data_size_1)]
    data_size_2 = 400   # 400个负样本
    x1_2 = np.random.normal(loc=10.0, scale=2.0, size=data_size_2)
    x2_2 = np.random.normal(loc=8.0, scale=2.0, size=data_size_2)
    y_2 = [1 for _ in range(data_size_2)]
    x1 = np.concatenate((x1_1, x1_2), axis=0)
    x2 = np.concatenate((x2_1, x2_2), axis=0)
    
    
    x = np.hstack((x1.reshape(-1,1), x2.reshape(-1,1)))  # reshape成一列，在水平方向上拼接  等价于：np.concatenate(tup, axis=1)
    y = np.concatenate((y_1, y_2), axis=0)  # 把 axis=0 这一维度的shape相加
    data_size_all = data_size_1+data_size_2
    shuffled_index = np.random.permutation(data_size_all)   # 随机打乱原来的元素顺序，返回一个新的打乱顺序的数组，并不改变原来的数组
    x = x[shuffled_index]
    y = y[shuffled_index]
    return x, y
def train_test_split(x, y):
    # 划分训练集和测试集
    split_index = int(len(y)*0.7)
    x_train = x[:split_index]
    y_train = y[:split_index]
    x_test = x[split_index:]
    y_test = y[split_index:]
    return x_train, y_train, x_test, y_test


# In[4]:


# -*- coding: utf-8 -*-

import numpy as np

import matplotlib.pyplot as plt

#import data_helper

#from logistic_regression import *

# data generation

x, y = generate_data(seed=272)
x_train, y_train, x_test, y_test = train_test_split(x, y)

print('x_train: \t{}'.format(x_train.shape))
print('y_train: \t{}'.format(y_train.shape))   
print('x_test: \t{}'.format(x_test.shape))
print('y_test: \t{}'.format(y_test.shape))


# visualize data
plt.scatter(x_train[:,0], x_train[:,1], c=y_train, marker='.')
plt.show()
plt.scatter(x_test[:,0], x_test[:,1], c=y_test, marker='.')
plt.show()


# data normalization 归一化
x_train = (x_train - np.min(x_train, axis=0)) / (np.max(x_train, axis=0) - np.min(x_train, axis=0))
x_test = (x_test - np.min(x_test, axis=0)) / (np.max(x_test, axis=0) - np.min(x_test, axis=0))

# Logistic regression classifier
clf = LogisticRegression(learning_rate=0.1, max_iter=500, seed=272)
clf.fit(x_train, y_train)

# plot the result
split_boundary_func = lambda x: (-clf.b - clf.w[0] * x) / clf.w[1]
xx = np.arange(0.1, 0.6, 0.1)
cValue = ['g','b'] 
plt.scatter(x_train[:,0], x_train[:,1], c=[cValue[i] for i in y_train], marker='o')
plt.plot(xx, split_boundary_func(xx), c='red')
plt.show()

# loss on test set
y_test_pred = clf.predict(x_test)
y_test_pred_proba = clf.predict_proba(x_test)
print(clf.score(y_test, y_test_pred))
print(clf.loss(y_test, y_test_pred_proba))
# print(y_test_pred_proba)

