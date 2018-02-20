# -*- coding: utf-8 -*-

import numpy as np
import re
from pandas import DataFrame
import time as time
import matplotlib.pyplot as plt
import math

filename='testSet.txt' #文件目录
def loadDataSet():   #读取数据（这里只有两个特征）
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])   #前面的1，表示方程的常量。比如两个特征X1,X2，共需要三个参数，W1+W2*X1+W3*X2
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat


class LogistReg:
    def __init__(self):
        self.samples = []
        self.label = []
        self.w = 0
        pass
    
    def load_data(self, filename):
        data = DataFrame(columns=['b', 'x0', 'x1', 'label'])
        f = open(filename)
        for line in f:
            sample = line.strip().split('\t')
            data.set_value(len(data), ['b', 'x0', 'x1', 'label'],
                           [1.0, float(sample[0]), float(sample[1]), int(sample[2])])  # 数据存入DataFrame
            pass
        f.close()
        self.data = np.array(data.loc[:, ['b', 'x0', 'x1']])
        self.label = np.array(data['label'])

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def grad_ascent(self):
        start_time = time.time()
        m, n = self.data.shape
        self.w = np.ones((n, 1))
        epochs = 100
        alpha = 0.01
        
        for i in range(epochs):
            h = self.sigmoid(np.dot(self.data, self.w).astype('int64'))
            self.label = self.label.reshape((100, 1))
            error = self.label - h
            self.w = self.w + alpha * np.dot(self.data.T, error)
        end_time = time.time()
        cost_time = end_time - start_time
        print('time: ', cost_time)

    def stoc_grad_ascent(self):
        epochs = 2000
        start_time = time.time()
        m, n = self.data.shape
        self.w = np.ones((n, 1))
        alpha = 0.1
        weight = []
        for i in range(epochs):
            for j in range(m):
                h = self.sigmoid(np.dot(self.data[j], self.w).astype('int64'))
                error = self.label[j] - h
                self.w = self.w + alpha * self.data[j].reshape((3, 1)) * error
                weight.append(self.w.reshape(1, 3).tolist()[0])
                
        self.plot_weight(weight, m*epochs)
        pass
    
    def better_stoc_grad_ascent(self):
        epochs = 50
        start_time = time.time()
        m, n = self.data.shape
        self.w = np.ones((n, 1))
        alpha = 0.01
        weight = []
        iter_num = 5000
        for i in range(iter_num):
                alpha = 1 / (1+i) + 0.01
                j = int(np.random.uniform(0, m))
                h = self.sigmoid(np.dot(self.data[j], self.w).astype('int64'))
                error = self.label[j] - h
                self.w = self.w + alpha * self.data[j].reshape((3, 1)) * error
                weight.append(self.w.reshape(1, 3).tolist()[0])

        self.plot_weight(weight, iter_num)
        pass
                
    def plot_weight(self, weight, iter_num):
        fig = plt.figure()
        weight = np.array(weight)
        m, n = weight.shape
        x = list(range(1, iter_num+1, 1))
        y = weight[:, 0]
        sub_fig1 = fig.add_subplot(311)
        sub_fig1.plot(x, weight[:, 0])
        sub_fig2 = fig.add_subplot(312)
        sub_fig2.plot(x, weight[:, 1])
        sub_fig3 = fig.add_subplot(313)
        sub_fig3.plot(x, weight[:, 2])

    def plot_best_fig(self):
        m, n = self.data.shape
        min_x = min(self.data[:, 1])
        max_x = max(self.data[:, 1])
        xcoord1 = []
        ycoord1 = []
        xcoord2 = []
        ycoord2 = []
        for i in range(m):
            if int(self.label[i]) == 0:
                xcoord1.append(self.data[i, 1])
                ycoord1.append(self.data[i, 2])
            else:
                xcoord2.append(self.data[i, 1])
                ycoord2.append(self.data[i, 2])

        fig = plt.figure()
        sub_fig = fig.add_subplot(111)
        sub_fig.scatter(xcoord1, ycoord1, s=30, c='red', marker='s')
        sub_fig.scatter(xcoord2, ycoord2, s= 30, c = 'green')
        x = np.arange(min_x, max_x, 0.1)
        y =(-float(self.w[0]) -float(self.w[1]) * x) / float(self.w[2])
        sub_fig.plot(x, y)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.show()
        pass
        
        


app = LogistReg()
app.load_data("testSet.txt")
app.grad_ascent()
app.stoc_grad_ascent()
app.better_stoc_grad_ascent()
app.plot_best_fig()

