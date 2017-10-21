#coding=utf-8

'''
Created on Monday Oct 18 2017
K-means clustering
@Author:Xu Xiaoliang

function:implement K means clustering algorithm
'''

import numpy as np

class Kmeans(object):
    def __init__(self, k, numMaxIte, dataSet):
        self.k = k
        self.numMaxIte = numMaxIte
        self.numPoints, self.numFeatures = dataSet.shape
        self.dataSet = np.zeros((self.numPoints, self.numFeatures+1))
        self.dataSet[:, :-1] = dataSet

    def kmeans(self):
        '''
        1.初始化分类
        2.迭代：
            2.1.更新中心点
            2.2.根据中心点重新分类
            迭代结束条件：（1）最大迭代次数；（2）中心点未变化
        '''
        self.initClassification()
        oldCentroids = None
        newCentroids = np.zeros((self.k, self.numFeatures + 1))
        for i in range(0, self.numMaxIte):
            self.updateCentroids(newCentroids)
            if np.array_equal(oldCentroids, newCentroids):
                break
            self.updateClassification(newCentroids)
            oldCentroids = np.copy(newCentroids)

        print(self.dataSet)
        return self.dataSet

    def initClassification(self):
        for i in range(0, self.numPoints):
            self.dataSet[i, -1] = np.random.randint(1, self.k + 1)

    def updateCentroids(self, centroids):
        #centroids = np.zeros((self.k, self.numFeatures+1))
        for i in range(1, self.k + 1):
            # 1.取出同类数据集
            oneCluster = self.dataSet[self.dataSet[:, -1] == i, :-1]
            # 2.按行取均值
            centroids[i-1, :-1] = np.mean(oneCluster, axis=0)
            centroids[i-1, -1]  = i

        return centroids

    def updateClassification(self, centroids):

        for i in range(0, self.numPoints):
            minDis = np.linalg.norm(self.dataSet[i, :-1] - centroids[0, :-1])
            label = 1
            for j in range(2, self.k + 1):
                dis = np.linalg.norm(self.dataSet[i, :-1] - centroids[j-1, :-1])
                if minDis > dis:
                    minDis = dis
                    label = j
            self.dataSet[i, -1] = label
