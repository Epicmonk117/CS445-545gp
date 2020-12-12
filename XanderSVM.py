#Xander's SVM for CS445/545

import os
import time
import numpy as np
from keras.datasets import cifar10
from sklearn import svm
import matplotlib.pyplot as plt
import matplotlib as mpl

class mysvm:
    def __init__(self, krnl):
        self.baseDir = os.path.dirname(os.path.abspath('__file__')) + '/'
        self.classesName = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        (self.TrainData, self.TrainLabels), (self.TestData, self.TestLabels) = cifar10.load_data()
        self.TrainData = self.TrainData[:49000, :].astype(np.float)
        self.TrainLabels = np.squeeze(self.TrainLabels[:49000, :])
        self.TestLabels = np.squeeze(self.TestLabels)
        self.TestData = self.TestData.astype(np.float)
        self.trainacchist = np.array([])
        self.testacchist = np.array([])
        self.TrainData = np.reshape(self.TrainData, (self.TrainData.shape[0], -1))
        self.TestData = np.reshape(self.TestData, (self.TestData.shape[0], -1))
        self.TrainData = ((self.TrainData / 255) * 2) - 1
        self.TrainData = self.TrainData[:3000, :]
        self.TrainLabels = self.TrainLabels[:3000]
        self.krnl = krnl

    def run(self):
        cVals = [0.0001,0.001,0.01,0.1,1,10,100]
        for c in cVals:
            if self.krnl == 'rbf':
                core = svm.SVC(probability=False, kernel=self.krnl, C=c, gamma='auto')
            else:
                core = svm.SVC(probability=False, kernel=self.krnl, C=c)
            core.fit(self.TrainData, self.TrainLabels)
            TrainPreds = core.predict(self.TrainData)
            Trainacc = np.mean(TrainPreds == self.TrainLabels)
            self.trainacchist = np.append(self.trainacchist, Trainacc)
            TestPreds = core.predict(self.TestData)
            Testacc = np.mean(TestPreds == self.TestLabels)
            self.testacchist = np.append(self.testacchist, Testacc)
        plt.plot(cVals, self.trainacchist, '.-', color='red')
        plt.plot(cVals, self.testacchist, '.-', color='green')
        plt.xlabel('c')
        plt.ylabel('Accuracy')
        title = "Accuracy vs. C for Training (Red) and Test (Green) with a " + self.krnl + " Kernel"
        plt.title(title)
        plt.grid()
        plt.show()

    def reset(self, krnl):
        self.trainacchist = np.array([])
        self.testacchist = np.array([])
        self.krnl = krnl

me = mysvm('linear')
me.run()
me.reset('poly')
me.run()
me.reset('rbf')
me.run()
