#Mahmoud Al Robiai
#ML Fall 2020
#Final Project / SVM portion.

import os
import numpy as np

import matplotlib.pyplot as plt
from keras.datasets import cifar10
from sklearn import svm


class svm_ML(object):
    def __init__(self):
        self.baseDir = os.path.dirname(os.path.abspath('__file__')) + '/'
        self.classesName = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        (xTrain, yTrain), (xTest, yTest) = cifar10.load_data()
        self.xVal = xTrain[49000:, :].astype(np.float)
        self.yVal = np.squeeze(yTrain[49000:, :])
        self.xTrain = xTrain[:49000, :].astype(np.float)
        self.yTrain = np.squeeze(yTrain[:49000, :])
        self.yTest = np.squeeze(yTest)
        self.xTest = xTest.astype(np.float)
        self.acc_train_svm_linear = []
        self.acc_test_svm_linear = []
        self.acc_train_svm_poly = []
        self.acc_test_svm_poly = []

    def normalize(self):
        self.xTrain = np.reshape(self.xTrain, (self.xTrain.shape[0], -1))
        self.xVal = np.reshape(self.xVal, (self.xVal.shape[0], -1))
        self.xTest = np.reshape(self.xTest, (self.xTest.shape[0], -1))

        self.xTrain = ((self.xTrain / 255) * 2) - 1
        self.xTrain = self.xTrain[:3000, :]
        self.yTrain = self.yTrain[:3000]
        return self.xTrain

    def svm_linear(self, c):

        self.xTrain = self.normalize()

        svc = svm.SVC(probability=False, kernel='linear', C=c)

        svc.fit(self.xTrain, self.yTrain)

        # Find the prediction and accuracy on the training set.
        Yhat_svc_linear_train = svc.predict(self.xTrain)
        acc_train = np.mean(Yhat_svc_linear_train == self.yTrain)
        self.acc_train_svm_linear.append(acc_train)
        print('Train Accuracy = {0:f}'.format(acc_train))

        # Find the prediction and accuracy on the test set.
        Yhat_svc_linear_test = svc.predict(self.xVal)
        acc_test = np.mean(Yhat_svc_linear_test == self.yVal)
        self.acc_test_svm_linear.append(acc_test)
        print('Test Accuracy = {0:f}'.format(acc_test))

    def run_linear(self):
        c_svm_linear = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]

        for c in c_svm_linear:
            self.svm_linear(c)

        plt.plot(c_svm_linear, self.acc_train_svm_linear, '.-', color='red')
        plt.plot(c_svm_linear, self.acc_test_svm_linear, '.-', color='orange')
        plt.xlabel('c')
        plt.ylabel('Accuracy')
        plt.title("Plot of accuracy vs c for training and test data")
        plt.grid()
        plt.show()

    def svm_polynomial(self, c):
        self.xTrain = self.normalize()

        svc_polynomial = svm.SVC(probability=False, kernel='poly', C=c)

        svc_polynomial.fit(self.xTrain, self.yTrain)

        # Find the prediction and accuracy on the training set.
        Yhat_svc_polynomial_train = svc_polynomial.predict(self.xTrain)
        acc_train = np.mean(Yhat_svc_polynomial_train == self.yTrain)
        self.acc_train_svm_poly.append(acc_train)
        print('Train Accuracy = {0:f}'.format(acc_train))

        # Find the prediction and accuracy on the test set.
        Yhat_svc_polynomial_test = svc_polynomial.predict(self.xVal)
        acc_test = np.mean(Yhat_svc_polynomial_test == self.yVal)
        self.acc_test_svm_poly.append(acc_test)
        print('Test Accuracy = {0:f}'.format(acc_test))

    def run_poly(self):

        c_svm_poly = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]

        for c in c_svm_poly:
            self.svm_polynomial(c)

        plt.plot(c_svm_poly, self.acc_train_svm_poly, '.-', color='red')
        plt.plot(c_svm_poly, self.acc_test_svm_poly, '.-', color='orange')
        plt.xlabel('c')
        plt.ylabel('Accuracy')
        plt.title("Plot of accuracy vs c for training and test data")
        plt.grid()
        plt.show()


if __name__ == "__main__":
    svm_ML = svm_ML()
    print("SVM - linear Kernel\n")
    svm_ML.run_linear()
    print("\nSVM - Polynomial Kernel\n")
    svm_ML.run_poly()