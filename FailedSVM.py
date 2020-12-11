from PIL import Image
import csv
import os
import numpy as np
import math

def createFileList(myDir, format='.png'):
    fileList = []
    #print(myDir)
    for root, dirs, files in os.walk(myDir, topdown=False):
        for name in files:
            if name.endswith(format):
                fullName = os.path.join(root, name)
                fileList.append(fullName)
    return fileList

# load the original image
def csvmaker():
    myFileList = createFileList('train')
    for file in myFileList:
        #print(file)
        img_file = Image.open(file)
        # img_file.show()
        # get original image parameters...
        width, height = img_file.size
        format = img_file.format
        mode = img_file.mode
        # Make image Greyscale
        img_grey = img_file.convert('L')
        #img_grey.save('result.png')
        #img_grey.show()
        # Save Greyscale values
        value = np.asarray(img_grey.getdata(), dtype=np.int).reshape((img_grey.size[1], img_grey.size[0]))
        value = value.flatten()
        #print(value)
        with open("train.csv", 'a') as f:
            writer = csv.writer(f)
            writer.writerow(value)

#datasquash did not end up actually being used
def datasquash():
    files = ['test.csv', 'train.csv']
    for file in files:
        with open(file, 'r') as f, open('new' + file, 'w') as f_out:
            reader = csv.reader(f)
            for row in reader:
                new_row = []
                for e in row:
                    new_row.append(str(float(e) / 255))
                f_out.write(','.join(new_row) + '\n')

divider = lambda x: x * 10
multiplier = lambda x: x * 2

class svm:
    def __init__(self, target):
        data = np.loadtxt('train.csv', delimiter = ",")
        self.train = data[:40000, : ]
        self.test = data[40000: , : ]
        labels = np.loadtxt('trainLabels.csv', delimiter = ",")
        self.trainlabels = labels[:40000, 1]
        self.testlabels = labels[40000:, 1]
        self.target = target
        #I could not figure out the multiclass svm, so I decided to allow the user to choose to identify "is this class" vs. "not this class"
        self.trainum = len(self.train)
        self.testnum = len(self.test)
        self.datalen = len(self.train[0])
        self.weights = np.random.uniform(low=-0.05, high=0.05,size=(self.datalen))
        self.bias = np.random.uniform(low=-0.05, high=0.05)
        self.margin = 1 / np.linalg.norm(self.weights)
        self.trainresults = np.array([0]*(self.trainum * 2)).reshape(self.trainum,2)
        self.testresults = np.array([0]*(self.testnum * 2)).reshape(self.testnum,2)

    def predict(self,row):
        result = self.margin * (np.dot(self.weights, row) + self.bias)
        if result > 0: return 1
        else: return 0

    def update(self):
        new = divider(self.weights)
        sums = np.array([0.0] * self.datalen)
        for i in range(0,self.trainum):
            if self.trainresults[i,1] == 0:
                sums = np.add(sums, divider(self.train[i]))
        new = np.divide(new, multiplier(sums))
        self.weights = new
        self.margin = 1/np.linalg.norm(new)
        return

    def accuracy(self):
        confusion = np.array([0] * 4).reshape(2,2)
        for i in range(0, self.testnum):
            temp = self.predict(self.test[i])
            self.testresults[i, 0] = temp
            if self.testlabels[i] == self.target:
                confusion[temp, 1] += 1
            else: confusion[temp,0] += 1
        return confusion


    def cycle(self):
        count = 0
        while(count < 10):
            print(self.accuracy())
            for i in range(0, self.trainum):
                temp = self.predict(self.train[i])
                self.trainresults[i,0] = temp
                if (temp == 1 and self.trainlabels[i] == self.target) or (temp == 0 and self.trainlabels[i] != self.target):
                    self.trainresults[i,1] = 1
                else: self.trainresults[i,1] = 0
            self.update()
            count += 1
        return

me = svm(0)
me.cycle()