import os
import torch
import warnings
import numpy as np
import random
import time

class dataloader:
    def __init__(self, batchsize=64, dataDIR="./data", fileformate="tsv", device=None, read_encoding="utf-8", tockenize=None):
        self.dataDIR = dataDIR
        self.trainPath = os.path.join(dataDIR, "train."+fileformate).replace("\\", "/")
        self.testPath = os.path.join(dataDIR, "test." + fileformate).replace("\\", "/")
        self.valPath = os.path.join(dataDIR, "val." + fileformate).replace("\\", "/")
        self.batchSize = batchsize
        if device==None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.trainFile = None
        self.testFile = None
        self.valFile = None
        self.read_encoding = read_encoding
        self.tockenize = tockenize
        self.currentData = "train"
        self.trainEnd = False
        self.testEnd = False
        self.valEnd = False
        random.see(time.localtime())

    def openFile(self, obj="train"):
        if obj=="train":
            try:
                self.trainFile.close()
            except:
                pass
            self.trainFile = open(self.trainPath, "r", encoding=self.read_encoding)
            self.trainEnd = False
        elif obj=="test":
            try:
                self.testFile.close()
            except:
                pass
            self.testFile = open(self.testPath, "r", encoding=self.read_encoding)
            self.testEnd = False
        elif obj=="val":
            try:
                self.valFile.close()
            except:
                pass
            self.valFile = open(self.valPath, "r", encoding=self.read_encoding)
            self.valEnd = False

    def closeFile(self, obj="train"):
        if obj=="train":
            try:
                self.trainFile.close()
            except:
                pass
            self.trainFile = None
            self.trainEnd = True
        elif obj=="test":
            try:
                self.testFile.close()
            except:
                pass
            self.testFile = None
            self.testEnd = True
        elif obj=="val":
            try:
                self.valFile.close()
            except:
                pass
            self.valFile = None
            self.valEnd = True

    def openTrain(self):
        self.openFile("train")
    def openTest(self):
        self.openFile("test")
    def openVal(self):
        self.openFile("val")
    def closeTrain(self):
        self.closeFile("train")
    def closeTest(self):
        self.closeFile("test")
    def closeVal(self):
        self.closeFile("val")

    def readTrainLine(self):
        if self.trainFile==None:
            self.openTrain()
        line = self.trainFile.readline()
        # print(line)
        if line:
            # print(line)
            if line!="":
                # print(line)
                line = line.split("	")
                label = line[0]
                sentence = line[1]
                try:
                    # print(label)
                    label = int(label)
                    if self.tockenize!=None:
                        sentence = self.tockenize(sentence)
                    return label, sentence
                except:
                    return None, None
            else:
                return None, None
        else:
            return None, None

    def readTestLine(self):
        if self.testFile==None:
            self.testTrain()
        line = self.testFile.readline()
        if line:
            if line!="":
                line = line.split("	")
                label = line[0]
                sentence = line[1]
                try:
                    label = int(label)
                    if self.tockenize != None:
                        sentence = self.tockenize(sentence)
                    return label, sentence
                except:
                    return None, None
            else:
                return None, None
        else:
            return None, None

    def readValLine(self):
        if self.valFile==None:
            self.openVal()
        line = self.valFile.readline()
        if line:
            if line!="":
                line = line.split("	")
                label = line[0]
                sentence = line[1]
                try:
                    label = int(label)
                    if self.tockenize != None:
                        sentence = self.tockenize(sentence)
                    return label, sentence
                except:
                    return self.readValLine()
            else:
                return self.readValLine()
        else:
            return None, None

    def readData(self, obj="train"):
        sentence, label = None, None
        if obj=="train":
            sentence, label = self.readTrainLine()
        elif obj=="test":
            sentence, label = self.readTestLine()
        elif obj=="val":
            sentence, label = self.readValLine()
        return sentence, label

    def readBatch(self, batchSize=None, obj="train"):
        sentences, labels = [], []
        if batchSize==None:
            batchSize=self.batchSize
        for _ in range(batchSize):
            label, sentence = self.readData(obj)
            if sentence==None or label==None:
                break
            else:
                sentences.append(sentence)
                labels.append(label)
        return np.asarray(sentences), np.asarray(labels)

    def nextBatch(self, batchSize=None, obj="train"):
        return self.readBatch(batchSize, obj)

    def __iter__(self):
        return self

    def __next__(self):
        if self.currentData=="train":
            if self.trainEnd:
                raise StopIteration()
        elif self.currentData=="test":
            if self.testEnd:
                raise StopIteration()
        elif self.currentData=="val":
            if self.testEnd:
                raise StopIteration()
        return self.nextBatch(obj=self.currentData)


class dataloader2:
    def __init__(self, batchsize=64, dataDIR="./data", fileformate="tsv", device=None, read_encoding="utf-8", mask=False, segment=False):
        self.dataDIR = dataDIR
        self.trainPath = os.path.join(dataDIR, "train."+fileformate).replace("\\", "/")
        self.testPath = os.path.join(dataDIR, "test." + fileformate).replace("\\", "/")
        self.valPath = os.path.join(dataDIR, "val." + fileformate).replace("\\", "/")
        self.batchSize = batchsize
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.trainFile = None
        self.testFile = None
        self.valFile = None
        self.read_encoding = read_encoding
        self.currentData = "train"
        self.trainData = None
        self.testData = None
        self.valData = None
        self.trainDataLength = None
        self.testDataLength = None
        self.valDataLength = None
        self.trainIndex = 0
        self.testIndex = 0
        self.valIndex = 0
        self.mask = mask
        self.segment = segment

    def readFile(self, obj="train", file=None):
        if file==None:
            if obj=="train":
                file = self.trainPath
            elif obj == "test":
                file = self.testPath
            elif obj == "val":
                file = self.valPath
        file = open(file, "r", encoding="utf-8")
        data = []
        for eachLine in file.read().split("\n"):
            try:
                # print(eachLine)
                line = eachLine.split("	")
                label = int(line[0])
                sentence = line[1]
                label = int(label)
                if self.tockenize!=None:
                    sentence = self.tockenize(sentence)
                data.append([label, sentence])
            except:
                pass
        random.shuffle(data)
        if obj == "train":
            self.trainData = data
            # print(data)
            self.trainDataLength = len(data)
            self.trainIndex = 0
            # print(self.trainDataLength)
        elif obj == "test":
            self.testData = data
            self.testDataLength = len(data)
            self.testIndex = 0
        elif obj == "val":
            self.valData = data
            self.valDataLength = len(data)
            self.valIndex = 0

    def nextData(self, batchSize=None, obj=None):
        if obj==None:
            obj = self.currentData
        if batchSize == None:
            batchSize = self.batchSize
        label_batch = []
        sentence_batch = []
        if obj == "train":
            if self.trainData==None:
                self.readFile(obj="train")
            for _ in range(batchSize):
                if self.trainIndex<self.trainDataLength:
                    label, sentence = self.trainData[self.trainIndex]
                    self.trainIndex += 1
                    label_batch.append(label)
                    sentence_batch.append(sentence)
                else:
                    break
        elif obj == "test":
            if self.testData == None:
                self.readFile("test")
            for _ in range(batchSize):
                if self.testIndex<self.testDataLength:
                    label, sentence = self.testData[self.testIndex]
                    self.testIndex += 1
                    label_batch.append(label)
                    sentence_batch.append(sentence)
                else:
                    break
        elif obj == "val":
            if self.valData == None:
                self.readFile("val")
            for _ in range(batchSize):
                if self.valIndex < self.valDataLength:
                    label, sentence = self.valData[self.valIndex]
                    self.valIndex += 1
                    label_batch.append(label)
                    sentence_batch.append(sentence)
                else:
                    break
        if len(label_batch) > 0 and len(sentence_batch) > 0:
            return torch.tensor(np.array(label_batch), dtype=torch.long).to(self.device), self.convertSentence(sentence_batch)
        else:
            return None, None

    def convertSentence(self, sentence):
        return sentence

    def tockenize(self, sentence):
        return sentence

    def __iter__(self):
        return self

    def __next__(self):
        label, batch = self.nextData(obj=self.currentData)
        # print(batch)
        if label is None or batch is None:
            if self.currentData == "train":
                self.trainIndex = 0
            elif self.currentData == "test":
                self.testIndex = 0
            elif self.currentData == "val":
                self.valIndex = 0
            raise StopIteration()
        return label, batch


if __name__ == '__main__':
    os.chdir("../")
    data_iter = dataloader2()
    for each in data_iter:
        print(each)