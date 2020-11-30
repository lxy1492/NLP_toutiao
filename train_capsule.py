import torch
import numpy as np
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn import CrossEntropyLoss
from dataset import data_loader
from src import capsule_args as args
from src.capsule import Model as CapsuleNetwork
from loss_functions import *
from train import Log
from torchnet.meter import ClassErrorMeter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_loss_function(LOSS_TYPE, NUM_CLASS=15):
    if LOSS_TYPE == 'margin':
        loss_criterion = [MarginLoss(NUM_CLASS)]
    elif LOSS_TYPE == 'focal':
        loss_criterion = [FocalLoss()]
    elif LOSS_TYPE == 'cross':
        loss_criterion = [CrossEntropyLoss()]
    elif LOSS_TYPE == 'mf':
        loss_criterion = [MarginLoss(NUM_CLASS), FocalLoss()]
    elif LOSS_TYPE == 'mc':
        loss_criterion = [MarginLoss(NUM_CLASS), CrossEntropyLoss()]
    elif LOSS_TYPE == 'fc':
        loss_criterion = [FocalLoss(), CrossEntropyLoss()]
    else:
        loss_criterion = [MarginLoss(NUM_CLASS), FocalLoss(), CrossEntropyLoss()]
    return loss_criterion

class Train():
    def __init__(self, batchSize=16, epoch=10, device=device, log_path="./"):
        self.device = device
        self.net = CapsuleNetwork(
            args.NUM_CODEBOOK,
            args.NUM_CODEWORD,
            args.HIDDEN_SIZE,
            args.IN_LENGTH,
            args.OUT_LENGTH,
            args.NUM_CLASS,
            args.ROUTING_TYPE,
            args.EMBEDDING_TYPE,
            args.CLASSIFIER_TYPE,
            args.NUM_ITERATIONS,
            args.NUM_REPEAT,
            args.DROP_OUT,
            args.VOCAB_SIZE,
            args.EMBEDDING_SIZE
        ).to(self.device)
        self.batchsize = batchSize
        self.epoch = epoch
        self.data = data_loader(batchsize=self.batchsize)
        self.optimizer = None
        self.loss_function = None
        self.lr_scheduler = None

        self.log_path = log_path
        self.log = Log(self.log_path)

        self.step = 0
        self.epoch_step = 0
        self.train_acc = 0.0
        self.train_loss = 0.0
        self.eval_acc = 0.0
        self.eval_loss = 0.0
        self.best_acc = 0.0
        self.best_loss = 0.0
        self.best_step = -1
        self.best_epoch = -1
        self.train_auc = 0.0

        self.acc_meter = ClassErrorMeter(accuracy=True)

    def writeTrainLog(self):
        self.log.writeTrain(
            step=self.step,
            epoch=self.epoch_step,
            epochNum=self.epoch,
            trainAcc=self.train_acc,
            trainLoss=self.train_loss,
            valAcc=self.eval_acc,
            valLoss=self.eval_loss,
            bestEpoch=self.best_epoch,
            bestStep=self.best_step,
            bestValAcc=self.best_acc,
            bestLoss=self.best_loss,
            dataNum=self.data.trainDataLength,
            batchsize=self.batchsize
            )

    def setOptimizer(self):
        if args.NUM_REPEAT is None:
            optim_configs = [{'params': self.net.embedding.parameters(), 'lr': 1e-4 * 10},
                             {'params': self.net.features.parameters(), 'lr': 1e-4 * 10},
                             {'params': self.net.classifier.parameters(), 'lr': 1e-4}]
        else:
            for param in self.net.embedding.parameters():
                param.requires_grad = False
            for param in self.net.features.parameters():
                param.requires_grad = False
            optim_configs = [{'params': self.net.classifier.parameters(), 'lr': 1e-4}]
        self.optimizer = Adam(optim_configs, lr=1e-4)
        self.lr_scheduler = MultiStepLR(self.optimizer, milestones=[int(self.epoch * 0.5), int(self.epoch * 0.7)], gamma=0.1)

    def set_loss(self, TYPE=None):
        if TYPE==None:
            TYPE = args.LOSS_TYPE
        def loss(outputs, labels):
            loss_criterion = get_loss_function(TYPE)
            return sum([criterion(outputs, labels) for criterion in loss_criterion])
        self.loss_function = loss

    def train_batch(self, batch=None):
        if batch==None:
            batch = self.batchsize
        if self.optimizer==None:
            self.setOptimizer()
        if self.loss_function == None:
            self.set_loss(args.LOSS_TYPE)

        labels, datas = self.data.nextData(batchSize=batch)
        if labels is None or datas is None:
            return None
        sentences = datas[0]
        del datas
        labels = labels.to(self.device)
        sentences = sentences.to(self.device)

        self.optimizer.zero_grad()
        # self.net.train()
        outputs = self.net(sentences)
        loss = self.loss_function(outputs, labels)
        loss.backward()
        self.optimizer.step()
        _, prediction = outputs.max(dim=1)
        # self.lr_scheduler.step()
        # self.optimizer.step()
        # self.acc_meter.add(outputs.detach().cpu(), labels.detach().cpu())
        acc = prediction.eq(labels).sum().item() / labels.shape[0]
        loss = loss.detach().tolist()
        self.train_loss = loss
        self.writeTrainLog()
        del labels
        del outputs
        del sentences
        del prediction
        # print(acc)
        self.train_acc = acc
        return acc

    def eval(self, batch_size=None, step=None, epoch_step=None):
        self.net.eval()
        if batch_size==None:
            batch_size = self.batchsize
        labels, inputs = self.data.nextData(batchSize=batch_size, obj="val")
        correct = 0
        num = 0
        while True:
            if not (labels is None or inputs is None):
                num += 1
                self.net.eval()
                sentences = inputs[0]
                del inputs
                labels = labels.to(self.device)
                sentences = sentences.to(self.device)
                # segments = segments.to(self.device)
                # masks = masks.to(self.device)
                outputs = self.net(sentences)
                _, prediction = outputs.max(dim=1)
                correct += prediction.eq(labels).sum().item() / labels.shape[0]
                labels, inputs = self.data.nextData(batchSize=batch_size, obj="val")
            else:
                self.data.valIndex = 0
                break
        if num >= 1:
            self.eval_acc = correct/num
            if self.eval_acc > self.best_acc:
                self.best_acc = self.eval_acc
                self.best_step = step
                self.best_epoch = epoch_step
        self.writeTrainLog()

    def train_epoch(self, step=1, epoch=1, batch=None):
        self.step = step
        self.epoch_step = epoch
        self.net.train()
        r = self.train_batch(batch)
        while r!=None:
            r = self.train_batch(batch)
            self.step += 1
        self.data.trainIndex = 0

    def train(self):
        for epoch_step in range(self.epoch):
            self.train_epoch(1, epoch_step+1)
            self.lr_scheduler.step()
            self.eval()
            if epoch_step==4:
                print("lr:1e-4")
                self.optimizer = Adam(self.net.parameters(), lr=1e-4)
                self.lr_scheduler = MultiStepLR(self.optimizer,
                                                milestones=[int(self.epoch * 0.5), int(self.epoch * 0.7)], gamma=0.1)
            elif epoch_step==6:
                print("lr:1e-5")
                self.optimizer = Adam(self.net.parameters(), lr=1e-5)
                self.lr_scheduler = MultiStepLR(self.optimizer,
                                                milestones=[int(self.epoch * 0.5), int(self.epoch * 0.7)], gamma=0.1)
            elif epoch_step==8:
                print("lr:1e-5")
                self.optimizer = Adam(self.net.parameters(), lr=1e-6)
                self.lr_scheduler = MultiStepLR(self.optimizer,
                                                milestones=[int(self.epoch * 0.5), int(self.epoch * 0.7)], gamma=0.1)

        try:
            self.log.close()
        except:
            pass

if __name__ == '__main__':
    train = Train(batchSize=256)
    train.train()

