import os
import torch
import warnings
from net import Net
from utils import NMTCriterion, Util
from dataset import data_loader
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule

class Log:

    def __init__(self, path, name=None, round_size=4, flushPrintInfo=False, printInfo=True):
        assert isinstance(path, str)
        if ".txt" in path.split("/")[-1]:
            self.name = path.split("/")[-1]
            self.dir = os.path.dirname(path)
        else:
            if name==None:
                self.name = "log.txt"
                self.dir = path
            else:
                assert isinstance(name, str)
                if "." in name:
                    if name.split(".")[-1]=="txt":
                        self.name = name
                    else:
                        self.name = name+".txt"
                else:
                    self.name = name + ".txt"
                self.dir = path
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        self.filePath = os.path.join(self.dir, self.name).replace("\\", "/")
        self.file = None
        self.start = False
        self.round_size = round_size
        self.flush_print_info = flushPrintInfo
        self.print_info = printInfo

    def openFile(self):
        self.file = open(self.filePath, "w")

    def writeLine(self, line:str):
        if self.file!=None:
            if self.start:
                self.file.write("\n")
            else:
                self.start=True
            self.file.write(line)
            if self.flush_print_info:
                print("\r" + line, end="", flush=True)
            elif self.print_info:
                print(line)
        else:
            self.openFile()
            self.writeLine(line)

    def get_str(self, value):
        if isinstance(value, int):
            return str(value)
        elif isinstance(value, float):
            value = round(value, self.round_size)
            value = str(value)
            l = len(value.split(".")[-1])
            while l<self.round_size:
                value += "0"
                l += 1
            return value
        else:
            return str(value)

    def close(self):
        try:
            self.file.close()
            self.start = False
        except Exception as m:
            warnings.warn("log文件关闭错误:"+m)
            warnings.warn("log path:"+self.filePath)

    def writeinfo(self, **kwargs):
        string = ""
        for eachKey in kwargs:
            string += str(eachKey)+":"+str(kwargs[eachKey])+" | "
        self.writeLine(string[:-1])

    def writeTrain(self, step=None, epoch=None, batchsize=None, epochNum=None, dataNum=None, trainLoss=None, trainAcc=None, trainAuc=None, valLoss=None, valAcc=None, bestValAcc=None, bestLoss=None, bestStep=None, bestEpoch=None):
        if isinstance(dataNum, int) and isinstance(batchsize, int):
            steps = dataNum/batchsize
            if dataNum % batchsize != 0:
                steps = int(steps)+1
            else:
                steps = int(steps)
        else:
            steps = None
        if steps!=None:
            string = "step:{}/{}".format(str(step), str(steps))
        else:
            string = "step:"+str(step)
        string += " for epoch:{}/{} | ".format(str(epoch), str(epochNum))
        string += "train loss:{} | train acc:{} | train auc:{} | val loss:{} | val acc:{} | best val acc:{} on step/epoch:{}/{} for best loss:{} |".format(
            self.get_str(trainLoss),
            self.get_str(trainAcc),
            self.get_str(valLoss),
            self.get_str(trainAuc),
            self.get_str(valAcc),
            self.get_str(bestValAcc),
            self.get_str(bestStep),
            self.get_str(bestEpoch),
            self.get_str(bestLoss)
        )
        self.writeLine(string)

class Train:
    def __init__(self, net, bath=None, epoch=None, data=None, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), loss_function=None, label_smoothing=0.07, optimizer=None, lr=None, log_path="./"):
        if not isinstance(bath, int):
            self.batch = 64
        elif bath <= 0:
            self.batch = 64
        else:
            self.batch = bath
        if isinstance(epoch, int) and epoch>0:
            self.epoch = epoch
        else:
            self.epoch = 10
        if data==None:
            self.data = data_loader(batchsize=self.batch)
        else:
            if isinstance(data, data_loader):
                self.data = data
            else:
                self.data = data_loader()
        self.device = device
        self.net = net
        if not self.net is None:
            try:
                self.net.to(self.device)
            except:
                pass
        if loss_function==None:
            self.loss_function = NMTCriterion(label_smoothing=label_smoothing).to(self.device)
        else:
            self.loss_funciton = loss_function
        if lr!=None:
            self.lr = lr
        else:
            self.lr = 0.01
        if optimizer==None:

            param_optimizer = list(self.net.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer if any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

            self.optimizer = AdamW(params=optimizer_grouped_parameters, lr=self.lr, correct_bias=False)
        else:
            self.optimizer = optimizer
        self.optimizer = WarmupLinearSchedule(optimizer=self.optimizer, warmup_steps=0.1,
                                         t_total=100)
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
            batchsize=self.batch
        )

    def train_batch(self, input=None, label=None, writeLog=True):
        if input==None or label==None:
            label, sentences = self.data.nextData()
            label = label.to(self.device)
            if label is None or sentences is None:
                return None
            else:
                inputs, masks, segments = sentences
                inputs = inputs.to(self.device)
                masks = masks.to(self.device)
                segments = segments.to(self.device)
                # outputs = self.net(inputs, segments, masks)
                # loss = self.loss_function(inputs=outputs, labels=label, normalization=1.0, reduce=False)
                # loss.backward(torch.ones_like(loss))
                # self.optimizer.step()
                # _, prediction = outputs.max(dim=1)
                # self.train_acc = prediction.eq(label).sum().item() / label.shape[0]
                # if writeLog:
                #     self.writeTrainLog()
        else:
            label = label.to(self.device)
            inputs, masks, segments = input
            inputs = inputs.to(self.device)
            masks = masks.to(self.device)
            segments = segments.to(self.device)
        outputs = self.net(inputs, segments, masks)
        loss = self.loss_function(inputs=outputs, labels=label, normalization=1.0, reduce=False)
        loss.backward(torch.ones_like(loss))
        self.optimizer.step()
        _, prediction = outputs.max(dim=1)
        train_acc, report, auc = Util.classifiction_metric(prediction.cpu(), labels=label.cpu(), label_list= ['news_culture', 'news_entertainment', 'news_sports', 'news_finance', 'news_house',
              'news_car', 'news_edu', 'news_tech', 'news_military', 'news_travel', 'news_world',
              'news_agriculture', 'news_game', 'stock', 'news_story'])
        if writeLog:
            self.writeTrainLog()
        self.train_acc = train_acc
        self.train_auc = auc
        del inputs
        del outputs
        del label
        del prediction
        del loss
        del train_acc
        del auc
        return self.train_acc

    def train_epoch(self, step=1, epoch=1):
        self.step = step
        self.epoch_step = epoch
        self.net.train()
        for label, inputs in self.data:
            self.train_batch(input=inputs, label=label)
            self.step += 1

    def train(self):
        for epoch_step in range(self.epoch):
            self.train_epoch(self.step, epoch_step+1)
            self.eval()

    def eval(self, batch_size=None, step=None, epoch_step=None):
        self.net.eval()
        if batch_size==None:
            batch_size = self.batch
        labels, inputs = self.data.nextData(batchSize=batch_size, obj="val")
        correct = 0
        num = 0
        while True:
            if not (labels is None or inputs is None):
                num += 1
                self.net.eval()
                sentences, masks, segments = inputs
                labels = labels.to(self.device)
                sentences = sentences.to(self.device)
                segments = segments.to(self.device)
                masks = masks.to(self.device)
                outputs = self.net(sentences, segments, masks)
                _, prediction = outputs.max(dim=1)
                correct += prediction.eq(labels).sum().item() / labels.shape[0]
            else:
                self.data.valIndex = 0
                break
        if num >= 1:
            self.eval_acc = correct/num
            if self.eval_acc > self.best_acc:
                self.best_acc = self.eval_acc
                self.best_step = step
                self.best_epoch = epoch_step

class BaseTrain:
    def __init__(self, batch, epoch, logPath="./", model_path="./model.pkl", device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        assert isinstance(batch, int)
        assert isinstance(epoch, int)
        self.batch = batch
        self.epoch = epoch
        self.model = None
        self.data_iter = None
        self.optimizer = None
        self.loss_function = None
        self.lr = None
        self.model_path = model_path

        self.log = Log(logPath)

        self.device = device
        self.loss_value = None

        self.trainLoss = 0.0
        self.trainAcc = 0.0
        self.trainAuc = 0.0
        self.valLoss = 0.0
        self.valAcc = 0.0
        self.valAuc = 0.0
        self.best_acc = 0.0
        self.best_loss = 0.0
        self.best_step = 0.0
        self.best_epoch = 0.0
        self.step = 0
        self.epoch_step = 0
        self.trainDataLength = 0
        self.evalDataLength = 0
        self.testDataLength = 0

    def set_model(self, model):
        self.model = model

    def set_optimizer(self):
        pass

    def set_loss_function(self):
        pass

    def set_dataset(self, batch=None):
        if batch==None:
            batch=self.batch
        self.data_iter = data_loader(batchsize=batch)
        self.data_iter.readFile("train")
        self.data_iter.readFile("val")
        self.trainDataLength = self.data_iter.trainDataLength
        self.evalDataLength = self.data_iter.valDataLength
        # self.testDataLength = self.data_iter.testDataLength



    def writeTrainLog(self):
        self.log.writeTrain(
            step=self.step,
            epoch=self.epoch_step,
            epochNum=self.epoch,
            trainAcc=self.trainAcc,
            trainLoss=self.trainLoss,
            valAcc=self.valAcc,
            valLoss=self.valLoss,
            bestEpoch=self.best_epoch,
            bestStep=self.best_step,
            bestValAcc=self.best_acc,
            bestLoss=self.best_loss,
            dataNum=self.trainDataLength,
            batchsize=self.batch
        )

    def get_batchData(self, batch=None):
        labels, inputs = self.data_iter.nextData(batch)
        return labels, inputs

    def get_loss(self, labels, outputs):
        # return self.loss_function(labels, outputs)
        self.loss_value = self.loss_function(outputs, labels)

    def loss(self, labels, outputs):
        # self.loss_value = self.get_loss(labels, outputs)
        self.get_loss(labels, outputs)
        self.loss_value.backward()

    def optimize(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    @staticmethod
    def get_acc(labels, outputs=None, prediction=None):
        if prediction==None:
            _, prediction = outputs.max(dim=1)
        return prediction.eq(labels).sum().item() / labels.shape[0]

    def get_train_info(self, labels, outputs, loss=None):
        self.trainAcc = self.get_acc(labels, outputs)
        if not loss is None:
            self.trainLoss = loss.detach().tolist()

    def before_batch(self):
        pass

    def affter_batch(self):
        pass

    def set_learning_rate(self):
        pass

    def train_batch(self, labels, inputs):
        self.before_batch()
        outputs = self.model(inputs)
        del inputs
        self.loss(labels, outputs)
        self.optimize()
        self.get_train_info(labels, outputs, self.loss_value)
        del labels
        del outputs
        self.writeTrainLog()
        self.affter_epoch()

    def before_epoch(self):
        pass

    def affter_epoch(self):
        pass

    def train_epoch(self, batch=None):
        if batch==None:
            batch = self.batch
        self.before_epoch()
        self.model.train()
        self.step = 1
        labels, inputs = self.get_batchData(batch)
        while True:
            if labels is None or inputs is None:
                break
            else:
                self.train_batch(labels, inputs)
                labels, inputs = self.get_batchData(batch)
                self.step += 1
        self.affter_epoch()
        self.set_learning_rate()
        self.data_iter.trainIndex = 0

    def train(self, epoch=None, batch=None):
        if epoch==None:
            epoch = self.epoch
        for epoch_step in range(epoch):
            self.epoch_step = epoch_step+1
            self.train_epoch(batch)
            self.eval(batch)
        self.log.close()

    def run(self, model=None, batch=None):
        self.init(model, batch)
        print("\r", end="\n", flush=True)
        print("start train for epoch:{}, batchsize:{}, train data length:{}, saving model at:{}".format(
            str(self.epoch),
            str(self.batch),
            str(self.trainDataLength),
            self.model_path
        ))
        self.train()
        print("end for best step:{}, best epoch:{}, best eval acc:{}, log_path:{}".format(
            str(self.best_step),
            str(self.best_epoch),
            str(self.best_acc),
            self.log.filePath,
        ))


    def predict(self, inputs):
        outputs = self.model(inputs)
        _, prediction = outputs.max(dim=1)
        del outputs
        return prediction

    def get_eval(self, labels, inputs):
        prediction = self.predict(inputs)
        correct = self.get_acc(labels, prediction=prediction)
        return correct

    def eval(self, batch=None):
        self.model.eval()
        if batch==None:
            batch = self.batch
        labels, inputs = self.data_iter.nextData(obj="val", batchSize=batch)
        correct = 0.0
        num = 0
        print("\rstart eval ... ", end="", flush=True)
        data_num = 0
        while not (labels is None or inputs is None):
            acc = self.get_eval(labels, inputs[0])
            correct += acc
            data_num += labels.shape[0]
            num += 1
            string = ">>> eval acc:{} | eval dataset:{}/{} rest:{}%".format(
                str(acc),
                str(data_num),
                str(self.evalDataLength),
                str(int(round((self.evalDataLength-data_num)/self.evalDataLength, 2)*100))
            )
            print("\r"+string, end="", flush=True)
            labels, inputs = self.data_iter.nextData(obj="val", batchSize=batch)
        self.valAcc = correct/num
        if self.best_acc<self.valAcc:
            self.best_acc = self.valAcc
            self.best_step = self.step
            self.best_epoch = self.epoch_step
            torch.save(self.model, self.model_path)
            print("\r>>> saved model :{}, at step/epoch:{}/{} for best eval acc:{}".format(
                self.model_path,
                str(self.step),
                str(self.best_epoch),
                str(self.best_acc)
            ),
                end="",
                flush=True
            )
        print("\r", end="\n", flush=True)
        self.writeTrainLog()
        self.data_iter.valIndex = 0

    def save_model(self):
        pass

    def init(self, model=None, batch=None):
        print("\rsetting model ...", end="", flush=True)
        self.set_model(model)
        print("\rsetting dataset ...", end="", flush=True)
        self.set_dataset(batch)
        print("\rsetting optimizer ...", end="", flush=True)
        self.set_optimizer()
        print("\rsetting loss function ...", end="", flush=True)
        self.set_loss_function()
        print("\r", end="\n", flush=True)


if __name__ == '__main__':
    model = Net.from_pretrained(pretrained_model_name_or_path=os.path.join("../2020-FlyAI-Today-s-Headlines-By-Category-master/BERT", 'data/input/model').replace("\\", "/"),
                                num_labels=15,
                                cache_dir=os.path.join("../2020-FlyAI-Today-s-Headlines-By-Category-master/BERT", "data/cache").replace("\\", "/"))
    trainer = Train(model, bath=16, lr=5e-5)
    trainer.train()
