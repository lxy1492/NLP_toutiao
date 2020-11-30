from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from train import BaseTrain
from src.dot_attention_capsule import config
from src.capsule_with_dot_product_attention import Model as capsule_model_with_dot_produc_attention_routing
from src.dot_attention_capsule.capsule_model import CapsModel
from train_capsule import get_loss_function
from dataset import data_loader

class Train_dot_attention_capsule_network(BaseTrain):
    def __init__(self, batch=128, epoch=10, lr=1e-2, log_path="./", model_path="./model.pkl"):
        super().__init__(batch, epoch, log_path, model_path)
        self.lr = lr

    def set_model(self, model=None):
        if model is None:
            self.model = CapsModel(
                vocab_size=config.VOCAB_SIZE,
                backbone=config.BACKBONE,
                num_routing=config.NUM_ROUTING,
                num_codebook=config.NUM_CODEBOOK,
                num_codeword=config.NUM_CODEWORD,
                in_channels=config.IN_CHANNELS,
                num_repeat=config.NUM_REPEAT,
                out_channels=config.OUT_CHANNELS,
                sequential_routing=config.SEQUENTIAL_ROUTING,
                embedding_dim=config.EMBEDDING_SIZE,
                sentenceLength=config.SENTENCE_LENGTH,
                classes=config.NUM_CLASS,
                embedding_type=config.EMBEDDING_TYPE,
                dp=config.DP
            )
        else:
            self.model = model
        self.model = self.model.to(self.device)

    def set_optimizer(self, repeat=config.NUM_REPEAT):
        if repeat is None:
            optim_configs = [{'params': self.model.embedding.parameters(), 'lr': self.lr},
                             {'params': self.model.pre_caps.parameters(), 'lr': self.lr/10},
                             {'params': self.model.pc_layer.parameters(), 'lr': self.lr/10},
                             {'params': self.model.nonlinear_act.parameters(), 'lr': self.lr/100}]
        else:
            for param in self.model.embedding.parameters():
                param.requires_grad = False
            for param in self.model.features.parameters():
                param.requires_grad = False
            optim_configs = [{'params': self.model.classifier.parameters(), 'lr': 1e-4}]
        self.optimizer = Adam(optim_configs, lr=1e-4)
        self.lr_scheduler = MultiStepLR(self.optimizer, milestones=[int(self.epoch * 0.5), int(self.epoch * 0.7)],
                                        gamma=0.1)

    def affter_epoch(self):
        self.lr_scheduler.step()

    def set_loss_function(self, TYPE=None):
        if TYPE==None:
            TYPE = config.LOSS_TYPE
        def loss():
            # print(outputs)
            # print(labels)
            return nn.CrossEntropyLoss()
            # loss_criterion = get_loss_function(TYPE)
            # return sum([criterion(outputs, labels) for criterion in loss_criterion])
        self.loss_function = loss()

    def set_dataset(self, batch=None):
        if batch==None:
            batch = self.batch
        self.data_iter = data_loader(batch, device=self.device)
        self.data_iter.readFile("train")
        # self.data_iter.readFile("test")
        self.data_iter.readFile("val")
        self.trainDataLength = self.data_iter.trainDataLength
        self.evalDataLength = self.data_iter.valDataLength
        # self.testDataLength = self.data_iter.testDataLength

    def get_batchData(self, batch=None):
        labels, inputs = self.data_iter.nextData(batch)
        if inputs is None:
            return None, None
        inputs = inputs[0]
        return labels, inputs

    def set_learning_rate(self):
        if self.epoch_step == 2:
            optim_configs = [{'params': self.model.embedding.parameters(), 'lr': self.lr/10},
                             {'params': self.model.pre_caps.parameters(), 'lr': self.lr / 100},
                             {'params': self.model.pc_layer.parameters(), 'lr': self.lr / 100},
                             {'params': self.model.nonlinear_act.parameters(), 'lr': self.lr / 1000}]
            self.optimizer = Adam(optim_configs, lr=1e-4)
            self.lr_scheduler = MultiStepLR(self.optimizer, milestones=[int(self.epoch * 0.5), int(self.epoch * 0.7)],
                                            gamma=0.1)
        elif self.epoch_step == 4:
            optim_configs = [{'params': self.model.embedding.parameters(), 'lr': self.lr / 100},
                             {'params': self.model.pre_caps.parameters(), 'lr': self.lr / 100},
                             {'params': self.model.pc_layer.parameters(), 'lr': self.lr / 100},
                             {'params': self.model.nonlinear_act.parameters(), 'lr': self.lr / 1000}]
            self.optimizer = Adam(optim_configs, lr=1e-4)
            self.lr_scheduler = MultiStepLR(self.optimizer, milestones=[int(self.epoch * 0.5), int(self.epoch * 0.7)],
                                            gamma=0.1)
        elif self.epoch_step == 6:
            optim_configs = [{'params': self.model.embedding.parameters(), 'lr': self.lr / 1000},
                             {'params': self.model.pre_caps.parameters(), 'lr': self.lr / 1000},
                             {'params': self.model.pc_layer.parameters(), 'lr': self.lr / 1000},
                             {'params': self.model.nonlinear_act.parameters(), 'lr': self.lr / 10000}]
            self.optimizer = Adam(optim_configs, lr=1e-4)
            self.lr_scheduler = MultiStepLR(self.optimizer, milestones=[int(self.epoch * 0.5), int(self.epoch * 0.7)],
                                            gamma=0.1)
        elif self.epoch_step == 8:
            self.optimizer = Adam(self.model.parameters(), lr=1e-4)
            self.lr_scheduler = MultiStepLR(self.optimizer, milestones=[int(self.epoch * 0.5), int(self.epoch * 0.7)],
                                            gamma=0.1)

        elif self.epoch_step == 10:
            self.optimizer = Adam(self.model.parameters(), lr=1e-5)
            self.lr_scheduler = MultiStepLR(self.optimizer, milestones=[int(self.epoch * 0.5), int(self.epoch * 0.7)],
                                            gamma=0.1)


if __name__ == '__main__':
    train = Train_dot_attention_capsule_network(batch=128, epoch=12, lr=1e-3, log_path="./train dot attention capsule network with dot product attention.txt")
    train.log.flush_print_info = True
    net = capsule_model_with_dot_produc_attention_routing()
    train.run()
