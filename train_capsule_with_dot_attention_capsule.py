from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from train_dot_attention_capsule import Train_dot_attention_capsule_network
from src.capsule_with_dot_product_attention import Model

class Train_capsule_with_dot_product_attention(Train_dot_attention_capsule_network):
    def __init__(self, batch=128, epoch=10, lr=1e-2, log_path="./", model_path="./model.pkl"):
        super().__init__(batch=batch, epoch=epoch, lr=lr, log_path=log_path, model_path=model_path)

    def set_model(self, model=None):
        self.model = Model(num_routing=3).to(self.device)

    def set_optimizer(self, repeat=None):
        if repeat is None:
            optim_configs = [{'params': self.model.embedding.parameters(), 'lr': self.lr},
                             {'params': self.model.features.parameters(), 'lr': self.lr},
                             {'params': self.model.nonlinear_act.parameters(), 'lr': self.lr/10},
                             {'params': self.model.capsule_layers.parameters(), 'lr': self.lr/10}]
                             # {'params': self.model.final_fc.parameters(), 'lr': self.lr/100}]
        else:
            for param in self.model.embedding.parameters():
                param.requires_grad = False
            for param in self.model.features.parameters():
                param.requires_grad = False
            optim_configs = [{'params': self.model.classifier.parameters(), 'lr': 1e-4}]
        self.optimizer = Adam(optim_configs, lr=1e-4)
        self.lr_scheduler = MultiStepLR(self.optimizer, milestones=[int(self.epoch * 0.5), int(self.epoch * 0.7)],
                                        gamma=0.1)

    def set_learning_rate(self):
        if self.epoch_step == 2:
            self.optimizer = Adam(self.model.parameters(), lr=1e-4)
            self.lr_scheduler = MultiStepLR(self.optimizer, milestones=[int(self.epoch * 0.5), int(self.epoch * 0.7)],
                                            gamma=0.1)
        elif self.epoch_step == 3:
            self.optimizer = Adam(self.model.parameters(), lr=1e-5)
            self.lr_scheduler = MultiStepLR(self.optimizer, milestones=[int(self.epoch * 0.5), int(self.epoch * 0.7)],
                                            gamma=0.1)
        elif self.epoch_step == 4:
            self.optimizer = Adam(self.model.parameters(), lr=1e-6)
            self.lr_scheduler = MultiStepLR(self.optimizer, milestones=[int(self.epoch * 0.5), int(self.epoch * 0.7)],
                                            gamma=0.1)


if __name__ == '__main__':
    Trainer = Train_capsule_with_dot_product_attention(batch=16, epoch=12, lr=1e-3, log_path="./train dot attention capsule network with dot product attention.txt")
    Trainer.run()

