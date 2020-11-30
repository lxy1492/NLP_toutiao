#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2019 Apple Inc. All Rights Reserved.
#
from src.dot_attention_capsule import layers
import torch.nn as nn
import torch.nn.functional as F
import torch

from src.capsule import CompositionalEmbedding

# Capsule model
class CapsModel(nn.Module):
    def __init__(self,
                 vocab_size,
                 backbone,
                 dp,
                 num_routing,
                 num_codebook,
                 num_codeword,
                 in_channels=1,
                 num_repeat=None,
                 out_channels=128,
                 sequential_routing=True,
                 embedding_dim=150,
                 sentenceLength=150,
                 classes=4,
                 embedding_type="cwc"
                 ):

        super(CapsModel, self).__init__()

        if embedding_type == 'cwc':
            self.embedding = CompositionalEmbedding(vocab_size, embedding_dim, num_codebook, num_codeword,
                                                    weighted=True)
        elif embedding_type == 'cc':
            self.embedding = CompositionalEmbedding(vocab_size, embedding_dim, num_codebook, num_codeword, num_repeat,
                                                    weighted=False)
        else:
            # 将id化后的语料库，映射到低维稠密的向量空间中
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        #### Parameters
        self.sequential_routing = sequential_routing
        self.embedding_dim = embedding_dim
        self.sentenceLength = sentenceLength
        ## Primary Capsule Layer
        self.pc_num_caps = 32
        self.pc_caps_dim = 16
        self.pc_output_dim = [sentenceLength // 2, 1]
        ## General
        self.num_routing = num_routing  # >3 may cause slow converging
        # print("self.num_routing", num_routing)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.classes = classes

        #### Building Networks
        ## Backbone (before capsule)
        if backbone == 'simple':
            self.pre_caps = layers.simple_backbone(self.in_channels, self.out_channels, (3, self.embedding_dim), 2,
                                                   (1, 0))
        elif backbone == 'resnet':
            self.pre_caps = layers.resnet_backbone(self.in_channels, self.out_channels, 2, self.embedding_dim)
        elif backbone == "gru":
            self.pre_caps = nn.GRU(self.embedding_dim, 128, num_layers=2, dropout=0.5, batch_first=True,
                                   bidirectional=True)
        ## Primary Capsule Layer (a single CNN)
        # 调整通道
        self.pc_layer = nn.Conv2d(in_channels=self.out_channels, out_channels=self.pc_num_caps * self.pc_caps_dim,
                                  kernel_size=1, stride=1, padding=0)

        self.nonlinear_act = nn.LayerNorm(self.pc_caps_dim)

        ## Main Capsule Layers
        self.capsule_layers = nn.ModuleList([])

        self.capsule_layers.append(
            layers.CapsuleCONV(
                in_n_capsules=self.pc_num_caps,
                in_d_capsules=self.pc_caps_dim,
                out_n_capsules=32,
                out_d_capsules=16,
                kernel_size=(3, 1),
                stride=2,
                matrix_pose=True,
                dp=dp,
                coordinate_add=False
            )
        )
        self.capsule_layers.append(
            layers.CapsuleCONV(
                in_n_capsules=self.pc_num_caps,
                in_d_capsules=self.pc_caps_dim,
                out_n_capsules=32,
                out_d_capsules=16,
                kernel_size=(3, 1),
                stride=1,
                matrix_pose=True,
                dp=dp,
                coordinate_add=False,
            )
        )

        if self.embedding_dim == 50:
            in_n_caps = 32 * 22 * 1
        elif self.embedding_dim == 100:
            in_n_caps = 32 * 28 * 1
        elif self.embedding_dim == 150:
            in_n_caps = 32 * 35 * 1
        else:
            in_n_caps = 32 * 32 * 1
        in_d_caps = 16

        self.capsule_layers.append(
            layers.CapsuleFC(
                in_n_capsules=in_n_caps,
                in_d_capsules=in_d_caps,
                out_n_capsules=self.classes,
                out_d_capsules=self.pc_caps_dim,
                matrix_pose=True,
                dp=dp
            )
        )

        ## After Capsule
        # fixed classifier for all class capsules
        self.final_fc = nn.Linear(16, 1)

    def forward(self, x, lbl_1=None, lbl_2=None):
        #### Forward Pass
        ## Backbone (before capsule)
        x = self.embedding(x) # 输入：[batch embedding dim, embedding_dim]
        # print(x.shape) # torch.Size([batch_size, 150, 150])
        x = torch.unsqueeze(x, 1)
        # print(x.shape) # torch.Size([batch_size, 1, 150, 150])
        c = self.pre_caps(x)
        # print(c.shape) # torch.Size([batch_size, 128, 75, 1])
        ## Primary Capsule Layer (a single CNN)
        u = self.pc_layer(c)
        # print(u.shape) # torch.Size([batch_size, 512, 75, 1])
        u = u.permute(0, 2, 3, 1) # 将通道放到后面，height和weight放在中间
        u = u.view(u.shape[0], self.pc_output_dim[0], self.pc_output_dim[1], self.pc_num_caps,
                   self.pc_caps_dim)
        # print(u.shape) torch.Size([batch size, 75, 1, 32, 16])
        u = u.permute(0, 3, 1, 2, 4)
        # print(u.shape) # torch.Size([batch size, 32, 75, 1, 16])
        init_capsule_value = self.nonlinear_act(u)  # capsule_utils.squash(u)

        ## Main Capsule Layers
        # concurrent routing
        if not self.sequential_routing:
            # first iteration
            # perform initilialization for the capsule values as single forward passing
            capsule_values, _val = [init_capsule_value], init_capsule_value
            for i in range(len(self.capsule_layers)):
                # print(_val.shape)
                _val = self.capsule_layers[i].forward(_val, 0)
                capsule_values.append(_val)  # get the capsule value for next layer

            # second to t iterations
            # perform the routing between capsule layers
            for n in range(self.num_routing - 1):
                _capsule_values = [init_capsule_value]
                for i in range(len(self.capsule_layers)):
                    _val = self.capsule_layers[i].forward(capsule_values[i], n, capsule_values[i + 1])
                    _capsule_values.append(_val)
                capsule_values = _capsule_values
        # sequential routing
        else:
            capsule_values, _val = [init_capsule_value], init_capsule_value
            for i in range(len(self.capsule_layers)):
                # first iteration
                # print(_val.shape) # torch.Size([128, 32, 35, 1, 16])
                # 以第三个维度调整 in_n_caps
                __val = self.capsule_layers[i].forward(_val, 0)
                # second to t iterations
                # perform the routing between capsule layers
                for n in range(self.num_routing - 1):
                    __val = self.capsule_layers[i].forward(_val, n, __val)
                _val = __val
                capsule_values.append(_val)
        ## After Capsule
        out = capsule_values[-1]
        # print(out.shape) # torch.Size([batch size, 15, 16])
        out = self.final_fc(out)  # fixed classifier for all capsules
        # print(out.shape)  # torch.Size([batch size, 15, 1])
        out = out.squeeze()  # fixed classifier for all capsules
        # print(out.shape) # torch.Size([batch size, 15])
        # out = torch.einsum('bnd, nd->bn', out, self.final_fc) # different classifiers for distinct capsules
        # print(out.shape) # torch.Size([batch size, 15])
        return out
