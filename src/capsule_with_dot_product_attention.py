import math

import torch
import torch.nn.functional as F
from capsule_layer import CapsuleLinear
from torch import nn
from src.capsule import CompositionalEmbedding
from torch.nn.parameter import Parameter
from src.dot_attention_capsule.layers import CapsuleCONV, CapsuleFC

def squash_fn(input, dim=-1):
    norm = input.norm(p=2, dim=dim, keepdim=True)
    scale = norm / (1 + norm ** 2)
    return scale * input

class Model(nn.Module):
    def __init__(self, num_codebook=8, num_codeword=None, hidden_size=128, in_length=8, out_length=16,
                embedding_type="cwc", caps_dim=16, num_repeat=None, num_routing=1, dropout=0.5, num_class=15, vocab_size=21128, embedding_size=150):
        super().__init__()
        if embedding_type == 'cwc':
            self.embedding = CompositionalEmbedding(vocab_size, embedding_size, num_codebook, num_codeword,
                                                    weighted=True)
        elif embedding_type == 'cc':
            self.embedding = CompositionalEmbedding(vocab_size, embedding_size, num_codebook, num_codeword, num_repeat,
                                                    weighted=False)
        else:
            # 将id化后的语料库，映射到低维稠密的向量空间中
            self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.hidden_size = hidden_size
        self.in_length = in_length
        self.out_length = out_length
        self.features = nn.GRU(embedding_size, self.hidden_size, num_layers=2, dropout=dropout, batch_first=True,
                               bidirectional=True)

        self.num_routing = num_routing
        self.pc_caps_dim = caps_dim
        self.pc_caps_num = int(hidden_size/self.pc_caps_dim)
        self.classes = num_class
        self.nonlinear_act = nn.LayerNorm(self.pc_caps_dim)
        self.capsule_layers = nn.ModuleList([])

        self.CapsLayer1 = CapsuleCONV(
            in_n_capsules=self.pc_caps_num,
            in_d_capsules=self.pc_caps_dim,
            out_d_capsules=self.pc_caps_dim,
            out_n_capsules=32,
            kernel_size=(3, 1),
            stride=2,
            matrix_pose=True,
            dp=0.0,
            coordinate_add=False
        )
        self.CapsLayer2 = CapsuleCONV(
            in_n_capsules=32,
            in_d_capsules=self.pc_caps_dim,
            out_d_capsules=self.pc_caps_dim,
            out_n_capsules=32,
            kernel_size=(3, 1),
            stride=1,
            matrix_pose=True,
            dp=0.0,
            coordinate_add=False
        )
        self.CapsFC = CapsuleFC(
            in_n_capsules=32*72,
            in_d_capsules=self.pc_caps_dim,
            out_n_capsules=self.classes,
            out_d_capsules=self.pc_caps_dim,
            matrix_pose=True,
            dp=0.0
        )

        self.capsule_layers.append(self.CapsLayer1)
        self.capsule_layers.append(self.CapsLayer2)
        self.capsule_layers.append(self.CapsFC)

        # self.final_fc = nn.Linear(self.pc_caps_dim, 1)
        self.final_fc = squash_fn

    def forward(self, x):
        # print(x.shape) #torch.Size([batchszie, 150])
        embed = self.embedding(x)
        # print(embed.shape) # torch.Size([batchszie, 150, 150])
        out, _ = self.features(embed)
        # print(out.shape) # torch.Size([batchszie, 150, 512])
        out = out[:, :, :self.hidden_size] + out[:, :, self.hidden_size:]
        # out = torch.unsqueeze(out, 2) # [batchszie, height, weight, 1, out_channels]
        out = out.view(out.size(0), out.size(1), 1, self.pc_caps_num, self.pc_caps_dim) # [batchsize, height, weight, 1, caps_num, caps_dim]
        out = out.permute(0, 3, 1, 2, 4) # [batchsize, caps_num, height, weight, 1, caps_dim]
        out = self.nonlinear_act(out)
        # print(out.shape) # torch.Size([batchsize, 16, 150, 1, 16])
        capsule_values, _val = [out], out
        for i in range(len(self.capsule_layers)):
            _val = self.capsule_layers[i].forward(_val, 0)
            capsule_values.append(_val)  # get the capsule value for next layer

        # second to t iterations
        # perform the routing between capsule layers
        for n in range(self.num_routing):
            _capsule_values = [out]
            for i in range(len(self.capsule_layers)):
                _val = self.capsule_layers[i].forward(capsule_values[i], n, capsule_values[i + 1])
                _capsule_values.append(_val)
            capsule_values = _capsule_values
        out = capsule_values[-1]
        out = self.final_fc(out)  # fixed classifier for all capsules
        out = out.norm(dim=-1)
        return out

