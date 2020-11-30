import os
import torch
from net import Net
from utils import NMTCriterion
from dataset import data_loader
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

