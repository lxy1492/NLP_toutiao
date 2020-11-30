import os
import torch
from net import Net
from utils import NMTCriterion
from dataset import data_loader
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bert_model_dir = os.path.join("../2020-FlyAI-Today-s-Headlines-By-Category-master/BERT", 'data/input/model').replace("\\", "/")
cache_dir = os.path.join("../2020-FlyAI-Today-s-Headlines-By-Category-master/BERT", "data/cache").replace("\\", "/")
num_labels = 15

data = data_loader()

model = Net.from_pretrained(pretrained_model_name_or_path=bert_model_dir,
                                num_labels=num_labels,
                                cache_dir=cache_dir)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

criterion = NMTCriterion(label_smoothing=0.07).to(device=DEVICE)
optimizer = AdamW(params=optimizer_grouped_parameters, lr=0.01,
                      correct_bias=False)
scheduler = WarmupLinearSchedule(optimizer=optimizer, warmup_steps=0.1,
                                     t_total=100)

for _ in range(10):
    model.train()
    for eachData in data:
        # print(eachData)
        label = eachData[0]
        sentence, mask, segment = eachData[1]
        output = model(sentence, segment, mask)
        loss =criterion(inputs=output, labels=label, normalization=1.0, reduce=False)
        loss.backward(torch.ones_like(loss))
        scheduler.step()
        _, prediction = output.max(dim=1)
        currentCorrect = prediction.eq(label).sum().item() / label.shape[0]
        print(torch.ones_like(loss))
        print(currentCorrect)