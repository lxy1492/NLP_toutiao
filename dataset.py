import os
import torch
import numpy as np
from data.loaddata import dataloader2
from pytorch_transformers import BertTokenizer

bert_vocab_file = os.path.join("../2020-FlyAI-Today-s-Headlines-By-Category-master/BERT", 'data/input/model/vocab.txt').replace("\\", "/")
bert_model_dir = os.path.join("../2020-FlyAI-Today-s-Headlines-By-Category-master/BERT", 'data/input/model').replace("\\", "/")

class data_loader(dataloader2):

    def __init__(self, batchsize=64, sentenceMaxLength=150, dataDIR="./data", fileformate="tsv", device=None, read_encoding="utf-8"):
        super().__init__(batchsize=batchsize, dataDIR=dataDIR, fileformate=fileformate, device=device, read_encoding=read_encoding)
        self.tockenizer = BertTokenizer(bert_vocab_file).from_pretrained(bert_model_dir, do_lower_case=True)
        self.sentence_maxlength = sentenceMaxLength

    def tockenize(self, sentence):
        sentence = np.asarray(["[CLS]"] + self.tockenizer.tokenize(sentence) + ["[SEP]"])
        sentence = self.tockenizer.convert_tokens_to_ids(sentence)
        return sentence

    def convertSentence(self, sentence, mask=None, segment=None):
        segments = []
        masks = []
        sentences = []
        if not isinstance(mask, bool):
            mask = self.mask
        if not isinstance(mask, bool):
            segment = self.segment
        for eachSentence in sentence:
            if mask==False and segment==False:
                break
            length = len(eachSentence)
            segment_ids = [0] * length
            input_mask = [1] * length
            padding = [0] * (self.sentence_maxlength - length)
            eachSentence += padding
            input_mask += padding
            segment_ids += padding
            assert len(eachSentence) == self.sentence_maxlength
            assert len(input_mask) == self.sentence_maxlength
            assert len(segment_ids) == self.sentence_maxlength
            sentences.append(eachSentence)
            segments.append(segment_ids)
            masks.append(input_mask)
        sentences = torch.tensor(sentences, dtype=torch.long).to(self.device)
        if masks==True:
            masks = torch.tensor(masks, dtype=torch.long).to(self.device)
        else:
            masks = None
        if segment:
            segments = torch.tensor(segments, dtype=torch.long).to(self.device)
        else:
            segments = None
        return sentences, masks, segments

    def rebatch(self, obj="train"):
        if obj=="train":
            self.trainIndex = 0
        elif obj=="test":
            self.testIndex = 0
        elif obj == "val":
            self.valIndex = 0
        else:
            self.trainIndex = 0
            self.testIndex = 0
            self.valIndex = 0

if __name__ == '__main__':
    data = data_loader()
    data.readFile(obj="train")
    # print(data.trainData)
    for each in data:
        print(each)