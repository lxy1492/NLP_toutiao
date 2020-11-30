import os
from train import Train
from src.BertATT.args import get_args
from src.BertATT.BertATT import BertATT

def getNet():
    data_dir = os.path.join("../2020-FlyAI-Today-s-Headlines-By-Category-master/BERT", "data/input").replace("\\", "/")
    output_dir = os.path.join("../2020-FlyAI-Today-s-Headlines-By-Category-master/BERT", "data/output").replace("\\", "/")
    cache_dir = os.path.join("../2020-FlyAI-Today-s-Headlines-By-Category-master/BERT", "data/cache").replace("\\", "/")
    bert_vocab_file = os.path.join("../2020-FlyAI-Today-s-Headlines-By-Category-master/BERT", 'data/input/model/vocab.txt').replace("\\", "/")
    bert_model_dir = os.path.join("../2020-FlyAI-Today-s-Headlines-By-Category-master/BERT", 'data/input/model').replace("\\", "/")
    log_dir = os.path.join("../2020-FlyAI-Today-s-Headlines-By-Category-master/BERT", "data/log").replace("\\", "/")
    args = get_args(
        data_dir=data_dir,
        output_dir=output_dir,
        cache_dir=cache_dir,
        bert_vocab_file=bert_vocab_file,
        bert_model_dir=bert_model_dir,
        log_dir=log_dir
    )
    # print(args)
    net = BertATT(args)
    return net

def run():
    train = Train(net=getNet(), bath=16)
    train.train_epoch()


if __name__ == '__main__':
    run()
