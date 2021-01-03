# Refactored from FEWREL github : https://github.com/thunlp/FewRel
import sys
import numpy as np
import json
import argparse
import os
import models
import torch
import transformers
sys.path.append('src/FewRel-master')

from fewshot_re_kit.data_loader import get_loader, get_loader_pair, get_loader_unsupervised
from fewshot_re_kit.framework import FewShotREFramework
from fewshot_re_kit.sentence_encoder import BERTSentenceEncoder, BERTPAIRSentenceEncoder
from models.pair import Pair
from transformers import AdamW
from torch import optim, nn
from dotmap import DotMap

def main():
    opt = {'train': 'train_wiki',
           'val': 'val_wiki',
           'test': 'val_semeval',
           'adv': None,
           'trainN': 10,
           'N': 5,
           'K': 1,
           'Q': 1,
           'batch_size': 1,
           'train_iter': 5,
           'val_iter': 1000,
           'test_iter': 10,
           'val_step': 2000,
           'model': 'pair',
           'encoder': 'bert',
           'max_length': 64,
           'lr': -1,
           'weight_decay': 1e-5,
           'dropout': 0.0,
           'na_rate':0,
           'optim': 'adam',
           'load_ckpt': './src/FewRel-master/Checkpoints/ckpt_5-Way-1-Shot_FewRel.pth',
           'save_ckpt': './src/FewRel-master/Checkpoints/post_ckpt_5-Way-1-Shot_FewRel.pth',
           'fp16':False,
           'only_test': False,
           'ckpt_name': 'Checkpoints/ckpt_5-Way-1-Shot_FewRel.pth',
           'pair': True,
           'pretrain_ckpt': '',
           'cat_entity_rep': False,
           'dot': False,
           'no_dropout': False,
           'mask_entity': False,
           'use_sgd_for_bert': False
           }
    opt = DotMap(opt)
    trainN = opt.trainN
    N = opt.N
    K = opt.K
    Q = opt.Q
    batch_size = opt.batch_size
    model_name = opt.model
    encoder_name = opt.encoder
    max_length = opt.max_length
    
    print("{}-way-{}-shot Few-Shot Relation Classification".format(N, K))
    print("model: {}".format(model_name))
    print("encoder: {}".format(encoder_name))
    print("max_length: {}".format(max_length))

    pretrain_ckpt = opt.pretrain_ckpt or 'bert-base-uncased'
    sentence_encoder = BERTPAIRSentenceEncoder(
            pretrain_ckpt,
            max_length)
    
    train_data_loader = get_loader_pair(opt.train, sentence_encoder,
            N=trainN, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size, encoder_name=encoder_name)
    val_data_loader = get_loader_pair(opt.val, sentence_encoder,
            N=N, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size, encoder_name=encoder_name)
    test_data_loader = get_loader_pair(opt.test, sentence_encoder,
            N=N, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size, encoder_name=encoder_name)
   
    if opt.optim == 'sgd':
        pytorch_optim = optim.SGD
    elif opt.optim == 'adam':
        pytorch_optim = optim.Adam
    elif opt.optim == 'adamw':
        pytorch_optim = AdamW
    else:
        raise NotImplementedError
    framework = FewShotREFramework(train_data_loader, val_data_loader, test_data_loader)
        
    prefix = '-'.join([model_name, encoder_name, opt.train, opt.val, str(N), str(K)])
    if opt.na_rate != 0:
        prefix += '-na{}'.format(opt.na_rate)
    if opt.dot:
        prefix += '-dot'
    if opt.cat_entity_rep:
        prefix += '-catentity'
    if len(opt.ckpt_name) > 0:
        prefix += '-' + opt.ckpt_name

    model = Pair(sentence_encoder, hidden_size=opt.hidden_size)
    
    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')
    ckpt = 'checkpoint/{}.pth.tar'.format(prefix)
    if opt.save_ckpt:
        ckpt = opt.save_ckpt

    if torch.cuda.is_available():
        model.cuda()

    if not opt.only_test:
        if encoder_name in ['bert', 'roberta']:
            bert_optim = True
        else:
            bert_optim = False

        if opt.lr == -1:
            if bert_optim:
                opt.lr = 2e-5
            else:
                opt.lr = 1e-1

        framework.train(model, prefix, batch_size, trainN, N, K, Q,
                pytorch_optim=pytorch_optim, load_ckpt=opt.load_ckpt, save_ckpt=ckpt,
                na_rate=opt.na_rate, val_step=opt.val_step, fp16=opt.fp16, pair=opt.pair, 
                train_iter=opt.train_iter, val_iter=opt.val_iter, bert_optim=bert_optim, 
                learning_rate=opt.lr, use_sgd_for_bert=opt.use_sgd_for_bert)
    else:
        ckpt = opt.load_ckpt
        if ckpt is None:
            print("Warning: --load_ckpt is not specified. Will load Huggingface pre-trained checkpoint.")
            ckpt = 'none'

    acc = framework.eval(model, batch_size, N, K, Q, opt.test_iter, na_rate=opt.na_rate, ckpt=ckpt, pair=opt.pair)
    print("RESULT: %.2f" % (acc * 100))

if __name__ == "__main__":
    main()