import torch
import json
import sys
from torch.utils.data import DataLoader
import os
import pickle
import random
from tqdm import tqdm
from functools import partial
sys.path.append("src/REDN")

from opennre import encoder, model, framework
from opennre.framework.data_loader import SentenceREDataset
from opennre.framework.f1_metric import F1Metric
from example import configs as cf
from opennre.model.para_loss import PARALoss
from opennre.model.para_loss_softmax import PARALossSoftmax

tqdm = partial(tqdm, position=0, leave=False)
def train(dataset_name, batch_size=50, num_workers=0, max_epoch=15, lr=3e-5, weight_decay=1e-5, add_subject_loss=False,
          eval=False, continue_train=False, large_bert=False, subject_1=False, use_cls=True, softmax=False,
          opt='adam', seed=31415926535897932, cuda_device=0, sort=True, metric="micro_f1"
          ):
    print("@@@@@@@@@@@ args @@@@@@@@@@@")
    print(locals())
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

    if seed is not None:
        torch.manual_seed(seed)

    root_path = 'src/REDN/Datasets'
    dataset_file = ["First/5%_train/train.json", "First/5%_train/dev.json", "First/5%_train/test.json"]
    dataset_pkl_file = ["First/5%_train/train.pkl", "First/5%_train/dev.pkl", "First/5%_train/test.pkl"]

    bert_path, bert_large = 'src/REDN/pretrain/bert-base-uncased', False

    ckpt = 'Checkpoints/%s_%s_%s_%s_%s_%s_%s_bert.th' % (
        "5%_train",
        cf.outputname,
        dataset_name,
        "softmax" if softmax else "sigmoid",
        "withCLS" if use_cls else "withoutCLS",
        "-1" if subject_1 else "-2",
        "1024" if bert_large else "768",)

    def get_dataset(_model):
        if all(map(lambda x: os.path.exists(os.path.join(root_path, dataset_name, x)), dataset_pkl_file)):
            dataset = list(
                map(lambda x: pickle.load(open(os.path.join(root_path, dataset_name, x), "rb")), dataset_pkl_file))

            if softmax:
                list(map(lambda x: x.split(), dataset))
        else:
            dataset = list(
                map(lambda x: SentenceREDataset(path=os.path.join(root_path, dataset_name, x), rel2id=rel2id,
                                                tokenizer=_model.sentence_encoder.tokenize, kwargs=None,
                                                sort=sort),
                    dataset_file))

            list(map(lambda x, y: pickle.dump(x, open(os.path.join(root_path, dataset_name, y), "wb")), dataset,
                     dataset_pkl_file))

        if dataset_name in ["nyt10", "nyt10_1", "nyt10_2"]:
            list(map(lambda x: x.set_max_words(100), dataset))
            list(map(lambda x: x.remove_na(), dataset))
            # list(map(lambda x: x.remove_repeat(), dataset))
            list(map(lambda x: x.char_idx_to_word_idx(), dataset))
            for d in dataset:
                d.NA_id = -1
        if dataset_name in ["semeval_1"]:
            for d in dataset:
                d.NA_id = -1
        if dataset_name in ["webnlg", "webnlg_1"]:
            for d in dataset:
                d.NA_id = -1
        if dataset_name in ["fewrel"]:
            # list(map(lambda x: x.set_max_words(64), dataset))
            for d in dataset:
                d.NA_id = -1

        dataset_loader = list(map(
            lambda x: DataLoader(dataset=x, batch_size=batch_size, shuffle=False, pin_memory=True,
                                 num_workers=num_workers, collate_fn=SentenceREDataset.collate_fn), dataset))

        return dataset_loader

    rel2id = json.load(open(os.path.join(root_path, dataset_name, 'First/rel2id.json')))

    sentence_encoder = encoder.BERTHiddenStateEncoder(pretrain_path=bert_path)
    _model = model.PARA(sentence_encoder, len(rel2id), rel2id, num_token_labels=2, subject_1=subject_1, use_cls=use_cls)

    train_loader, val_loader, test_loader = get_dataset(_model)

    _framework = framework.SentenceRE(
        train_loader=train_loader,
        val_loader=val_loader if dataset_name not in ["nyt10", "nyt10_1"] else test_loader,
        test_loader=test_loader,
        model=_model,
        ckpt=ckpt,
        max_epoch=max_epoch,
        lr=lr,
        weight_decay=weight_decay,
        opt=opt,
        add_subject_loss=add_subject_loss,
        loss_func=PARALossSoftmax() if softmax else PARALoss(),
        metric=F1Metric(multi_label=not softmax,
                        na_id=train_loader.dataset.NA_id,
                        ignore_na=dataset_name == "semeval",
                        rel2id=rel2id,
                        print_error_prob=1
                        ),
    )

    if not eval:
        if continue_train:
            _framework.parallel_model.load_state_dict(torch.load(ckpt).state_dict())
        _framework.train_model(metric=metric)
    _framework.parallel_model.load_state_dict(torch.load(ckpt).state_dict())

    print("TRAIN---------------------------")
    result = _framework.eval_model(_framework.train_loader)
    print('Accuracy on test set: {}'.format(result['acc']))
    print('Micro Precision: {}'.format(result['micro_p']))
    print('Micro Recall: {}'.format(result['micro_r']))
    print('Micro F1: {}'.format(result['micro_f1']))
    
    print("DEV---------------------------")
    result = _framework.eval_model(_framework.val_loader)
    print('Accuracy on test set: {}'.format(result['acc']))
    print('Micro Precision: {}'.format(result['micro_p']))
    print('Micro Recall: {}'.format(result['micro_r']))
    print('Micro F1: {}'.format(result['micro_f1']))

    print("TEST---------------------------")
    result = _framework.eval_model(_framework.test_loader)
    print('Accuracy on test set: {}'.format(result['acc']))
    print('Micro Precision: {}'.format(result['micro_p']))
    print('Micro Recall: {}'.format(result['micro_r']))
    print('Micro F1: {}'.format(result['micro_f1']))

    ### Uncomment this section and comment other test sections
    ### if use Few Shot Evaluation. In the "main" function, 
    ### set eval=True
    # print("Few Shot Evaluation------------")
    # torch.save(_framework.parallel_model, ckpt)
    # _framework.parallel_model.load_state_dict(torch.load(ckpt).state_dict())
    # t = tqdm(range(1000), leave=False)
    # acc = []
    # for i in t:
    #   result = _framework.eval_model_few_shot(_framework.test_loader, K=5, Q=1, iter=1)
    #   acc.append(result['acc'])
    #   _framework.parallel_model.load_state_dict(torch.load(ckpt).state_dict())
    #   t.set_postfix(acc=sum(acc)/len(acc))
    # print('Accuracy on test set: {}'.format(sum(acc)/len(acc)))

    if os.path.exists(os.path.join(root_path, dataset_name, "test.txt")):
        test_sample_dataset = SentenceREDataset(path=os.path.join(root_path, dataset_name, "test.txt"), rel2id=rel2id,
                                                tokenizer=_model.sentence_encoder.tokenize, kwargs=None,
                                                sort=sort)
        test_sample_loader = DataLoader(dataset=test_sample_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                        num_workers=num_workers, collate_fn=SentenceREDataset.collate_fn)
        print("TEST-Sample--------------------")
        result = _framework.eval_model(test_sample_loader)
        print('Accuracy on test set: {}'.format(result['acc']))
        print('Micro Precision: {}'.format(result['micro_p']))
        print('Micro Recall: {}'.format(result['micro_r']))
        print('Micro F1: {}'.format(result['micro_f1']))
        _framework.metric.df.to_excel(os.path.join(root_path, dataset_name, "res.xlsx"))


def get_ablation_args(dataset, max_epoch, batch_size, **kwargs):
    _args_list = []

    args = {"dataset_name": dataset, "max_epoch": max_epoch, "batch_size": batch_size,
            "subject_1": False, "use_cls": True, "softmax": False, }
    args.update(kwargs)
    _args_list.append(args.copy())
    args["subject_1"] = True
    _args_list.append(args.copy())
    args["use_cls"] = False
    _args_list.append(args.copy())
    args["softmax"] = True
    _args_list.append(args.copy())
    return _args_list


if __name__ == '__main__':
    dataset_name = 'pubmed'
    is_train = "t"

    if dataset_name in ["semeval", "semeval_1"]:
        max_epoch = 20
        batch_size = 1
        args_list = get_ablation_args(dataset_name,
                                      max_epoch=max_epoch,
                                      batch_size=batch_size,
                                      cuda_device=1,
                                      # continue_train=True,
                                      # seed=None,
                                      eval=not is_train,
                                      )
        train(**args_list[0])
    
    elif dataset_name in ["pubmed"]:
        max_epoch = 20
        batch_size = 1
        args_list = get_ablation_args(dataset_name,
                                      max_epoch=max_epoch,
                                      batch_size=batch_size,
                                      cuda_device=1,
                                      sort=False,
                                      eval=not is_train,
                                      )
        train(**args_list[0])
      
    elif dataset_name in ["fewrel"]:
        max_epoch = 30
        batch_size = 10
        args_list = get_ablation_args(dataset_name,
                                      max_epoch=max_epoch,
                                      batch_size=batch_size,
                                      cuda_device=1,
                                      sort=False,
                                      eval=not is_train,
                                      )
        train(**args_list[0])

    elif dataset_name in ["nyt10", "nyt10_1", "nyt10_2"]:
        max_epoch = 20
        batch_size = 1
        args_list = get_ablation_args(dataset_name,
                                      max_epoch=max_epoch,
                                      batch_size=batch_size,
                                      cuda_device=3,
                                      lr=5e-5,
                                      sort=False,
                                      eval=not is_train,
                                      )
        train(**args_list[0])