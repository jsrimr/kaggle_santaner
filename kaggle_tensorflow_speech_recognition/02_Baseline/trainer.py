"""
model trainer
"""
from torch.autograd import Variable
from data import SpeechDataset
from torch.utils.data import DataLoader
import torch
from time import time
from torch.nn import Softmax
import numpy as np
import pandas as pd
import os
from random import choice
from resnet import ResModel
from tqdm import tqdm

def create_directory(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def get_time(now, start):
    time_in_min = int((now - start) / 60)
    return time_in_min

# set params
BATCH_SIZE = 32
mGPU = False
epochs = 20
mode = 'test' # 'cv' or 'test'
model_name = 'model/model_resnet.pth'

# load model
loss_fn = torch.nn.CrossEntropyLoss()
model = ResModel
speechmodel = torch.nn.DataParallel(model()) if mGPU else model()
speechmodel = speechmodel.cuda()


# load data
labels = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
label_to_int = dict(zip(labels, range(len(labels))))
int_to_label = dict(zip(range(len(labels)), labels))
int_to_label.update({len(labels): 'unknown', len(labels) + 1: 'silence'})

trn = 'input/trn.txt' if mode == 'cv' else 'input/trn_all.txt'
tst = 'input/val.txt' if mode == 'cv' else 'input/tst.txt'

trn = [line.strip() for line in open(trn, 'r').readlines()]
wav_list = [line.split(',')[-1] for line in trn]
label_list = [line.split(',')[0] for line in trn]

traindataset = SpeechDataset(mode='train', label_to_int=label_to_int, wav_list=wav_list, label_list=label_list)

# train
start_time = time()
for e in range(epochs):
    print("training epoch ", e)
    learning_rate = 0.01 if e < 10 else 0.001
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, speechmodel.parameters()), lr=learning_rate, momentum=0.9, weight_decay=0.00001)
    speechmodel.train()

    total_correct = 0
    num_labels = 0
    trainloader = DataLoader(traindataset, BATCH_SIZE, shuffle=True)
    for batch_idx, batch_data in enumerate(tqdm(trainloader)):
        spec = batch_data['spec']
        label = batch_data['label']
        spec, label = Variable(spec.cuda()), Variable(label.cuda())
        y_pred = speechmodel(spec)
        _, pred_labels = torch.max(y_pred.data, 1)
        correct = (pred_labels == label.data).sum()
        loss = loss_fn(y_pred, label)

        total_correct += correct
        num_labels += len(label)
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print("training accuracy:", 100. * total_correct / num_labels, get_time(time(), start_time))
    if mode == 'cv':
        torch.save(speechmodel.state_dict(), '{}_cv'.format(model_name))
        
        softmax = Softmax()
        tst_list = [line.strip() for line in open(tst, 'r').readlines()]
        wav_list = [line.split(',')[-1] for line in tst_list]
        label_list = [line.split(',')[0] for line in tst_list]
        cvdataset = SpeechDataset(mode='test', label_to_int=label_to_int, wav_list=wav_list)
        cvloader = DataLoader(cvdataset, BATCH_SIZE, shuffle=False)

        speechmodel = torch.nn.DataParallel(model()) if mGPU else model()
        speechmodel.load_state_dict(torch.load('{}_cv'.format(model_name)))
        speechmodel = speechmodel.cuda()
        speechmodel.eval()

        fnames, preds = [], []
        for batch_idx, batch_data in enumerate(tqdm(cvloader)):
            spec = Variable(batch_data['spec'].cuda())
            fname = batch_data['id']
            y_pred = softmax(speechmodel(spec))
            preds.append(y_pred.data.cpu().numpy())
            fnames += fname

        preds = np.vstack(preds)
        preds = [int_to_label[x] for x in np.argmax(preds, 1)]
        fnames = [fname.split('/')[-2] for fname in fnames]
        num_correct = 0
        for true, pred in zip(fnames, preds):
            if true == pred:
                num_correct += 1
        print("cv accuracy:", 100. * num_correct / len(preds), get_time(time(), start_time))

# save model
create_directory("model")
torch.save(speechmodel.state_dict(), model_name)


if mode != 'cv':
    # eval on test
    print("doing prediction...")
    softmax = Softmax()
    tst = [line.strip() for line in open(tst, 'r').readlines()]
    wav_list = [line.split(',')[-1] for line in tst]
    testdataset = SpeechDataset(mode='test', label_to_int=label_to_int, wav_list=wav_list)
    testloader = DataLoader(testdataset, BATCH_SIZE, shuffle=False)

    speechmodel = torch.nn.DataParallel(model()) if mGPU else model()
    speechmodel.load_state_dict(torch.load(model_name))
    speechmodel = speechmodel.cuda()
    speechmodel.eval()
    
    test_fnames, test_labels = [], []
    pred_scores = []

    ## (tst) save submission
    for batch_idx, batch_data in enumerate(tqdm(testloader)):
        spec = Variable(batch_data['spec'].cuda())
        fname = batch_data['id']
        y_pred = softmax(speechmodel(spec))
        pred_scores.append(y_pred.data.cpu().numpy())
        test_fnames += fname

    final_pred = np.vstack(pred_scores)
    final_labels = [int_to_label[x] for x in np.argmax(final_pred, 1)]
    test_fnames = [x.split("/")[-1] for x in test_fnames]

    create_directory("sub")
    pd.DataFrame({'fname': test_fnames, 'label': final_labels}).to_csv("sub/{}.csv".format(model_name.split('/')[-1]), index=False)

