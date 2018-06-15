# -*- encoding : utf-8 -*-

'''

    Generate files that contain path and label of trn, dev, trn_all, tst dataset

'''

labels = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
data_path = '~/.kaggle/competitions/tensorflow-speech-recognition-challenge'

from glob import glob
import random
import os
import numpy as np

SEED = 2018

def random_shuffle(lst):
    random.seed(SEED)
    random.shuffle(lst)
    return lst


# get trn_all
if not os.path.exists('input'):
    os.mkdir('input')

trn_all = []
trn_all_file = open('input/trn_all.txt', 'w')
files = glob(data_path + '/train/audio/*/*.wav')
for f in files:
    if '_background_noise_' in f:
        continue
    label = f.split('/')[-2]
    speaker = f.split('/')[-1].split('_')[0]
    if label not in labels:
        label = 'unknown'
        if random.random() < 0.2:
            trn_all.append((label, speaker, f))
            trn_all_file.write('{},{},{}\n'.format(label, speaker, f))
    else:
        trn_all.append((label, speaker, f))
        trn_all_file.write('{},{},{}\n'.format(label, speaker, f))
trn_all_file.close()


# divide trn_all into trn and val by speaker (9:1)
uniq_speakers = list(set([speaker for (label, speaker, path) in trn_all]))

random_shuffle(uniq_speakers)
cutoff = int(len(uniq_speakers) * 0.9)
speaker_val = uniq_speakers[cutoff:]

trn_file = open('input/trn.txt', 'w')
val_file = open('input/val.txt', 'w')
for (label, speaker, path) in trn_all:
    if speaker not in speaker_val:
        trn_file.write('{},{},{}\n'.format(label, speaker, path))
    else:
        val_file.write('{},{},{}\n'.format(label, speaker, path))
trn_file.close()
val_file.close()


# get tst
tst_all_file = open('input/tst.txt', 'w')
files = glob(data_path + '/test/audio/*.wav')
for f in files:
    tst_all_file.write(',,{}\n'.format(f))
tst_all_file.close()

