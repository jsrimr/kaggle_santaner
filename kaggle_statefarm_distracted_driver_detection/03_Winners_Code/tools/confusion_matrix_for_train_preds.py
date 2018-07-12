
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from glob import glob
from sklearn.metrics import confusion_matrix
import keras

true_path = 'input/driver_imgs_list.csv'
pred_path = 'resnet_fold9_test.csv'
mode = 'test'
nclasses = 10
columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']

if mode == 'train': 
    # get true
    true = []
    drivers = pd.read_csv(true_path)
    img_to_driver = {}
    for i, row in drivers.iterrows():
        label_n_driver = {}
        label_n_driver['label'] = int(row['classname'].split('c')[1])
        label_n_driver['driver'] = row['subject']
        img_to_driver[row['img']] = label_n_driver

# get pred
pred = pd.read_csv(pred_path)
pred_probs = []
for i, row in pred.iterrows():
    if mode == 'train':
        temp = np.zeros(nclasses)
        label = img_to_driver[row['img']]['label']
        temp[label] = 1.
        true.append(temp)
    else:
        label = 0
    pred_probs.append([max(row[columns].values), label])

if mode == 'train':
    true = np.argmax(np.array(true), axis=-1)
    pred = np.argmax(pred[columns].values, axis=-1)
    print(confusion_matrix(true, pred))

pred_probs = pd.DataFrame(pred_probs, columns=['prob', 'label'])
pred_probs_plot = pred_probs['prob'].hist(bins=100)
plt.savefig('myplot.{}.png'.format(mode))

print(pred_probs['prob'].mean(), pred_probs['prob'].max(), pred_probs['prob'].min(), pred_probs['prob'].std())
