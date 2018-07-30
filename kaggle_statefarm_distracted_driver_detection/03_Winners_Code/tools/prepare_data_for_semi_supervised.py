import pandas as pd
import numpy as np
import os
import sys

test_pred_fname = sys.argv[1]
test_pred = pd.read_csv(test_pred_fname)
test_pred_probs = test_pred.iloc[:, :-1]
test_pred_probs_max = np.max(test_pred_probs.values, axis=1)

for thr in range(1,10):
  thr = thr / 10.
  count = sum(test_pred_probs_max > thr)
  print('# Thre : {} | count : {} ({}%)'.format(thr, count, 1. * count / len(test_pred_probs_max)))

print('=' * 50)
threshold = 0.90
count = {}
print('# Extracting data with threshold : {}'.format(threshold))
cmd = 'cp -r input/train input/semi_train_{}'.format(os.path.basename(test_pred_fname))
os.system(cmd)
for i, row in test_pred.iterrows():
  img = row['img']
  row = row.iloc[:-1]
  if np.max(row) > threshold:
    label = row.values.argmax()
    cmd = 'cp input/test/imgs/{} input/semi_train_{}/c{}/{}'.format(img, os.path.basename(test_pred_fname), label, img)
    os.system(cmd)
    count[label] = count.get(label, 0) + 1

print('# Added semi-supservised labels: \n{}'.format(count))
