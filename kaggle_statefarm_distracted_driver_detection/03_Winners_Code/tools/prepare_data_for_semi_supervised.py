import pandas as pd
import numpy as np
import os
import sys

test_pred = sys.argv[1]
test_pred = pd.read_csv(test_pred)
test_pred_probs = test_pred.iloc[:, :-1]
test_pred_probs_max = np.max(test_pred_probs.values, axis=1)

for thr in range(1,10):
  thr = thr / 10.
  count = sum(test_pred_probs_max > thr)
  print('# Thre : {} | count : {} ({}%)'.format(thr, count, 1. * count / len(test_pred_probs_max)))

print('=' * 50)
threshold = 0.7
count = {}
print('# Extracting data with threshold : {}'.format(threshold))
for i, row in test_pred.iterrows():
  img = row['img']
  row = row.iloc[:-1]
  if np.max(row) > threshold:
    label = row.values.argmax()
    cmd = 'cp input/test/imgs/{} input/semi_train/c{}/{}'.format(img, label, img)
    os.system(cmd)
    count[label] = count.get(label, 0) + 1

print('# Added semi-supservised labels: \n{}'.format(count))
