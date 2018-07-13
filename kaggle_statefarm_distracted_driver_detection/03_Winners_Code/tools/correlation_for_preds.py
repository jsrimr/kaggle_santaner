import pandas as pd
import numpy as np
import sys

pre = sys.argv[1]
suf = sys.argv[2]

pre = pd.read_csv(pre, index_col=-1)
suf = pd.read_csv(suf, index_col=-1)

pre = np.argmax(pre.values, axis=1)
suf = np.argmax(suf.values, axis=1)

print('# Corr : \n{}'.format(np.corrcoef(pre, suf)))
