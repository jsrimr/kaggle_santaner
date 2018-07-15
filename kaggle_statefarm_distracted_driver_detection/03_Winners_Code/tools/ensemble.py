from glob import glob
import pandas as pd
import os

fls = glob('subm/*/*.csv')
ensemble = 0
for i, fl in enumerate(fls):
    ensemble += pd.read_csv(fl, index_col=-1).values * 1. / len(fls)
test_id = [os.path.basename(fl) for fl in glob('input/test/imgs/*.jpg')]
ensemble = pd.DataFrame(ensemble, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
ensemble.loc[:, 'img'] = pd.Series(test_id, index=ensemble.index)
sub_file = 'ensemble.csv'
ensemble.to_csv(sub_file, index=False)

