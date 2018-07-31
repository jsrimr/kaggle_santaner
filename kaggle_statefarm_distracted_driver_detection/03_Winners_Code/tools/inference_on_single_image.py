import sys
import keras
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from glob import glob
import os

model = load_model('cache/mvgg16.wimagenet.s18.nf10.tTrue.d2018-07-14-02-05/mini_weight.fold_1.h5')
path = 'input/train_in_one'
test_id = [os.path.basename(fl) for fl in glob('{}/imgs/*.jpg'.format(path))]

test_datagen = ImageDataGenerator(
        rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        path,
        target_size=(224, 224),
        batch_size=1,
        class_mode=None,
        shuffle=False)

preds = model.predict_generator(
        test_generator,
        steps=len(test_id),
        verbose=1)

import pandas as pd
import numpy as np
#preds = np.argmax(preds, axis=-1)
result = pd.DataFrame(preds, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
result.loc[:, 'img'] = pd.Series(test_id, index=result.index)
result.to_csv('vgg16_fold1_train.csv', index=False)
