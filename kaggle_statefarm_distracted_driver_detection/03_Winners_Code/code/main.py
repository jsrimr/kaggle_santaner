# run train set and evaluate

import keras
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers import add
from keras.models import Model
from keras.optimizers import SGD, Adam
import argparse
import subprocess
import shutil
import os
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=False, default='vgg16', help='Model Architecture')
parser.add_argument('--weights', required=False, default='imagenet')
args = parser.parse_args()

fc_size = 4096
n_class = 10
seed = 18
nfolds = 10
batch_size = 16
suffix = 'm{}.w{}.s{}.nf{}.d{}'.format(args.model, args.weights, seed, nfolds, datetime.now().strftime("%Y-%m-%d-%H-%M"))
os.mkdir('../cache/{}'.format(suffix))
os.mkdir('../subm/{}'.format(suffix))
temp_train_fold = '../input/temp_train_fold_{}'.format(suffix)
temp_valid_fold = '../input/temp_valid_fold_{}'.format(suffix)

print('# Define Model')
if args.weights == 'None':
    args.weights = None
if args.model in ['vgg16']:
    base_model = keras.applications.vgg16.VGG16(include_top=False, weights=args.weights, input_shape=(224,224,3))
    out = Flatten()(base_model.output)
elif args.model in ['resnet50']:
    base_model = keras.applications.resnet50.ResNet50(include_top=False, weights=args.weights, input_shape=(224,224,3))
    out = Flatten()(base_model.output)
elif args.model in ['xception']:
    base_model = keras.applications.xception.Xception(include_top=False, weights=args.weights, input_shape=(224,224,3))
    out = GlobalAveragePooling2D()(base_model.output)
elif args.model in ['densenet']:
    base_model = keras.applications.densenet.DenseNet201(include_top=False, weights=args.weights, input_shape=(224,224,3))
    out = GlobalAveragePooling2D()(base_model.output)
elif args.model in ['inceptionresnet']:
    base_model = keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights=args.weights, input_shape=(224,224,3))
    out = GlobalAveragePooling2D()(base_model.output)
elif args.model in ['inceptionv3']:
    base_model = keras.applications.inception_v3.InceptionV3(include_top=False, weights=args.weights, input_shape=(224,224,3))
    out = GlobalAveragePooling2D()(base_model.output)
else:
    print('# {} is not a valid value for "--model"'.format(args.model))
    exit()
out = Dense(fc_size, activation='relu')(out)
out = Dropout(0.5)(out)
out = Dense(fc_size, activation='relu')(out)
out = Dropout(0.5)(out)
output = Dense(n_class, activation='softmax')(out)
model = Model(inputs=base_model.input, outputs=output)

sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

from glob import glob
import numpy as np
import cv2
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.preprocessing import image

# get drivers
import pandas as pd

print('# Data Load')
drivers = pd.read_csv('../input/driver_imgs_list.csv')
img_to_driver = {}
uniq_drivers = []

for i, row in drivers.iterrows():
    label_n_driver = {}
    label_n_driver['label'] = row['classname']
    label_n_driver['driver'] = row['subject']
    img_to_driver[row['img']] = label_n_driver

    if row['subject'] not in uniq_drivers:
        uniq_drivers.append(row['subject'])

from sklearn.cross_validation import KFold
from keras.preprocessing.image import ImageDataGenerator

def generate_driver_based_split(img_to_driver, train_drivers):
    # remove existing train_temp folder
    if os.path.exists(temp_train_fold):
        shutil.rmtree(temp_train_fold)
        shutil.rmtree(temp_valid_fold)

    # make temp folder for train
    def _generate_temp_folder(root_path):
        os.mkdir(root_path)
        for i in range(n_class):
            os.mkdir('{}/c{}'.format(root_path, i))
    _generate_temp_folder(temp_train_fold)
    _generate_temp_folder(temp_valid_fold)

    train_samples = 0
    valid_samples = 0
    # iterate over 'img_to_driver' dict for each image and its path
    for img_path in img_to_driver.keys():
        cmd = 'cp ../input/train/{}/{} {}/{}/{}'
        label = img_to_driver[img_path]['label']
        if not os.path.exists('../input/train/{}/{}'.format(label, img_path)):
            continue
        if img_to_driver[img_path]['driver'] in train_drivers:
            cmd = cmd.format(label, img_path, temp_train_fold, label, img_path)
            train_samples += 1
        else:
            cmd = cmd.format(label, img_path, temp_valid_fold, label, img_path)
            valid_samples += 1
        # copy image
        subprocess.call(cmd, stderr=subprocess.STDOUT, shell=True)

    # show stat
    print('# {} train samples | {} valid samples'.format(train_samples, valid_samples))
    return train_samples, valid_samples

print('# Train Model')
test_datagen = ImageDataGenerator(
        rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        '../input/test',
        target_size=(224, 224),
        batch_size=1,
        class_mode=None,
        shuffle=False)
test_id = [os.path.basename(fl) for fl in glob('../input/test/imgs/*.jpg')]

kf = KFold(len(uniq_drivers), n_folds=nfolds, shuffle=True, random_state=2018)
for i, (train_drivers, valid_drivers) in enumerate(kf):
    train_drivers = [uniq_drivers[j] for j in train_drivers]

    train_samples, valid_samples = generate_driver_based_split(img_to_driver, train_drivers)

    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2)
    train_generator = train_datagen.flow_from_directory(
            temp_train_fold,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical',
            seed=seed)
    valid_generator = train_datagen.flow_from_directory(
            temp_valid_fold,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical',
            seed=seed)

    weight_path = '../cache/{}/mini_weight.fold_{}.h5'.format(suffix, i)
    callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=0),
            ModelCheckpoint(weight_path, monitor='val_loss', save_best_only=True, verbose=0)]
    model.fit_generator(
            train_generator,
            steps_per_epoch=train_samples/batch_size,
            epochs=500,
            validation_data=valid_generator,
            validation_steps=valid_samples/batch_size,
            shuffle=True,
            callbacks=callbacks,
            verbose=1)

    # pred on test data
    preds = model.predict_generator(
            test_generator,
            steps=len(test_id),
            verbose=1)

    result = pd.DataFrame(preds, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
    result.loc[:, 'img'] = pd.Series(test_id, index=result.index)
    sub_file = '../subm/{}/f{}.csv'.format(suffix, i)
    result.to_csv(sub_file, index=False)

    # remove temp folders
    if os.path.exists(temp_train_fold):
        shutil.rmtree(temp_train_fold)
        shutil.rmtree(temp_valid_fold)

print('# Ensemble')
# simple average all folds
ensemble = 0
for fold in range(nfolds):
    ensemble += pd.read_csv('../subm/{}/f{}.csv'.format(suffix, fold), index_col=-1).values * 1. / nfolds
ensemble = pd.DataFrame(ensemble, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
ensemble.loc[:, 'img'] = pd.Series(test_id, index=ensemble.index)
sub_file = '../subm/{}/ens.csv'.format(suffix)
ensemble.to_csv(sub_file, index=False)

# submit to kaggle
submit_cmd = 'kaggle competitions submit -c state-farm-distracted-driver-detection -f {} -m {}'.format(sub_file, suffix)
subprocess.call(submit_cmd, stderr=subprocess.STDOUT, shell=True)
