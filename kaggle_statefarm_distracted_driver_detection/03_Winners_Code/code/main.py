# run train set and evaluate

import keras
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Add
from keras.layers import add
from keras.models import Model
from keras.optimizers import SGD, Adam
import argparse
import subprocess
import shutil
import os
from datetime import datetime
from scipy.ndimage import rotate
import scipy.misc

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=False, default='vgg16', help='Model Architecture')
parser.add_argument('--weights', required=False, default='imagenet')
parser.add_argument('--semi-train', required=False, default=None)
args = parser.parse_args()

fc_size = 4096
n_class = 10
seed = 20
nfolds = 5
test_nfolds = 3
batch_size = 8
img_row_size, img_col_size = 224, 224
preprocess = False
random_split = True

suffix = 'm{}.w{}.s{}.nf{}.t{}.b{}.row{}col{}.p{}.d{}'.format(args.model, args.weights, seed, nfolds, args.semi_train, batch_size, img_row_size, img_col_size, preprocess, datetime.now().strftime("%Y-%m-%d-%H-%M"))
os.mkdir('../cache/{}'.format(suffix))
os.mkdir('../subm/{}'.format(suffix))
temp_train_fold = '../input/temp_train_fold_{}'.format(suffix)
temp_valid_fold = '../input/temp_valid_fold_{}'.format(suffix)
train_path = '../input/train'
if args.semi_train is not None:
    train_path = '../input/semi_train_vgg16.semi-train.v1.f7-ens'
test_path = '../input/test'
labels = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']


print('# Define Model')
if args.weights == 'None':
    args.weights = None
if args.model in ['vgg16']:
    base_model = keras.applications.vgg16.VGG16(include_top=False, weights=args.weights, input_shape=(img_row_size, img_col_size,3))
elif args.model in ['vgg19']:
    base_model = keras.applications.vgg19.VGG19(include_top=False, weights=args.weights, input_shape=(img_row_size, img_col_size,3))
elif args.model in ['resnet50']:
    base_model = keras.applications.resnet50.ResNet50(include_top=False, weights=args.weights, input_shape=(img_row_size, img_col_size,3))
else:
    print('# {} is not a valid value for "--model"'.format(args.model))
    exit()
out = Flatten()(base_model.output)
out = Dense(fc_size, activation='relu')(out)
out = Dropout(0.5)(out)
out = Dense(fc_size, activation='relu')(out)
out = Dropout(0.5)(out)

output = Dense(n_class, activation='softmax')(out)
model = Model(inputs=base_model.input, outputs=output)

sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

from glob import glob
import numpy as np
import cv2
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.preprocessing import image

# get drivers
import pandas as pd

def read_image(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

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
    if not random_split or args.semi_train is None:
        # iterate over 'img_to_driver' dict for each image and its path
        for img_path in img_to_driver.keys():
            cmd = 'cp {}/{}/{} {}/{}/{}'
            label = img_to_driver[img_path]['label']
            if not os.path.exists('{}/{}/{}'.format(train_path, label, img_path)):
                continue
            if img_to_driver[img_path]['driver'] in train_drivers:
                cmd = cmd.format(train_path, label, img_path, temp_train_fold, label, img_path)
                train_samples += 1
            else:
                cmd = cmd.format(train_path, label, img_path, temp_valid_fold, label, img_path)
                valid_samples += 1
            # copy image
            subprocess.call(cmd, stderr=subprocess.STDOUT, shell=True)
    else:
        # semi supervised train does naive random split
        for label in labels:
            files = glob('{}/{}/*jpg'.format(train_path, label))
            for fl in files:
        	cmd = 'cp {} {}/{}/'
                if np.random.randint(nfolds) != 1:
                    cmd = cmd.format(fl, temp_train_fold, label)
                    train_samples += 1
                else:
                    cmd = cmd.format(fl, temp_valid_fold, label)
                    valid_samples += 1
                # copy image
                subprocess.call(cmd, stderr=subprocess.STDOUT, shell=True)

    # show stat
    print('# {} train samples | {} valid samples'.format(train_samples, valid_samples))
    return train_samples, valid_samples

def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

def preprocess(image):
    # rescale
    image /= 255.

    # rotate
    rotate_angle = np.random.randint(60) - 30
    image = rotate(image, rotate_angle)

    # translate
    rows, cols, _ = image.shape
    width_translate = np.random.randint(100) - 50
    height_translate = np.random.randint(100) - 50
    M = np.float32([[1,0,width_translate],[0,1,height_translate]])
    image = cv2.warpAffine(image,M,(cols,rows))    

    # zoom
    width_zoom = int(img_row_size * (0.7 + 0.3 * (1 - np.random.random())))
    height_zoom = int(img_col_size * (0.7 + 0.3 * (1 - np.random.random())))
    final_image = np.zeros((height_zoom, width_zoom, 3))
    final_image[:,:,0] = crop_center(image[:,:,0], width_zoom, height_zoom)
    final_image[:,:,1] = crop_center(image[:,:,1], width_zoom, height_zoom)
    final_image[:,:,2] = crop_center(image[:,:,2], width_zoom, height_zoom)

    # resize
    image = cv2.resize(final_image, (img_row_size, img_col_size))
    return image

print('# Train Model')
if preprocess:
    datagen = ImageDataGenerator(preprocessing_function=preprocess)
else:
    datagen = ImageDataGenerator()

test_generator = datagen.flow_from_directory(
        test_path,
        target_size=(img_row_size, img_col_size),
        batch_size=1,
        class_mode=None,
        shuffle=False)
test_id = [os.path.basename(fl) for fl in glob('{}/imgs/*.jpg'.format(test_path))]

kf = KFold(len(uniq_drivers), n_folds=nfolds, shuffle=True, random_state=20)
for i, (train_drivers, valid_drivers) in enumerate(kf):
    train_drivers = [uniq_drivers[j] for j in train_drivers]

    train_samples, valid_samples = generate_driver_based_split(img_to_driver, train_drivers)

    train_generator = datagen.flow_from_directory(
            temp_train_fold,
            target_size=(img_row_size, img_col_size),
            batch_size=batch_size,
            class_mode='categorical',
            seed=seed)
    valid_generator = datagen.flow_from_directory(
            temp_valid_fold,
            target_size=(img_row_size, img_col_size),
            batch_size=batch_size,
            class_mode='categorical',
            seed=seed)

    weight_path = '../cache/{}/mini_weight.fold_{}.h5'.format(suffix, i)
    callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=0),
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

    # pred on test data n times with data augmentation
    for j in range(test_nfolds):
        preds = model.predict_generator(
                test_generator,
                steps=len(test_id),
                verbose=1)

        if j == 0:
            result = pd.DataFrame(preds, columns=labels)
        else:
            result += pd.DataFrame(preds, columns=labels)

    result /= test_nfolds
    result.loc[:, 'img'] = pd.Series(test_id, index=result.index)
    sub_file = '../subm/{}/f{}.csv'.format(suffix, i)
    result.to_csv(sub_file, index=False)

    # submit to kaggle
    submit_cmd = 'kaggle competitions submit -c state-farm-distracted-driver-detection -f {} -m {}.fold{}'.format(sub_file, suffix, i)
    subprocess.call(submit_cmd, stderr=subprocess.STDOUT, shell=True)

    # remove temp folders
    if os.path.exists(temp_train_fold):
        shutil.rmtree(temp_train_fold)
        shutil.rmtree(temp_valid_fold)

print('# Ensemble')
# simple average all folds
ensemble = 0
for fold in range(nfolds):
    ensemble += pd.read_csv('../subm/{}/f{}.csv'.format(suffix, fold), index_col=-1).values * 1. / nfolds
ensemble = pd.DataFrame(ensemble, columns=labels)
ensemble.loc[:, 'img'] = pd.Series(test_id, index=ensemble.index)
sub_file = '../subm/{}/ens.csv'.format(suffix)
ensemble.to_csv(sub_file, index=False)

# submit to kaggle
submit_cmd = 'kaggle competitions submit -c state-farm-distracted-driver-detection -f {} -m {}'.format(sub_file, suffix)
subprocess.call(submit_cmd, stderr=subprocess.STDOUT, shell=True)
