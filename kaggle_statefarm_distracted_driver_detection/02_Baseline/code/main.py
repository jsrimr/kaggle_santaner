# -*- encoding: utf-8 -*-

# 딥러닝 관련 Keras 라이브러리
import keras
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.preprocessing.image import ImageDataGenerator

# File I/O 
import subprocess
import shutil
import os
from glob import glob

# 데이터 처리
import pandas as pd
import numpy as np

# 학습 파라미터를 설정한다
nfolds = 5
n_class = 10
fc_size = 2048
batch_size = 8
img_row_size, img_col_size = 224, 224
temp_train_fold = '../input/temp_train'
temp_valid_fold = '../input/temp_valid'
cache = '../cache/vgg16.baseline'
subm = '../subm/vgg16.baseline'
train_path = '../../03_Winners_Code/input/train'
test_path = '../../03_Winners_Code/input/test'
seed = 10
labels = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']

def _clear_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)
for path in [temp_train_fold, temp_valid_fold, cache, subm]:
    _clear_dir(path)

def get_model():
    # 최상위 전결층을 제외한 vgg16 모델을 불러온다
    base_model = keras.applications.vgg16.VGG16(include_top=False, weights=None, input_shape=(img_row_size,img_col_size,3))

    # 최상위 전결층을 정의한다
    out = Flatten()(base_model.output)
    out = Dense(fc_size, activation='relu')(out)
    out = Dropout(0.5)(out)
    out = Dense(fc_size, activation='relu')(out)
    out = Dropout(0.5)(out)
    output = Dense(n_class, activation='softmax')(out)
    model = Model(inputs=base_model.input, outputs=output)

    # SGD Optimizer를 사용하여, 모델을 compile한다
    sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def generate_split():
    # 이미지 생성기를 위하여 임시 훈련/검증 폴더를 생성한다
    def _generate_temp_folder(root_path):
        os.mkdir(root_path)
        for i in range(n_class):
            os.mkdir('{}/c{}'.format(root_path, i))
    _generate_temp_folder(temp_train_fold)
    _generate_temp_folder(temp_valid_fold)

    # 임시 훈련/검증 폴더에 데이터를 랜덤하게 복사한다
    train_samples = 0
    valid_samples = 0
    for label in labels:
        files = glob('{}/{}/*jpg'.format(train_path, label))
        for fl in files:
            cmd = 'cp {} {}/{}/{}'
            # 데이터의 4/5를 훈련 데이터에 추가한다
            if np.random.randint(nfolds) != 1:
                cmd = cmd.format(fl, temp_train_fold, label, os.path.basename(fl))
                train_samples += 1
            # 데이터의 1/5를 검증 데이터에 추가한다
            else:
                cmd = cmd.format(fl, temp_valid_fold, label, os.path.basename(fl))
                valid_samples += 1
            # 원본 훈련 데이터를 임시 훈련/검증 데이터에 복사한다
            subprocess.call(cmd, stderr=subprocess.STDOUT, shell=True)

    # 훈련/검증 데이터 개수를 출력한다
    print('# {} train samples | {} valid samples'.format(train_samples, valid_samples))
    return train_samples, valid_samples

print('# Train Model')
# 이미지 데이터 전처리를 수행하는 함수를 정의한다
datagen = ImageDataGenerator()
# 테스트 데이터를 불러오는 ImageGenerator를 생성한다
test_generator = datagen.flow_from_directory(
        test_path,
        target_size=(img_row_size, img_col_size),
        batch_size=1,
        class_mode=None,
        shuffle=False)
test_id = [os.path.basename(fl) for fl in glob('{}/imgs/*.jpg'.format(test_path))]

# 5-Fold 교차 검증을 진행한다
for fold in range(nfolds):
    # 새로운 모델을 정의한다
    model = get_model()
    # 훈련/검증 데이터를 생성한다
    train_samples, valid_samples = generate_split()

    # 훈련/검증 데이터 생성기를 정의한다
    train_generator = datagen.flow_from_directory(
            directory=temp_train_fold,
            target_size=(img_row_size, img_col_size),
            batch_size=batch_size,
            class_mode='categorical',
            seed=seed)
    valid_generator = datagen.flow_from_directory(
            directory=temp_valid_fold,
            target_size=(img_row_size, img_col_size),
            batch_size=batch_size,
            class_mode='categorical',
            seed=seed)

    weight_path = '../cache/vgg16.baseline/weight.fold_{}.h5'.format(fold)
    callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=0),
            ModelCheckpoint(weight_path, monitor='val_loss', save_best_only=True, verbose=0)]
    # 모델을 학습한다. val_loss 값이 3 epoch 연속 개악되면, 학습을 멈추고 최적 weight를 저장한다
    model.fit_generator(
            train_generator,
            steps_per_epoch=train_samples/args.batch_size,
            epochs=500,
            validation_data=valid_generator,
            validation_steps=valid_samples/args.batch_size,
            shuffle=True,
            callbacks=callbacks,
            verbose=1)

    # 테트스 테이터에 대한 예측값을 생성한다
    preds = model.predict_generator(
            test_generator,
            steps=len(test_id),
            verbose=1)
    result = pd.DataFrame(preds, columns=labels)
    result.loc[:, 'img'] = pd.Series(test_id, index=result.index)
    sub_file = '../subm/vgg16.baseline/f{}.csv'.format(fold)
    result.to_csv(sub_file, index=False)

    # 캐글에 제출한다
    submit_cmd = 'kaggle competitions submit -c state-farm-distracted-driver-detection -f {} -m vgg16.baseline.fold{}'.format(sub_file, fold)
    subprocess.call(submit_cmd, stderr=subprocess.STDOUT, shell=True)

    # 5-Fold 교차 검증 과정에서 생성한 훈련/검증 데이터를 삭제한다
    shutil.rmtree(temp_train_fold)
    shutil.rmtree(temp_valid_fold)

print('# Ensemble')
# 5-Fold 교차 검증의 결과물을 단순 앙상블한다
ensemble = 0
for fold in range(nfolds):
    ensemble += pd.read_csv('../subm/vgg16.baseline/f{}.csv'.format(fold), index_col=-1).values * 1. / nfolds
ensemble = pd.DataFrame(ensemble, columns=labels)
ensemble.loc[:, 'img'] = pd.Series(test_id, index=ensemble.index)
sub_file = '../subm/vgg16.baseline/ens.csv'
ensemble.to_csv(sub_file, index=False)

# 캐글에 제출한다
submit_cmd = 'kaggle competitions submit -c state-farm-distracted-driver-detection -f {} -m vgg16.baseline.ensemble'.format(sub_file)
subprocess.call(submit_cmd, stderr=subprocess.STDOUT, shell=True)
