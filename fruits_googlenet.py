import argparse
import glob
import os
import time

import cv2
from PIL import Image
from keras import Sequential
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.layers import *
from keras.utils import to_categorical
from keras_preprocessing.image import ImageDataGenerator
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from functools import partial

image_size = (224, 224)
batch_size = 32
NAME = 'Fruits-GoogleNet'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='../data/FIDS30/', help='Root folder for the (unprocessed) data set.')
    parser.add_argument('--log_dir', default='C:\\Users\\Patrick\\Documents\\TU\\2019S\\ML\\ML_Exercise3\\ML_Exercise3',
                        help='Root folder for TensorBoard logging.')
    dir = parser.parse_args().dir
    log_dir = parser.parse_args().log_dir

    os.chdir(dir)
    fileNames = glob.glob("*/*.jpg")
    targetLabels = []
    imageList = []
    for fileName in fileNames:
        pathSepIndex = fileName.index(os.path.sep)
        targetLabels.append(fileName[:pathSepIndex])
        # print(np.array(Image.open(fileName)).shape)
        image = cv2.resize(np.array(Image.open(fileName)), image_size)
        imageList.append(np.array(image))

    toDelete = np.where(np.array([x.shape for x in imageList]) == 4)[0][0]
    del imageList[toDelete]
    imageArr = np.array(imageList)

    le = preprocessing.LabelEncoder()
    le.fit(targetLabels)
    target = le.transform(targetLabels)
    target = np.delete(target, toDelete, 0)
    target_C = to_categorical(target)

    # imageArr = np.array(imageList)
    X_train, X_test, y_train, y_test = train_test_split(imageArr, target_C, random_state=42)

    datagen_train = ImageDataGenerator(rescale=1. / 255,
                                       #featurewise_center=True,
                                       #featurewise_std_normalization=True,
                                       rotation_range=20,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True
                                       )
    datagen_train.fit(X_train)
    # datagen_train.fit(X_train)
    # generator_train = datagen_train.flow(
    #     X_train,
    #     y_train,
    #     batch_size=batch_size
    # )
    generator_train = datagen_train.flow(X_train, np.array([y_train, y_train, y_train]), batch_size=batch_size)

    datagen_test = ImageDataGenerator(rescale=1. / 255)
    generator_test = datagen_test.flow(
        X_test,
        y_test,
        batch_size=batch_size
    )

    conv1x1 = partial(Conv2D, kernel_size=1, activation='relu')
    conv3x3 = partial(Conv2D, kernel_size=3, padding='same', activation='relu')
    conv5x5 = partial(Conv2D, kernel_size=5, padding='same', activation='relu')

    def inception_module(in_tensor, c1, c3_1, c3, c5_1, c5, pp):
        conv1 = conv1x1(c1)(in_tensor)

        conv3_1 = conv1x1(c3_1)(in_tensor)
        conv3 = conv3x3(c3)(conv3_1)

        conv5_1 = conv1x1(c5_1)(in_tensor)
        conv5 = conv5x5(c5)(conv5_1)

        pool_conv = conv1x1(pp)(in_tensor)
        pool = MaxPool2D(3, strides=1, padding='same')(pool_conv)

        merged = Concatenate(axis=-1)([conv1, conv3, conv5, pool])
        return merged

    def aux_clf(in_tensor):
        avg_pool = AvgPool2D(5, 3)(in_tensor)
        conv = conv1x1(128)(avg_pool)
        flattened = Flatten()(conv)
        dense = Dense(1024, activation='relu')(flattened)
        dropout = Dropout(0.7)(dense)
        out = Dense(1000, activation='softmax')(dropout)
        return out

    def inception_net(in_shape=(224, 224, 3), n_classes=1000, opt='sgd'):
        in_layer = Input(in_shape)

        conv1 = Conv2D(64, 7, strides=2, activation='relu', padding='same')(in_layer)
        pad1 = ZeroPadding2D()(conv1)
        pool1 = MaxPool2D(3, 2)(pad1)
        conv2_1 = conv1x1(64)(pool1)
        conv2_2 = conv3x3(192)(conv2_1)
        pad2 = ZeroPadding2D()(conv2_2)
        pool2 = MaxPool2D(3, 2)(pad2)

        inception3a = inception_module(pool2, 64, 96, 128, 16, 32, 32)
        inception3b = inception_module(inception3a, 128, 128, 192, 32, 96, 64)
        pad3 = ZeroPadding2D()(inception3b)
        pool3 = MaxPool2D(3, 2)(pad3)

        inception4a = inception_module(pool3, 192, 96, 208, 16, 48, 64)
        inception4b = inception_module(inception4a, 160, 112, 224, 24, 64, 64)
        inception4c = inception_module(inception4b, 128, 128, 256, 24, 64, 64)
        inception4d = inception_module(inception4c, 112, 144, 288, 32, 48, 64)
        inception4e = inception_module(inception4d, 256, 160, 320, 32, 128, 128)
        pad4 = ZeroPadding2D()(inception4e)
        pool4 = MaxPool2D(3, 2)(pad4)

        aux_clf1 = aux_clf(inception4a)
        aux_clf2 = aux_clf(inception4d)

        inception5a = inception_module(pool4, 256, 160, 320, 32, 128, 128)
        inception5b = inception_module(inception5a, 384, 192, 384, 48, 128, 128)
        pad5 = ZeroPadding2D()(inception5b)
        pool5 = MaxPool2D(3, 2)(pad5)

        avg_pool = GlobalAvgPool2D()(pool5)
        dropout = Dropout(0.4)(avg_pool)
        preds = Dense(1000, activation='softmax')(dropout)

        model = Model(in_layer, [preds, aux_clf1, aux_clf2])
        model.compile(loss="categorical_crossentropy", optimizer=opt,
                      metrics=["accuracy"])
        return model

    model = inception_net(n_classes=30)

    now = time.strftime("%b%d_%H-%M")

    # for e in range(30):
    #     batches = 0
    #     for X_batch, Y_batch in generator_train.flow(X_train, y_train, batch_size=batch_size):
    #         loss = model.train_on_batch(X_batch, [Y_batch, Y_batch, Y_batch])  # note the three outputs
    #         batches += 1
    #         if batches >= len(X_train) / batch_size:
    #             # we need to break the loop by hand because
    #             # the generator loops indefinitely
    #             break

    model.fit_generator(
        generator_train,
        steps_per_epoch=1000 // batch_size,
        epochs=50,
        validation_data=generator_test,
        validation_steps=500 // batch_size,
        callbacks=[TensorBoard(histogram_freq=0, log_dir=os.path.join(log_dir, 'logs', now+'-'+NAME),
                               write_graph=True)]
    )


if __name__ == '__main__':
    main()
