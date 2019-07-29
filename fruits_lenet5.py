import argparse
import glob
import os
import time

import cv2
from PIL import Image
from keras import Sequential
from keras.callbacks import TensorBoard
from keras.layers import *
from keras.utils import to_categorical
from keras_preprocessing.image import ImageDataGenerator
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

image_size = (64, 64)
batch_size = 32
NAME = 'LeNet5-RotateShiftShearZoom'


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
    generator_train = datagen_train.flow(
        X_train,
        y_train,
        batch_size=batch_size
    )

    datagen_test = ImageDataGenerator(rescale=1. / 255)
    generator_test = datagen_test.flow(
        X_test,
        y_test,
        batch_size=batch_size
    )

    model = Sequential()
    model.add(Conv2D(6, (3, 3), input_shape=(image_size[0], image_size[1], 3), activation='relu'))
    model.add(AveragePooling2D())

    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(AveragePooling2D())

    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(30, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    now = time.strftime("%b%d_%H-%M")
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
