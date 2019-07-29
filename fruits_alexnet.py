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
batch_size = 8
NAME = 'Fruits-AlexNet'


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
    #imageArr = imageArr / 255.0

    le = preprocessing.LabelEncoder()
    le.fit(targetLabels)
    target = le.transform(targetLabels)
    target = np.delete(target, toDelete, 0)
    target_C = to_categorical(target)

    # imageArr = np.array(imageList)
    X_train, X_test, y_train, y_test = train_test_split(imageArr, target_C, random_state=42)

    datagen_train = ImageDataGenerator(rescale=1. / 255,
                                       featurewise_center=True,
                                       featurewise_std_normalization=True,
                                       rotation_range=10,
                                       #width_shift_range=0.1,
                                       #height_shift_range=0.1,
                                       #shear_range=0.1,
                                       #zoom_range=0.1,
                                       horizontal_flip=True,
                                       #vertical_flip=True
                                       )
    datagen_train.fit(X_train)
    generator_train = datagen_train.flow(
        X_train,
        y_train,
        batch_size=batch_size
    )

    datagen_test = ImageDataGenerator(rescale=1. / 255,
                                       featurewise_center=True,
                                       featurewise_std_normalization=True)
    datagen_test.fit(X_train)
    generator_test = datagen_test.flow(
        X_test,
        y_test,
        batch_size=batch_size
    )

    # Instantiate an empty model
    model = Sequential()

    # 1st Convolutional Layer
    model.add(Conv2D(filters=96, input_shape=(224, 224, 3), kernel_size=(11, 11), strides=(4, 4), padding='valid'))
    model.add(Activation('relu'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(11, 11), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    # 3rd Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))

    # 4th Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))

    # 5th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    # Passing it to a Fully Connected layer
    model.add(Flatten())
    # 1st Fully Connected Layer
    model.add(Dense(4096, input_shape=(224 * 224 * 3,)))
    model.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    # model.add(Dropout(0.2))

    # 2nd Fully Connected Layer
    model.add(Dense(4096))
    model.add(Activation('relu'))
    # Add Dropout
    # model.add(Dropout(0.2))

    # 3rd Fully Connected Layer
    model.add(Dense(1000))
    model.add(Activation('relu'))
    # Add Dropout
    # model.add(Dropout(0.2))

    # Output Layer
    model.add(Dense(30))
    model.add(Activation('softmax'))

    model.summary()

    # Compile the model
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    now = time.strftime("%b%d_%H-%M")
    model.fit_generator(
        generator_train,
        steps_per_epoch=512 // batch_size,
        epochs=50,
        validation_data=generator_test,
        validation_steps=500 // batch_size,
        callbacks=[TensorBoard(histogram_freq=0, log_dir=os.path.join(log_dir, 'logs', now+'-'+NAME),
                               write_graph=True)]
    )


if __name__ == '__main__':
    main()
