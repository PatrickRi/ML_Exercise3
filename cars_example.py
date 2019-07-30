import argparse
import glob
import os
import time

import cv2
import yaml
from PIL import Image
from keras import Sequential
from keras.callbacks import TensorBoard
from keras.layers import *
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', default='config.yaml', help='YAML Config File')
    parser.add_argument('--expname', default='Cars-Example-FullAugNoVert-StdNorm-100x40',
                        help='Name of the experiment (Tensorboard).')
    with open(parser.parse_args().config_file, 'r') as ymlfile:
        _config = yaml.load(ymlfile)

    log_dir = _config["tensorboard-log-path"]
    expname = parser.parse_args().expname
    np.random.seed(0)
    batch_size = _config["batch-size"]

    os.chdir(_config["cars-image-path"])
    fileNames = glob.glob("*/*.pgm")
    targetLabels = []
    imageList = []
    for fileName in fileNames:
        if fileName.find("neg") > 0:
            targetLabels.append(0)
        else:
            targetLabels.append(1)
        # print(np.array(Image.open(fileName)).shape)
        image = cv2.resize(np.array(Image.open(fileName)), (int(_config["image-size-y"]), int(_config["image-size-x"])))
        imageList.append(np.array(image))

    img_array = np.array(imageList)
    img_array = img_array.reshape(img_array.shape[0], img_array.shape[1], img_array.shape[2], 1)
    target_C = targetLabels

    X_train, X_test, y_train, y_test = train_test_split(img_array, target_C, random_state=42)

    datagen_train = ImageDataGenerator(rescale=1. / 255,
                                       #featurewise_center=True,
                                       #featurewise_std_normalization=True,
                                       rotation_range=20,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       #vertical_flip=True,
                                       )
    datagen_train.fit(X_train)
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

    n_filters = 16
    # this applies n_filters convolution filters of size 5x5 resp. 3x3 each in the 2 layers below

    # Layer 1
    model.add(Convolution2D(n_filters, 3, 3, border_mode='valid', input_shape=(100, 40, 1)))
    # input shape: 100x100 images with 3 channels -> input_shape should be (3, 100, 100)
    model.add(BatchNormalization())
    model.add(Activation('relu'))  # ReLu activation
    model.add(MaxPooling2D(pool_size=(2, 2)))  # reducing image resolution by half
    model.add(Dropout(0.3))  # random "deletion" of %-portion of units in each batch

    # Layer 2
    model.add(Convolution2D(n_filters, 3, 3))  # input_shape is only needed in 1st layer
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())  # Note: Keras does automatic shape inference.

    # Full Layer
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer=_config["optimizer"],
        metrics=['accuracy']
    )
    now = time.strftime("%b%d_%H-%M")
    model.fit_generator(
        generator_train,
        steps_per_epoch=int(_config["steps-per-epoch"]) // batch_size,
        epochs=int(_config["epochs"]),
        validation_data=generator_test,
        validation_steps=500 // batch_size,
        callbacks=[TensorBoard(histogram_freq=0, log_dir=os.path.join(log_dir, now + '-' + expname),
                               write_graph=True)]
    )


if __name__ == '__main__':
    main()
