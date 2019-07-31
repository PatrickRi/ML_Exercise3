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
    parser.add_argument('--expname', default='Cars-LeNet5-NoAug-100x40',
                        help='Name of the experiment (Tensorboard).')
    with open(parser.parse_args().config_file, 'r') as ymlfile:
        _config = yaml.load(ymlfile)

    log_dir = _config["tensorboard-log-path"]
    expname = parser.parse_args().expname
    np.random.seed(0)
    batch_size = _config["batch-size"]
    image_size = (int(_config["cars-image-size-x"]), int(_config["cars-image-size-y"]))

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
        image = cv2.resize(np.array(Image.open(fileName)), (int(_config["cars-image-size-y"]), int(_config["cars-image-size-x"])))
        imageList.append(np.array(image))

    img_array = np.array(imageList)
    img_array = img_array.reshape(img_array.shape[0], img_array.shape[1], img_array.shape[2], 1)
    target_C = targetLabels

    X_train, X_test, y_train, y_test = train_test_split(img_array, target_C, random_state=42)

    datagen_train = ImageDataGenerator(rescale=1. / 255,
                                       # featurewise_center=True,
                                       # featurewise_std_normalization=True,
                                       #rotation_range=20,
                                       #width_shift_range=0.2,
                                       #height_shift_range=0.2,
                                       #shear_range=0.2,
                                       #zoom_range=0.2,
                                       #horizontal_flip=True,
                                       # vertical_flip=True,
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

    if _config["fruits-model"] == 'lenet5':
        model = get_LeNet5_model(image_size)
    elif _config["fruits-model"] == 'example':
        model = get_example_model(image_size)
    else:
        raise Exception('model not valid')

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


def get_example_model(image_size):
    model = Sequential()
    n_filters = 16
    # Layer 1
    model.add(Convolution2D(n_filters, 3, 3, border_mode='valid', input_shape=(image_size[0], image_size[1], 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    # Layer 2
    model.add(Convolution2D(n_filters, 3, 3))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    model.add(Flatten())

    # Full Layer
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))
    return model


def get_LeNet5_model(image_size):
    model = Sequential()
    model.add(Conv2D(6, (3, 3), input_shape=(image_size[0], image_size[1], 1), activation='relu'))
    model.add(AveragePooling2D())

    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(AveragePooling2D())

    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model


if __name__ == '__main__':
    main()
