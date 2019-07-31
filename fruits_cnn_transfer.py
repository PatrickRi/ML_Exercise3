import argparse
import glob
import os
import time

import cv2
import yaml
from PIL import Image
from keras import Sequential, Model
from keras.applications import MobileNet
from keras.applications import ResNet50
from keras.applications.vgg16 import VGG16
from keras.callbacks import TensorBoard
from keras.layers import *
from keras.utils import to_categorical
from keras_preprocessing.image import ImageDataGenerator
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', default='config.yaml', help='YAML Config File')
    parser.add_argument('--expname', default='VGG16_Transfer',
                        help='Name of the experiment (Tensorboard).')

    with open(parser.parse_args().config_file, 'r') as ymlfile:
        _config = yaml.load(ymlfile)

    log_dir = _config["tensorboard-log-path"]
    expname = parser.parse_args().expname
    np.random.seed(0)
    image_size = (int(_config["fruits-image-size-x"]), int(_config["fruits-image-size-y"]))
    batch_size = _config["batch-size"]

    os.chdir(_config["fruits-image-path"])
    fileNames = glob.glob("*/*.jpg")
    targetLabels = []
    imageList = []
    for fileName in fileNames:
        pathSepIndex = fileName.index(os.path.sep)
        targetLabels.append(fileName[:pathSepIndex])
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

    X_train, X_test, y_train, y_test = train_test_split(imageArr, target_C, random_state=42)

    datagen_train = ImageDataGenerator(rescale=1. / 255,
                                       # featurewise_center=True,
                                       # featurewise_std_normalization=True,
                                        rotation_range=20,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        horizontal_flip=True,
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

    if _config["transfer-model"] == 'mobilenet':
        model = get_mobilenet()
    elif _config["transfer-model"] == 'resnet50':
        model = get_resnet50()
    elif _config["transfer-model"] == 'vgg16':
        model = get_vgg16()
    else:
        raise Exception('model not valid')

    base_model = MobileNet(weights='imagenet',
                           include_top=False)  # imports the mobilenet model and discards the last 1000 neuron layer.

    model.compile(
        loss='categorical_crossentropy',
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


def get_vgg16():
    base_model = VGG16(weights='imagenet',
                          include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    preds = Dense(30, activation='softmax')(x)  # final layer with softmax activation
    model = Model(inputs=base_model.input, outputs=preds)
    for layer in model.layers:
        layer.trainable = True
    return model

def get_resnet50():
    base_model = ResNet50(weights='imagenet',
                           include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    preds = Dense(30, activation='softmax')(x)  # final layer with softmax activation
    model = Model(inputs=base_model.input, outputs=preds)
    for layer in model.layers:
        layer.trainable = True
    return model


def get_mobilenet():
    base_model = MobileNet(weights='imagenet',
                           include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    preds = Dense(30, activation='softmax')(x)  # final layer with softmax activation
    model = Model(inputs=base_model.input, outputs=preds)
    for layer in model.layers[:20]:
        layer.trainable = False
    for layer in model.layers[20:]:
        layer.trainable = True
    return model


if __name__ == '__main__':
    main()
