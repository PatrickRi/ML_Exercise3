import argparse
import time

from keras import Sequential
from keras.callbacks import TensorBoard
from keras.layers import *
from keras_preprocessing.image import ImageDataGenerator

image_size = (64, 64)
batch_size = 32
NAME = 'LeNet5-ValGenerator'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='../data/FIDS30/', help='Root folder for the (unprocessed) data set.')
    dir = parser.parse_args().dir

    datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.20)
    generator = datagen.flow_from_directory(
        dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
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
        generator,
        steps_per_epoch=1000 // batch_size,
        epochs=30,
        validation_data=datagen.flow_from_directory(dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'),
        validation_steps = 500 // batch_size,
        callbacks=[TensorBoard(histogram_freq=0, log_dir='./logs/' + now + '-' + NAME, write_graph=True)]
    )


if __name__ == '__main__':
    main()
