import tensorflow as tf
import numpy as np
import os
import math
import re


def load_img(img_path, target_size, color_mode):
    obj = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size, color_mode=color_mode)
    arr = tf.keras.preprocessing.image.img_to_array(obj, data_format='channels_last', dtype='float32')
    return arr


def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm'):
    return [os.path.join(root, f) for root, _, files in os.walk(directory) for f in files if re.match(r'([\w]+\.(?:' + ext + '))', f.lower())]


def load_data(images, labels, img_height, img_width, begin=None, end=None):
    assert len(images) == len(labels)

    # Load everything
    if not begin and not end:
        begin = 0
        end = len(images)
    # Load from the interval [begin, end]
    else:
        begin = max(begin, 0)
        end = min(end, len(images))
    assert begin < end

    x = np.array([load_img(image, (img_height, img_width), 'rgb') for image in images[begin:end]], dtype='float32')
    assert x.shape == (end-begin, img_height, img_width, 3)

    y = np.array([load_img(label, (img_height, img_width), 'grayscale') for label in labels[begin:end]], dtype='float32')
    assert y.shape == (end-begin, img_height, img_width, 1)

    return x, y


def load_split_stratified_data(images_filenames, labels_filenames, img_height, img_width, split):
    # Organize images and labels by their category encoded in the filename
    data = {}
    for (image, label) in zip(images_filenames, labels_filenames):
        category = os.path.basename(image).split('.')[0].rstrip('0123456789')
        data[category] = data.get(category, []) + [(image, label)]

    train = []
    validation = []
    test = []
    for category in data.keys():
        # Shuffle data (deterministically if seed has been set)
        data[category].sort()
        np.random.shuffle(data[category])

        # Repeat until all leftovers are split as well
        while len(data[category]) > 0:
            # Favor train dataset in edge case
            if len(data[category]) == 1:
                train += [data[category].pop()]
                break

            # Compute indices from percentual splits
            split_train = int(split[0] * len(data[category]))
            split_validation = split_train + int(split[1] * len(data[category]))
            split_test = split_validation + int(split[2] * len(data[category]))

            # Add samples to train, validation and test sets
            train += data[category][:split_train]
            validation += data[category][split_train:split_validation]
            test += data[category][split_validation:split_test]

            # Delete taken samples so that indices are correctly calculated in the next potential iteration
            data[category][split_validation:split_test] = []
            data[category][split_train:split_validation] = []
            data[category][:split_train] = []

    def split_x_y(s):
        return list(map(lambda e: e[0], s)), list(map(lambda e: e[1], s))

    x_train, y_train = split_x_y(train)
    x_train, y_train = load_data(x_train, y_train, img_height, img_width)

    x_validation, y_validation = split_x_y(validation)
    x_validation, y_validation = load_data(x_validation, y_validation, img_height, img_width)

    x_test, y_test = split_x_y(test)
    x_test, y_test = load_data(x_test, y_test, img_height, img_width)

    return x_train, y_train, x_validation, y_validation, x_test, y_test


class MaskedImageSequence(tf.keras.utils.Sequence):
    def __init__(self, x, y, img_height, img_width, batch_size, augment, seed=None):
        self.x = x
        self.y = y

        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.augment = augment
        self.seed = seed

        self.imgaug = tf.keras.preprocessing.image.ImageDataGenerator(
            # Standardization
            preprocessing_function=tf.keras.applications.vgg16.preprocess_input,
            rescale=None,
            samplewise_center=False,
            samplewise_std_normalization=False,
            featurewise_center=False,
            featurewise_std_normalization=False,
            zca_whitening=False,

            # Allowed transformations
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            shear_range=0.5,
            zoom_range=0.2,
            channel_shift_range=0.005,
        )

    def __len__(self):
        return int(math.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        x_batch = self.x[idx*self.batch_size : (1+idx)*self.batch_size]
        y_batch = self.y[idx*self.batch_size : (1+idx)*self.batch_size]

        # Standardize and augment batch
        for i in range(len(x_batch)):
            x_batch[i] = self.imgaug.standardize(x_batch[i])
            if self.augment:
                params = self.imgaug.get_random_transform(x_batch[i].shape, self.seed)
                x_batch[i] = self.imgaug.apply_transform(x_batch[i], params)
                y_batch[i] = self.imgaug.apply_transform(y_batch[i], params)

        return x_batch, y_batch


def generators(images_path, labels_path, img_height, img_width, split, batch_size, augmentation):
    x_train, y_train, x_validation, y_validation, x_test, y_test = load_split_stratified_data(
        images_filenames=list_pictures(images_path),
        labels_filenames=list_pictures(labels_path),
        img_height=img_height,
        img_width=img_width,
        split=split,
    )

    train_generator = MaskedImageSequence(x=x_train, y=y_train, img_height=img_height, img_width=img_width, batch_size=batch_size, augment=augmentation)
    validation_generator = MaskedImageSequence(x=x_validation, y=y_validation, img_height=img_height, img_width=img_width, batch_size=batch_size, augment=False)
    test_generator = MaskedImageSequence(x=x_test, y=y_test, img_height=img_height, img_width=img_width, batch_size=batch_size, augment=False)

    return train_generator, validation_generator, test_generator
