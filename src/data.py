import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import helpers
import math
import random
import re


SEED = 2**10
#np.random.seed(SEED)


def load_img(img_path, target_size, color_mode):
    obj = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size, color_mode=color_mode)
    arr = tf.keras.preprocessing.image.img_to_array(obj, data_format='channels_last', dtype='float32')
    return arr


def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm'):
    return [os.path.join(root, f) for root, _, files in os.walk(directory) for f in files if re.match(r'([\w]+\.(?:' + ext + '))', f.lower())]


def load_data(images, labels, img_height, img_width, begin, end):
    assert len(images) == len(labels)

    begin = max(begin, 0)
    end = min(end, len(images))
    assert begin < end

    x = np.array([load_img(image, (img_height, img_width), 'rgb') for image in images[begin:end]], dtype='float32')
    assert x.shape == (end-begin, img_height, img_width, 3)

    y = np.array([load_img(label, (img_height, img_width), 'grayscale') for label in labels[begin:end]], dtype='float32')
    assert y.shape == (end-begin, img_height, img_width, 1)

    return x, y


class MaskedImageSequence(tf.keras.utils.Sequence):
    def __init__(self, images_path, labels_path, img_height, img_width, batch_size):
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size

        self.images = list_pictures(images_path)
        self.labels = list_pictures(labels_path)

        self.imgaug = tf.keras.preprocessing.image.ImageDataGenerator(
            # Standardization
            featurewise_center=True,
            featurewise_std_normalization=True,

            # Allowed transformations
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.2,

            # Make sure the values are floats in channels_last order within [0,1]
            dtype='float32',
            rescale=1./255,
            data_format='channels_last',
            preprocessing_function=lambda x: x,
        )

        x, _ = load_data(self.images, self.labels, img_height, img_width, 0, len(self.images))
        self.imgaug.fit(x, augment=True, seed=SEED)

    def __len__(self):
        return int(math.ceil(len(self.images) / float(self.batch_size)))

    def __getitem__(self, idx):
        x, y = load_data(images=self.images, labels=self.labels, img_height=self.img_height, img_width=self.img_width, begin=idx*self.batch_size, end=(1+idx)*self.batch_size)

        for i in range(len(x)):
            params = self.imgaug.get_random_transform(x[i].shape)
            x[i] = self.imgaug.apply_transform(self.imgaug.standardize(x[i]), params)
            y[i] = self.imgaug.apply_transform(y[i], params)

        return x, y


if __name__ == '__main__':
    plt.ion()

    pwd = os.path.realpath(__file__)
    images_path = os.path.abspath(os.path.join(pwd, '../../data/images/')) + '/'
    labels_path = os.path.abspath(os.path.join(pwd, '../../data/labels/')) + '/'
    img_height = 224
    img_width = 224
    batch_size = 1

    for xb, yb in MaskedImageSequence(images_path=images_path, labels_path=labels_path, img_height=img_height, img_width=img_width, batch_size=batch_size):
        fig, axis = plt.subplots(nrows=batch_size, ncols=2, squeeze=False, subplot_kw={'xticks': [], 'yticks': []})
        for i in range(len(xb)):
            xb = (xb - np.min(xb))/np.ptp(xb) # make normalized image somewhat plottable
            axis[i,0].imshow(xb[i,:,:,:])
            axis[i,1].imshow(yb[i,:,:,0])
        input('Press [Enter] to visualize another augmented mini-batch...')
        plt.close()
