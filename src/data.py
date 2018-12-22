import tensorflow as tf
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import helpers
import tqdm
import math
import random


SEED = 2**10
#np.random.seed(SEED)


def load_img(img_path, target_size, color_mode):
    obj = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size, color_mode=color_mode)
    arr = tf.keras.preprocessing.image.img_to_array(obj, data_format='channels_last', dtype='float32')
    return arr


def load_data(images_path, labels_path, img_height, img_width):
    all_images = [x for x in sorted(os.listdir(images_path)) if x[-4:] == '.png']
    x = np.empty(shape=(len(all_images), img_height, img_width, 3), dtype='float32')
    y = np.empty(shape=(len(all_images), img_height, img_width, 1), dtype='float32')

    for i, name in tqdm.tqdm(enumerate(all_images), total=len(all_images)):
        assert os.path.isfile(images_path+name) == os.path.isfile(labels_path+name)
        x[i] = load_img(images_path + name, (img_height, img_width), 'rgb')
        y[i] = load_img(labels_path + name, (img_height, img_width), 'grayscale')

    return x, y


class MaskedImageSequence(tf.keras.utils.Sequence):
    def __init__(self, images_path, labels_path, img_height, img_width, batch_size, x=None, y=None, shuffle=True):
        self.images_path = images_path
        self.labels_path = labels_path
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.basenames = [x for x in sorted(os.listdir(images_path)) if x[-4:] == '.png']

        do_featurewise_normalization = x is not None and y is not None
        self.imgaug = tf.keras.preprocessing.image.ImageDataGenerator(
            # Standardization
            featurewise_center=do_featurewise_normalization,
            featurewise_std_normalization=do_featurewise_normalization,

            # Allowed transformations
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.2,

            # Allowed percentage of data for validation
            validation_split=0.2,

            # Make sure the values are floats in channels_last order within [0,1]
            dtype='float32',
            rescale=1./255,
            data_format='channels_last',
            preprocessing_function=lambda x: x,
        )

        if do_featurewise_normalization:
            self.imgaug.fit(x, augment=True, seed=SEED)

    def __len__(self):
        return int(math.ceil(len(self.basenames) / float(self.batch_size)))

    def on_epoch_end(self):
        self.indices = range(len(self.basenames))
        # Shuffles indices after each epoch
        if self.shuffle:
            self.indices = random.sample(self.indices, k=len(self.indices))

    def __getitem__(self, idx):
        x = np.array([load_img(self.images_path + basename, (self.img_height, self.img_width), 'rgb') for basename in self.basenames[idx * self.batch_size: (1 + idx) * self.batch_size]])
        y = np.array([load_img(self.labels_path + basename, (self.img_height, self.img_width), 'grayscale') for basename in self.basenames[idx * self.batch_size: (1 + idx) * self.batch_size]])

        for i in range(len(x)):
            params = self.imgaug.get_random_transform(x[i].shape)
            x[i] = self.imgaug.apply_transform(self.imgaug.standardize(x[i]), params)
            y[i] = self.imgaug.apply_transform(y[i], params)

        return x, y




if __name__ == '__main__':
    plt.ion()

    pwd = os.path.realpath(__file__)
    images_path = os.path.abspath(os.path.join(pwd, '../../data/images/all/')) + '/'
    labels_path = os.path.abspath(os.path.join(pwd, '../../data/labels/all/')) + '/'
    img_height = 224
    img_width = 224
    batch_size = 1
    x, y = load_data(images_path, labels_path, img_height, img_width)

    for xb, yb in MaskedImageSequence(images_path=images_path, labels_path=labels_path, img_height=img_height, img_width=img_width, x=x, y=y, batch_size=batch_size):
        fig, axis = plt.subplots(nrows=batch_size, ncols=2, squeeze=False, subplot_kw={'xticks': [], 'yticks': []})
        for i in range(batch_size):
            xb = (xb - np.min(xb))/np.ptp(xb) # make normalized image somewhat plottable
            axis[i,0].imshow(xb[i,:,:,:])
            axis[i,1].imshow(yb[i,:,:,0])
        input('Press [Enter] to visualize another augmentated mini-batch...')
        plt.close()
