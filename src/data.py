import tensorflow as tf
import numpy as np
import os
import glob
import matplotlib.pyplot as plt


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

    for i, name in enumerate(all_images):
        assert os.path.isfile(images_path+name) == os.path.isfile(labels_path+name)
        x[i] = load_img(images_path + name, (img_height, img_width), 'rgb')
        y[i] = load_img(labels_path + name, (img_height, img_width), 'grayscale')

    return x, y


def training_generator(images_path, labels_path, img_height, img_width, x=None, y=None, batch_size=1):
    do_featurewise_normalization = x is not None and y is not None

    datagen_args = dict(
        # Allowed transformations
        rotation_range=90,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,

        # Allowed percentage of data for validation
        validation_split=0.2,

        # Make sure the values are floats in channels_last order within [0,1]
        dtype='float32',
        rescale=1./255,
        data_format='channels_last',
    )

    image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_args, featurewise_center=do_featurewise_normalization, featurewise_std_normalization=do_featurewise_normalization)
    label_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_args)

    # Compute quantities required for featurewise normalization (std, mean, and principal components if ZCA whitening is applied)
    if do_featurewise_normalization:
        image_datagen.fit(x, augment=True, seed=SEED)
        label_datagen.fit(y, augment=True, seed=SEED)

    # Combine generators into one which yields image and masks
    f = lambda d: os.path.abspath(os.path.join(d, '..')) + '/'
    image_generator = image_datagen.flow_from_directory(f(images_path), target_size=(img_height, img_width), color_mode='rgb', class_mode=None, batch_size=batch_size, shuffle=True, seed=SEED)
    label_generator = label_datagen.flow_from_directory(f(labels_path), target_size=(img_height, img_width), color_mode='grayscale', class_mode=None, batch_size=batch_size, shuffle=True, seed=SEED)

    return zip(image_generator, label_generator)


if __name__ == '__main__':
    pwd = os.path.realpath(__file__)
    images_path = os.path.abspath(os.path.join(pwd, '../../data/images/all/')) + '/'
    labels_path = os.path.abspath(os.path.join(pwd, '../../data/labels/all/')) + '/'
    img_height = 224
    img_width = 224

    plt.ion()
    x, y = load_data(images_path, labels_path, img_height, img_width)
    batch_size = 1
    for x, y in training_generator(images_path=images_path, labels_path=labels_path, img_height=img_height, img_width=img_width, x=x, y=y, batch_size=batch_size):
        fig, axis = plt.subplots(nrows=batch_size, ncols=2, squeeze=False, subplot_kw={'xticks': [], 'yticks': []})
        for i in range(batch_size):
            x = (x - np.min(x))/np.ptp(x)
            axis[i,0].imshow(x[i,:,:,:])
            axis[i,1].imshow(y[i,:,:,0])
        input('Press [Enter] to visualize another augmentated mini-batch...')
        plt.close()
