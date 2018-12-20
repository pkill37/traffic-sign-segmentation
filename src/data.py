import tensorflow as tf
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


SEED = 2**10
#np.random.seed(SEED)


def load_data(images_path, labels_path, img_height, img_width):
    all_images = [x for x in sorted(os.listdir(images_path)) if x[-4:] == '.png']

    x = np.empty(shape=(len(all_images), img_height, img_width, 3), dtype='float32')
    for i, name in enumerate(all_images):
        x[i] = mpimg.imread(images_path + name)

    y = np.empty(shape=(len(all_images), img_height, img_width, 1), dtype='float32')
    for i, name in enumerate(all_images):
        y[i] = mpimg.imread(labels_path + name).reshape((img_width, img_height, 1))

    return x, y


def training_generator(images_path, labels_path, img_height, img_width, x=None, y=None, batch_size=1):
    do_featurewise_normalization = x is not None and y is not None

    datagen_args = dict(
        # Allowed transformations
        featurewise_center=do_featurewise_normalization,
        featurewise_std_normalization=do_featurewise_normalization,
        rotation_range=90,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,

        # Allowed percentage of data for validation
        validation_split=0.2,

        # Make sure the values are floats within [0,1] in channels_last order
        data_format='channels_last',
        dtype='float32',
        rescale=1./255,
    )

    image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_args)
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
            axis[i,0].imshow(x[i,:,:,:])
            axis[i,1].imshow(y[i,:,:,0])
        input('Press [Enter] to visualize another augmentated mini-batch...')
        plt.close()
