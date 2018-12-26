import os
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import data
import helpers


if __name__ == '__main__':
    helpers.seed()

    pwd = os.path.realpath(__file__)
    out_dir = os.path.abspath(os.path.join(pwd, '../../out/')) + '/'
    data_dir = os.path.abspath(os.path.join(pwd, '../../data/')) + '/'

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=out_dir+'models/best.hdf5')
    parser.add_argument('--images_path', type=str, default=data_dir+'images/')
    parser.add_argument('--labels_path', type=str, default=data_dir+'labels/')
    parser.add_argument('--img_height', type=int, default=224)
    parser.add_argument('--img_width', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)

    plt.ion()

    for x_batch, y_batch in data.MaskedImageSequence(images_path=args.images_path, labels_path=args.labels_path, img_height=args.img_height, img_width=args.img_width, batch_size=args.batch_size, augment=False):
        y_pred = model.predict_on_batch(x_batch)
        #y_pred = (y_pred > 0.5).astype(np.uint8)

        for i in range(len(x_batch)):
            plt.imshow(x_batch[i,:,:,:])
            plt.imshow(y_pred[i,:,:,0], alpha=0.7)
            input('Press [Enter] to predict another mini-batch...')
            plt.close()
