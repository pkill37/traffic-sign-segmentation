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

    # Load best trained model
    model = tf.keras.models.load_model(args.model, custom_objects={'dice_coef_loss': helpers.dice_coef_loss, 'dice_coef': helpers.dice_coef})

    # Load test set
    _, __, ___, ____, x_test, y_test = data.load_split_stratified_data(
        images=data.list_pictures(args.images_path),
        labels=data.list_pictures(args.labels_path),
        img_height=args.img_height,
        img_width=args.img_width,
        split=(0.8, 0.1, 0.1)
    )

    # Evaluate model on test set
    y_pred = model.predict(x_test)
    dice = tf.keras.backend.get_value(helpers.dice_coef(y_test, y_pred))
    print(f'Test set Dice coefficient: {dice}')

    # Visualize the model's predictions
    plt.ion()
    for x, y in zip(x_test, y_pred):
        print(x)
        plt.imshow(x[:,:,:])
        plt.imshow(y[:,:,0], alpha=0.7)
        input('Press [Enter] to predict another mini-batch...')
        plt.close()
