import os
import multiprocessing
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

    # Build test generator
    _, __, test_generator = data.generators(
        images_path=args.images_path,
        labels_path=args.labels_path,
        img_height=args.img_height,
        img_width=args.img_width,
        split=(0.8, 0.1, 0.1),
        batch_size=args.batch_size,
        augmentation=True,
    )

    # Evaluate model on test generator
    results = model.evaluate_generator(
        generator=test_generator,
        verbose=1,
        workers=multiprocessing.cpu_count()-1 or 1,
        use_multiprocessing=True,
    )
    print(f'Soft Dice Loss: {results[0]}')
    print(f'Dice Coefficient: {results[1]}')

    # Visualize the model's predictions
    plt.ion()
    for x_batch, y_batch in test_generator:
        y_pred = model.predict_on_batch(x_batch)
        for i in range(len(x_batch)):
            plt.imshow(x_batch[i,:,:,:])
            plt.imshow(y_pred[i,:,:,0], alpha=0.7)
            input('Press [Enter] to predict another mini-batch...')
            plt.close()
