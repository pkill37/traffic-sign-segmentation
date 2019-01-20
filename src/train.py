import os
import argparse
import multiprocessing
import tensorflow as tf
import models
import data
import helpers


if __name__ == '__main__':
    helpers.seed()

    pwd = os.path.realpath(__file__)
    out_dir = os.path.abspath(os.path.join(pwd, '../../out/')) + '/'
    data_dir = os.path.abspath(os.path.join(pwd, '../../data/')) + '/'

    parser = argparse.ArgumentParser()
    parser.add_argument('--tensorboard', type=str, default=out_dir+'tensorboard/')
    parser.add_argument('--log', type=str, default=out_dir+'training_log.csv')
    parser.add_argument('--plots', type=str, default=out_dir+'plots/')
    parser.add_argument('--model', type=str, default=out_dir+'models/best.hdf5')
    parser.add_argument('--images_path', type=str, default=data_dir+'images/')
    parser.add_argument('--labels_path', type=str, default=data_dir+'labels/')
    parser.add_argument('--img_height', type=int, default=224)
    parser.add_argument('--img_width', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--output_activation', type=str, default='relu')
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--loss', type=str, default='dice_coef_loss')
    parser.add_argument('--metrics', type=str, default='dice_coef')
    args = parser.parse_args()

    model = models.vgg16(
        img_height=args.img_height,
        img_width=args.img_width,
        output_activation=args.output_activation,
        loss=helpers.dice_coef_loss if args.loss == 'dice_coef_loss' else args.loss,
        metrics=[helpers.dice_coef if m == 'dice_coef' else m for m in args.metrics.split(',')],
        optimizer=args.optimizer,
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_dice_coef', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None),
        tf.keras.callbacks.LearningRateScheduler(lambda epoch: args.learning_rate*(0.1**int(epoch/10))),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_dice_coef', factor=0.2, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.001),
        tf.keras.callbacks.ModelCheckpoint(filepath=args.model, monitor='val_dice_coef', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1),
        tf.keras.callbacks.TensorBoard(log_dir=args.tensorboard, histogram_freq=0, write_graph=True, write_images=True),
        tf.keras.callbacks.CSVLogger(filename=args.log, separator=',', append=False),
    ]

    train_generator, validation_generator, _ = data.generators(
        images_path=args.images_path,
        labels_path=args.labels_path,
        img_height=args.img_height,
        img_width=args.img_width,
        split=(0.8, 0.1, 0.1),
        batch_size=args.batch_size,
    )

    model.fit_generator(
        generator=train_generator,
        epochs=args.epochs,
        validation_data=validation_generator,
        shuffle=True,
        verbose=1,
        callbacks=callbacks,
        workers=multiprocessing.cpu_count()-1 or 1,
        use_multiprocessing=True,
    )
