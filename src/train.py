import argparse
import multiprocessing
import tensorflow as tf
import models
import data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensorboard', type=str, default='./out/tensorboard')
    parser.add_argument('--logs', type=str, default='out/training_logs.csv')
    parser.add_argument('--dataset_images', type=str, default='data/images')
    parser.add_argument('--dataset_labels', type=str, default='data/labels')
    parser.add_argument('--img_height', type=int, default=224)
    parser.add_argument('--img_width', type=int, default=224)

    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--model_name', type=str, default='TSS')
    parser.add_argument('--optimizer', type=str, default='adadelta')
    parser.add_argument('--output_activation', type=str, default='sigmoid')
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--loss', type=str, default='binary_crossentropy')

    args = parser.parse_args()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None),
        tf.keras.callbacks.LearningRateScheduler(lambda epoch: args.learning_rate*(0.1**int(epoch/10))),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.001),
        tf.keras.callbacks.TensorBoard(log_dir=args.tensorboard, histogram_freq=0, write_graph=True, write_images=True),
        tf.keras.callbacks.ModelCheckpoint(filepath='{model_name:s}_weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1),
        tf.keras.callbacks.CSVLogger(filename=args.logs, separator=',', append=False),
    ]

    model = models.vgg16(output_neurons=args.img_width*args.img_height, output_activation=args.output_activation)
    model.compile(loss=args.loss, optimizer=args.optimizer, metrics=['accuracy'])

    # Fits the model on batches with real-time data augmentation:
    x, y = data.load_data(images_path=args.images_path, labels_path=args.labels_path)
    model.fit_generator(
        data.training_generator(x=x, y=y, batch_size=args.batch_size),
        epochs=args.epochs,
        steps_per_epoch=len(x)/args.batch_size,
        #workers=multiprocessing.cpu_count()-1,
        #use_multiprocessing=True,
        callbacks=callbacks,
    )
