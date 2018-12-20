import tensorflow as tf


def vgg16(output_neurons, output_activation, nb_layers=None):
    # Freeze the model's first n weights
    vgg16 = tf.keras.applications.vgg16.VGG16(weights='imagenet')
    for layer in vgg16.layers[:nb_layers]:
        layer.trainable = False

    # Stitch VGG16 to our own fully-connected layer for segmentation
    x = vgg16.get_layer('fc2').output
    x = tf.keras.layers.Dense(output_neurons, name='pilita', activation=output_activation)(x)
    x = tf.keras.layers.Reshape((224, 224, 1))(x)
    model = tf.keras.models.Model(inputs=vgg16.input, outputs=x)
    model.summary()
    return model
