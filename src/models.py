import tensorflow as tf


def vgg16(img_height, img_width, output_activation, loss, optimizer, nb_layers=None):
    # Freeze the model's first nb_layers layers
    vgg16 = tf.keras.applications.vgg16.VGG16(weights='imagenet')
    for layer in vgg16.layers[:nb_layers]:
        layer.trainable = False

    # Stitch VGG16 to our own fully-connected layer for segmentation
    x = vgg16.get_layer('fc2').output
    x = tf.keras.layers.Dense(units=img_height*img_width, activation=output_activation)(x)
    x = tf.keras.layers.Reshape((img_height, img_width, 1))(x)
    model = tf.keras.models.Model(inputs=vgg16.input, outputs=x)
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model
