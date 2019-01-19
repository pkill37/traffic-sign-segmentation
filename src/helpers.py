import numpy as np
from tensorflow.keras import backend as K


def seed():
    from random import seed
    seed(1)
    import numpy.random
    numpy.random.seed(2)
    from tensorflow import set_random_seed
    set_random_seed(3)


def dice_coef(y_true, y_pred, smooth=1.):
    axes = tuple(range(1, len(y_pred.shape)-1)) # skip the batch and class axis for calculating Dice score
    numerator = 2. * K.sum(y_pred * y_true, axes)
    denominator = K.sum(K.square(y_pred) + K.square(y_true), axes)
    return K.mean((numerator + smooth) / (denominator + smooth)) # average over classes and batch


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)
