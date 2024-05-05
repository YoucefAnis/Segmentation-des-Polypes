import numpy as np
import tensorflow as tf

smooth = 1e-15

def dice_coef(y_true, y_pred):
    """
    Compute Dice coefficient.

    Dice coefficient is a measure of similarity between two sets.
    This function calculates the Dice coefficient between ground truth (y_true) and predictions (y_pred).

    Args:
        y_true : The ground truth values.
        y_pred : The predicted values.

    Returns:
        float: The computed Dice coefficient.
    """
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    """
    Compute loss based on Dice coefficient.

    Dice loss is used for optimizing a segmentation model.
    This function computes the loss by subtracting the Dice coefficient from 1.0.

    Args:
        y_true : The ground truth values.
        y_pred : The predicted values.

    Returns:
        float: The Dice-based loss.
    """
    return 1.0 - dice_coef(y_true, y_pred)