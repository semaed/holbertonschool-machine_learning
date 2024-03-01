#!/usr/bin/env python3
"""
Defines class NST that performs tasks for neural style transfer
"""

import numpy as np
import tensorflow as tf


class NST:
    """
    Performs tasks for Neural Style Transfer

    public class attributes:
        style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                        'block4_conv1', 'block5_conv1']
        content_layer = 'block5_conv2'

    instance attributes:
        style_image: preprocessed style image
        content_image: preprocessed content image
        alpha: weight for content cost
        beta: weight for style cost
        model: the Keras model used to calculate cost
    """

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Class constructor for Neural Style Transfer class
        """
        if not isinstance(style_image, np.ndarray) or style_image.shape[-1] != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(content_image, np.ndarray) or content_image.shape[-1] != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")
        if not (isinstance(alpha, (float, int)) and alpha >= 0):
            raise TypeError("alpha must be a non-negative number")
        if not (isinstance(beta, (float, int)) and beta >= 0):
            raise TypeError("beta must be a non-negative number")

        tf.compat.v1.enable_eager_execution()

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.model = self.load_model()

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its pixel values are between 0 and 1
        and its largest side is 512 pixels
        """
        max_dim = 512
        initial_shape = tf.cast(tf.shape(image)[:-1], tf.float32)
        scale_ratio = min(max_dim / initial_shape.max(), 1)
        new_shape = tf.cast(initial_shape * scale_ratio, tf.int32)
        image = tf.image.resize(image, new_shape)
        image = image[tf.newaxis, :]
        image = tf.clip_by_value(image / 255.0, 0.0, 1.0)
        return image

    def load_model(self):
        """
        Creates the model used to calculate cost from VGG19 Keras base model
        """
        vgg = tf.keras.applications.VGG19(
            include_top=False, weights='imagenet')
        vgg.trainable = False

        style_outputs = [vgg.get_layer(
            name).output for name in self.style_layers]
        content_output = vgg.get_layer(self.content_layer).output
        model_outputs = style_outputs + [content_output]

        return tf.keras.models.Model(vgg.input, model_outputs)
