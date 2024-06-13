#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras


class Simple_GAN:
    def __init__(self, ...):  # your existing parameters
        ...
        self.history = {'discriminator_loss': [], 'adversarial_loss': []}
        ...

    def train(self, epochs, verbose=True):
        for epoch in range(epochs):
            ...
            # existing code
            ...
            generated_images = self.generator(random_latent_vectors)

            # Combine them with real images
            combined_images = tf.concat(
                [generated_images, self.real_examples], axis=0)

            # Assemble labels discriminating real from fake images
            labels = tf.concat([tf.ones((generated_images.shape[0], 1)), tf.zeros(
                (self.real_examples.shape[0], 1))], axis=0)

            # Train the discriminator
            d_loss = self.discriminator.train_on_batch(
                combined_images, labels)

            # Assemble labels that say "all real images"
            misleading_labels = tf.zeros((20, 1))

            # Train the generator (via the gan model, where the discriminator weights are frozen)
            a_loss = self.gan.train_on_batch(
                random_latent_vectors, misleading_labels)

            # append losses to history
            self.history['discriminator_loss'].append(d_loss)
            self.history['adversarial_loss'].append(a_loss)

            if verbose:
                print(
                    f'epoch: {epoch+1}, discriminator loss: {d_loss}, adversarial loss: {a_loss}')
