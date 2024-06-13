#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras


class Simple_GAN:
    def __init__(self, generator, discriminator, latent_generator, real_examples, learning_rate=0.001):
        self.generator = generator
        self.discriminator = discriminator
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.learning_rate = learning_rate
        self.gan = self.build_gan()
        self.history = {'discriminator_loss': [], 'adversarial_loss': []}

    def build_gan(self):
        model = keras.Sequential()
        model.add(self.generator)
        model.add(self.discriminator)
        return model

    def compile(self):
        self.discriminator.compile(
            loss='binary_crossentropy', optimizer=keras.optimizers.Adam(self.learning_rate))
        self.gan.compile(loss='binary_crossentropy',
                         optimizer=keras.optimizers.Adam(self.learning_rate))

    def fit(self, real_examples, epochs, steps_per_epoch, verbose=1):
        for epoch in range(epochs):
            for step in range(steps_per_epoch):
                # Sample random points in the latent space
                random_latent_vectors = self.latent_generator(20)

                # Decode them to fake images
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

    def train(self, epochs, verbose=True):
        for epoch in range(epochs):
            # Generate random points in the latent space
            random_latent_vectors = tf.random.normal(shape=(20, 100))

            # Decode them to fake images
            generated_images = self.generator(random_latent_vectors)

            # Combine them with real images
            combined_images = tf.concat(
                [generated_images, self.real_examples], axis=0)

            # Assemble labels discriminating real from fake images
            labels = tf.concat([tf.ones((generated_images.shape[0], 1)), tf.zeros(
                (self.real_examples.shape[0], 1))], axis=0)

            # Train the discriminator
            d_loss = self.discriminator.train_on_batch(combined_images, labels)

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
