'''VAE'''
import os

from keras import backend as K
from keras.layers import Lambda, Input, Dense, Conv2D, Conv2DTranspose, Flatten, Reshape
from keras.losses import mse, binary_crossentropy
from keras.models import Model
# from keras.regularizers import l1_l2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


class VAE(object):
    '''Variational Auto Encoder'''
    class Params(object):
        '''Params for VAE'''
        def __init__(self):
            self.loss = 'binary_crossentropy'
            self.latent_dim = 2
            self.kernel_size = 3

            self.layers = 2
            self.encoder_input_shape = (28, 28, 1)

            # Below defined by compile methods
            self.z_mean, self.z_log_var = None, None # Filled when compiled
            self.last_conv_shape = None


    def __init__(self, input_shape):
        '''Init a VAE'''
        self.params = VAE.Params()
        self.params.encoder_input_shape = input_shape
        self.decoder = None
        self.encoder = None
        self.inputs = None
        self.vae = None
        self.z_log_var = None
        self.z_mean = None
        self.reconstruction_loss = None

    # def train(self, train, test, batch_size=50):
    #     pass

    @staticmethod
    def sampling(args):
        """Reparameterization trick by sampling fr an isotropic unit Gaussian.
        # Arguments:
            args (tensor): mean and log of variance of Q(z|X)
        # Returns:
            z (tensor): sampled latent vector
        """

        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def encoder_model(self):
        self.inputs = Input(shape=self.params.encoder_input_shape, name='vae_input')
        x = self.inputs
        for i in range(self.params.layers):
            x = Conv2D(filters=(2 ** (5 + i)),
                       kernel_size=self.params.kernel_size,
                       activation='relu',
                       strides=2,
                       padding='same')(x)

        # shape info needed to build decoder model
        self.params.last_conv_shape = K.int_shape(x)

        # generate latent vector Q(z|X)
        x = Flatten()(x)
        x = Dense(16, activation='relu')(x)
        self.z_mean = Dense(self.params.latent_dim, name='z_mean')(x)
        self.z_log_var = Dense(self.params.latent_dim, name='z_log_var')(x)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(VAE.sampling, output_shape=(self.params.latent_dim,), name='z')([self.z_mean, self.z_log_var])

        # instantiate encoder model
        encoder = Model(self.inputs, [self.z_mean, self.z_log_var, z], name='encoder')
        return encoder

    def decoder_model(self):
        latent_inputs = Input(shape=(self.params.latent_dim,), name='z_sampling')
        x = Dense(np.product(self.params.last_conv_shape[1:]), activation='relu')(latent_inputs)
        x = Reshape(self.params.last_conv_shape[1:])(x)

        for i in range(self.params.layers):
            x = Conv2DTranspose(
                filters=(2 ** (5 + self.params.layers - i)),
                kernel_size=self.params.kernel_size,
                activation='relu',
                strides=2,
                padding='same')(x)
        x = Conv2DTranspose(
            filters=1,
            kernel_size=self.params.kernel_size,
            activation='sigmoid',
            padding='same',
            name='decoder_output')(x)
        decoder = Model(latent_inputs, x, name='decoder')
        return decoder

    def create_model(self):
        self.encoder = self.encoder_model()
        self.decoder = self.decoder_model()
        self.vae = Model(self.encoder.input, self.decoder(self.encoder(self.inputs)[2]), name='vae_mlp')
        self.add_losses()

    def add_losses(self):
        if self.params.loss == 'mse':
            self.reconstruction_loss = mse(K.flatten(self.inputs), K.flatten(self.vae.outputs[0]))
        elif self.params.loss == 'binary_crossentropy':
            self.reconstruction_loss = binary_crossentropy(
                K.flatten(self.inputs), K.flatten(self.vae.outputs[0]))
        else:
            raise ValueError("Not a valid VAE loss.")

        self.reconstruction_loss *= np.prod(self.params.encoder_input_shape) ** 2
        kl_loss = 1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(self.reconstruction_loss + kl_loss)
        self.vae.add_loss(vae_loss)
        self.vae.compile(optimizer='nadam')

