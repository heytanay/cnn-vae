# Main file for CNN-VAE

# Imports
import numpy as np
import tensorflow as tf

# Main class
class ConvVae(object):

    def __init__(self, latent_size=32, batch_size=8, learning_rate=1e-3, kl_tolerance=0.5, is_training=False, reuse=False, gpu=False):
        """
        Constructor
        @param latent_size: Size of the Sample latent Dimension
        @param batch_size: Size of the Batch passed into the model
        @param learning_rate: Learning rate for the model
        @param kl_tolerance: A Paramter for Kullback-leibler loss
        @param is_training: Boolean to specify if model is in training or inference mode
        @param reuse: Boolearn to specify if the model graph can be reused of not
        @param gpu: Boolean to specify if GPU should be used or not
        """

        self.latent_size = latent_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.kl_tolerance = kl_tolerance
        self.is_training = is_training
        self.reuse = reuse

        with tf.variable_scope('conv_vae', reuse=self.reuse):
            if not gpu:
                with tf.device('/cpu:0'):
                        tf.logging.info("Model is using CPU")
                        self.build_graph()
            else:
                tf.logging.info("Model is using GPU")
                self.build_graph()

        self._init_session()

    def build_graph(self):
        """
        Builds the Graph and VAE Model Architecture itself
        """
        self.graph = tf.Graph()
        
        # Set the new graph as default
        with self.graph.as_default():
            # Input Placeholder that gets RGB Colored Images of Dimension -> 64*64
            self.x = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])

            # Encoder
            h = tf.layers.conv2d(self.x, 32, 4, 2, activation=tf.nn.relu, name='enc_conv1')
            h = tf.layers.conv2d(h, 64, 4, 2, activation=tf.nn.relu, name='enc_conv2')
            h = tf.layers.conv2d(h, 128, 4, 2, activation=tf.nn.relu, name='enc_conv3')
            h = tf.layers.conv2d(h, 256, 4, 2, activation=tf.nn.relu, name='enc_conv4')
            h = tf.reshape(h, [-1, 2*2*256])

            # Adding Variations
            self.mu = tf.layers.dense(h, self.latent_size, name='enc_mu')
            self.logvar = tf.layers.dense(h, self.latent_size, name='enc_logvar')
            self.sigma = tf.exp(self.logvar / 2.0)
            self.epsilon = tf.random_normal([self.batch_size, self.latent_size])
            self.z = self.mu + self.sigma * self.epsilon

            # Decoder
            h = tf.layers.dense(self.z, 1024, name='dec_fc1')
            h = tf.reshape(h, [-1, 1, 1, 1024])
            h = tf.layers.conv2d_transpose(h, 128, 5, strides=2, activation=tf.nn.relu, name='dec_deconv1')
            h = tf.layers.conv2d_transpose(h, 64, 5, strides=2, activation=tf.nn.relu, name='dec_deconv2')
            h = tf.layers.conv2d_transpose(h, 32, 6, strides=2, activation=tf.nn.relu, name='dec_deconv3')
            self.y = tf.layers.conv2d_transpose(h, 3, 6, strides=2, activation=tf.nn.relu, name='dec_deconv4')

            # Training Procedure
            if self.is_training:
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.x - self.y), reduction_indices=[1,2,3]))
                self.kl_loss = - 0.5 * tf.reduce_sum((1 + self.logvar - tf.square(self.mu) - tf.exp(self.logvar)), reduction_indices=1)
                self.kl_loss = tf.maximum(self.kl_loss, self.latent_size * self.kl_tolerance)
                self.kl_loss = tf.reduce_mean(self.kl_loss)
                self.loss = self.reconstruction_loss + self.kl_loss
                self.lr = tf.Variable(self.learning_rate, trainable=False)
                self.optim = tf.train.AdamOptimizer(self.lr)
                grads = self.optim.compute_gradients(self.loss)
                self.train_op = self.optim.apply_gradients(grads, global_step=self.global_step, name='train_step')
            
            self.init = tf.global_variables_initializer()
