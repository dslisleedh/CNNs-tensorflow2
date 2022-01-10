import tensorflow as tf
from Blocks import *


class Stage(tf.keras.layers.Layer):
    def __init__(self, n_blocks, c_reduce, c_recon, r, Downsampling = True):
        super(Stage, self).__init__()
        self.n_blocks = n_blocks
        self.c_reduce = c_reduce
        self.c_recon = c_recon
        self.r = r
        self.Downsampling = Downsampling


        self.Stage = tf.keras.Sequential([
            SEConvBlock(self.c_reduce,
                        self.c_recon,
                        self.r,
                        Downsampling = self.Downsampling
                        )
        ] + [
            SEIdentityBlock(self.c_reduce,
                            self.c_recon,
                            self.r
                            ) for _ in range(self.n_blocks - 1)
        ])

    def call(self, X):
        return self.Stage(X)


class SE_ResNet_50(tf.keras.models.Model):
    def __init__(self, n_labels, activation):
        super(SE_ResNet_50, self).__init__()
        self.n_labels = n_labels
        self.activation = activation

        self.Conv1 = Conv(filters = 64,
                          kernel_size = (7,7),
                          strides = (2,2)
                          )

        self.MP1 = tf.keras.layers.MaxPool2D(pool_size = (3, 3),
                                             strides = (2, 2)
                                             )
        self.Stages = tf.keras.Sequential([
            Stage(3, 64, 256, 16, Downsampling = False),
            Stage(4, 128, 512, 32),
            Stage(6, 256, 1024, 64),
            Stage(3, 512, 2048, 128)
        ])
        self.GAP = tf.keras.layers.GlobalAvgPool2D()
        self.Classifier = tf.keras.layers.Dense(self.n_labels, activation = self.activation)

    def call(self, X):
        X = self.Conv1(X)
        X = self.MP1(X)
        X = self.Stages(X)
        head = self.GAP(X)
        y = self.Classifier(head)
        return y