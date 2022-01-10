import tensorflow as tf


class SEBlock(tf.keras.layers.Layer):
    def __init__(self, r, C):
        super(SEBlock, self).__init__()
        self.r = r
        self.C = C

        self.GAP = tf.keras.layers.GlobalAvgPool2D()
        self.FC1 = tf.keras.layers.Dense(self.C / self.r,
                                         activation = 'relu',
                                         kernel_initializer = 'he_normal'
                                         )
        self.FC2 = tf.keras.layers.Dense(self.C,
                                         activation = 'sigmoid'
                                         )

    def call(self, X):
        se = self.GAP(X)
        se = self.FC1(se)
        se = self.FC2(se)
        return X * se


class Conv(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides = (1, 1)):
        super(Conv,self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

        self.C = tf.keras.layers.Conv2D(filters = self.filters,
                                        kernel_size = self.kernel_size,
                                        strides = self.strides,
                                        activation = 'linear',
                                        kernel_initializer = 'he_normal',
                                        use_bias = False,
                                        padding = 'same'
                                        )
        self.B = tf.keras.layers.BatchNormalization()

    def call(self, X):
        X = self.C(X)
        X = self.B(X)
        X = tf.nn.relu(X)
        return X


class SEIdentityBlock(tf.keras.layers.Layer):
    def __init__(self, c_reduce, c_recon, r):
        super(SEIdentityBlock, self).__init__()
        self.c_reduce = c_reduce
        self.c_recon = c_recon
        self.r = r

        self.Conv1 = Conv(filters = self.c_reduce,
                          kernel_size = (1,1)
                          )
        self.Conv2 = Conv(filters = self.c_reduce,
                          kernel_size = (3,3)
                          )
        self.Conv3 = Conv(filters = self.c_recon,
                          kernel_size = (1,1)
                          )
        self.SE = SEBlock(self.r, self.c_recon)

    def call(self, X):
        U = self.Conv1(X)
        U = self.Conv2(X)
        U = self.Conv3(X)
        U = self.SE(U)
        return X + U

class SEConvBlock(tf.keras.layers.Layer):
    def __init__(self, c_reduce, c_recon, r, Downsampling = True):
        super(SEConvBlock, self).__init__()
        self.c_reduce = c_reduce
        self.c_recon = c_recon
        self.r = r
        self.Downsampling = Downsampling

        self.Conv1 = Conv(filters = self.c_reduce,
                          kernel_size = (1, 1),
                          strides = (2, 2) if self.Downsampling else (1, 1)
                          )
        self.Conv2 = Conv(filters = self.c_reduce,
                          kernel_size = (3, 3)
                          )
        self.Conv3 = Conv(filters = self.c_recon,
                          kernel_size = (1, 1)
                          )
        self.SE = SEBlock(self.r, self.c_recon)
        self.linear_projection = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters = self.c_recon,
                                   kernel_size = (1, 1),
                                   strides = (2, 2) if self.Downsampling else (1, 1)
                                   ),
            tf.keras.layers.BatchNormalization()
        ])

    def call(self, X):
        U = self.Conv1(X)
        U = self.Conv2(U)
        U = self.Conv3(U)
        U = self.SE(U)
        X = self.linear_projection(X)
        return X + U