import tensorflow as tf

from keras.models import Model, Sequential
from keras.layers import Layer, Input, Reshape, Dense, Conv1D, Convolution1D, Dropout
from keras.layers import Activation, AveragePooling1D, MaxPooling1D, Flatten, Lambda, Multiply, Add
from keras.regularizers import l1, l2
from keras import backend as K


class CompositionVAE:
	"""
	Variational autoencoder to dense chemical compositions
	"""
    def __init__(
        self, original_dim=200, latent_dim=64, initializer='he_normal',
        last_layer_activation='linear', activation='relu', optimizer='rmsprop',
        num_hidden=[1024, 512], dropout_rate=0.02, l1=1e-5, l2=1e-5,
        epsilon_std=1e-4,
    ):
        self.original_dim = original_dim
        self._input_shape = (self.original_dim,)
        self.latent_dim = latent_dim
        self.initializer = initializer
        self.last_layer_activation = last_layer_activation
        self.activation = activation
        self.num_hidden = num_hidden
        self.dropout_rate = dropout_rate
        self.epsilon_std = epsilon_std
        self.l1 = l1
        self.l2 = l2
        self.optimizer = optimizer
        
        self.vae, self.encoder, self.decoder = self.compile()
        
    def compile(self):
        input_tensor = Input(shape=self._input_shape, name='input')

        layer_1 = Dense(units=self.num_hidden[0], kernel_regularizer=l1(self.l1),
                        kernel_initializer=self.initializer, activation=self.activation)(input_tensor)
        dropout_1 = Dropout(self.dropout_rate)(layer_1)
        layer_2 = Dense(units=self.num_hidden[1], kernel_regularizer=l1(self.l1),
                        kernel_initializer=self.initializer, activation=self.activation)(dropout_1)
        dropout_2 = Dropout(self.dropout_rate)(layer_2)
        z_mu = Dense(units=self.latent_dim, activation="linear", name='mean_output',
                     kernel_initializer=self.initializer,
                     kernel_regularizer=l1(self.l1))(dropout_2)
        z_log_var = Dense(units=self.latent_dim, activation='linear', name='sigma_output',
                          kernel_initializer=self.initializer,
                          kernel_regularizer=l1(self.l1))(dropout_2)

        z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])
        z_sigma = Lambda(lambda t: tf.exp(.5 * t))(z_log_var)

        eps = Input(
            tensor=tf.random_normal(mean=0., stddev=self.epsilon_std, shape=(tf.shape(z_mu)[0], self.latent_dim)))
        z_eps = Multiply()([z_sigma, eps])
        z = Add()([z_mu, z_eps])

        decoder = Sequential(name='output')
        decoder.add(Dense(units=self.num_hidden[1], kernel_regularizer=l1(self.l1),
                          kernel_initializer=self.initializer, activation=self.activation, input_dim=self.latent_dim))
        decoder.add(Dropout(self.dropout_rate))
        decoder.add(Dense(units=self.num_hidden[0], kernel_regularizer=l2(self.l2),
                          kernel_initializer=self.initializer, activation=self.activation))
        decoder.add(Dropout(self.dropout_rate))
        decoder.add(Dense(units=self.original_dim, kernel_regularizer=l2(self.l2),
                          kernel_initializer=self.initializer, activation=self.last_layer_activation, name='output'))

        x_pred = decoder(z)
        x_pred = Reshape((self.original_dim,), input_shape=(self.original_dim, 1))(x_pred)

        vae = Model(inputs=[input_tensor, eps], outputs=x_pred)
        
        vae.compile(optimizer=self.optimizer, loss=nll)
        encoder = Model(input_tensor, z_mu)
        
        print(vae.summary())

        return vae, encoder, decoder


class XRDVAE:
	"""
	Convolutional variational autoencoder to dense 1D diffraction patterns
	Original implementation: https://github.com/henrysky/astroNN (arXiv:1808.04428)
	"""
    def __init__(
        self, original_dim=200, latent_dim=64, initializer='he_normal',
        filter_len=2, last_layer_activation='linear', activation='relu',
        num_filters=[16, 32, 64], num_hidden=[1024, 512], dropout_rate=0.02,
        batch_size=64, epsilon_std=1e-4, l1=1e-5, l2=1e-5, pool_length=3,
        optimizer='rmsprop',
    ):
        self.original_dim = original_dim
        self._input_shape = (self.original_dim,)
        self.latent_dim = latent_dim
        self.initializer = initializer
        self.filter_len = filter_len
        self.last_layer_activation = last_layer_activation
        self.activation = activation
        self.num_filters = num_filters
        self.num_hidden = num_hidden
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.epsilon_std = epsilon_std
        self.l1 = l1
        self.l2 = l2
        self.pool_length = pool_length
        self.optimizer = optimizer
        
        self.vae, self.encoder, self.decoder = self.compile()
        
    def compile(self):
        input_tensor = Input(shape=self._input_shape, name='input')
        input_internal = Reshape((self.original_dim, 1), input_shape=self._input_shape)(input_tensor)
        cnn_layer_1 = Conv1D(kernel_initializer=self.initializer, activation=self.activation, padding="same",
                             filters=self.num_filters[0],
                             kernel_size=self.filter_len, kernel_regularizer=l2(self.l2))(input_internal)
        dropout_1 = Dropout(self.dropout_rate)(cnn_layer_1)
        maxpool_1 = MaxPooling1D(pool_size=self.pool_length)(dropout_1)
        cnn_layer_2 = Conv1D(kernel_initializer=self.initializer, activation=self.activation, padding="same",
                             filters=self.num_filters[1],
                             kernel_size=self.filter_len, kernel_regularizer=l2(self.l2))(maxpool_1)
        dropout_2 = Dropout(self.dropout_rate)(cnn_layer_2)
        maxpool_2 = MaxPooling1D(pool_size=self.pool_length)(dropout_2)
        cnn_layer_3 = Conv1D(kernel_initializer=self.initializer, activation=self.activation, padding="same",
                             filters=self.num_filters[2],
                             kernel_size=self.filter_len, kernel_regularizer=l2(self.l2))(maxpool_2)
        dropout_3 = Dropout(self.dropout_rate)(cnn_layer_3)
        maxpool_3 = MaxPooling1D(pool_size=self.pool_length)(dropout_3)    
        flattener = Flatten()(maxpool_3)
        layer_4 = Dense(units=self.num_hidden[0], kernel_regularizer=l1(self.l1),
                        kernel_initializer=self.initializer, activation=self.activation)(flattener)
        dropout_3 = Dropout(self.dropout_rate)(layer_4)
        layer_5 = Dense(units=self.num_hidden[1], kernel_regularizer=l1(self.l1),
                        kernel_initializer=self.initializer, activation=self.activation)(dropout_3)
        dropout_4 = Dropout(self.dropout_rate)(layer_5)
        z_mu = Dense(units=self.latent_dim, activation="linear", name='mean_output',
                     kernel_initializer=self.initializer,
                     kernel_regularizer=l1(self.l1))(dropout_4)
        z_log_var = Dense(units=self.latent_dim, activation='linear', name='sigma_output',
                          kernel_initializer=self.initializer,
                          kernel_regularizer=l1(self.l1))(dropout_4)

        z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])
        z_sigma = Lambda(lambda t: tf.exp(.5 * t))(z_log_var)

        eps = Input(
            tensor=tf.random_normal(mean=0., stddev=self.epsilon_std, shape=(tf.shape(z_mu)[0], self.latent_dim)))
        z_eps = Multiply()([z_sigma, eps])
        z = Add()([z_mu, z_eps])

        decoder = Sequential(name='output')
        decoder.add(Dense(units=self.num_hidden[1], kernel_regularizer=l1(self.l1),
                          kernel_initializer=self.initializer, activation=self.activation, input_dim=self.latent_dim))
        decoder.add(Dropout(self.dropout_rate))
        decoder.add(Dense(units=self._input_shape[0]*self.num_filters[1], kernel_regularizer=l2(self.l2),
                          kernel_initializer=self.initializer, activation=self.activation))
        decoder.add(Dropout(self.dropout_rate))
        output_shape = (self.batch_size, self._input_shape[0], self.num_filters[1])
        decoder.add(Reshape(output_shape[1:]))
        decoder.add(Conv1D(kernel_initializer=self.initializer, activation=self.activation, padding="same",
                           filters=self.num_filters[2],
                           kernel_size=self.filter_len, kernel_regularizer=l2(self.l2)))
        decoder.add(Dropout(self.dropout_rate))
        decoder.add(Conv1D(kernel_initializer=self.initializer, activation=self.activation, padding="same",
                           filters=self.num_filters[1],
                           kernel_size=self.filter_len, kernel_regularizer=l2(self.l2)))
        decoder.add(Dropout(self.dropout_rate))    
        decoder.add(Conv1D(kernel_initializer=self.initializer, activation=self.activation, padding="same",
                           filters=self.num_filters[0],
                           kernel_size=self.filter_len, kernel_regularizer=l2(self.l2)))
        decoder.add(Conv1D(kernel_initializer=self.initializer, activation=self.last_layer_activation, padding="same",
                           filters=1, kernel_size=self.filter_len, name='output'))

        x_pred = decoder(z)
        x_pred = Reshape((self.original_dim,), input_shape=(self.original_dim, 1))(x_pred)

        vae = Model(inputs=[input_tensor, eps], outputs=x_pred)
        
        vae.compile(optimizer=self.optimizer, loss=nll)
        encoder = Model(input_tensor, z_mu)
        
        print(vae.summary())

        return vae, encoder, decoder
		

class SpaceGroupClassifier():
	"""
	Fully-connected neural network to classify 1D diffraction patterns
	Original implementation: https://doi.org//10.1107/S205225251700714X/fc5018sup1.pdf
	Park, Woon Bae and Chung, Jiyong and Jung, Jaeyoung and Sohn, Keemin and Singh, Satendra Pal and Pyo, Myoungho and Shin, Namsoo and Sohn, Kee-Sun
	Classification of crystal structure using a convolutional neural network
	IUCrJ, 4, 4, 2017, 486--494, 10.1107/S205225251700714X
	"""
    def __init__(
        self, input_shape=(8001, 1),
        num_filters=[80, 80, 80], strides=[5, 5, 2], kernel_size=[100, 50, 25],
        num_hidden=[2300, 1150], conv_dropout=0.3, conn_dropout=0.5, pool_size=3,
        activation='relu', last_layer_activation='softmax', optimizer='Adam',
        initializer='he_normal'
    ):
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.strides = strides
        self.kernel_size = kernel_size
        self.conv_dropout = conv_dropout
        self.conn_dropout = conn_dropout
        self.num_hidden = num_hidden
        self.activation = activation
        self.last_layer_activation = last_layer_activation
        self.pool_size = pool_size
        self.optimizer = optimizer
        self.initializer = initializer
        
        self.model = self.compile()
        
    def compile(self):
        model = Sequential()
        
        model.add(Convolution1D(self.num_filters[0], self.kernel_size[0], strides=self.strides[0], padding='same',
                                input_shape=self.input_shape, kernel_initializer=self.initializer))
        model.add(Activation(self.activation))
        model.add(Dropout(self.conv_dropout))
        model.add(AveragePooling1D(pool_size=self.pool_size, strides=None))
        model.add(Convolution1D(self.num_filters[1], self.kernel_size[1], strides=self.strides[1], padding='same',
                                kernel_initializer=self.initializer))
        model.add(Activation(self.activation))
        model.add(Dropout(self.conv_dropout))
        model.add(AveragePooling1D(pool_size=self.pool_size, strides=None))
        model.add(Convolution1D(self.num_filters[2], self.kernel_size[2], strides=self.strides[2], padding='same',
                                kernel_initializer=self.initializer))
        model.add(Activation(self.activation))
        model.add(Dropout(self.conv_dropout))
        model.add(AveragePooling1D(pool_size=self.pool_size, strides=None))
        model.add(Flatten())
        
        model.add(Dense(self.num_hidden[0], kernel_initializer=self.initializer))
        model.add(Activation(self.activation))
        model.add(Dropout(self.conn_dropout))
        model.add(Dense(self.num_hidden[1], kernel_initializer=self.initializer))
        model.add(Activation(self.activation))
        model.add(Dropout(self.conn_dropout))
        model.add(Dense(230))
        model.add(Activation(self.last_layer_activation))
        
        model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        
        print(model.summary())
        
        return model    
    
    
class FormationEnergyEstimator:
	"""
	Fully-connected neural network to estimate stability of structure (formation energy )
	"""
    def __init__(
        self, input_dim=88, kernel_initializer='he_normal',
        last_layer_activation='sigmoid', activation='relu',
        num_hidden=64, dropout_rate=0.5, optimizer='rmsprop'
    ):
        self.input_dim = input_dim
        self.last_layer_activation=last_layer_activation
        self.kernel_initializer=kernel_initializer
        self.activation=activation
        self.num_hidden = num_hidden
        self.dropout_rate = dropout_rate
        self.optimizer = optimizer
        
        self.model = self.compile()
        
    def compile(self):
        model = Sequential()
        model.add(Dense(self.num_hidden, input_dim=self.input_dim, activation=self.activation,
                        kernel_initializer=self.kernel_initializer))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(self.num_hidden, activation=self.activation, kernel_initializer=self.kernel_initializer))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(self.num_hidden, activation=self.activation, kernel_initializer=self.kernel_initializer))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(1, activation=self.last_layer_activation, kernel_initializer=self.kernel_initializer))
        
        model.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        
        print(model.summary())

        return model    


class KLDivergenceLayer(Layer):
    """
    | Identity transform layer that adds KL divergence to the final model losses.
    | KL divergence used to force the latent space match the prior (in this case its unit gaussian)
    :return: A layer
    :rtype: object
    :History: 2018-Feb-05 - Written - Henry Leung (University of Toronto)
    """
    def __init__(self, name=None, **kwargs):
        self.is_placeholder = True
        if not name:
            prefix = self.__class__.__name__
            name = prefix + '_' + str(K.get_uid(prefix))
        super().__init__(name=name, **kwargs)

    def call(self, inputs, training=None):
        """
        :Note: Equivalent to __call__()
        :param inputs: Tensor to be applied, concatenated tf.tensor of mean and std in latent space
        :type inputs: tf.Tensor
        :return: Tensor after applying the layer
        :rtype: tf.Tensor
        """	
        mu, log_var = inputs
        kl_batch = - .5 * tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), axis=-1)
        self.add_loss(tf.reduce_mean(kl_batch), inputs=inputs)

        return inputs

    def get_config(self):
        """
        :return: Dictionary of configuration
        :rtype: dict
        """	
        config = {'None': None}
        base_config = super().get_config()
        return {**dict(base_config.items()), **config}

    def compute_output_shape(self, input_shape):
        return input_shape
    

def nll(y_true, y_pred):
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)
