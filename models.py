# General imports
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

# Keras imports
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2
from keras import layers as KL
from keras.models import Model

class VineyardClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 n_dates=70,
                 conv2_num=1,
                 conv2_num_filters=4,
                 conv2_size=(2, 2),
                 conv1_num=2,
                 conv1_cons_num=1,
                 conv1_num_filters=5,
                 conv1_size=5,
                 dense_size=100,
                 dense_num=1,
                 hidden_activation='tanh',
                 dropout=0.4,
                 kernel_regularizer=1e-4,
                 act_regularizer=0,
                 batch_size=128,
                 epochs=100,
                 keras_path='conv_weights.hdf5'
                 ):

        # Features that define the model topology
        self.n_dates = n_dates
        self.conv2_num = conv2_num
        self.conv2_num_filters = conv2_num_filters
        self.conv2_size = conv2_size

        self.conv1_num = conv1_num
        self.conv1_cons_num = conv1_cons_num
        self.conv1_num_filters = conv1_num_filters
        self.conv1_size = conv1_size

        self.dense_size = dense_size
        self.dense_num = dense_num
        self.hidden_activation = hidden_activation

        # Features that define the regularization
        self.dropout = dropout
        self.kernel_regularizer = kernel_regularizer
        self.act_regularizer = act_regularizer

        # Features that define the training strategy.
        # They can be considered as forms of regularization by some authors.
        # However, I prefer to associate them to the optimization stage and
        # instead of to the loss function.
        self.batch_size = batch_size
        self.epochs = epochs
        self.keras_path = keras_path

    def fit(self, X, y):
        self.model = self.build_model()

        # Learn the normalization constants using the training set only
        self.mean_per_input = {k: i.mean() for k, i in X.items()}
        self.std_per_input = {k: i.std() for k, i in X.items()}

        # Preprocess the dataset
        X_ = self.preprocess(X)

        # Train the model and keep the weights from the best iteration based
        # on the validation set
        checkpoint = ModelCheckpoint(self.keras_path, save_best_only=True,
                                     monitor='val_loss',
                                     verbose=1
                                     )
        history = self.model.fit(X_, y, batch_size=self.batch_size,
                                 epochs=self.epochs, validation_split=0.2,
                                 shuffle=True, verbose=2,
                                 callbacks=[checkpoint])
        self.history = history.history

        # Load the best weights
        self.model.load_weights(self.keras_path)

        return self

    def preprocess(self, X):
        # Normalize the time-based features and aggregate them as a sequence
        # of feature vectors. This allows using a Conv1D or LSTM layer to model
        # the temporal dependences between dates
        conv1d = np.stack([(X[v] - self.mean_per_input[v]) /
                           self.std_per_input[v]
                           for v in ['wind_dir',  # 'wind_spd',
                                     'precip',  # 'temp',
                                     'min_temp', 'max_temp',
                                     'clouds', 'ghi', 'rh']],
                          axis=2
                          )

        ret = {'elevation': X['elevation'],
               'map_coords': X['map_coords'],
               'date-features': conv1d,
               }
        return ret

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return probs.round().astype(np.int)

    def predict_proba(self, X):
        X_ = self.preprocess(X)
        probs = self.model.predict(X_)
        ret = np.hstack((1 - probs, probs))
        return ret

    def build_model(self):
        def get_stream(name, shape, stream_type):
            input_ = KL.Input(shape=shape, name=name)
            last_ = input_

            last_ = KL.GaussianNoise(0.025)(last_)

            if stream_type == 'Conv2D':
                for conv in range(self.conv2_num):
                    last_ = KL.Conv2D(self.conv2_num_filters,
                                      kernel_size=self.conv2_size)(input_)
                    last_ = KL.BatchNormalization()(last_)
                    last_ = KL.Activation(self.hidden_activation)(last_)
                last_ = KL.GlobalMaxPool2D()(last_)
            elif stream_type == 'Conv1D':
                for conv in range(self.conv1_num):
                    for _ in range(self.conv1_cons_num):
                        last_ = KL.Conv1D(self.conv1_num_filters,
                                          kernel_size=self.conv1_size)(last_)
                        last_ = KL.Activation(self.hidden_activation)(last_)
                    # last_ = KL.MaxPooling1D()(last_)
                last_ = KL.Flatten()(last_)
            elif stream_type == 'Norm':
                last_ = KL.BatchNormalization()(input_)
                # last_ = KL.Dense(10, activation='linear')(last_)
                last_ = KL.Activation(self.hidden_activation)(last_)
            elif stream_type == 'LSTM':
                last_ = KL.LSTM(self.conv1_num_filters, activation='tanh',
                                use_bias=True,
                                kernel_regularizer=l2(self.kernel_regularizer),
                                recurrent_regularizer=None,
                                dropout=0.0, recurrent_dropout=0.0,
                                return_sequences=False,
                                return_state=False)(last_)
            else:
                raise NotImplementedError

            return input_, last_

        streams = [('elevation', (5, 5, 1), 'Conv2D'),
                   ('map_coords', (2,), 'Norm'),
                   ('date-features', (self.n_dates, 7), 'Conv1D'),
                   ]

        # Build a stream per input according to the data type
        stream_input, stream_last = zip(*[get_stream(*s) for s in streams])
        stream_input = list(stream_input)
        stream_last = list(stream_last)

        # Concatenate all the streams
        last_ = KL.Concatenate()(stream_last)
        last_ = KL.Dropout(self.dropout)(last_)

        # Add a hidden dense layer
        for dense in range(self.dense_num):
            last_ = KL.Dense(self.dense_size,
                             activation=self.hidden_activation,
                             kernel_regularizer=l2(self.kernel_regularizer),
                             activity_regularizer=l2(self.act_regularizer)
                             )(last_)

        # Output layer
        out = KL.Dense(1, activation='sigmoid',
                       # kernel_regularizer=l2(self.kernel_regularizer)
                       )(last_)

        model = Model(inputs=stream_input, outputs=out)
        model.compile('adadelta', loss='binary_crossentropy',
                      metrics=['accuracy'])

        return model
