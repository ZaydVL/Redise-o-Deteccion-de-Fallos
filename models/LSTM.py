import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR

import logging
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import keras
import pandas as pd
pd.set_option('display.max_columns', None)

from keras import Model
from keras.layers import Conv1D, Dense, Dropout, Input, Concatenate, GlobalMaxPooling1D, MaxPooling1D, Flatten, BatchNormalization, GlobalAveragePooling1D, LSTM, ConvLSTM2D
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras_tuner import RandomSearch

from tensorflow.keras.losses import CategoricalFocalCrossentropy, CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC, Precision, Recall
from metrics import F1ScoreMetric, MatthewsCorrelationCoefficient


def Modelo_QPV_LSTM(hp, X_shape, num_clases):
    units_LSTM = hp.Choice("filters", [5, 25, 50, 75, 100, 125]) 
    val_dropout = hp.Float("dropout", min_value=0.0, max_value=0.4, step=0.1)
    lr = hp.Choice('lr', [1e-4, 1e-5])

    modelo = Sequential([
        LSTM(
            units=units_LSTM,
            return_sequences=False,
            activation='tanh',
            recurrent_activation='sigmoid',
            input_shape=(X_shape[1], X_shape[2]) 
        ),
        Dropout(val_dropout),
        Dense(num_clases, activation='sigmoid')
    ])

    modelo.compile(
        optimizer=Adam(learning_rate=lr),
        loss=CategoricalCrossentropy(),
        metrics=[
            'accuracy',
            AUC(name='auc'),
            Precision(name='precision'),
            Recall(name='recall'),
            F1ScoreMetric(name='f1_score'),
            MatthewsCorrelationCoefficient(name='mcc')
        ]
    )
    return modelo