import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR

import logging
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import keras
import pandas as pd
pd.set_option('display.max_columns', None)

from keras.layers import Conv1D, Dense, Dropout, Input, Concatenate, GlobalAveragePooling1D, GlobalMaxPooling1D, MaxPooling1D, Flatten, MultiHeadAttention, LayerNormalization, BatchNormalization
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras_tuner import RandomSearch

from tensorflow.keras.losses import CategoricalFocalCrossentropy, CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC, Precision, Recall
from metrics import F1ScoreMetric, MatthewsCorrelationCoefficient

def crear_modelo1(hp, X_shape, num_clases):
    input_layer = Input(shape=(X_shape[1], X_shape[2]))
    num_filtros = hp.Choice("filters", [8, 16, 32, 64, 128, 256, 512]) 
    tam_núcleo = hp.Int("kernel_size", min_value=3, max_value=15, step=2)
    val_dropout = hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.1)
    num_dense = hp.Int("dense_units", min_value=16, max_value=256, step=16)
    modelo = Sequential([
        input_layer,
        Conv1D(filters=num_filtros, kernel_size=tam_núcleo, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(val_dropout),
        Conv1D(filters=num_filtros*2, kernel_size=tam_núcleo, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(val_dropout),
        Flatten(),
        Dense(num_dense, activation='relu'),
        Dropout(val_dropout),
        Dense(num_clases, activation='softmax')
    ])
    modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return modelo

# ================ MODELO DE QPV ======================================

def Modelo_QPV_Conv1D(hp, X_shape, num_clases):
    input_layer = Input(shape=(X_shape[1], X_shape[2]))
    num_filtros = hp.Choice("filters", [8, 16, 32, 64, 128, 256, 512]) 
    tam_núcleo = hp.Int("kernel_size", min_value=3, max_value=10, step=2)
    val_dropout = hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.1)
    lr = hp.Choice('lr', [1e-4, 1e-5])

    modelo = Sequential([
        input_layer,
        Conv1D(filters=num_filtros, kernel_size=tam_núcleo, activation='relu'),
        Dropout(val_dropout),
        GlobalAveragePooling1D(),
        Dense(num_clases, activation='softmax')
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

def crear_QPV(X_shape, num_clases):
    input_layer = Input(shape=(X_shape[1], X_shape[2]))
    modelo = Sequential([
        input_layer,
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=256, kernel_size=3, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dense(num_clases, activation='softmax')
    ])
    modelo.compile(
        optimizer=Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=[
            'accuracy',
            AUC(name='auc'),
            Precision(name='precision'),
            Recall(name='recall'),
            F1ScoreMetric(name='f1_score'),
            MatthewsCorrelationCoefficient(name='mcc')
        ])
    return modelo




