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