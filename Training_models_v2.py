#%%

# ———————————————— Librerías ——————————————————————————————————————————————————————
import os
import shutil
import config_global
import logging
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras_tuner
import tensorflow as tf

from rutinas import cargar_datos, cargar_datos_sanos_mas_cercanos, preparar_deteccion, preparar_clasificacion
from evaluation import evaluar_modelo
from visualization import dibujar_fallos, dibujar_historial
# from rutinas import permutation_importance_model, cargar_datos_sanos_mas_cercanos

from keras.layers import Conv1D, Dense, Dropout, Input, Concatenate, GlobalMaxPooling1D, MaxPooling1D, Flatten, BatchNormalization, GlobalAveragePooling1D, LSTM, ConvLSTM2D
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras_tuner import RandomSearch
from datetime import datetime, timedelta

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import CategoricalFocalCrossentropy, CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC, Precision, Recall, Metric
from metrics import F1ScoreMetric, MatthewsCorrelationCoefficient
from sklearn.utils import class_weight
from pyts.image import MarkovTransitionField, RecurrencePlot, GramianAngularField
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, Dropout, Dense, Input)
from sklearn.preprocessing import KBinsDiscretizer
from pathlib import Path

CONFIG = config_global.ConfigGlobal()
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# —————————————————————————————————————————————————————————————————————————————————

# ———————————————— Hipermodelo y modelos de ML ———————————————————————————————————

class Hipermodelo(keras_tuner.HyperModel):

    def __init__(self, model, X_shape, num_clases):
        self. X_shape = X_shape
        self.num_clases = num_clases
        self.model = model

    def build(self, hp):
        return self.model(hp, self.X_shape, self.num_clases)
    
    #def fit(self, hp, model, *args, **kwargs):
    #    batch_size = hp.Choice("batch_size", [16, 32, 64])
    #    patience = hp.Int("patience", 5, 10)
    #    # Callbacks desde keras_tuner
    #    callbacks = kwargs.pop("callbacks", [])
    #    # Añadir
    #    callbacks += [
    #        EarlyStopping(
    #            monitor='val_loss',
    #            patience=patience,
    #            restore_best_weights=True
    #        ),
    #        ReduceLROnPlateau(
    #            monitor='val_loss',
    #            factor=0.5,
    #            patience=3
    #        )
    #    ]
    #    return model.fit(
    #        *args,
    #        batch_size=batch_size,
    #        callbacks=callbacks,
    #        **kwargs
    #    )


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


def Modelo_QPV_ConvLSTM2D(hp, X_shape, num_clases):
    input_layer = Input(shape=(X_shape[1], 1, X_shape[3], 1))
    filters_cnn_lstm = hp.Choice("filters", [8, 16, 32, 64, 128])
    kernel_size = (1,3)
    val_dropout = hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.1)
    num_dense = hp.Int("dense_units", min_value=2, max_value=64, step=8)
    lr = hp.Choice('lr', [1e-4, 1e-5])

    modelo = Sequential([
        input_layer,
        ConvLSTM2D(
            filters = filters_cnn_lstm,
            kernel_size = kernel_size,
            padding = 'same',
            return_sequences = False,
            activation = 'relu',
            recurrent_activation = 'sigmoid'
        ),
        BatchNormalization(),
        Dropout(val_dropout),
        Flatten(),
        Dense(
            num_dense,
            activation  = 'sigmoid',
        ),
        Dense(
            num_clases,
            activation  = 'softmax',
        )
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


def Modelo_PVOP_personalizado1(hp, X_shape, num_clases):
    """
    AQUI SE PUEDE CREAR UN MODELO PERSONALIZADO PARA EVALUAR
    CON LA CONDICIÓN QUE LUEGO DEBE AGREGARSE EN EL DICCIONARIO INFERIOR 
    LLAMADO "MODELOS" JUNTO CON UNA FUNCIÓN ANÓNIMA QUE DESCRIBA CÓMO SERÁN
    INTRODUCIDOS SUS DATOS DE ENTRADA (DIMENSIONES QUE VARIARÁN DE ACUERDO A CADA MODELO)
    """
    return None


MODELOS = {
    "LSTM": {
        "model": Modelo_QPV_LSTM,
        "preprocesar": lambda X: X  
    },
    "Conv1D": {
        "model": Modelo_QPV_Conv1D,
        "preprocesar": lambda X: X  
    },
    "ConvLSTM2D": {
        "model": Modelo_QPV_ConvLSTM2D,
        "preprocesar": lambda X: X[:, :, np.newaxis, :, np.newaxis] 
    },
    "Modelo_PVOP": {
        "model": Modelo_PVOP_personalizado1,
        "preprocesar": lambda X: X  
    },
}

# —————————————————————————————————————————————————————————————————————————————————
# —————— Tranformaciones de espacio de datos de entrada (Markov y Grammian) ———————
# —————————————————————————— y función de entrenamiento ———————————————————————————

def aplicar_transformada(datos, transform_type):
    """
    Aplica la transformada 2D al X_train y X_test.
    Entrada:  X shape (N, T, V)
    Salida:   X shape (N, T, T, V) para gramian/recurrence
              X shape (N, n_bins, n_bins, V) para markov
    """
    transformadores = {
        'gramian'    : GramianAngularField(method='summation'),
        'recurrence' : RecurrencePlot(dimension=1, threshold='point', percentage=20),
        'markov'     : MarkovTransitionField(image_size=20, n_bins=8, strategy='quantile'),
    }

    if transform_type not in transformadores:
        print(f"transform_type='{transform_type}' no reconocido, sin transformada")
        return datos

    t = transformadores[transform_type]

    for split in ('X_train', 'X_test'):
        X = datos[split]
        num_samples, seq_len, n_features = X.shape
        # Transformar feature a feature y apilar como canales
        canales = []
        for f in range(n_features):
            img = t.fit_transform(X[:, :, f])  # (N, size, size)
            canales.append(img)
        # Resultado: (N, size, size, V)
        datos[split] = np.stack(canales, axis=-1)

    return datos

def entrenar_modelo(modelo, X_train, y_train, X_test, y_test, epochs=200, batch_size=32, patience=5, class_weight_dict=None):
    """
    Entrena el modelo con early stopping.
    
    y_train e y_test deben estar en el formato que espera el modelo:
    - one-hot si usa CategoricalCrossentropy
    - enteros si usa SparseCategoricalCrossentropy
    """
    early_stop = EarlyStopping(monitor='val_loss', patience=patience, min_delta=0.001, restore_best_weights=True)
    history = modelo.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        class_weight=class_weight_dict 
    )
    return history


# —————————————————————————————————————————————————————————————————————————————————
# ———————— CARGA DE CONFIGURACIÓN Y FICHEROS   ————————————————————————————————————



def cargar_config(args, config_base='config/config_gen1.py'):
    """Carga el config base y luego el config_rn encima."""
    config_global.ConfigGlobal(config_base)
    CONFIG = config_global.ConfigGlobal(args[0])
    print(f'CONFIG usada:\n{CONFIG}')
    return CONFIG


def preparar_directorio_planta(CONFIG, nombre_grupo):
    """
    Crea el directorio de resultados para un grupo de plantas y guarda la config usada.
    Devuelve el path resuelto del directorio.
    """
    dir_resultados = CONFIG.dir_resultados.replace('{planta}', nombre_grupo)
    os.makedirs(dir_resultados, exist_ok=True)
    with open(f'{dir_resultados}/config_usada.txt', 'w') as f:
        f.write(str(CONFIG) + '\n')
    return dir_resultados


def construir_patron_ficheros(dir_resultados, nombre_grupo, nombre_modelo, tipo_disp, diag, transform_type):
    """
    Construye el prefijo de rutas para guardar resultados.
    Ejemplo: resultados/planta_A/res-cnn2d/gramian/res-cnn2d-inversor-001-gramian
    """    
    transform_str = transform_type if transform_type else 'sin_transform'  # ← fix None
    base = os.path.join(dir_resultados, f'{nombre_modelo}', transform_str)
    os.makedirs(base, exist_ok=True)
    patron = os.path.join(base, f'{nombre_modelo}')
    if tipo_disp:
        patron += f'-{tipo_disp}'
    if diag:
        diag_str = '-'.join(f'{int(d):03}' for d in diag) if isinstance(diag, list) else f'{int(diag):03}'
        patron += f'-{diag_str}'
    patron += f'-{transform_str}'
    return patron

 
# ─────────────────────────────────────────────────────────────────────────────
# CARGA DE DATOS DE PLANTAS
# ─────────────────────────────────────────────────────────────────────────────
 
def combinar_plantas(lista_df):
    """
    Combina DataFrames de varias plantas tomando solo las columnas
    comunes a todas ellas. Avisa de las columnas descartadas.
    
    Args:
        lista_df: lista de DataFrames, uno por planta
    Returns:
        DataFrame combinado con solo las columnas comunes
    """
    if len(lista_df) == 1:
        return lista_df[0]

    # Columnas de metadata que siempre deben estar
    cols_meta = {
        'id_caso', 'id_fallo', 'planta', 'pvet_id', 'pvet_disp',
        'tipo_disp', 'diag', 'diag_txt', 'ini_fallo', 'fin_fallo',
        'duration', 'fallo_continuo', 'ope_ck', 'fallo'
    }

    # Columnas de operación por planta (excluye metadata)
    cols_op_por_planta = [
        set(df.columns) - cols_meta for df in lista_df
    ]

    # Intersección de columnas de operación
    cols_comunes = cols_op_por_planta[0].intersection(*cols_op_por_planta[1:])
    cols_descartadas = cols_op_por_planta[0].union(*cols_op_por_planta[1:]) - cols_comunes

    if cols_descartadas:
        print(f"Columnas descartadas por no ser comunes a todas las plantas: {sorted(cols_descartadas)}")

    cols_finales = sorted(cols_comunes | cols_meta)
    dfs_alineados = [df[[c for c in cols_finales if c in df.columns]] for df in lista_df]
    return pd.concat(dfs_alineados)
 

  
def cargar_y_preparar(CONFIG):
    """
    Carga CSVs de todas las plantas, intersecta columnas,
    y llama a preparar_deteccion o preparar_clasificacion.
    Devuelve el dict de datos listo para entrenar o None.
    """
    lista_df = []
    for planta in CONFIG.plantas_train:
        fich = CONFIG.fich_datos.replace('{planta}', planta)
        if not os.path.exists(fich):
            print(f"No existe fichero para planta={planta}: {fich}")
            continue
        df = cargar_datos(CONFIG, fich)
        if df is not None and len(df) > 0:
            lista_df.append(df)

    if not lista_df:
        print("Sin datos para ninguna planta.")
        return None

    # Combinar plantas descartando columnas no comunes
    df = combinar_plantas(lista_df)

    # Preparar según modo
    if CONFIG.modo == 'detection':
        datos = preparar_deteccion(
            df_fallos_base = df,
            diags          = CONFIG.diags,
            tipo_disp      = CONFIG.tipo_disp,
            transform_type = CONFIG.transform_type,
        )
    elif CONFIG.modo == 'classification':
        datos = preparar_clasificacion(
            df_fallos_base = df,
            diags          = CONFIG.diags,
            tipo_disp      = CONFIG.tipo_disp,
            transform_type = CONFIG.transform_type,
        )
    else:
        raise ValueError(f"modo desconocido: '{CONFIG.modo}'")

    if datos is None:
        return None

    # Aplicar transformada 2D si procede (después de preparar, no dentro)
    if CONFIG.transform_type in ('gramian', 'markov'):
        datos = aplicar_transformada(datos, CONFIG.transform_type)

    return datos

 
# ─────────────────────────────────────────────────────────────────────────────
# TUNING Y ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────────────
 
def hacer_tuning(datos, CONFIG, patron, nombre_modelo):
    """
    Ejecuta la búsqueda bayesiana de hiperparámetros.
    Devuelve (mejor_modelo, mejores_hp).
    """
    config_modelo = MODELOS[nombre_modelo]
    X_train = config_modelo['preprocesar'](datos['X_train'])

    project_name = f'{nombre_modelo}_{CONFIG.transform_type}'  # 'Conv1D_None'
    tuner_dir = str(Path(patron).parent / 'tuning')            # 'results/br03-IN-246/tuning'
    os.makedirs(f'{tuner_dir}/{project_name}', exist_ok=True)
 
    hipermodelo = Hipermodelo(
        model   = config_modelo['model'],
        X_shape    = X_train.shape,
        num_clases = datos['num_clases']
    )
 
    tuner = keras_tuner.BayesianOptimization(
        hipermodelo,
        objective            = 'val_loss',
        max_trials           = CONFIG.max_trials,
        num_initial_points   = CONFIG.num_initial_points,
        executions_per_trial = CONFIG.executions_per_trial,
        directory            = tuner_dir,
        project_name         = project_name,
    )
 
    tuner.search_space_summary(extended=True)
    tuner.search(
        X_train,
        datos['y_train_onehot'],
        validation_data = (config_modelo['preprocesar'](datos['X_test']), datos['y_test_onehot']),
        epochs          = CONFIG.epochs_tuning
    )
    tuner.results_summary()
 
    best_hp    = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.get_best_models(num_models=1)[0]
    print(f"  Mejores hiperparámetros: {best_hp.values}")
 
    return best_model, best_hp
 
 
def entrenar_modelo_final(modelo, datos, best_hp, CONFIG, nombre_modelo):
    """
    Reentrena el mejor modelo con los hiperparámetros encontrados.
    """
    config_modelo = MODELOS[nombre_modelo]
    X_train = config_modelo['preprocesar'](datos['X_train'])
    X_test  = config_modelo['preprocesar'](datos['X_test'])
 
    historia = entrenar_modelo(
        modelo,
        X_train,
        datos['y_train_onehot'],
        X_test,
        datos['y_test_onehot'],
        epochs     = CONFIG.epochs_final,
        batch_size = CONFIG.batch_size,
        patience   = CONFIG.patience,
    )
    return historia
 
 
# ─────────────────────────────────────────────────────────────────────────────
# GUARDAR RESULTADOS
# ─────────────────────────────────────────────────────────────────────────────
 
def guardar_resultados(modelo, historia, datos, patron, nombre_modelo):
    config_modelo = MODELOS[nombre_modelo]
    X_test = config_modelo['preprocesar'](datos['X_test'])
    datos_eval = {**datos, 'X_test': X_test}

    # patron pasa a ser un directorio, no un prefijo de fichero
    dir_salida = Path(patron)
    dir_salida.mkdir(parents=True, exist_ok=True)
    patron_ficheros = dir_salida / 'resultados'  # en lugar de dir_salida / nombre_modelo

    keras.utils.plot_model(modelo, show_shapes=True, to_file=str(patron_ficheros) + '-'+ nombre_modelo + '-arquitectura.png')
    dibujar_historial(historia, nombre_modelo, patron_ficheros=str(patron_ficheros))
    evaluar_modelo(modelo, nombre_modelo, datos_eval, patron_ficheros=str(patron_ficheros))
 
# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENTO
# ─────────────────────────────────────────────────────────────────────────────
 
def ejecutar_experimento(CONFIG, datos, patron):
    """
    Orquesta tuning → entrenamiento → guardado para un conjunto de datos ya preparado.
    """
    nombre_modelo = CONFIG.nombre_modelo  # 'LSTM' | 'Conv1D' | 'ConvLSTM2D'
 
    modelo, best_hp = hacer_tuning(datos, CONFIG, patron, nombre_modelo)
    historia        = entrenar_modelo_final(modelo, datos, best_hp, CONFIG, nombre_modelo)
    guardar_resultados(modelo, historia, datos, patron, nombre_modelo)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
 
def main(args):
    CONFIG = cargar_config(args, config_base='config/config_gen1.py')

    keras.utils.set_random_seed(CONFIG.semilla)

    # Nombre del experimento para guardar resultados
    plantas_str = '_'.join(CONFIG.plantas_train)
    diags_str   = '_'.join(str(d) for d in CONFIG.diags)
    nombre_exp  = f"{plantas_str}-{CONFIG.tipo_disp}-{diags_str}-{CONFIG.modo}"
    dir_resultados = Path(
        CONFIG.dir_resultados
        .replace('{plantas}',   plantas_str)
        .replace('{tipo_disp}', CONFIG.tipo_disp)
        .replace('{diags}',     diags_str)
    )
    dir_resultados.mkdir(parents=True, exist_ok=True)

    datos = cargar_y_preparar(CONFIG)
    if datos is None:
        print("Sin datos suficientes para entrenar. Abortando.")
        return

    patron = dir_resultados / nombre_exp
    ejecutar_experimento(CONFIG, datos, patron)
 

# %%

main(args= ["config/config_rn1.py"])
# %%
