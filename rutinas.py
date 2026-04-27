import os
import time
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cliente_pgsql import ClientePostgres
import config_global
CONFIG = config_global.ConfigGlobal()

from datetime import datetime, timedelta
from preprocesado import cargar_PVET_ids, PVET_id, PVET_ids
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, accuracy_score, f1_score, recall_score, precision_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


"""

│
├── # ── CARGA DE DATOS ──────────────────────────────────────────
│   ├── cargar_datos()
│   └── cargar_datos_sanos_mas_cercanos()
│
├── # ── PREPROCESADO ────────────────────────────────────────────
│   ├── completar_timestamps()
│   ├── imputar_nans()
│   └── normalizar_X()
│
├── # ── PREPARACIÓN PARA ENTRENAMIENTO ──────────────────────────
│   ├── separar_df_train_test_caso()       <- principal
│   ├── separar_df_train_test()            <- legacy, puede eliminarse
│   ├── extraer_xy_df()
│   ├── preparar_deteccion()               <- para clase binaria
│   └── preparar_clasificacion()           <- para multiclase
"""



# ── CARGA DE DATOS ──────────────────────────────────────────

def cargar_datos(CONFIG, nom_fich_datos, planta=None):
    # nom_fich_datos = CONFIG.fich_datos
    if '{planta}' in nom_fich_datos and planta is not None:
        nom_fich_datos = nom_fich_datos.replace('{planta}', planta)
    if not os.path.exists(nom_fich_datos):
        return None
    df_fallos = pd.read_csv(nom_fich_datos, index_col=0, parse_dates=['_time', 'ini_fallo', 'fin_fallo'], on_bad_lines='error')
    # Elimina columnas que son completamente NaN
    # Pone el resto de NaN a 0
    df_fallos = df_fallos.dropna(axis=1, how='all')
    df_fallos = df_fallos.fillna(0)
    # Elimina columnas que son solo ceros
    df_fallos = df_fallos.loc[:, (df_fallos != 0).any(axis=0)]
    # Se queda solo con una cierta cantidad máxima de casos sanos para cada fallo,
    # y, si se quiere, más todos los que sean sintéticos (promedio..., tienen pvet_id = 0)
    max_num_casos_sanos = CONFIG.max_disp_sanos_por_fallo if hasattr(CONFIG, 'max_disp_sanos_por_fallo') else 5
    for id_fallo in df_fallos['id_fallo'].unique():
        id_casos_sanos = df_fallos.query(f'(id_fallo == {id_fallo}) & (~ fallo) & pvet_id > 0')['id_caso'].unique()
        if len(id_casos_sanos) > max_num_casos_sanos:
            id_casos_sanos_a_eliminar = id_casos_sanos[max_num_casos_sanos:]
            df_fallos = df_fallos[~df_fallos['id_caso'].isin(id_casos_sanos_a_eliminar)]
    return df_fallos


def cargar_datos_sanos_mas_cercanos(config, nom_fich_datos, planta=None, max_casos_sanos=None):
    if planta is not None and '{planta}' in nom_fich_datos:
        nom_fich_datos = nom_fich_datos.replace('{planta}', planta)
    dir_salida = os.path.dirname(nom_fich_datos)
    if not os.path.exists(nom_fich_datos):
        print(f"File {nom_fich_datos} not found")
        return None
    df = pd.read_csv(
        nom_fich_datos,
        index_col='_time',
        parse_dates=['_time', 'ini_fallo', 'fin_fallo'],
        on_bad_lines='error'
        )
    cols_orig = list(df.columns)
    print(f"DF de BD: {df.shape}, Fallos: {df['id_fallo'].nunique()}/{sum(df['fallo'])}, Casos de fallo: {df['id_caso'].nunique()}")
    # Elimina columnas que son completamente NaN
    df = df.dropna(axis=1, how='all')
    cols_after_na = list(df.columns)
    removed_na = [c for c in cols_orig if c not in cols_after_na]
    print(f"Columnas eliminadas por ser todos NaN: {removed_na}")
    # tratamiento de NaN
    print("Imputación de Nan")
    start_nan = time.time()
    df, tracking = imputar_nans(
        df,
        margen_temporal_h=config.margen_temporal_h, 
        columnas_a_imputar=None,
        time_col='_time',
        id_caso_col='id_caso',
        id_fallo_col='id_fallo',
        id_dispositivo='pvet_id',
        col_fallo='fallo',
        claves_=['ct', 'in', 'id_caso', 'id_fallo', 'pvet_id', 'diag', 'duration', 'ope_ck'],
        max_gap_interpolacion=4
    )
    end_nan = time.time()
    print(f"Tiempo de ejecución de imputar Nan: {end_nan - start_nan:.4f} segundos")
    # Elimina columnas que son solo ceros
    df = df.loc[:, (df != 0).any(axis=0)]
    cols_after_zero = list(df.columns)
    removed_zero = [c for c in cols_after_na if c not in cols_after_zero]
    print(f"Columnas eliminadas por ser todos ceros: {removed_zero}")
    print(f"Columnas finales ({len(cols_after_zero)}): {cols_after_zero}")
    print(f"DF tras eliminar Nan y 0 cols: {df.shape}, Casos: {df['id_caso'].nunique()}, Fallos: {df['id_fallo'].nunique()}/{sum(df['fallo'])}")
    if df.isna().any().any():
        print("WARNIGN: Hay al menos un NaN en df")
    ruta_ids = os.path.join(dir_salida, f"{planta}-PVET_ids.csv") if dir_salida else f"{planta}-PVET_ids.csv"
    df_pvet = pd.read_csv(ruta_ids)
    df_pvet = df_pvet.drop_duplicates()
    cols_num = ["id", "ct", "in", "tr", "sb", "st", "pos"]
    df_pvet[cols_num] = df_pvet[cols_num].apply(pd.to_numeric, errors="coerce")
    pvet_lookup = df_pvet.set_index("id")
    # Matriz de distancias
    pvet_ids_unicos = df['pvet_id'].unique()
    dist_matrix = {}
    for pid1 in pvet_ids_unicos:
        if pid1 not in pvet_lookup.index:
            print(f"Warning: id {pid1} not in {planta}-PVET_ids.csv")
            continue
        row1 = pvet_lookup.loc[pid1]
        for pid2 in pvet_ids_unicos:
            if pid2 not in pvet_lookup.index:
                print(f"Warning: id {pid2} not in {planta}-PVET_ids.csv")
                continue
            row2 = pvet_lookup.loc[pid2]
            dist_matrix[(pid1, pid2)] = distancia_rows(row1, row2)
    lista_final = []
    # Procesar por DIAG
    for diag, grupo_diag in df.groupby('diag'):
        print(f"\n[INFO] Procesando DIAG: {diag}")
        # Determinar mínimo número de sanos por id_fallo para el mismo diag
        sanos_por_fallo = {}
        for id_fallo, grupo_fallo in grupo_diag.groupby('id_fallo'):
            sanos_unicos = grupo_fallo.loc[~grupo_fallo['fallo'] & (grupo_fallo['pvet_id'] > 0), 'pvet_id'].unique()
            sanos_por_fallo[id_fallo] = len(sanos_unicos)
        if not sanos_por_fallo:
            print(f"[WARN] No se encontraron sanos para DIAG {diag}")
            continue
        min_sanos_diag = min(sanos_por_fallo.values())
        if max_casos_sanos is not None:
            min_sanos_diag = min(min_sanos_diag, max_casos_sanos)
        if min_sanos_diag == 0:
            print(f"[WARN] Número mínimo de sanos es 0 para DIAG {diag}, se omite")
            continue
        print(f"[INFO] Se usarán {min_sanos_diag} sanos por fallo para DIAG {diag}")
        #  DF final por fallo
        for id_fallo, grupo_fallo in grupo_diag.groupby('id_fallo'):
            df_fallo = grupo_fallo.loc[grupo_fallo['fallo']].copy()
            pid_fallo = df_fallo['pvet_id'].iloc[0]
            # Sanos disponibles
            df_sanos = grupo_fallo.loc[~grupo_fallo['fallo'] & (grupo_fallo['pvet_id'] > 0)].copy()
            if len(df_sanos['pvet_id'].unique()) < min_sanos_diag:
                continue
            # Calcular distancia usando matriz
            df_sanos['distancia'] = df_sanos['pvet_id'].map(lambda pid: dist_matrix.get((pid, pid_fallo), float('inf')))
            # Seleccionar N sanos más cercanos y fijos
            pvet_ids_seleccionados = (
                df_sanos.groupby('pvet_id').first()
                .sort_values('distancia')
                .head(min_sanos_diag)
                .index
                .tolist()
            )
            df_sanos_filtrados = df_sanos[df_sanos['pvet_id'].isin(pvet_ids_seleccionados)]
            # Promedio opcional (pvet_id == 0)
            # df_prom = grupo_fallo[grupo_fallo['pvet_id'] == 0]
            partes = [df_fallo]
            # if not df_prom.empty:
            #     partes.append(df_prom.iloc[[0]])
            partes.append(df_sanos_filtrados)
            lista_final.append(pd.concat(partes))
    if not lista_final:
        print("[WARN] No se encontraron datos válidos tras el filtrado")
        return None
    df_filtrado = pd.concat(lista_final)
    df_filtrado.index.name = '_time'
    # Filtrar casos 
    # pasos_por_caso = (
    #     df_filtrado
    #     .reset_index()
    #     .groupby('id_caso')['_time']
    #     .nunique()
    # )
    # pasos_moda = pasos_por_caso.mode()[0]
    # casos_validos = pasos_por_caso[pasos_por_caso == pasos_moda].index
    # df_filtrado = df_filtrado[df_filtrado['id_caso'].isin(casos_validos)]
    print(f"[INFO] DF final: {df_filtrado.shape}, Casos válidos: {df_filtrado['id_caso'].nunique()}, Fallos: {df_filtrado['id_fallo'].nunique()}//{sum(df_filtrado['fallo'])}")
    return df_filtrado.drop(columns=['distancia', 'fecha'], errors='ignore').sort_index()


# ── PREPROCESADO ────────────────────────────────────────────

def completar_timestamps(
        df,
        margen_temporal_h,
        freq='15min',
        time_col='_time',
        outer_group_column='id_fallo',
        inner_group_column='id_caso',
        invariable_columns=[
            'id_caso',
            'id_fallo',
            'planta',
            'pvet_id',
            'pvet_disp',
            'tipo_disp',
            'diag',
            'diag_txt',
            'ini_fallo',
            'fin_fallo',
            'duration',
            'fallo_continuo',
            'ope_ck',
            'fallo',
            'ct',
            'in'
            ]
    ):
    df = df.copy()
    if time_col in df.columns:
        df = df.set_index(time_col)
    if invariable_columns is None:
        invariable_columns = []
    lista = []
    pasos_por_hora = pd.Timedelta('1h') / pd.Timedelta(freq)
    N = int(pasos_por_hora * (24 + margen_temporal_h))
    df = df.sort_index()
    lista = []
    for id_fallo, df_fallo in df.groupby(outer_group_column):
        t_min = df_fallo.index.min().floor(freq)
        t_max_real = df_fallo.index.max()
        # rango natural real
        full_index_fallo = pd.date_range(start=t_min,
                                        end=t_max_real,
                                        freq=freq)
        full_index_fallo_len = len(full_index_fallo)
        if full_index_fallo_len >= N:
            if full_index_fallo_len > N:
                print(f"Warning: id_fallo {id_fallo} con {str(full_index_fallo_len)} filas > a {str(N)} filas")
                full_index_fallo = full_index_fallo[:N]
        else:
                print(f"Warning: id_fallo {id_fallo} con {str(full_index_fallo_len)} filas < a {str(N)} filas")
                full_index_fallo = pd.date_range(start=t_min,
                                       periods=N,
                                       freq=freq)
        for id_caso, df_caso in df_fallo.groupby(inner_group_column):
            if len(df_caso) == N and df_caso.index.equals(full_index_fallo):
                lista.append(df_caso)
                continue
            valores_constantes = {
                col: df_caso[col].iloc[0]
                for col in invariable_columns
                if col in df_caso.columns
            }
            df_caso = df_caso.reindex(full_index_fallo)
            for col, val in valores_constantes.items():
                df_caso[col] = val
            lista.append(df_caso)
    df_final = pd.concat(lista)
    df_final.index.name = time_col
    return df_final.sort_index()

def imputar_nans(
        df,
        margen_temporal_h,
        columnas_a_imputar=None,
        time_col='_time',
        id_caso_col='id_caso', # identifica un único dispositivo
        id_fallo_col='id_fallo',
        id_dispositivo='pvet_id',
        col_fallo='fallo',
        claves_=['ct', 'in', 'id_caso', 'id_fallo', 'pvet_id', 'diag', 'duration', 'ope_ck'],
        max_gap_interpolacion=4
    ):
    df = df.copy()
    df.replace([' ', 'nan', 'NaN', ''], np.nan, inplace=True)
    # completar timestamps
    df = completar_timestamps(df,margen_temporal_h,time_col=time_col)
    if time_col not in df.columns and df.index.name == time_col:
        df = df.reset_index()
    df.sort_values(by=[id_fallo_col, id_caso_col, time_col], inplace=True)
    if columnas_a_imputar is None:
        columnas_a_imputar = df.select_dtypes(include=[np.number]).columns.difference(claves_).tolist()
    tracking = {}
    for col in columnas_a_imputar:
        before = df[col].isna().sum()
        tracking[col] = {
            'nan_iniciales': before,
            'interpolacion_simple': 0,
            'media_fallo_timestamp': 0,
            'interpolacion_forzada': 0,
            'media_final': 0
        }
        
        # Paso 1: Interpolación simple por dispositivo
        df[col] = df.groupby(id_caso_col)[col].transform(lambda s: s.interpolate(
            method='linear',
            #   limit=max_gap_interpolacion,
            limit_direction='both'
        ))
        after = df[col].isna().sum()
        tracking[col]['interpolacion_simple'] = before - after
        
        # Paso 2: Imputación por timestamp (mismo grupo)
        mask_nan = df[col].isna()
        mean_fallo_timestamp = (
            df.groupby([id_fallo_col, time_col])[col]
              .transform('mean')
        )
        df.loc[mask_nan, col] = mean_fallo_timestamp[mask_nan]
        after2 = df[col].isna().sum()
        tracking[col]['media_fallo_timestamp'] = after - after2

        # Paso 3: Interpolación completa (forzada)
        before3 = df[col].isna().sum()
        df[col] = df.groupby(id_caso_col)[col].transform(lambda s: s.interpolate(
            method='linear',
            limit_direction='both'
        ))
        after3 = df[col].isna().sum()
        tracking[col]['interpolacion_forzada'] = before3 - after3

        # Paso 4: Media global final
        before4 = df[col].isna().sum()
        df[col] = df[col].fillna(df[col].mean())
        after4 = df[col].isna().sum()
        tracking[col]['media_final'] = before4 - after4
        df[col] = df[col].astype(float)

        if pd.api.types.is_integer_dtype(df[col]) or pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype(float)
    
    # Resumen final
    print("\nResumen de imputación por columna:")
    for col in columnas_a_imputar:
        print(f"\n· Columna: {col}")
        for k, v in tracking[col].items():
            print(f"   → {k}: {v}")

    if time_col in df.columns:
        df.set_index(time_col, inplace=True)
    return df, tracking


def normalizar_X(X, transform_type):
    # X.shape = (N, T, V)
    N, T, V = X.shape
    X_reshaped = X.reshape(-1, V)  # (N*T, V)
    
    # scaler = StandardScaler()
    if transform_type == 'gramian':
        scaler = MinMaxScaler(feature_range=(-1, 1))
    else:
        scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X_reshaped)
    
    X_norm = X_scaled.reshape(N, T, V)
    return X_norm, scaler


# ── PREPARACIÓN PARA ENTRENAMIENTO ──────────────────────────



# ==============================================================================================
# ==============================================================================================

"""
Mantenemos registro de la función anterior que existía en el repositorio
original con el fin de mantener una referencia en caso se requiera volver a el.
"""

def separar_df_train_test(df_fallos, frac_train=0.8):
    """Separa los datos de fallos en conjuntos de entrenamiento y prueba.
    Se asegura de que ambos conjuntos tengan al menos un fallo.
    Si no, reduce el porcentaje de entrenamiento hasta un mínimo del 50%."""
    min_frac_train = 0.5
    separar = True
    while separar and frac_train >= min_frac_train:
        id_casos = df_fallos['id_caso'].unique()
        np.random.shuffle(id_casos)
        n_train = int(len(id_casos) * frac_train)
        train_ids = id_casos[:n_train]
        test_ids = id_casos[n_train:]
        df_train_ids = df_fallos['id_caso'].isin(train_ids)
        df_test_ids = df_fallos['id_caso'].isin(test_ids)
        df_train = df_fallos[df_train_ids]
        df_test = df_fallos[df_test_ids]
        if df_train['fallo'].sum() == 0 or df_test['fallo'].sum() == 0:
            frac_train -= 0.05
            if frac_train < min_frac_train:
                print('No se puede separar el conjunto de datos en entrenamiento y prueba con suficientes fallos.')
                return None, None
        else:
            separar = False
    num_casos_entrenamiento = df_train["id_caso"].nunique()
    num_casos_prueba = df_test["id_caso"].nunique()
    num_fallos_entrenamiento = df_train.loc[df_train["fallo"], "id_caso"].nunique()
    num_fallos_prueba = df_test.loc[df_test["fallo"], "id_caso"].nunique()
    print(f'Número de casos de entrenamiento: {num_casos_entrenamiento}, número de fallos: {num_fallos_entrenamiento}')
    print(f'Número de casos de prueba: {num_casos_prueba}, número de fallos: {num_fallos_prueba}')
    return df_train, df_test

def generar_datos_aprendizaje(df_fallos_base, planta, diag, transform_type = None):
    # df_fallos_base['fallo_acotado'] = (
    #     (df_fallos_base['fallo']) &
    #     (df_fallos_base.index >= df_fallos_base['ini_fallo']) &
    #     (df_fallos_base.index <= df_fallos_base['fin_fallo'])
    # ).astype(int)
    df_fallos = df_fallos_base[df_fallos_base['diag'] == diag]
    diag_txt = df_fallos[df_fallos['fallo']]['diag_txt'].iloc[0]
    tipo_disp = df_fallos[df_fallos['fallo']]['tipo_disp'].iloc[0]
    num_casos = df_fallos['id_caso'].nunique()
    num_fallos = df_fallos['id_fallo'].nunique()
    print(f'Número de casos con diagnóstico {diag}/{diag_txt}: {num_casos} total ({num_casos-num_fallos} sanos, {num_fallos} fallos)')
    if num_casos < 2 or num_fallos < 2:
        print(f'No hay suficientes casos o fallos para entrenar un modelo. Número de casos: {num_casos}, número de fallos: {num_fallos}')
        return None
    if df_fallos.isna().any().any():
        print("Hay NaNs en df_fallos antes de separar train/test")
        print(df_fallos.isna().sum()[df_fallos.isna().sum() > 0])
    df_train, df_test = separar_df_train_test_caso(df_fallos, frac_train=0.8)
    if df_train is None or df_test is None:
        print("No se pudo separar en train/test con fallos en ambos")
        return None
    print("NaNs en df_train:", df_train.isna().sum()[df_train.isna().sum() > 0])
    print("NaNs en df_test:", df_test.isna().sum()[df_test.isna().sum() > 0])
    X_train, y_train, id_casos_train, var_list = extraer_xy_df(df_train, return_var_list=True)
    X_test, y_test, id_casos_test, _ = extraer_xy_df(df_test, return_var_list=True)  
    print("NaNs en X_train (numpy array):", np.isnan(X_train).any())
    print("NaNs en X_test (numpy array):", np.isnan(X_test).any())
    print("pd.NA en X_train:", pd.isna(X_train).any())
    print("pd.NA en X_test:", pd.isna(X_test).any())
    if len(var_list) == 0:
            print("No quedan variables válidas para entrenamiento tras filtrar")
            return None
    print("Tipos en X_train antes de normalizar:", type(X_train), X_train.dtype)

    if transform_type is None:
        scaler = keras.utils.normalize
        X_train = scaler(X_train, axis=1)
        X_test = scaler(X_test, axis=1)
    else:
        X_train, scaler = normalizar_X(X_train, transform_type)
        X_test = scaler.transform(X_test.reshape(-1, X_test.shape[2])).reshape(X_test.shape)
    datos_aprendizaje = {
        'diag': diag,
        'diag_txt': diag_txt,
        'planta': planta,
        'tipo_disp': tipo_disp,
        'df_fallos': df_fallos,
        'df_train': df_train,
        'df_test': df_test,
        'X_train': X_train,
        'y_train': y_train,
        'id_casos_train': id_casos_train,
        'X_test': X_test,
        'y_test': y_test,
        'id_casos_test': id_casos_test,
        'scaler': scaler,
        'var_list': var_list
    }
    return datos_aprendizaje

# ==============================================================================================
# ==============================================================================================


def separar_df_train_test_caso(df_fallos, frac_train=0.8, random_state=None):
    """
    Separa los datos de fallos en conjuntos de entrenamiento y prueba, teniendo
    en cuenta bloques de id_fallo.
    Cada id_fallo (fallo + sus sanos asociados) va completo a train o test.
    El porcentaje se aplica sobre el número de fallos (bloques).
    """
    # Obtener bloques únicos (cada uno contiene 1 fallo + k sanos)
    bloques = df_fallos['id_fallo'].unique()
    if len(bloques) < 2:
        print("No hay suficientes bloques (id_fallo) para dividir.")
        return None, None
    bloques_train, bloques_test = train_test_split(
        bloques,
        train_size=frac_train,
        random_state=random_state
    )
    df_train = df_fallos[df_fallos['id_fallo'].isin(bloques_train)]
    df_test  = df_fallos[df_fallos['id_fallo'].isin(bloques_test)]
    # Estadísticas
    n_bloques_train = df_train['id_fallo'].nunique()
    n_bloques_test  = df_test['id_fallo'].nunique()
    n_casos_train = df_train['id_caso'].nunique()
    n_casos_test  = df_test['id_caso'].nunique()
    n_fallos_train = df_train[df_train['fallo']]['id_caso'].nunique()
    n_fallos_test  = df_test[df_test['fallo']]['id_caso'].nunique()
    print(f"Bloques train: {n_bloques_train}, casos: {n_casos_train}, fallos: {n_fallos_train}")
    print(f"Bloques test:  {n_bloques_test}, casos: {n_casos_test}, fallos: {n_fallos_test}")
    return df_train, df_test


def extraer_xy_df(df, return_var_list=True, var_entrada_override=None):
    if df.empty:
        raise ValueError("El DataFrame pasado a extraer_xy_df está vacío")
    
    tipo_disp = df['tipo_disp'].iloc[0]

    if var_entrada_override and tipo_disp in var_entrada_override:
        var_entrada = var_entrada_override[tipo_disp]
    else:
        excluir = {'ct', 'in', 'tr', 'st', 'sb',
            'id_caso', 'id_fallo', 'planta', 'pvet_id', 'pvet_disp', 'tipo_disp', 'diag', 'diag_txt',
            'ini_fallo', 'fin_fallo', 'duration', 'fallo_continuo', 'ope_ck', 'fallo'}
        var_entrada = [col for col in df.columns
                    if col not in excluir and pd.api.types.is_numeric_dtype(df[col])]

    cols_df = set(df.columns)
    var_entrada_existentes = [v for v in var_entrada if v in cols_df]
    missing_vars = [v for v in var_entrada if v not in cols_df]
    if missing_vars:
        print(f"Advertencia: variables esperadas no están en el DF: {missing_vars}")
    if not var_entrada_existentes:
        raise ValueError(f"No quedan variables de entrada válidas para tipo_disp={tipo_disp}")

    var_entrada = sorted(var_entrada_existentes)
    var_salida = 'fallo'

    X_list, y_list, id_casos = [], [], []
    for id_caso in df['id_caso'].unique():
        df_caso = df[df['id_caso'] == id_caso]
        x = df_caso[var_entrada].astype(float).values
        valores_fallo = df_caso[var_salida].unique()
        if len(valores_fallo) > 1:
            print(f"Inconsistencia en caso {id_caso}: múltiples valores de 'fallo': {valores_fallo}")
        y = int(valores_fallo[0])
        X_list.append(x)
        y_list.append(y)
        id_casos.append(id_caso)

    shape0 = X_list[0].shape
    for i, x in enumerate(X_list):
        if x.shape != shape0:
            raise ValueError(f"El caso {id_casos[i]} tiene forma {x.shape}, esperado: {shape0}")

    X = np.stack(X_list)
    y = np.array(y_list, dtype=int)

    if return_var_list:
        return X, y, id_casos, var_entrada
    return X, y, id_casos


def preparar_deteccion(df_fallos_base, diags, tipo_disp, transform_type=None, frac_train=0.8, random_state=None):
    """
    Prepara datos para DETECCIÓN binaria de fallos.
    
    - Clase 0 (sano):  casos con fallo=False del tipo_disp indicado
    - Clase 1 (fallo): casos con fallo=True cuyo diag esté en la lista diags
    
    Solo trabaja con un tipo de dispositivo a la vez para garantizar
    que las variables de entrada sean homogéneas.
    
    Args:
        df_fallos_base: DataFrame completo cargado del CSV
        diags:          lista de códigos de diagnóstico que se consideran fallo
                        (ej: [241, 242] → ambos son clase 1)
        tipo_disp:      tipo de dispositivo a considerar ('IN', 'ST', 'SB'...)
        transform_type: None usa keras.utils.normalize (legacy),
                        'minmax' o 'gramian' usa MinMaxScaler
        frac_train:     fracción de bloques (id_fallo) para entrenamiento
        random_state:   semilla para reproducibilidad
    """
    # Filtrar por tipo de dispositivo
    df = df_fallos_base[df_fallos_base['tipo_disp'] == tipo_disp].copy()
    if df.empty:
        print(f"No hay datos para tipo_disp={tipo_disp}")
        return None

    # Filtrar: solo los fallos de los diags de interés + todos los sanos
    df = df[(~df['fallo']) | (df['diag'].isin(diags))]

    df_con_fallo = df[df['fallo']]
    if len(df_con_fallo) == 0:
        print(f"No hay casos con fallo=True para diags={diags}, tipo_disp={tipo_disp}")
        return None

    diag_txt  = df_con_fallo['diag_txt'].unique().tolist()
    num_casos  = df['id_caso'].nunique()
    num_fallos = df['id_fallo'].nunique()
    print(f"Detección {tipo_disp}, diags={diags}/{diag_txt}: {num_casos} casos ({num_casos-num_fallos} sanos, {num_fallos} fallos)")

    if num_casos < 2 or num_fallos < 2:
        print(f"No hay suficientes casos para entrenar.")
        return None

    if df.isna().any().any():
        print("NaNs en df antes de separar:")
        print(df.isna().sum()[df.isna().sum() > 0])

    df_train, df_test = separar_df_train_test_caso(df, frac_train=frac_train, random_state=random_state)
    if df_train is None or df_test is None:
        print("No se pudo separar en train/test con fallos en ambos.")
        return None

    X_train, y_train, id_casos_train, var_list = extraer_xy_df(
        df_train, return_var_list=True,
        var_entrada_override=getattr(CONFIG, 'var_entrada', None)
    )
    X_test, y_test, id_casos_test, _ = extraer_xy_df(
        df_test, return_var_list=True,
        var_entrada_override=getattr(CONFIG, 'var_entrada', None)
    )

    if len(var_list) == 0:
        print("No quedan variables válidas para entrenamiento.")
        return None

    scaler = keras.utils.normalize
    X_train = scaler(X_train, axis=1)
    X_test  = scaler(X_test,  axis=1)
    
    #if transform_type is None:
    #    scaler = keras.utils.normalize
    #    X_train = scaler(X_train, axis=1)
    #    X_test  = scaler(X_test,  axis=1)
    #else:
    #    X_train, scaler = normalizar_X(X_train, transform_type)
    #    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[2])).reshape(X_test.shape)

    enc = OneHotEncoder(handle_unknown='ignore')

    return {
        'modo': 'detection',
        'diags': diags,
        'diag_txt': diag_txt,
        'tipo_disp': tipo_disp,
        'num_clases': 2,
        'df_fallos': df,
        'df_train': df_train,
        'df_test': df_test,
        'X_train': X_train, 'y_train': y_train, 'y_train_onehot' : enc.fit_transform(y_train.reshape(-1,1)).toarray(), 'id_casos_train': id_casos_train,
        'X_test':  X_test,  'y_test':  y_test, 'y_test_onehot' : enc.fit_transform(y_test.reshape(-1,1)).toarray(),  'id_casos_test':  id_casos_test,
        'scaler': scaler,
        'var_list': var_list
    }


def preparar_clasificacion(df_fallos_base, diags, tipo_disp, transform_type=None, frac_train=0.8, random_state=None):
    """
    Prepara datos para CLASIFICACIÓN multiclase de tipos de fallo.
    
    Solo trabaja con casos que tienen fallo=True.
    Cada diag es una clase distinta. No incluye sanos.
    
    Args:
        df_fallos_base: DataFrame completo cargado del CSV
        diags:          lista de códigos de diagnóstico a clasificar
                        (ej: [241, 242, 243] → 3 clases)
        tipo_disp:      tipo de dispositivo ('IN', 'ST', 'SB'...)
        transform_type: None, 'minmax' o 'gramian'
        frac_train:     fracción de bloques para entrenamiento
        random_state:   semilla para reproducibilidad
    """
    # Solo fallos del tipo y diags de interés
    df = df_fallos_base[
        (df_fallos_base['tipo_disp'] == tipo_disp) &
        (df_fallos_base['fallo']) &
        (df_fallos_base['diag'].isin(diags))
    ].copy()

    if df.empty:
        print(f"No hay casos de fallo para diags={diags}, tipo_disp={tipo_disp}")
        return None

    # Mapeo diag → clase numérica (ordenado para reproducibilidad)
    diags_presentes = sorted(df['diag'].unique().tolist())
    mapa_clases = {d: i for i, d in enumerate(diags_presentes)}
    df['clase'] = df['diag'].map(mapa_clases)
    num_clases = len(diags_presentes)

    diag_txt   = df['diag_txt'].unique().tolist()
    num_fallos = df['id_fallo'].nunique()
    print(f"Clasificación {tipo_disp}, diags={diags_presentes}/{diag_txt}: {num_fallos} fallos, {num_clases} clases")
    print(f"Mapa de clases: {mapa_clases}")

    if num_fallos < 2:
        print("No hay suficientes fallos para entrenar.")
        return None

    if df.isna().any().any():
        print("NaNs en df antes de separar:")
        print(df.isna().sum()[df.isna().sum() > 0])

    df_train, df_test = separar_df_train_test_caso(df, frac_train=frac_train, random_state=random_state)
    if df_train is None or df_test is None:
        print("No se pudo separar en train/test.")
        return None

    # Para clasificación la variable salida es 'clase', no 'fallo'
    X_train, y_train, id_casos_train, var_list = extraer_xy_df(
        df_train, return_var_list=True,
        var_entrada_override=getattr(CONFIG, 'var_entrada', None)
    )
    X_test, y_test, id_casos_test, _ = extraer_xy_df(
        df_test, return_var_list=True,
        var_entrada_override=getattr(CONFIG, 'var_entrada', None)
    )
    # Sobreescribe y con la clase multiclase
    y_train = df_train.loc[df_train['id_caso'].isin(id_casos_train)].groupby('id_caso')['clase'].first().loc[id_casos_train].values
    y_test  = df_test.loc[df_test['id_caso'].isin(id_casos_test)].groupby('id_caso')['clase'].first().loc[id_casos_test].values

    if len(var_list) == 0:
        print("No quedan variables válidas para entrenamiento.")
        return None

    if transform_type is None:
        scaler = keras.utils.normalize
        X_train = scaler(X_train, axis=1)
        X_test  = scaler(X_test,  axis=1)
    else:
        X_train, scaler = normalizar_X(X_train, transform_type)
        X_test = scaler.transform(X_test.reshape(-1, X_test.shape[2])).reshape(X_test.shape)

    return {
        'modo': 'classification',
        'diags': diags_presentes,
        'diag_txt': diag_txt,
        'mapa_clases': mapa_clases,
        'tipo_disp': tipo_disp,
        'num_clases': num_clases,
        'df_fallos': df,
        'df_train': df_train,
        'df_test': df_test,
        'X_train': X_train, 'y_train': y_train, 'id_casos_train': id_casos_train,
        'X_test':  X_test,  'y_test':  y_test,  'id_casos_test':  id_casos_test,
        'scaler': scaler,
        'var_list': var_list
    }

