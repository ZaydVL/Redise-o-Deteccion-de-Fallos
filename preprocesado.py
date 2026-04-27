import os
import json
from datetime import datetime,timedelta
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from cliente_influx import ClienteInflux
from cliente_pgsql import ClientePostgres
import random
import config_global

depurar = True if "DEPURAR" in os.environ and os.environ["DEPURAR"].lower() == "true" else False

###################################################################

def corregir_fecha(fecha):
    ''' Corrige el formato de fecha para que sea compatible con InfluxDB.
    Por ejemplo, convierte '2023-10-01 12:00:00' en '2023-10-01T12:00:00Z'.
    Si ya es un objeto datetime, lo convierte a string en el formato correcto.'''
    if isinstance(fecha, str):
        if 'T' not in fecha:
            fecha = fecha.replace(' ', 'T')
    elif isinstance(fecha, datetime):
        fecha = fecha.strftime('%Y-%m-%dT%H:%M:%SZ')
    return fecha

###################################################################

def cargar_df(cliente_influx: ClienteInflux, nom_bucket: str, nom_medida, t_inicio, t_final) -> pd.DataFrame:
    ''' Carga de InfluxDB los datos de una variable de operación entre dos fechas.'''
    t_inicio = corregir_fecha(t_inicio)
    t_final = corregir_fecha(t_final)
    consulta = f'''
        from(bucket:"{nom_bucket}") |>
        range(start: {t_inicio}, stop: {t_final}) |>
        filter(fn: (r) => r["_measurement"] == "{nom_medida}") |>
        pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")|> group()
    '''
    if depurar:
        print(f'CONSULTA INFLUX: {consulta}')
    df = cliente_influx.cargar_df(consulta=consulta)
    for campo in [ 'ct', 'in', 'tr', 'sb', 'st', 'pos' ]:
        if campo in df.columns:
            df[campo] = pd.to_numeric(df[campo])
    df.index = df.index.tz_localize(None)
    return df

###################################################################

def cargar_meteo(cliente_influx: ClienteInflux, nom_bucket: str, nom_medida, t_inicio, t_final) -> pd.DataFrame:
    ''' Carga de InfluxDB los datos de una variable de operación entre dos fechas.'''
    t_inicio = corregir_fecha(t_inicio)
    t_final = corregir_fecha(t_final)
    consulta = f'''
        from(bucket:"{nom_bucket}") |>
        range(start: {t_inicio}, stop: {t_final}) |>
        filter(fn: (r) => (r["_measurement"] == "vop_ms" or r["_measurement"] == "vop_rm")) |>
        pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''
    if depurar:
        print(f'CONSULTA INFLUX: {consulta}')
    df = cliente_influx.cargar_df(consulta=consulta)
    # si se combinan vop_ms y vop_rm, salen dos filas por timestamps, por lo que hay que combinar el resultado en una
    # first() devuelve el primer valor no nulo por columna
    df = df.groupby(df.index).first()
    campos = [ 'ct', 'in', 'tr', 'sb', 'st', 'pos' ]
    for campo in campos:
        if campo in df.columns:
            df[campo] = pd.to_numeric(df[campo])
    df = df.query('ct == 0 & pos == 0')
    df = df.drop(columns=['ct', 'pos'])
    if isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
    return df

###################################################################

# Los campos CT/IN..POS están en mayúsculas porque IN no puede ser minúsculas (palabra reservada Python)

"""
CT (Centro de Transformación)
└── IN (Inversor)
    └── TR (Tracker / seguidor solar)
        └── SB (StringBox / caja de strings)
            └── ST (String / cadena de paneles)
                └── POS (posición del panel en el string)

Entonces un PVET_id con CT=1, IN=3, TR=0, SB=2, ST=5, POS=0 identifica unívocamente un dispositivo 
concreto dentro de la planta. El campo type indica qué nivel de la jerarquía es ese dispositivo 
(si es un inversor, un string, etc.), y id es simplemente la clave numérica con la que ese dispositivo 
está registrado en PostgreSQL.


Entonces el flujo será de la forma:


PostgreSQL                          InfluxDB
    │                                   │
    │  "El dispositivo 47 (CT1/IN2/...) │
    │   tuvo el fallo 201 el 2023-06-15"│
    │                                   │
    ▼                                   │
Sabemos QUÉ falló y CUÁNDO             │
    │                                   │
    └──────────────────────────────────▶│
                    "Dame los datos de operación
                     del 2023-06-15 para CT1/IN2/..."
                                        │
                                        ▼
                            Serie temporal del día del fallo
                            + datos de dispositivos sanos
                            + datos meteorológicos
"""

@dataclass
class PVET_id:
    id: int
    CT: int
    IN: int
    TR: int
    SB: int
    ST: int
    POS: int
    type: int

    def __str__(self):
        return f'D{self.id}:CT{self.CT}/IN{self.IN}/TR{self.TR}/SB{self.SB}/ST{self.ST}/POS{self.POS}/type{self.type}'

PVET_ids = {}

def cargar_PVET_ids(cliente_sql: ClientePostgres, planta:str, usar_cache=False, ruta_salida=False) -> dict[int,PVET_id]:
    ''' Carga los identificadores de los dispositivos PVET desde la base de datos SQL.
    Si usar_cache es True, intenta cargar los datos desde un fichero JSONL.'''
    if len(PVET_ids) > 0:
        PVET_ids.clear() # Limpia el diccionario si ya hay datos cargados
    nom_fich_pvet_ids = f'pvet_ids-{planta}.jsonl'
    if usar_cache and os.path.exists(nom_fich_pvet_ids):
        with open(nom_fich_pvet_ids, 'r') as f:
            for o in f:
                aux1 = json.loads(o)
                aux2 = PVET_id(**aux1)
                PVET_ids[aux2.id] = aux2
    else:
        consulta_sql = f"SELECT * FROM PVET_ids"
        cursor = cliente_sql.obtener_cursor(consulta_sql)
        registros = [] 
        for fila in cursor:
            elem = PVET_id(id=fila['id'], CT=fila['ct'], IN=fila['in'], TR=fila['tr'], SB=fila['sb'], ST=fila['st'], POS=fila['pos'], type=fila['type'])
            PVET_ids[elem.id] = elem
            registro = {
                "id": fila['id'],
                "ct": fila['ct'],
                "in": fila['in'],
                "tr": fila['tr'],
                "sb": fila['sb'],
                "st": fila['st'],
                "pos": fila['pos'],
                "type": fila['type']
            }
            registros.append(registro)
        if usar_cache:
            with open(nom_fich_pvet_ids, 'w') as f:
                for o in PVET_ids.values():
                    print(json.dumps(asdict(o)), file=f)
        if ruta_salida:
            df_pvet = pd.DataFrame(registros)
            df_pvet.to_csv(ruta_salida, index=False)
    return PVET_ids

# FUNCIONES AUXILIARES ############################################

def seleccionar_dispositivo(df: pd.DataFrame, disp: PVET_id) -> pd.DataFrame:
    ''' Devuelve un DataFrame filtrado por el dispositivo indicado.'''
    consulta = ''
    for campo in [ 'ct', 'in', 'tr', 'sb', 'st', 'pos' ]:
        if campo in df.columns:
            if len(consulta) > 0:
                consulta += ' & '
            consulta += f"(`{campo}` == {getattr(disp, campo.upper())})"
    if depurar:
        print(f'CONSULTA PANDAS: {consulta}')
    return df.query(consulta)

###################################################################

#def escoger_otro_dispositivo(PVET_ids, disp_fallo: PVET_id):
#    ''' Escoge un dispositivo diferente al indicado, pero del mismo tipo.'''
#    buscar = True
#    while buscar:
#        id_otro = random.randint(1, len(PVET_ids)-1)
#        buscar = id_otro != disp_fallo.id and id_otro in PVET_ids and PVET_ids[id_otro].type == disp_fallo.type
#    return PVET_ids[id_otro]


"""
El id es simplemente un número de registro, como el DNI: único, sin significado estructural, 
solo sirve para referenciar rápidamente al dispositivo en las consultas SQL.

Las coordenadas CT/IN/TR/SB/ST/POS son la dirección física del dispositivo dentro de la planta, 
y sí te dicen exactamente dónde está en la jerarquía. Por eso cuando un dispositivo es, por ejemplo, 
un StringBox, los niveles por debajo (ST y POS) están a 0, porque no aplican.

"""

def escoger_otro_dispositivo(PVET_ids: dict, disp_fallo: PVET_id) -> PVET_id:
    '''Escoge aleatoriamente un dispositivo del mismo tipo que disp_fallo,
    distinto a él.'''
    candidatos = [
        disp for disp in PVET_ids.values()
        if disp.id != disp_fallo.id and disp.type == disp_fallo.type
    ]
    if not candidatos:
        raise ValueError(f'No hay otros dispositivos del tipo {disp_fallo.type}')
    return random.choice(candidatos)

###################################################################

num_id_fallo = num_id_caso = 1

def obtener_datos_casos(cliente_sql:ClientePostgres, cliente_influx:ClienteInflux, nom_planta:str, tipo_disp:str, diag_interes=None, margen_temporal_h:int=0) -> pd.DataFrame:
    ''' Devuelve un DataFrame con los datos de cada fallo y de varios dispositivos sanos.
    '''
    global num_id_fallo, num_id_caso
    CONFIG = config_global.ConfigGlobal()
    ratio_datos_min = CONFIG.ratio_datos_min  # Porcentaje mínimo de datos para guardar los datos del dispositivo
    nom_tabla_fallos = 'DDA_DIA'
    tabla_disp = f'vop_{tipo_disp}'.lower()
    filtro_interes = f"Type = '{tipo_disp}' AND ope_ck >= 1"#
    if diag_interes is not None:
        if isinstance(diag_interes, int):
            filtro_interes += f" AND diag = {diag_interes}"
        elif isinstance(diag_interes, list):
            filtro_interes += f" AND diag IN ({','.join(map(str, diag_interes))})"
    consulta = f"SELECT COUNT(*) FROM {nom_tabla_fallos} WHERE {filtro_interes}"
    # cliente_sql.obtener_cursor(consulta, as_dict=False)
    cliente_sql.obtener_cursor(consulta)
    num_fallos = int(cliente_sql.leer_registro()['count'])
    # Obtiene todos los fallos validados de ese tipo
    consulta_sql = f""" SELECT * FROM {nom_tabla_fallos}
                        JOIN diagnosis ON {nom_tabla_fallos}.diag=diagnosis.code
                        WHERE {filtro_interes}
                        ORDER BY diag,Duration DESC"""
    if depurar:
        print(f'CONSULTA SQL: {consulta_sql}')
    cursor = cliente_sql.obtener_cursor(consulta_sql)

    # Para cada fallo, guardará los datos de ese fallo y de varios dispositivos sanos
    # A cada fallo se le asigna un id de grupo de fallo, y a cada dispositivo un id de caso.
    # Por tanto, todos los dispositivos de un mismo fallo tendrán el mismo id de grupo de fallo,
    # y cada dispositivo tendrá un id de caso único.
    lista_casos = []
    num_fallo = 1
    datos_comp = 4 * (24 + margen_temporal_h) # Ñapa: 96 datos por día, más los de margen temporal
    stats = {
        "fallos_totales": 0,
        "fallos_guardados": 0,
        "dispositivos_fallidos_rechazados_ratio": 0,
        "dispositivos_fallidos_sin_datos": 0,
        "dispositivos_sanos_rechazados_ratio": 0
    }
    for fila in cursor:
        print(f'{num_fallo}/{num_fallos} FALLOS', flush=True)
        stats["fallos_totales"] += 1
        ini_time = fila['ini_time']
        end_time = fila['end_time']
        id_dispositivo_fallo = fila['id']
        diag_fallo = fila['diag']
        diag_fallo_txt = fila['esp']
        duración_fallo = fila['duration']
        # fallo_continuo es True si el fallo existe durante todo el período ini-end_time
        # fallo_continuo = (duración_fallo - 15) == ((end_time - ini_time).total_seconds() / 60)
        fallo_continuo = abs((duración_fallo - 15) - ((end_time - ini_time).total_seconds() / 60)) < 0.1

        # Carga los datos de todos los dispositivos de ese día (00:00:00 a 23:59:59)
        # Añade un margen temporal de N horas antes
        ini_dia = datetime(ini_time.year, ini_time.month, ini_time.day)
        fin_dia = datetime(ini_time.year, ini_time.month, ini_time.day) + timedelta(days=1, seconds=-1)
        df_dia = cargar_df(cliente_influx, nom_planta, tabla_disp, ini_dia - timedelta(hours=margen_temporal_h), fin_dia)
        df_meteo = cargar_meteo(cliente_influx, nom_planta, None, ini_dia - timedelta(hours=margen_temporal_h), fin_dia)
        # print(len(df_dia.index.unique()))
        # print(len(df_meteo.index.unique()))
        df_dia = df_dia.join(df_meteo, how='outer', rsuffix='_meteo')
        df_dia = df_dia.sort_index()
        # print(len(df_dia.index.unique()))
        # Ahora va a guardar los datos del dispositivo que ha fallado y de varios sanos
        disp_fallo = PVET_ids[id_dispositivo_fallo]
        dispositivos_guardar = [ disp_fallo ]
        # Escoge varios dispositivos sanos, al azar si hay max_disp_sanos_por_fallo.
        # Si hay max_disp_sanos_por_fallo, itenta que sean dicha cantidad mínima, pero a veces hay menos.
        dispositivos_sanos = obtener_dispositivos_sanos(cliente_sql, tipo_disp, fecha_fallo=ini_dia.strftime('%Y-%m-%d'))
        dispositivos_keys = list(dispositivos_sanos.keys())
        if CONFIG.max_disp_sanos_por_fallo:
            num_disp_sanos = min(CONFIG.max_disp_sanos_por_fallo, len(dispositivos_keys))
            ids_dispositivos_sanos = random.sample(dispositivos_keys, num_disp_sanos)
        else:
            ids_dispositivos_sanos = dispositivos_keys
        dispositivos_guardar.extend(dispositivos_sanos[i] for i in ids_dispositivos_sanos)

        # Para cada dispositivo, selecciona los datos del día y los guarda en el DataFrame
        # Si el dispositivo no tiene datos suficientes, lo ignora.
        # Si el dispositivo que ha fallado no tiene datos suficientes, no guarda nada.
        # Si es el dispositivo que ha fallado, guarda los datos del fallo.
        # Si es un dispositivo sano, guarda los datos como si no hubiera fallado.
        # En principio, mejor coger todos los fallos que cumplan un ratio mínimo, y luego ver si se pueden rellenar los faltantes,
        # ya que, al no haber tantos fallos, mejor no quitar los fallos que no estén muy incompletos
        num_dispositivos_guardados = 0
        for dispositivo in dispositivos_guardar:
            datos_guardar = seleccionar_dispositivo(df_dia, dispositivo).copy()
            num_datos_guardar = len(datos_guardar)
            ratio_datos = num_datos_guardar/datos_comp
            if datos_guardar.empty:
                if dispositivo.id == disp_fallo.id:
                    print(f"Dispositivo que falló {disp_fallo.id} sin datos (EMPTY)")
                    stats["dispositivos_fallidos_sin_datos"] += 1
                    break
                continue
            elif num_datos_guardar != datos_comp:
                # tener en cuenta que, si no se limitan a datos completos, en el análisis de los modelos habrá que hacer
                # un tratamiento posterior para que los datos tengan la misma logntiud temporal
                if dispositivo.id == disp_fallo.id:
                    print(f"Dispositivo que falló {disp_fallo.id} sin datos completos: {str(num_datos_guardar)}/{str(datos_comp)} ({str(round(ratio_datos,2))})")
                    if ratio_datos < ratio_datos_min:
                        print(f"No se coge el dispositivo que falló {disp_fallo.id}")
                        stats["dispositivos_fallidos_rechazados_ratio"] += 1
                        break # Si el dispositivo que ha fallado no tiene datos suficientes, no guarda nada.
                else:
                    print(f"Dispositivo sano {dispositivo.id} sin datos completos: {str(num_datos_guardar)}/{str(datos_comp)} ({str(round(ratio_datos,2))})")
                    if ratio_datos < ratio_datos_min:
                        print(f"No se coge el dispositivo sano {dispositivo.id}")
                        stats["dispositivos_sanos_rechazados_ratio"] += 1
                        continue # Si un dispositivo sano no tiene datos suficientes, lo ignora.
            datos_guardar['id_caso'] = num_id_caso
            datos_guardar['id_fallo'] = num_id_fallo
            datos_guardar['planta'] = nom_planta
            datos_guardar['pvet_id'] = dispositivo.id
            datos_guardar['pvet_disp'] = str(dispositivo)
            datos_guardar['tipo_disp'] = tipo_disp
            datos_guardar['diag'] = diag_fallo
            datos_guardar['diag_txt'] = diag_fallo_txt
            datos_guardar['ini_fallo'] = ini_time
            datos_guardar['fin_fallo'] = end_time
            datos_guardar['duration'] = duración_fallo
            datos_guardar['fallo_continuo'] = fallo_continuo
            datos_guardar['ope_ck'] = fila['ope_ck']
            datos_guardar['fallo'] = (dispositivo.id == disp_fallo.id)
            lista_casos.append(datos_guardar)
            num_id_caso += 1
            num_dispositivos_guardados += 1
        if num_dispositivos_guardados > 0 and False: # JMR: pendiente revisar, quizá no vale para mucho
            # Si se han guardado datos de algún dispositivo,
            # guarda el promedio para cada instante del día como un caso más
            datos_promedio = df_dia.groupby('_time').mean()
            datos_promedio['id_caso'] = num_id_caso
            datos_promedio['id_fallo'] = num_id_fallo
            datos_promedio['planta'] = nom_planta
            datos_promedio['pvet_id'] = 0
            datos_promedio['pvet_disp'] = 'PROMEDIO'
            datos_promedio['tipo_disp'] = tipo_disp
            datos_promedio['diag'] = diag_fallo
            datos_promedio['diag_txt'] = diag_fallo_txt
            datos_promedio['ini_fallo'] = ini_time
            datos_promedio['fin_fallo'] = end_time
            datos_promedio['duration'] = duración_fallo
            datos_promedio['fallo_continuo'] = fallo_continuo
            datos_promedio['ope_ck'] = fila['ope_ck']
            datos_promedio['fallo'] = False
            lista_casos.append(datos_promedio)
            num_id_caso += 1
        if num_dispositivos_guardados > 0:
            num_id_fallo += 1 # Solo incrementa si se han guardado datos de algún dispositivo
            stats["fallos_guardados"] += 1
        num_fallo += 1
    if lista_casos:
        df_casos = pd.concat(lista_casos)
        return df_casos.sort_index(), stats
    else:
        return None, stats

###################################################################

#def obtener_dispositivos_sanos(cliente_sql: ClientePostgres, disp_fallo: str, fecha_fallo:str=None) -> dict[int,PVET_id]:
#    """
#    Obtiene los dispositivos sanos de un tipo de fallo específico.
#    """
#    # Primero obtiene todos los dispositivos de ese tipo
#    consulta = f"SELECT * FROM PVET_ids WHERE Type = '{disp_fallo}'"
#    if depurar:
#        print(f'CONSULTA SQL: {consulta}')
#    cursor = cliente_sql.obtener_cursor(consulta)
#    dispositivos_sanos = {}
#    for fila in cursor:
#        dispositivo = PVET_id(id=fila['id'], CT=fila['ct'], IN=fila['in'], TR=fila['tr'], SB=fila['sb'], ST=fila['st'], POS=fila['pos'], type=fila['type'])
#        dispositivos_sanos[dispositivo.id] = dispositivo
#
#    # Luego elimina los que tengan fallos registrados, opcionalmente filtrando por fecha
#    consulta = f"SELECT DISTINCT ID FROM DDA_DIA WHERE Type = '{disp_fallo}'"
#    if fecha_fallo:
#        consulta += f" AND date = '{fecha_fallo}'"
#    if depurar:
#        print(f'CONSULTA SQL: {consulta}')
#    cursor = cliente_sql.obtener_cursor(consulta)
#    for fila in cursor:
#        del dispositivos_sanos[fila['id']]
#
#    return dispositivos_sanos


def obtener_dispositivos_sanos(cliente_sql: ClientePostgres, tipo_disp: str, fecha_fallo: str = None) -> dict[int, PVET_id]:
    """Obtiene los dispositivos sanos de un tipo dado, opcionalmente filtrando por fecha.
    Un dispositivo se considera sano si no tiene fallos registrados en DDA_DIA ese día."""
    # Primero obtiene todos los dispositivos de ese tipo
    consulta = f"SELECT * FROM PVET_ids WHERE Type = '{tipo_disp}'"
    if depurar:
        print(f'CONSULTA SQL: {consulta}')
    cursor = cliente_sql.obtener_cursor(consulta)
    dispositivos_sanos = {}
    for fila in cursor:
        dispositivo = PVET_id(id=fila['id'], CT=fila['ct'], IN=fila['in'], TR=fila['tr'],
                              SB=fila['sb'], ST=fila['st'], POS=fila['pos'], type=fila['type'])
        dispositivos_sanos[dispositivo.id] = dispositivo

    consulta = f"SELECT DISTINCT ID FROM DDA_DIA WHERE Type = '{tipo_disp}'"
    if fecha_fallo:
        consulta += f" AND date = '{fecha_fallo}'"
    if depurar:
        print(f'CONSULTA SQL: {consulta}')
    cursor = cliente_sql.obtener_cursor(consulta)
    for fila in cursor:
        dispositivos_sanos.pop(fila['id'], None)  # pop evita KeyError si el id no existe

    return dispositivos_sanos

###################################################################

if __name__ == "__main__":
    pass
