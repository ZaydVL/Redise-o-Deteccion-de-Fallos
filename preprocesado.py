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
        filter(fn: (r) => (r["_measurement"] == "vop_ms" or r["_measurement"] == "vop_ms")) |>
        pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")|> group()
    '''
    if depurar:
        print(f'CONSULTA INFLUX: {consulta}')
    df = cliente_influx.cargar_df(consulta=consulta)
    for campo in [ 'ct', 'in', 'tr', 'sb', 'st', 'pos' ]:
        if campo in df.columns:
            df[campo] = pd.to_numeric(df[campo])
    df = df.query('ct == 0 & pos == 0')
    df = df.drop(columns=['ct', 'pos'])
    df.index = df.index.tz_localize(None)
    return df

###################################################################

# Los campos CT/IN..POS están en mayúsculas porque IN no puede ser minúsculas (palabra reservada Python)
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

def cargar_PVET_ids(cliente_sql: ClientePostgres, planta:str, usar_cache=False) -> dict[int,PVET_id]:
    ''' Carga los identificadores de los dispositivos PVET desde la base de datos SQL.
    Si usar_cache es True, intenta cargar los datos desde un fichero JSONL.'''
    if len(PVET_ids) > 0:
        PVET_ids.clear()  # Limpia el diccionario si ya hay datos cargados
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
        for fila in cursor:
            elem = PVET_id(id=fila['id'], CT=fila['ct'], IN=fila['in'], TR=fila['tr'], SB=fila['sb'], ST=fila['st'], POS=fila['pos'], type=fila['type'])
            PVET_ids[elem.id] = elem
        if usar_cache:
            with open(nom_fich_pvet_ids, 'w') as f:
                for o in PVET_ids.values():
                    print(json.dumps(asdict(o)), file=f)
    return PVET_ids

###################################################################

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

def escoger_otro_dispositivo(PVET_ids, disp_fallo: PVET_id):
    ''' Escoge un dispositivo diferente al indicado, pero del mismo tipo.'''
    buscar = True
    while buscar:
        id_otro = random.randint(1, len(PVET_ids)-1)
        buscar = id_otro != disp_fallo.id and id_otro in PVET_ids and PVET_ids[id_otro].type == disp_fallo.type
    return PVET_ids[id_otro]

###################################################################

num_id_fallo = num_id_caso = 1

def obtener_datos_casos(cliente_sql:ClientePostgres, cliente_influx:ClienteInflux, nom_planta:str, tipo_disp:str, diag_interés=None, margen_temporal_h:int=0) -> pd.DataFrame:
    ''' Devuelve un DataFrame con los datos de cada fallo y de varios dispositivos sanos.
    '''
    global num_id_fallo, num_id_caso
    CONFIG = config_global.ConfigGlobal()
    nom_tabla_fallos = 'DDA_DIA'
    tabla_disp = f'vop_{tipo_disp}'.lower()
    filtro_interés = f"Type = '{tipo_disp}' AND ope_ck = 1"
    if diag_interés is not None:
        if isinstance(diag_interés, int):
            filtro_interés += f" AND diag = {diag_interés}"
        elif isinstance(diag_interés, list):
            filtro_interés += f" AND diag IN ({','.join(map(str, diag_interés))})"
    consulta = f"SELECT COUNT(*) FROM {nom_tabla_fallos} WHERE {filtro_interés}"
    #cliente_sql.obtener_cursor(consulta, as_dict=False)
    cliente_sql.obtener_cursor(consulta)
    num_fallos = int(cliente_sql.leer_registro()['count'])
    # Obtiene todos los fallos validados de ese tipo
    consulta_sql = f""" SELECT * FROM {nom_tabla_fallos}
                        JOIN diagnosis ON {nom_tabla_fallos}.diag=diagnosis.code
                        WHERE {filtro_interés}
                        ORDER BY diag,Duration DESC"""
    if depurar:
        print(f'CONSULTA SQL: {consulta_sql}')
    cursor = cliente_sql.obtener_cursor(consulta_sql)

    # Para cada fallo, guardará los datos de ese fallo y de varios dispositivos sanos
    # A cada fallo se le asigna un id de grupo de fallo, y a cada dispositivo un id de caso.
    # Por tanto, todos los dispositivos de un mismo fallo tendrán el mismo id de grupo de fallo,
    # y cada dispositivo tendrá un id de caso único.
    df_casos = None
    num_fallo = 1
    for fila in cursor:
        print(f'{num_fallo}/{num_fallos} FALLOS', flush=True)
        ini_time = fila['ini_time']
        end_time = fila['end_time']
        id_dispositivo_fallo = fila['id']
        diag_fallo = fila['diag']
        diag_fallo_txt = fila['esp']
        duración_fallo = fila['duration']
        # fallo_continuo es True si el fallo existe durante todo el período ini-end_time
        fallo_continuo = (duración_fallo - 15) == ((end_time - ini_time).total_seconds() / 60)

        # Carga los datos de todos los dispositivos de ese día (00:00:00 a 23:59:59)
        # Añade un margen temporal de N horas antes
        ini_día = datetime(ini_time.year, ini_time.month, ini_time.day)
        fin_día = datetime(ini_time.year, ini_time.month, ini_time.day) + timedelta(days=1, seconds=-1)
        df_día = cargar_df(cliente_influx, nom_planta, tabla_disp, ini_día - timedelta(hours=margen_temporal_h), fin_día)
        df_meteo = cargar_meteo(cliente_influx, nom_planta, None, ini_día - timedelta(hours=margen_temporal_h), fin_día)
        df_día = df_día.join(df_meteo, how='outer', rsuffix='_meteo')

        # Ahora va a guardar los datos del dispositivo que ha fallado y de varios sanos
        disp_fallo = PVET_ids[id_dispositivo_fallo]
        dispositivos_guardar = [ disp_fallo ]
        # Escoge varios dispositivos sanos al azar. Intenta que sean una cierta cantidad mínima, pero a veces hay menos.
        dispositivos_sanos = obtener_dispositivos_sanos(cliente_sql, tipo_disp, fecha_fallo=ini_día.strftime('%Y-%m-%d'))
        num_disp_sanos = min(CONFIG.max_disp_sanos_por_fallo, len(dispositivos_sanos))
        ids_dispositivos_sanos = random.sample(list(dispositivos_sanos.keys()), num_disp_sanos)
        for i in ids_dispositivos_sanos:
            dispositivos_guardar.append(dispositivos_sanos[i])

        # Para cada dispositivo, selecciona los datos del día y los guarda en el DataFrame
        # Si el dispositivo no tiene datos suficientes, lo ignora.
        # Si el dispositivo que ha fallado no tiene datos suficientes, no guarda nada.
        # Si es el dispositivo que ha fallado, guarda los datos del fallo.
        # Si es un dispositivo sano, guarda los datos como si no hubiera fallado.
        num_dispositivos_guardados = 0
        for dispositivo in dispositivos_guardar:
            datos_guardar = seleccionar_dispositivo(df_día, dispositivo).copy()
            if len(datos_guardar) != 96 + 2 * margen_temporal_h * 4: # Ñapa: 96 datos por día, más los de margen temporal
                if dispositivo.id == disp_fallo.id:
                    break # Si el dispositivo que ha fallado no tiene datos suficientes, no guarda nada.
                else:
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
            if df_casos is None:
                df_casos = datos_guardar
            else:
                df_casos = pd.concat([df_casos, datos_guardar])
            num_id_caso += 1
            num_dispositivos_guardados += 1
        if num_dispositivos_guardados > 0 and False: # JMR: pendiente revisar, quizá no vale para mucho.
            # Si se han guardado datos de algún dispositivo,
            # guarda el promedio para cada instante del día como un caso más
            datos_promedio = df_día.groupby('_time').mean()
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
            df_casos = pd.concat([df_casos, datos_promedio])
            num_id_caso += 1
            num_id_fallo += 1 # Solo incrementa si se han guardado datos de algún dispositivo
        num_fallo += 1
    return df_casos.sort_index() if df_casos is not None else None

###################################################################

def obtener_dispositivos_sanos(cliente_sql: ClientePostgres, disp_fallo: str, fecha_fallo:str=None) -> dict[int,PVET_id]:
    """
    Obtiene los dispositivos sanos de un tipo de fallo específico.
    """
    # Primero obtiene todos los dispositivos de ese tipo
    consulta = f"SELECT * FROM PVET_ids WHERE Type = '{disp_fallo}'"
    if depurar:
        print(f'CONSULTA SQL: {consulta}')
    cursor = cliente_sql.obtener_cursor(consulta)
    dispositivos_sanos = {}
    for fila in cursor:
        dispositivo = PVET_id(id=fila['id'], CT=fila['ct'], IN=fila['in'], TR=fila['tr'], SB=fila['sb'], ST=fila['st'], POS=fila['pos'], type=fila['type'])
        dispositivos_sanos[dispositivo.id] = dispositivo

    # Luego elimina los que tengan fallos registrados, opcionalmente filtrando por fecha
    consulta = f"SELECT DISTINCT ID FROM DDA_DIA WHERE Type = '{disp_fallo}'"
    if fecha_fallo:
        consulta += f" AND date = '{fecha_fallo}'"
    if depurar:
        print(f'CONSULTA SQL: {consulta}')
    cursor = cliente_sql.obtener_cursor(consulta)
    for fila in cursor:
        del dispositivos_sanos[fila['id']]

    return dispositivos_sanos

###################################################################

if __name__ == "__main__":
    pass
