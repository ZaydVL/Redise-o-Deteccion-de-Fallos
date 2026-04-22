import psycopg
import sys
import os
import json
from datetime import datetime,timedelta

###################################################################

class ClientePostgres:

    def __init__(self, fich_params:str=None, basedatos:str=''):
        self.params = self.__obtener_params_entorno()
        if fich_params is not None:
            params2 = self.__obtener_params_fichero(fich_params)
            self.params.update(params2)
        self.__validar_params()
        self.basedatos = basedatos

###################################################################

    def __enter__(self):
        self.conectar(basedatos=self.basedatos)
        return self

###################################################################

    def __exit__(self, ex_type, ex_value, ex_traceback):
        self.desconectar()
        return False # Si devuelve True, se silenciarán todas las excepciones.

###################################################################

    def conectar(self, basedatos='') -> psycopg.Connection:
        servidor = self.params['PGSQL_SERVER']
        usuario = self.params['PGSQL_USER']
        contraseña = self.params['PGSQL_PASSWORD']
        puerto = 5432 if 'PGSQL_PORT' not in self.params else int(self.params['PGSQL_PORT'])
        self.conexión = psycopg.connect(
            f'postgresql://{usuario}:{contraseña}@{servidor}:{puerto}/{basedatos}',
            row_factory=psycopg.rows.dict_row,
            sslmode="require",  # El verify-full me da problemas, a pesar de que el certificado es correcto.
            #sslmode="verify-full",
            #sslrootcert='system'
        )
        return self.conexión

###################################################################

    def desconectar(self):
        self.conexión.close()

###################################################################

    def __obtener_params_entorno(self):
        params = {}
        for var_entorno in ['PGSQL_SERVER', 'PGSQL_USER', 'PGSQL_PASSWORD', 'PGSQL_PORT']:
            params[var_entorno]   = os.environ[var_entorno]   if var_entorno   in os.environ else None
        return params

###################################################################

    def __obtener_params_fichero(self, nom_fich_params):
        with open(nom_fich_params, 'r') as fich_params:
            params = json.load(fich_params)
        return params

###################################################################

    def __validar_params(self):
        for param in ['PGSQL_SERVER', 'PGSQL_USER', 'PGSQL_PASSWORD']:
            if param not in self.params or self.params[param] is None:
                raise RuntimeError(f'ERROR: es necesario definir {param}')

###################################################################

    def abrir_tabla(self, nom_tabla:str):
        self.cursor = self.conexión.cursor()
        self.cursor.execute(f"SELECT * FROM {nom_tabla}")

###################################################################

    def cerrar_tabla(self):
        self.cursor.close()

###################################################################

    def leer_registro(self):
        return self.cursor.fetchone()

###################################################################

    def obtener_cursor(self, consulta_sql: str) -> psycopg.Cursor:
        self.cursor = self.conexión.cursor()
        self.cursor.execute(consulta_sql)
        return self.cursor

###################################################################

    def obtener_tablas(self) -> list:
        with self.conexión.cursor() as cursor:
            cursor.execute("SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE='BASE TABLE' AND table_schema='public'")
            return [fila['table_name'] for fila in cursor]

###################################################################

    def obtener_esquema_tabla(self, nom_tabla:str):
        with self.conexión.cursor() as cursor:
            cursor.execute("SELECT * FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = %s", (nom_tabla,))
            columnas = []
            for fila in cursor:
                nom_columna = fila['column_name']
                if fila['data_type'] in ['real', 'bigint']:
                    tipo = float
                elif fila['data_type'] in ['integer', 'smallint', 'bigint']:
                    tipo = int
                elif fila['data_type'].startswith('time') or fila['data_type'] in ['date']:
                    tipo = datetime
                elif fila['data_type'].endswith('char') or fila['data_type'].startswith('char'):
                    tipo = str
                elif fila['data_type'] == 'bit':
                    tipo = None
                else:
                    raise RuntimeError(f"Tipo {nom_columna}:{fila['data_type']} desconocido.")
                if tipo is not None:
                    columnas.append((nom_columna, tipo))
        return columnas

###################################################################

def main1(args):
    with ClientePostgres('pvet-sp10', 'params-pgsql.json') as cliente_sql:
        print('LISTA DE TABLAS')
        tablas = cliente_sql.obtener_tablas()
        for t in tablas:
            print(t)
        print()
        print()
        print(f'Esquema de {tablas[0]}')
        esquema = cliente_sql.obtener_esquema_tabla(tablas[0])
        for c in esquema:
            print(c)
        print()
        print()
        consulta_sql = f"SELECT * FROM DDA_DIA WHERE Type = 'SB' AND ope_ck <> 0"
        cursor = cliente_sql.obtener_cursor(consulta_sql)
        for fila in cursor:
            ini_time = fila['ini_time']
            end_time = fila['end_time']
            id_dispositivo = fila['ID']
            print(f'ID: {id_dispositivo}, Ini: {ini_time}, Fin: {end_time}')

###################################################################

if __name__ == "__main__":
    main1(sys.argv[1:])
