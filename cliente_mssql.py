import pymssql
import sys
import os
import json
from datetime import datetime,timedelta

###################################################################

class LectorSqlServer:
    
    def __init__(self, basedatos:str, fich_params:str=None):
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

    def conectar(self, basedatos='') -> pymssql.Connection:
        servidor = self.params['MSSQL_SERVER']
        usuario = self.params['MSSQL_USER']
        contraseña = self.params['MSSQL_PASSWORD']
        puerto = 1433 if 'MSSQL_PORT' not in self.params else int(self.params['MSSQL_PORT'])
        self.conexión = pymssql.connect(server=servidor, user=usuario, password=contraseña, database=basedatos, port=puerto)
        return self.conexión

###################################################################

    def desconectar(self):
        self.conexión.close()

###################################################################

    def __obtener_params_entorno(self):
        params = {}
        for var_entorno in ['MSSQL_SERVER', 'MSSQL_USER', 'MSSQL_PASSWORD', 'MSSQL_PORT']:
            params[var_entorno]   = os.environ[var_entorno]   if var_entorno   in os.environ else None
        return params

###################################################################

    def __obtener_params_fichero(self, nom_fich_params):
        with open(nom_fich_params, 'r') as fich_params:
            params = json.load(fich_params)
        return params

###################################################################

    def __validar_params(self):
        for param in ['MSSQL_SERVER', 'MSSQL_USER', 'MSSQL_PASSWORD']:
            if param not in self.params or self.params[param] is None:
                raise RuntimeError(f'ERROR: es necesario definir {param}')

###################################################################

    def abrir_tabla(self, nom_tabla:str):
        self.cursor = self.conexión.cursor(as_dict=True)
        self.cursor.execute(f"SELECT * FROM {nom_tabla}")
        
###################################################################

    def cerrar_tabla(self):
        self.cursor.close()
        
###################################################################

    def leer_registro(self):
        return self.cursor.fetchone()

###################################################################

    def obtener_cursor(self, consulta_sql: str, as_dict=True) -> pymssql.Cursor:
        self.cursor = self.conexión.cursor(as_dict=as_dict)
        self.cursor.execute(consulta_sql)
        return self.cursor
    
###################################################################

    def obtener_tablas(self) -> list:
        with self.conexión.cursor(as_dict=True) as cursor:
            cursor.execute("SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE='BASE TABLE'")
            return [fila['TABLE_NAME'] for fila in cursor]
    
###################################################################

    def obtener_esquema_tabla(self, nom_tabla:str):
        with self.conexión.cursor(as_dict=True) as cursor:
            cursor.execute("SELECT * FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = %s", nom_tabla)
            columnas = []
            for fila in cursor:
                nom_columna = fila['COLUMN_NAME']
                if fila['DATA_TYPE'] in ['real', 'float', 'bigint']:
                    #tipo = f"{fila['DATA_TYPE']}({fila['NUMERIC_PRECISION_RADIX']})"
                    tipo = float
                elif fila['DATA_TYPE'] in ['tinyint', 'int', 'smallint', 'bigint']:
                    tipo = int
                elif fila['DATA_TYPE'].startswith('date'):
                    #tipo = f"{fila['DATA_TYPE']}({fila['DATETIME_PRECISION']})"
                    tipo = datetime
                elif fila['DATA_TYPE'].endswith('char'):
                    #tipo = f"{fila['DATA_TYPE']}({fila['CHARACTER_MAXIMUM_LENGTH']})"
                    tipo = str
                elif fila['DATA_TYPE'] == 'bit':
                    tipo = None
                else:
                    raise f"Tipo {nom_columna}:{fila['DATA_TYPE']} desconocido."
                if tipo is not None:
                    columnas.append((nom_columna, tipo))
        return columnas

###################################################################

def main1(args):
    with LectorSqlServer('eng-pvet-br02') as cliente_sql:
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
