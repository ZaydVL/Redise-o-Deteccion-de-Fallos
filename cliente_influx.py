import pandas as pd
import influxdb_client
import sys
import os
import json

###################################################################

class ClienteInflux:
    
    def __init__(self, fich_params=None):
        self.params = self.__obtener_params_entorno()
        if fich_params is not None:
            params2 = self.__obtener_params_fichero(fich_params)
            self.params.update(params2)
        self.__validar_params()

###################################################################

    def __enter__(self):
        self.conectar()
        return self

###################################################################

    def __exit__(self, ex_type, ex_value, ex_traceback):
        self.desconectar()
        return False

###################################################################

    def conectar(self) -> influxdb_client.InfluxDBClient:
        self.cliente_influx = influxdb_client.InfluxDBClient(
            url=f"{self.params['INFLUX_HOST']}",
            token=self.params["INFLUX_TOKEN"],
            org=self.params["INFLUX_ORG"],
            timeout=60000,
            ssl=True,
            verify_ssl=True,
            ssl_ca_cert='letsencrypt.pem',
        )
        return self.cliente_influx

###################################################################

    def desconectar(self):
        self.cliente_influx.close()

###################################################################

    def __obtener_params_entorno(self):
        params = {}
        for var_entorno in ['INFLUX_HOST', 'INFLUX_ORG', 'INFLUX_TOKEN']:
            params[var_entorno]   = os.environ[var_entorno]   if var_entorno   in os.environ else None
        return params

###################################################################

    def __obtener_params_fichero(self, nom_fich_params):
        with open(nom_fich_params, 'r') as fich_params:
            params = json.load(fich_params)
        return params

###################################################################

    def __validar_params(self):
        for param in ['INFLUX_HOST', 'INFLUX_ORG', 'INFLUX_TOKEN']:
            if param not in self.params or self.params[param] is None:
                raise RuntimeError(f'ERROR: es necesario definir {param}')

###################################################################

    def cargar_df(self, consulta=None, nom_bucket=None, ini_periodo=None, fin_periodo='now()', nom_medida=None) -> pd.DataFrame:
        query_api = self.cliente_influx.query_api()
        if consulta is None:
            consulta = f'''
                from(bucket:"{nom_bucket}")
                |> range(start:{ini_periodo}, stop:{fin_periodo})
                |> filter(fn: (r) => r["_measurement"] == "{nom_medida}")
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
                |> group()
            '''
        df = query_api.query_data_frame(consulta)
        df = df.drop(['result', 'table', '_start', '_stop', '_measurement'], axis=1).set_index('_time')
        return df

###################################################################

def mostrar_df(df:pd.DataFrame):
    print(df.info())
    print('=' * 60)
    print(df.describe())
    print('=' * 60)
    print(df)
    print('=' * 60)
    print('\n' * 3)

###################################################################

# Documentación Flux:
# 
# https://docs.influxdata.com/flux/v0/release-notes/

def main1(args):
    nom_bu_influx = 'gsf-f0001'
    cliente_influx = ClienteInflux('params-influx.json')
    cliente_influx.conectar()
    
    # Ejemplo: cargar todos los datos de las últimas 24h.
    df = cliente_influx.cargar_df(nom_bucket=nom_bu_influx, ini_periodo='-1d', nom_medida='m')
    mostrar_df(df)

    # Ejemplo: cargar datos del último mes, experimento
    # y variables concretas.
    consulta = f'''
        from(bucket:"{nom_bu_influx}")
        |> range(start: -30d)
        |> filter(fn: (r) => r["_measurement"] == "m")
        |> filter(fn: (r) => r.experiment == "1")
        |> filter(fn: (r) => r._field =~ /^(B_ws1|G_ws1)$/)
        |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        |> group()
    '''
    df = cliente_influx.cargar_df(consulta=consulta)
    mostrar_df(df)

    # Ejemplo: cargar datos de una cierta fecha, experimento
    # y variables cuyo nombre coincide con una expresión regular.
    consulta = f'''
        from(bucket:"{nom_bu_influx}")
        |> range(start: 2025-05-20T00:00:00Z, stop: 2025-05-20T23:59:59Z)
        |> filter(fn: (r) => r["_measurement"] == "m")
        |> filter(fn: (r) => r.experiment == "1")
        |> filter(fn: (r) => r._field =~ /ws1$/)
        |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        |> group()
    '''
    df = cliente_influx.cargar_df(consulta=consulta)
    mostrar_df(df)

###################################################################

if __name__ == "__main__":
    main1(sys.argv[1:])
