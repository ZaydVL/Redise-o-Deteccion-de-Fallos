import os
os.chdir(r'C:\Users\minio\Doctorado UPM\Proyecto PVOP\Rediseño')
from cliente_influx import ClienteInflux

plantas = ['cl02', 'cl03', 'br02', 'br03', 'sp08', 'sp09', 'sp10', 'mx05', 'mx06', 'rd02']

"""
    Ejecutar este script imprimirá aquellas variables metereológicas existentes en cada
    planta considerada en la lista de arriba "plantas". Si la planta no las posee
    pues la lista de vop_rm (Refenrece Module) y vop_ms (Metereological sensor) 
    aparecerán vacías.
"""

with ClienteInflux('params-influx.json') as influx:
    query_api = influx.cliente_influx.query_api()
    
    for planta in plantas:
        for medida in ['vop_ms', 'vop_rm']:
            consulta = f'''
                import "influxdata/influxdb/schema"
                schema.measurementFieldKeys(
                    bucket: "pvet-{planta}",
                    measurement: "{medida}",
                    start: -365d
                )
            '''
            tablas = query_api.query(consulta)
            campos = [row.values['_value'] for tabla in tablas for row in tabla.records]
            print(f"pvet-{planta} / {medida}: {campos}")
