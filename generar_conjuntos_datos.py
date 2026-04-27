#%%
import sys
import os
import config_global
CONFIG = config_global.ConfigGlobal() 

import numpy as np
import pandas as pd
from cliente_influx import ClienteInflux
from cliente_pgsql import ClientePostgres
from preprocesado import cargar_PVET_ids, obtener_datos_casos

###################################################################

def uso():
    print(f'Uso: {sys.argv[0]} fich_config')
    print(f'  fich_config : Fichero de configuración')
    print(f'Ej: {sys.argv[0]} config/config_gen1.py')

###################################################################

def main1(args):
    if len(args) != 1:
        uso()
        sys.exit(1)
    ''' Genera un conjunto de casos de fallo para las plantas indicadas en la configuración.
    En cada caso aparece el dispositivo que ha fallado y varios más sanos.
    Los datos se guardan en un directorio configurable.'''
    config_global.ConfigGlobal(args[0])
    fich_salida = CONFIG.fich_salida
    dir_salida = os.path.dirname(fich_salida)
    if dir_salida and not os.path.exists(dir_salida):
        os.makedirs(dir_salida)

    diag_interes_conf = CONFIG.diags if hasattr(CONFIG, 'diags') else None
    guardar_por = 'tipo_disp' if '{tipo_disp}' in fich_salida else 'planta' if '{planta}' in fich_salida else 'total'

    lista_stats = []       # acumula stats de todas las combinaciones → CSV al final
    lista_fallos_total = [] # para el modo 'total'

    with ClientePostgres('params-pgsql.json') as cliente_postgres:
        with ClienteInflux('params-influx.json') as cliente_influx:
            for planta in CONFIG.plantas:
                lista_fallos_planta = []
                nom_bd_pgsql = f'pvet_{planta}'
                nom_bu_influx = f'pvet-{planta}'
                cliente_postgres.conectar(nom_bd_pgsql)
                ruta_salida = os.path.join(dir_salida, f"{planta}-PVET_ids.csv") if dir_salida else f"{planta}-PVET_ids.csv"
                cargar_PVET_ids(cliente_postgres, planta, usar_cache=False, ruta_salida=ruta_salida)

                for tipo_disp in CONFIG.tipos_disp:
                    diag_interes = diag_interes_conf.get(tipo_disp, None) if isinstance(diag_interes_conf, dict) else diag_interes_conf
                    diag_interes_txt = 'Todos' if diag_interes is None else str(diag_interes)
                    print(f'\n\nBuscando fallos del tipo {tipo_disp} en la planta {planta}, diags={diag_interes_txt} ...')

                    df_fallos, stats = obtener_datos_casos(
                        cliente_postgres, cliente_influx, nom_bu_influx, tipo_disp,
                        diag_interes=diag_interes, margen_temporal_h=CONFIG.margen_temporal_h
                    )

                    # Acumula stats con contexto de planta y tipo
                    lista_stats.append({
                        'planta': planta,
                        'tipo_disp': tipo_disp,
                        'diag_interes': diag_interes_txt,
                        **stats
                    })

                    print(f'''
                        Resumen {planta} - {tipo_disp} - {diag_interes_txt}:
                        Fallos totales evaluados:              {stats['fallos_totales']}
                        Fallos guardados:                      {stats['fallos_guardados']}
                        Dispositivos fallidos sin datos:       {stats['dispositivos_fallidos_sin_datos']}
                        Dispositivos fallidos rechazados:      {stats['dispositivos_fallidos_rechazados_ratio']}
                        Dispositivos sanos rechazados:         {stats['dispositivos_sanos_rechazados_ratio']}
                    ''')

                    if df_fallos is None:
                        print(f'No se han encontrado fallos del tipo {tipo_disp} en la planta {planta}.')
                        continue

                    print(f'Casos obtenidos: {df_fallos["id_caso"].nunique()}, fallos: {df_fallos["id_fallo"].nunique()}')

                    if guardar_por == 'tipo_disp':
                        fich = fich_salida.replace('{tipo_disp}', tipo_disp).replace('{planta}', planta)
                        print(f'Guardando datos en {fich}')
                        df_fallos.to_csv(fich, date_format='%Y-%m-%d %H:%M:%S')
                    elif guardar_por == 'planta':
                        lista_fallos_planta.append(df_fallos)
                    else:
                        lista_fallos_total.append(df_fallos)

                if guardar_por == 'planta' and lista_fallos_planta:
                    df_planta = pd.concat(lista_fallos_planta)
                    fich = fich_salida.replace('{planta}', planta)
                    print(f'Guardando datos en {fich}')
                    df_planta.to_csv(fich, date_format='%Y-%m-%d %H:%M:%S')

    if guardar_por == 'total' and lista_fallos_total:
        df_total = pd.concat(lista_fallos_total)
        print(f'Guardando datos en {fich_salida}')
        df_total.to_csv(fich_salida, date_format='%Y-%m-%d %H:%M:%S')
        print(f'\nNúmero TOTAL de casos: {df_total["id_caso"].nunique()}, fallos: {df_total["id_fallo"].nunique()}')

    # Stats → CSV en disco / y ya no en memoria para poder recorrer lo descargado
    if lista_stats:
        df_stats = pd.DataFrame(lista_stats)
        fich_stats = os.path.join(dir_salida, 'fallos_stats.csv') if dir_salida else 'fallos_stats.csv'
        df_stats.to_csv(fich_stats, index=False)
        print(f'\nEstadísticas guardadas en {fich_stats}')

    print('\n' + '='*60)
    print('RESUMEN FINAL')
    for s in lista_stats:
        print(f"  {s['planta']}/{s['tipo_disp']}: {s['fallos_guardados']}/{s['fallos_totales']} fallos guardados")

###################################################################
#%%

if __name__ == "__main__":
    if len(sys.argv) == 1:
        main1(["config/config_gen1.py"])
    else:
        main1(sys.argv[1:])