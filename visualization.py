import os
from datetime import datetime,timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import pandas as pd

depurar = True if "DEPURAR" in os.environ and os.environ["DEPURAR"].lower() == "true" else False

###################################################################

def corregir_fecha(fecha):
    if isinstance(fecha, str):
        if 'T' not in fecha:
            fecha = fecha.replace(' ', 'T')
    elif isinstance(fecha, datetime):
        fecha = fecha.strftime('%Y-%m-%dT%H:%M:%SZ')
    return fecha

###################################################################

campos_gráficas_por_tipo = {
#    'SB' : [ 'vdc', 'idc', 'pdc' ],
#    'IN' : [ 'vdc', 'idc', 'pdc' ],
    'SB' : [ 'idc' ],
    'IN' : [ 'idc' ],
    'TR' : [ 'pos' ],
    'ST' : [ 'idc' ]
}

#campos_gráficas_por_diag = {
#    201 : [ 'vdc', 'idc', 'pdc' ],             # String abierto (ST)
#    202 : [ 'vdc', 'idc', 'pdc' ],             # Diodo defectuoso (ST)
#    221 : [ 'vdc', 'idc', 'pdc' ],             # Stringbox abierto
#    222 : [ 'vdc', 'idc', 'pdc' ],             # String abierto (SB)
#    241 : [ 'vdc', 'idc', 'pdc' ],             # Parada de inversor
#    242 : [ 'vdc', 'idc', 'pdc' ],             # Fallo MPPT
#    245 : [ 'temp_pot', 'temp_cab', 'pdc' ],   # MPPT Temperatura
#    246 : [ 'temp_pot', 'temp_cab', 'pdc'  ],  # MPPT VDC
#    260 : [ 'pos' ],                           # Posición bandera
#    265 : [ 'pos' ],                           # Posición máxima no alcanzada
#    343 : [ 'temp_pot', 'temp_cab', 'pdc' ],   # Temperatura anómala cabina
#    345 : [ 'temp_pot', 'temp_cab', 'pdc' ],   # Temperatura anómala potencia
#    0 :   [ 'pdc', 'pos' ]                     # Cualquier otro
#}


campos_gráficas_por_diag = {
    201 : [ 'vdc', 'idc', 'pdc' ],             # String abierto (ST)
    202 : [ 'vdc', 'idc', 'pdc' ],             # Diodo defectuoso (ST)
    221 : [ 'vdc', 'idc', 'pdc' ],             # Stringbox abierto
    222 : [ 'vdc', 'idc', 'pdc' ],             # String abierto (SB)
    224 : [ 'vdc', 'idc', 'pdc' ],             # String defectuoso
    241 : [ 'vdc', 'idc', 'pdc' ],             # Parada de inversor
    242 : [ 'vdc', 'idc', 'pdc' ],             # Fallo MPPT
    245 : [ 'temp_pot', 'temp_cab', 'pdc' ],   # MPPT Temperatura
    246 : [ 'temp_pot', 'temp_cab', 'pdc'  ],  # MPPT VDC
    260 : [ 'pos', 'pos_obj', 'pos_the' ],     # Posición bandera
    261 : [ 'pos', 'pos_obj', 'pos_the' ],     # Parada de seguimiento
    262 : [ 'pos', 'pos_obj', 'pos_the' ],     # Desalineamiento
    263 : [ 'pos', 'pos_obj', 'pos_the' ],     # Posición objetivo errónea
    264 : [ 'wd', 'ws' ],                      # Alarma de viento
    265 : [ 'pos' ],                           # Posición máxima no alcanzada
    341:  [ 'temp_pot', 'temp_cab', 'pdc' ],   # Temperatura anómala
    342:  [ 'temp_pot', 'temp_cab', 'pdc' ],   # Desvío temperatura fase
    343 : [ 'temp_pot', 'temp_cab', 'pdc' ],   # Temperatura anómala cabina
    344 : [ 'temp_pot', 'temp_cab', 'pdc' ],   # Temperatura anómala filtro LC
    345 : [ 'temp_pot', 'temp_cab', 'pdc' ],   # Temperatura anómala potencia
    0 :   [ 'pdc', 'pos' ]                     # Cualquier otro (280: restricciones de red, 281: Saturación de red)
}

def dibujar_fallo(df:pd.DataFrame, gráfica:plt.Axes, tipo_comparación:str=None, comentario=None, nom_fich_guardar_df=None):
    '''Dibuja un fallo en la gráfica proporcionada y también otro caso sano para comparación.
    Si se pasa "PROMEDIO" como tipo_comparación, se dibuja el caso promedio de los dispositivos sanos.
    '''

    # Obtiene diversas informaciones generales sobre el fallo
    id_fallo = df['id_fallo'].iloc[0]
    datos_disp_fallo = df[df['fallo'] == True]
    disp_fallo = datos_disp_fallo['pvet_disp'].iloc[0]
    diag_fallo = datos_disp_fallo['diag'].iloc[0]
    diag_fallo_txt = datos_disp_fallo['diag_txt'].iloc[0]
    tipo_disp = datos_disp_fallo['tipo_disp'].iloc[0]
    datos_disp_refer = None
    if tipo_comparación is not None:
        datos_disp_refer = df[df['pvet_disp'].str.lower() == tipo_comparación.lower()]
    else:
        # Si no se especifica tipo_comparación, coge el primer dispositivo sano.
        pvet_ids_refer = df[(df['fallo'] == False) & (df['pvet_id'] > 0)]['pvet_id'].unique()
        if len(pvet_ids_refer) > 0:
            datos_disp_refer = df[df['pvet_id'] == pvet_ids_refer[0]]
    if datos_disp_refer is not None and len(datos_disp_refer) > 0:
        disp_refer = datos_disp_refer['pvet_disp'].iloc[0]
    else:
        # Ocasionalmente no hay datos de dispositivos sanos
        disp_refer = 'NO'
    ini_time = datos_disp_fallo['ini_fallo'].iloc[0]
    end_time = datos_disp_fallo['fin_fallo'].iloc[0]
    if end_time == ini_time:
        end_time = ini_time + timedelta(minutes=15) # Para que al menos sea un intervalo

    # Obtiene los campos que se dibujarán y descarta los que no estén en los datos
    campos_interés = campos_gráficas_por_diag[diag_fallo] if diag_fallo in campos_gráficas_por_diag else campos_gráficas_por_diag[0]
    campos_interés = [c for c in campos_interés if c in datos_disp_fallo.columns]

    # Prepara la gráfica
    formato_x = mdates.DateFormatter('%H:%M')
    gráfica.xaxis.set_major_formatter(formato_x)
    gráfica.tick_params(axis='both', labelsize=8)
    gráfica.tick_params(axis='x', rotation=45)

    # Si hay campos cuyo valor máximo difiere bastante con respecto a otros,
    # usa dos gráficas: una para los valores pequeños y otra para los grandes.
    gráfica1 = gráfica
    gráfica2 = None
    gráfica_campo = {} # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/two_scales.html
    for v in campos_interés:
        if len(gráfica_campo) == 0:
            máx_ref = max(1,abs(datos_disp_fallo[v].max()))
            g = gráfica1
        else:
            máx_v = max(1, abs(datos_disp_fallo[v].max()))
            if máx_ref / máx_v > 5 or máx_v / máx_ref > 5:
                if gráfica2 is None:
                    gráfica2 = gráfica1.twinx()
                g = gráfica2
        gráfica_campo[v] = g

    colores = list(mcolors.TABLEAU_COLORS.keys()) # https://matplotlib.org/stable/gallery/color/named_colors.html
    for v in campos_interés:
        g = gráfica_campo[v]
        color = colores.pop(0) # Coge un color para cada variable (supone que habrá suficientes para ir sacando)
        # Dibuja la variable del dispositivo en fallo con línea punteada.
        g.plot(datos_disp_fallo.index, datos_disp_fallo[v], '--', label=f'{v} F',color=color)
        # Selecciona las filas cuyo índice está entre ini_time y end_time (aunque no existan exactamente esos timestamps)
        # Las dibuja encima con puntos gordos, para que se vean claramente.
        máscara_t = (datos_disp_fallo.index >= ini_time) & (datos_disp_fallo.index <= end_time)
        g.plot(datos_disp_fallo.index[máscara_t], datos_disp_fallo.loc[máscara_t, v], 'o--', color=color)
        # Si se compara con otro dispositivo, dibuja su variable con línea continua
        if datos_disp_refer is not None and not datos_disp_refer.empty:
            g.plot(datos_disp_refer.index, datos_disp_refer[v], '-', label=v, color=color)

    # Termina poniendo título, leyenda, etc.
    título = f'Fallo {id_fallo}, {diag_fallo_txt}, {ini_time.strftime("%Y-%m-%d")}\nF={disp_fallo}\nC={disp_refer}'
    if comentario is not None:
        título = f'{título}\n{comentario}'
    gráfica1.set_title(título, fontsize=8)
    gráfica1.set_visible(True)
    gráfica1.legend()
    if gráfica2 is not None:
        gráfica2.legend()

###################################################################

def dibujar_fallos(df_fallos: pd.DataFrame, tipo_comparación:str=None, dir_ficheros='png'):
    if not os.path.exists(dir_ficheros):
        os.makedirs(dir_ficheros)
    num_filas_gráficas = 2
    num_cols_gráficas = 2
    num_gráficas = num_filas_gráficas * num_cols_gráficas
    tam_fig_X = 7 + 3.5 * (num_cols_gráficas - 1)
    tam_fig_Y = 5 + 2.5 * (num_filas_gráficas - 1)
    num_gráfica = 0
    tipo_disp = df_fallos['tipo_disp'].iloc[0]
    # Ordena los id_fallo por diagnóstico
    id_fallo_ordenado = df_fallos[df_fallos['diag'] != 0].groupby(['id_fallo', 'diag']).size().reset_index().sort_values('diag')['id_fallo'].values
    for id_fallo in id_fallo_ordenado:
        if num_gráfica % num_gráficas == 0:
            figura, gráficas = plt.subplots(nrows=num_filas_gráficas, ncols=num_cols_gráficas, squeeze=False, figsize = (tam_fig_X, tam_fig_Y), subplot_kw={'visible':False})
            plt.subplots_adjust(left=0.15, wspace=0.3, hspace=0.4)
        df_fallo = df_fallos[df_fallos["id_fallo"] == id_fallo]
        if df_fallo[df_fallo['fallo'] == True].shape[0] > 0:
            dibujar_fallo(df_fallo, gráficas[(num_gráfica // num_cols_gráficas) % num_filas_gráficas, num_gráfica % num_cols_gráficas], tipo_comparación=tipo_comparación, nom_fich_guardar_df=f'csv/caso-fallo-{tipo_disp}-{num_gráfica:03}.csv')
            num_gráfica += 1
            if num_gráfica % num_gráficas == 0:
    #            plt.show(block=False)
                plt.savefig(f'{dir_ficheros}/fallos-{tipo_disp}-{num_gráfica // num_gráficas}.png', dpi=300)
                plt.close()
    if num_gráfica % num_gráficas != 0:
#        plt.show(block=False)
        plt.savefig(f'{dir_ficheros}/fallos-{tipo_disp}-{num_gráfica // num_gráficas + 1}.png', dpi=300)
        plt.close()


def dibujar_historial(historia, nombre_modelo, patron_ficheros=None):
    """Dibuja el historial de entrenamiento del modelo."""
    metricas = historia.history.keys()
    metricas_a_graficar = []
    for metrica in metricas:
        if not metrica.startswith('val_'):
            val_metrica = f'val_{metrica}'
            if val_metrica in metricas:
                metricas_a_graficar.append((metrica, val_metrica))
    print("Métricas a graficar:", metricas_a_graficar)
    n_graficos = len(metricas_a_graficar)
    plt.figure(figsize=(6 * n_graficos, 5))
    for idx, (metrica_ent, metrica_val) in enumerate(metricas_a_graficar, start=1):
        plt.subplot(1, n_graficos, idx)
        plt.plot(historia.history[metrica_ent], label=f'{metrica_ent} (entrenamiento)')
        plt.plot(historia.history[metrica_val], label=f'{metrica_val} (validación)')
        plt.title(f'{metrica_ent.capitalize()} del modelo')
        plt.xlabel('Épocas')
        plt.ylabel(metrica_ent.capitalize())
        plt.legend()
    plt.tight_layout()
    if patron_ficheros is not None:
        plt.savefig(f'{patron_ficheros}-{nombre_modelo}-historial_entrenamiento.png')
    else:
        plt.show()
    plt.close()