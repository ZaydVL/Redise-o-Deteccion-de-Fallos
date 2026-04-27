# Configuración de ejemplo para el programa de generar conjuntos de datos.

import config_global
CONFIG = config_global.ConfigGlobal()

# ———————————————————————————————————————————————————————————————
# —— Plantas que se consideran ——————————————————————————————————
# ———————————————————————————————————————————————————————————————
"""
Las plantas definidas de forma global (es decir como CONFIG.plantas_all) son las que apararecen a continuación:

 >> plantas_all = ["br02", "br03", "cl02", "cl03", "mx05", "mx06", "rd02", "rd03", "rd04", "rd05", "sp08", "sp09", "sp10", "sp12", "sp13", "sp15", "sp16"]
"""

plantas = CONFIG.plantas_all
# plantas_all = ["br02", "br03", "cl02"]


# ———————————————————————————————————————————————————————————————
# —— Tipos de dispositivos ——————————————————————————————————————
# ———————————————————————————————————————————————————————————————

"""
Aquí nuevamente puede seleccionarse que para cada planta se considere la importación de datos de fallos y casos sanos 
para cada dispositivo que se requiera (naturalmente en una lista). Puede resultar útil recordad el ámbito jerarquico
que existe detrás de los dispositivos existentes:

CT (Centro de Transformación)
└── IN (Inversor)
    └── TR (Tracker / seguidor solar)
        └── SB (StringBox / caja de strings)
            └── ST (String / cadena de paneles)
                └── POS (posición del panel en el string)
"""

tipos_disp = CONFIG.tipos_disp_all
#tipos_disp = [ 'ST', 'IN', 'TR', 'SB', 'CT' ]


# ———————————————————————————————————————————————————————————————
# —— Tipos de diagnóstico ———————————————————————————————————————
# ———————————————————————————————————————————————————————————————
"""
Esta variable global indicará los diagnósticos considerados al momento de la importación de los datos desde las bases de 
datos de InfluxDB y PostgreSQL. En esencia, el módulo generar_conjunto_datos.py contempla dos formas de establecer los
códigos de diagnósticos (diags) a considerar: 
    - Una es una lista (la cual es general y sencillamente se declara aquellos códigos que nos interesa importar).
    - La segunda forma es un diccionario donde podemos personalizar aquellos diags que se incluyan en su respectivo 
        dispositivo. Aquellos pares key-value (dispositivo - lista de diags) que no se escriban pues se asumirá que 
        queremos descargar todos sus diagnósticos.

Si no se declara la variable "diags", generar_conjunto_datos.py asumirá que queremos todos los diags de todos los dispositivos
que hemos seleccionado arriba.

"""

#diags = [ 241, 242 ]

#diags = {
#    'ST': [201, 202],
#    'IN': [241, 242, 243, 244, 245, 246, 341, 342, 343, 344, 345],
#    'TR': [260, 261, 262, 263, 264],
#    'SB': [221, 222, 224, 320],
#    'CT': [280, 281, 282]  #En la hoja de diagnósticos de QPV sale como 'GL'
#}


# ———————————————————————————————————————————————————————————————
# —— Ruta de fichero de guardado ————————————————————————————————
# ———————————————————————————————————————————————————————————————
"""
Se declara el patrón de fichero de guardado, habiendo entonces 3 combinaciones las cuales serán interpretadas en generar_conjunto_datos.py

    1) fich_salida = 'datos/fallos-{planta}-{tipo_disp}.csv' --> Se guardarán archivos por cada DISPOSITIVO de cada PLANTA
    2) fich_salida = 'datos/fallos-{planta}.csv'  -------------> Se guardarán archivos por cada PLANTA (dentro estarán incluidos todos sus dispositivos)
    3) fich_salida = 'datos/fallos.csv'  ----------------------> Se guarda UN SOLO archivo que incluye todas las plantas y los dipositivos seleccionados.

"""

fich_salida = 'datos/fallos-{planta}.csv'


# ———————————————————————————————————————————————————————————————
# —— Configuración para la importación de datos —————————————————
# ———————————————————————————————————————————————————————————————


"""
1) Margen temporal: Naturalmente se intenta conseguir bloques de 96 datos para cada para cada caso de fallo en el sentido de que se ha dividido
    temporalmente un día en registros de 15 minutos (4*24 = 96), permitiendo así descargar un día completo de registro.
    Entonces, este valor de margen_temporal_h agregará una hora anterior al inicio del día que se está guardando por si se requiere de mayor
    contexto temporal para detectar o clasificar el fallo.  ----> Esto vive en "preprocesado.py" en la función: "obtener_datos_casos"

2) max_disp_sanos_por_fallo: Debemos recordar que al importar los datos de la DB, se cumple que se obtiene el registro de un día de un dispositivo
    que falló, asimimo se buscan otros dispositivos del mismo tipo y se guardan tus registros de datos de ese mismo día
                IN_n01 falló         ---------------- entonces también se guarda   IN_n02   
                                                                                   IN_n03   los cuales no han fallado 
                                                                                   IN_n04
    Este parámetro vive en "preprocesado.py" en la función "obtener_datos_casos". Si es None no se toma en cuenta y se cogen todos los sanos disponibles.

3) ratio_datos_min = Es un valor porcentual añadido que tiene relación directa con "margen temporal" ya que si los registros de un fallo no superan
    los 96 (o más) fragmentos de 15 minutos de registro pues no se consideran. Así que se introduce una tolerancia porcentual para evitar que se
    pierdan dichos datos. Esto aplica a los dispositivos que fallan como a los sanos que se toman también. 
 
"""

margen_temporal_h = 0
max_disp_sanos_por_fallo = None
ratio_datos_min = 1

######################################################################################
# Variables de entrada por tipo de dispositivo (comentar si se quieren usar todas)
#variables_por_tipo = {
#    'ST': ['vdc', 'idc', 'pdc', 'temp'],
#    'IN': ['vdc', 'idc', 'pdc', 'temp', 'temp_cab'],
#    'TR': ['pos', 'angulo', 'volt_motor'],
#    'SB': ['temp'],
#    'CT': ['vdc', 'idc']  # Ejemplo para CT
#}
#####################################################################################
