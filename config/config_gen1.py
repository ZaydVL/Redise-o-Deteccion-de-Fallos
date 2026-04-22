# Configuración de ejemplo para el programa de generar conjuntos de datos.
#

import config_global
CONFIG = config_global.ConfigGlobal()

# Plantas que se consideran.
plantas = CONFIG.plantas_all
#plantas = [ 'sp08', 'sp09', 'sp10' ]

# Tipos de dispositivos que se consideran.
#tipos_disp = CONFIG.tipos_disp_all
tipos_disp = [ 'ST', 'IN', 'TR', 'SB', 'CT' ]

# Códigos de diagnóstico que se consideran.
# Si no se define, se considerarán todos.
# Ej:
#diags = [ 241, 242 ]

# Ej, diferentes según el tipo disp:
# Para los que no vienen, se considerarán todos.
#diags = {
#    'ST': [201, 202],
#    'IN': [241, 242, 243, 244, 245, 246, 341, 342, 343, 344, 345],
#    'TR': [260, 261, 262, 263, 264],
#    'SB': [221, 222, 224, 320],
#    'CT': [280, 281, 282]  #En la hoja de diagnósticos de QPV sale como 'GL'
#}

# Variables de entrada por tipo de dispositivo (comentar si se quieren usar todas)
#variables_por_tipo = {
#    'ST': ['vdc', 'idc', 'pdc', 'temp'],
#    'IN': ['vdc', 'idc', 'pdc', 'temp', 'temp_cab'],
#    'TR': ['pos', 'angulo', 'volt_motor'],
#    'SB': ['temp'],
#    'CT': ['vdc', 'idc']  # Ejemplo para CT
#}

# Fichero donde se guardarán los ficheros generados.
fich_salida = 'datos/prueba1.csv'

# Guardar en ficheros separados. Si se omite {tipo_disp}, será uno por planta.
#fich_salida = 'datos/fallos-{planta}-{tipo_disp}.csv'

# Margen temporal en horas para los datos de casos de fallo.
margen_temporal_h = 0

# Número máximo de dispositivos sanos por cada dispositivo en fallo
max_disp_sanos_por_fallo = 5
