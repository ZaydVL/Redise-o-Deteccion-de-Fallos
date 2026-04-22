import importlib

###################################################################
#
# Definiciones por defecto para la configuración global
#
# Si aquí se define una variable, clase o función llamada xyz,
# en el resto de módulos se podrá usa como CONFIG.xyz
#
# Ej: en general, se usará así:
#
# En config/prog1.py:  <-- Por defecto, las configuraciones se buscan en el directorio config/
#
# a = 1
# b = ... <-- Pueden ser literales, expresiones, calculado con código, etc.
#
# En el programa que usa esa configuración:
#
# import config_global
# CONFIG = config_global.ConfigGlobal('prog1')  <-- También valdría 'config/prog1.py'
# print(CONFIG.a)  <-- Imprime 1
#
# CONFIG es un singleton, por lo que hay una instancia única y global en todo el programa.
#
# Ej: lo siguiente devuelve su valor actual, no el valor inicial ni un valor vacío.
#
# CONFIG = config_global.ConfigGlobal()
#
# Ej: combinar varias configuraciones.
# Ojo: las variables que tengan el mismo nombre serán machacadas por la última que se cargue.
#
# config_global.ConfigGlobal(fichero_de_configuración_1)
# config_global.ConfigGlobal(fichero_de_configuración_2)
# config_global.ConfigGlobal(fichero_de_configuración_3)
# CONFIG = config_global.ConfigGlobal()
#
# Ej: si en uno de los ficheros de configuración se quiere actualizar una variable de configuración
# o crear otra a partir de una definida en alguna configuración anterior,
# se puede poner esto en dicho fichero de configuración:
#
# import config_global
# CONFIG = config_global.ConfigGlobal()  <-- Obtiene la configuración definida hasta este momento.
# v1 = CONFIG.v1 * 2  <-- Actualiza v1
# v2 = CONFIG.v1 + 3  <-- Crea v2 a partir de v1

import sys
import os

depurar = True if "DEPURAR" in os.environ and os.environ["DEPURAR"].lower() == "true" else False

# Directorio donde se buscan los ficheros de configuración
dir_config = os.path.dirname(os.path.abspath(__file__)) + '/config'

# Todas las plantas que se usan en el análisis
# plantas_all = ["br02", "br03", "cl02", "cl03", "mx05", "mx06", "rd02", "rd03", "rd04", "rd05", "sp08", "sp09", "sp10", "sp12", "sp13", "sp15", "sp16"]
plantas_all = [ 'br02', 'br03', 'sp08', 'sp09', 'sp10', 'cl02', 'cl03', 'mx05', 'mx06', 'rd02' ]

# Todos los tipos de dispositivos que se usan en el análisis
tipos_disp_all = [ 'ST', 'IN', 'TR', 'SB', 'CT' ]

###################################################################
#                  NO MODIFICAR A PARTIR DE AQUÍ
###################################################################

# Singleton para la configuración global
#
# Cuando se instancia por primera vez, define todo lo que haya en este módulo.
#
# Además, si se le pasa un fichero de configuración, lo importa y añade sus definiciones,
# machacando si alguna ya estaba definida.
#
class ConfigGlobal:
    _instancia = None

    def __new__(clase, fich_config:str=None):
        if clase._instancia is None:
            clase._instancia = super(ConfigGlobal, clase).__new__(clase)
            for k, v in globals().items():
                if not k.startswith('__'):
                    setattr(clase._instancia, k, v)
            sys.path.append(dir_config)
        if fich_config is not None:
            if fich_config.endswith('.py'):
                fich_config = fich_config[:-3]
            _dir_config = None
            if '/' in fich_config:
                componentes = fich_config.split('/')
                fich_config = componentes[-1]
                _dir_config = '/'.join(componentes[:-1])
            elif '\\' in fich_config:
                componentes = fich_config.split('\\')
                fich_config = componentes[-1]
                _dir_config = '\\'.join(componentes[:-1])
            if _dir_config is not None:
                sys.path.append(_dir_config)
            config = importlib.import_module(fich_config)
            if _dir_config is not None:
                sys.path.remove(_dir_config)
            for k,v in config.__dict__.items():
                if not k.startswith('__'):
                    setattr(clase._instancia, k, v)
        return clase._instancia

    def __str__(self):
        attrs = {k: v for k, v in self.__dict__.items() 
             if not k.startswith('_') and not callable(v) and not isinstance(v, type(importlib)) and k != 'CONFIG'}
        return f"ConfigGlobal({attrs})"

    @classmethod
    def reset(cls):
        """Resetea el singleton en caso querramos correr pruebas encadenadas"""
        cls._instancia = None

###################################################################
