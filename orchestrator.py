
"""
En principio, el diseño modular individual por experimento mantiene a cada uno de ellos aislado de forma total
del SO, porlo que se pueden ejecutar las búsquedas que se requiera sin necesidad de perjudicarnos el rendimiento o,
incluso mucho peor, abortar el entrenamiento el el tuning. 

Asimismo, hay que tener en cuenta que al tener subprocesos separados se incrementa el tiempo de arranque porque
las librerias, y principalmente tensorflow, se instancia nuevamente cada vez, pero considerando entonces este 
fraccionamiento nos podemos permitir búsquedas incluso más grandes por experimento. 

Hacer clear_session() y gc.collect() se puede inclur dentro aliberando aún más memoria por proceso, lo cual creo 
que sería el tope que podría optimizarse el procedimiento.
"""


import subprocess
import sys

experimentos = [
    'config/config_lstm.py',
    'config/config_conv1d.py',
    'config/config_convlstm.py',
    'config/config_lstm_gramian.py',
]

for config in experimentos:
    print(f'\n{"="*60}')
    print(f'Lanzando experimento: {config}')
    print(f'{"="*60}')
    resultado = subprocess.run(
        [sys.executable, 'Training_models_v2.py', config],
        check=False  # No aborta si uno falla
    )
    if resultado.returncode != 0:
        print(f'ADVERTENCIA: {config} terminó con error (código {resultado.returncode})')