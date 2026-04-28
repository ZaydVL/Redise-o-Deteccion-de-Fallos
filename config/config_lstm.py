import config_global
CONFIG = config_global.ConfigGlobal()

# ── Datos ──────────────────────────────────────────────────────
fich_datos      = 'datos/fallos-{planta}.csv'  # patrón por planta

# Plantas cuyos datos se usan para entrenar+evaluar (split interno 80/20)
plantas_train   = ['br03']

# ── Qué se entrena ─────────────────────────────────────────────

"""
Mapa de tipo de dispositivos junto con sus diags

diags_all = {
    'ST': [201, 202],
    'IN': [241, 242, 243, 244, 245, 246, 341, 342, 343, 344, 345],
    'TR': [260, 261, 262, 263, 264],
    'SB': [221, 222, 224, 320],
    'CT': [280, 281, 282],
}
 
"""


tipo_disp       = 'IN'
diags           = [246]   # mismo tipo_disp siempre
modo            = 'detection'  # 'detection' | 'classification'

# ── Modelo ─────────────────────────────────────────────────────
nombre_modelo   = 'LSTM'     # 'LSTM' | 'Conv1D' | 'ConvLSTM2D'
transform_type  = None         # None | 'gramian' | 'markov'

# ── Hiperparámetros ────────────────────────────────────────────
max_trials           = 10
num_initial_points   = 2
executions_per_trial = 2
epochs_tuning        = 30
epochs_final         = 100
batch_size           = 32
patience             = 10
semilla              = 42

max_disp_sanos_por_fallo = 5

# ── Salida ─────────────────────────────────────────────────────
#dir_resultados  = 'results/{plantas}-{tipo_disp}-{diags}'
dir_resultados  = 'results/{plantas}-{tipo_disp}-{diags}'