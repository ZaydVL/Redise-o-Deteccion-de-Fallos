# pip install keras tensorflow scikit-learn matplotlib pandas numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    roc_curve,
    auc,
    precision_recall_curve,
)
 
from visualization import dibujar_fallo  # mover dibujar_fallo a visualization.py
 
###################################################################
#
# evaluation.py
#
# Funciones de evaluación de modelos de detección y clasificación
# de fallos en plantas fotovoltaicas.
#
# Compatibles con los datos_aprendizaje devueltos por:
#   - preparar_deteccion()     → modo = 'deteccion'
#   - preparar_clasificacion() → modo = 'clasificacion'
#
###################################################################
 
 
def evaluar_modelo(modelo, nombre_modelo, datos_aprendizaje, patron_ficheros):
    """
    Evalúa un modelo de detección o clasificación de fallos.
 
    Detecta automáticamente el modo a partir de datos_aprendizaje['modo']:
        - 'deteccion':      binario, clase 0 = sano, clase 1 = fallo
        - 'clasificacion':  multiclase, cada entero corresponde a un diag
                            según datos_aprendizaje['mapa_clases']
 
    Genera y guarda:
        - {patron_ficheros}-matriz_confusion.csv
        - {patron_ficheros}-metricas.csv
        - {patron_ficheros}-info-pruebas.csv
        - {patron_ficheros}-fallo-{id_fallo}.png  (solo casos con error o baja confianza)
        - {patron_ficheros}-roc.png               (solo detección binaria)
        - {patron_ficheros}-precision_recall.png  (solo detección binaria)
        - {patron_ficheros}-hist_probs.png        (solo detección binaria)
 
    Args:
        modelo:              modelo Keras entrenado
        datos_aprendizaje:   dict devuelto por preparar_deteccion() o preparar_clasificacion()
        patron_ficheros:     prefijo de ruta para los ficheros de salida
                             Ej: 'resultados/exp001/res-cnn-241'
    """
    # ── Extracción de datos ───────────────────────────────────────
    modo          = datos_aprendizaje.get('modo', 'deteccion')
    X_test        = datos_aprendizaje['X_test']
    y_test        = datos_aprendizaje['y_test_onehot']
    id_casos_test = datos_aprendizaje['id_casos_test']
    df_test       = datos_aprendizaje['df_test']
    df_fallos     = datos_aprendizaje['df_fallos']
    planta        = datos_aprendizaje.get('planta', '?')
    tipo_disp     = datos_aprendizaje['tipo_disp']
    diags         = datos_aprendizaje.get('diags', datos_aprendizaje.get('diag', '?'))
    diag_txt      = datos_aprendizaje.get('diag_txt', '?')
    mapa_clases   = datos_aprendizaje.get('mapa_clases', {})  # solo clasificación
 
    # ── Evaluación y predicción ───────────────────────────────────
    resultados = modelo.evaluate(X_test, y_test, verbose=0)
    nombres_metricas = modelo.metrics_names
    metrics_eval = dict(zip(nombres_metricas, resultados))
    print(f"\nEvaluación sobre test ({modo}):")
    for nombre, valor in metrics_eval.items():
        print(f"  {nombre}: {valor:.4f}")
 
    predicciones = modelo.predict(X_test)
    y_pred = np.argmax(predicciones, axis=1)
    y_true = np.argmax(y_test, axis=1)  # ya son etiquetas enteras en ambos modos
 
    # ── Matriz de confusión ───────────────────────────────────────
    conf_matrix = confusion_matrix(y_true, y_pred)
    print("\nMatriz de confusión:")
    print(conf_matrix)
    pd.DataFrame(conf_matrix).to_csv(
        f"{patron_ficheros}-{nombre_modelo}-matriz_confusion.csv", index=False
    )
 
    # ── MCC ──────────────────────────────────────────────────────
    mcc_value = matthews_corrcoef(y_true, y_pred)
    print(f"\nMatthews Correlation Coefficient (MCC): {mcc_value:.4f}")
 
    # ── Informe de clasificación ──────────────────────────────────
    report_dict = classification_report(
        y_true, y_pred, digits=3, zero_division=np.nan, output_dict=True
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_df['mcc']      = mcc_value
    report_df['modo']     = modo
    report_df['planta']   = str(planta)
    report_df['tipo_disp']= tipo_disp
    report_df['diags']    = str(diags)
    report_df['diag_txt'] = str(diag_txt)
    print("\nMétricas relevantes:")
    print(report_df)
    report_df.to_csv(f"{patron_ficheros}-{nombre_modelo}-metricas.csv")
 
    # ── Curvas adicionales (solo detección binaria) ───────────────
    if modo == 'deteccion' and predicciones.shape[1] == 2:
        pos_probs = predicciones[:, 1]
 
        # ROC
        fpr, tpr, _ = roc_curve(y_true, pos_probs)
        roc_auc_value = auc(fpr, tpr)
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc_value:.3f}")
        plt.plot([0, 1], [0, 1], '--', color='gray')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC — {tipo_disp} diags={diags}')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{patron_ficheros}-{nombre_modelo}-roc.png')
        plt.close()
 
        # Precision-Recall
        prec_vals, rec_vals, _ = precision_recall_curve(y_true, pos_probs)
        no_skill = sum(y_true == 1) / len(y_true)
        plt.figure(figsize=(6, 6))
        plt.plot(rec_vals, prec_vals, lw=2)
        plt.axhline(no_skill, linestyle='--', color='gray', label='No skill')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall — {tipo_disp} diags={diags}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{patron_ficheros}-{nombre_modelo}-precision_recall.png')
        plt.close()
 
        # Histograma de probabilidades
        plt.figure(figsize=(6, 4))
        plt.hist(
            [pos_probs[y_true == 0], pos_probs[y_true == 1]],
            bins=20,
            label=['Clase 0 (sano)', 'Clase 1 (fallo)'],
            alpha=0.7,
            density=True,
        )
        plt.xlabel('Probabilidad predicha clase 1')
        plt.ylabel('Densidad')
        plt.title(f'Distribución de probabilidades — {tipo_disp} diags={diags}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{patron_ficheros}-{nombre_modelo}-hist_probs.png')
        plt.close()
 
    # ── Iteración caso a caso ─────────────────────────────────────
    epsilon = 1e-8
    lista_info_pruebas = []
 
    for i in range(X_test.shape[0]):
        id_caso   = id_casos_test[i]
        id_fallo  = df_test[df_test['id_caso'] == id_caso]['id_fallo'].iloc[0]
        clase_pred = y_pred[i]
        clase_real = y_true[i]
        confianza  = predicciones[i][clase_pred] / (predicciones[i][1 - clase_pred] + epsilon) \
                     if modo == 'deteccion' else predicciones[i][clase_pred]
 
        # Etiqueta legible de la clase real
        if modo == 'deteccion':
            etiqueta_real = diag_txt if clase_real == 1 else 'SANO'
        else:
            # Invertir mapa_clases para obtener el diag original
            mapa_inverso = {v: k for k, v in mapa_clases.items()}
            etiqueta_real = str(mapa_inverso.get(clase_real, clase_real))
 
        if clase_pred != clase_real:
            comentario = 'Predicción errónea'
        elif confianza < 1 and modo == 'deteccion':
            comentario = 'Confianza reducida'
        else:
            comentario = ''
 
        print(
            f'Prueba {i+1} {planta}/{tipo_disp}: '
            f'ID_FALLO={id_fallo}, '
            f'Real={etiqueta_real}, Pred={clase_pred}, '
            f'Probs={predicciones[i]}, Conf={confianza:.2f} <{comentario}>'
        )
 
        info_prueba = {
            'id_prueba' : i + 1,
            'id_fallo'  : id_fallo,
            'planta'    : planta,
            'tipo_disp' : tipo_disp,
            'diags'     : str(diags),
            'diag_txt'  : str(diag_txt),
            'y_pred'    : clase_pred,
            'y_real'    : clase_real,
            'confianza' : confianza,
            'comentario': comentario,
        }
        for j in range(len(predicciones[i])):
            info_prueba[f'prob_{j}'] = predicciones[i][j]
 
        lista_info_pruebas.append(info_prueba)
 
        # Dibuja el fallo si hubo error o baja confianza
        if comentario:
            try:
                figura, grafica = plt.subplots(1, 1, figsize=(8, 8))
                dibujar_fallo(
                    df_fallos[df_fallos['id_fallo'] == id_fallo],
                    grafica,
                    comentario=comentario,
                    tipo_comparacion='PROMEDIO',
                )
                plt.savefig(f'{patron_ficheros}-{nombre_modelo}-fallo-{id_fallo}.png', dpi=300)
                plt.close()
            except Exception as e:
                print(f"No se pudo dibujar el fallo {id_fallo}: {e}")
                plt.close()
 
    # ── Guardar info de pruebas ───────────────────────────────────
    pd.DataFrame(lista_info_pruebas).to_csv(
        f"{patron_ficheros}-{nombre_modelo}-info-pruebas.csv", index=False
    )
 