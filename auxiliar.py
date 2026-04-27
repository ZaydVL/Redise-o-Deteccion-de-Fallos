def evaluar_modelo(modelo, datos_aprendizaje, patron_ficheros):
        X_test = datos_aprendizaje['X_test']
        y_test = datos_aprendizaje['y_test']
        planta = datos_aprendizaje['planta']
        id_casos_test = datos_aprendizaje['id_casos_test']
        df_test = datos_aprendizaje['df_test']
        planta = datos_aprendizaje['planta']
        tipo_disp = datos_aprendizaje['tipo_disp']
        diag = datos_aprendizaje['diag']
        diag_txt = datos_aprendizaje['diag_txt']
        df_fallos = datos_aprendizaje['df_fallos']

        if False: # DDD
            print(f'EVALM: X_test shape: {X_test.shape}, y_test shape: {y_test.shape}')
            # Dibujar X_test para inspección visual
            for i in range(X_test.shape[0] - 1):
                plt.figure(figsize=(8, 3))
                plt.plot(X_test[i].squeeze(), color='blue')
                plt.plot(X_test[i+1].squeeze(), color='orange')
                plt.title(f'X_test[{i}] - id_caso: {id_casos_test[i]}-{id_casos_test[i+1]}, y_test: {y_test[i]}-{y_test[i+1]}')
                plt.xlabel('Timestep')
                plt.ylabel('Valor normalizado')
                plt.tight_layout()
                plt.savefig(f"{patron_ficheros}-X_test-{i}.png")
                plt.close()

        loss, accuracy = modelo.evaluate(X_test, y_test)
        print(f'Loss: {loss}, Accuracy: {accuracy}')

        predicciones = modelo.predict(X_test)
        y_pred = np.argmax(predicciones, axis=1)
        print("Matriz de confusión:")
        print(confusion_matrix(y_test, y_pred))
        print("\nMétricas relevantes:")
        print(classification_report(y_test, y_pred, digits=3, zero_division=np.nan))
        # Guardar matriz de confusión y métricas relevantes en CSV
        conf_matrix = confusion_matrix(y_test, y_pred)
        conf_matrix_df = pd.DataFrame(conf_matrix)
        conf_matrix_df.to_csv(f"{patron_ficheros}-matriz_confusion.csv", index=False)

        report_dict = classification_report(y_test, y_pred, digits=3, zero_division=np.nan, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        for campo in [ 'planta', 'diag', 'diag_txt', 'tipo_disp' ]:
            label = str()
            for o in datos_aprendizaje[campo]:
                label = label + str(o) + ' / ' 
            report_df[campo] = label[:-2]

        report_df.to_csv(f"{patron_ficheros}-metricas.csv")

        df_info_pruebas = None
        for i in range(X_test.shape[0]):
            id_caso = id_casos_test[i]
            id_fallo = df_test[df_test['id_caso'] == id_caso]['id_fallo'].iloc[0]
            clase_pred = np.argmax(predicciones[i])
            clase_real = y_test[i]
            confianza = predicciones[i][clase_pred] / predicciones[i][1-clase_pred]
            if clase_pred != clase_real:
                comentario = 'Predicción errónea'
            elif confianza < 1:
                comentario = 'Confianza reducida'
            else:
                comentario = ''
            print(f'Prueba {i+1} {planta}/{tipo_disp}: ID_FALLO={id_fallo}, Diag={diag}/{diag_txt if clase_real else "SANO"}, Pred: {clase_pred}, Real: {clase_real}, Probs: {predicciones[i]}, Conf: {confianza:.2f} <{comentario}>')
            info_prueba = { 'id_prueba' : i+1,
                            'id_fallo' : id_fallo,
                            'planta' : planta,
                            'tipo_disp' : tipo_disp,
                            'diag' : diag,
                            'diag_txt' : diag_txt,
                            'y_pred' : clase_pred,
                            'y_real' : clase_real,
                            'confianza' : confianza}
            for j in range(len(predicciones[i])):
                info_prueba[f'prob_{j}'] = predicciones[i][j]
            if df_info_pruebas is None:
                df_info_pruebas = pd.DataFrame([info_prueba])
            else:
                df_info_pruebas = pd.concat([df_info_pruebas, pd.DataFrame([info_prueba])])
            if len(comentario) > 0:
                figura, gráfica = plt.subplots(1, 1, figsize = (8, 8))
                #with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                    #df = df_test[df_test['id_caso'] == id_caso]
                    #print(df)
                    #print(df.info())
                #dibujar_fallo(df_test[df_test['id_caso'] == id_caso], gráfica, tipo_comparación='PROMEDIO')
                try: 
                    dibujar_fallo(df_fallos[df_fallos['id_fallo'] == id_fallo], gráfica, comentario=comentario, tipo_comparación='PROMEDIO')
                    plt.savefig(f'{patron_ficheros}-fallo-{id_fallo}.png', dpi=300)
                    #plt.show()
                    plt.close()
                except:
                    continue
        df_info_pruebas.to_csv(f"{patron_ficheros}-info-pruebas.csv", index=False)
