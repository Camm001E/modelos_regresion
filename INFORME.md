# Informe de laboratorio de minería de datos

**Estudiante:** por completar  
**Tema:** regresión lineal múltiple para dólar, glucosa y energía.

## Preparación de datos

Se verificaron columnas, tipos, nulos, duplicados y reglas de rango. Se conservaron 500, 2 000 y 10 000 filas respectivamente. Los extremos IQR se marcaron para examen, sin eliminación automática: inflación 4 y tasa 7; IMC 16 y glucosa 3; temperatura 90 y consumo 66.

## Evaluación en prueba

| Caso | Separación | Error | R² |
| --- | --- | ---: | ---: |
| Dólar | Temporal: días 1–400 / 401–500 | MSE 2905.71 | 0.8788 |
| Glucosa | Aleatoria 80/20, semilla 17 | MSE 220.62 | 0.7165 |
| Energía | Aleatoria 80/20, semilla 17 | RMSE 20.39 kWh | 0.8942 |

Las métricas pertenecen a los modelos ajustados solo en entrenamiento. Los modelos exportados se reajustaron después con todas las filas.

## Lectura de los coeficientes

- **Dólar:** día +4.9594; inflación -255.2842; tasa 0.4131. El día es el factor de mayor efecto por una desviación estándar. La tasa es pequeña e inestable entre los dos ajustes.
- **Glucosa:** edad +1.2198, IMC +0.8483, actividad física -1.9595. La edad presenta el mayor impacto por una desviación estándar (+21.46 mg/dL). No es una herramienta diagnóstica.
- **Energía:** temperatura +10.0322, hora +5.0116, día semanal -3.1128. La temperatura presenta el mayor impacto (+50.20 kWh por una desviación estándar).

## Gráficas e interfaz

Las nueve gráficas independientes frente a dependiente están en [`salidas/figuras`](salidas/figuras). La aplicación [`inicio.py`](inicio.py) contiene tres pestañas con entradas y predicciones basadas en los tres modelos exportados. Los archivos del laboratorio se guardan en este repositorio.

## Conclusión

Los tres ajustes sirven para explorar relaciones y hacer predicciones dentro de los rangos observados. Las asociaciones no establecen causa. El modelo de glucosa tiene mayor dispersión residual; el de dólar refleja principalmente una tendencia temporal; y el de energía depende en mayor medida de la temperatura.
