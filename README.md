# Laboratorio de minería de datos: tres modelos de regresión

Este proyecto desarrolla tres ejercicios académicos de regresión lineal: estimar el precio del dólar, el nivel de glucosa y el consumo de energía. Incluye revisión de los datos, gráficas por variable, evaluación de los modelos y una aplicación interactiva hecha con Streamlit.

## Organización

```text
modelos_regresion_companera/
├── fuentes/                    # CSV originales: dolar.csv, glucosa.csv, energia.csv
├── preparados/                 # Copias validadas, generadas por el script
├── codigo/
│   └── ejecutar_laboratorio.py # Validación, análisis, entrenamiento y exportación
├── salidas/
│   ├── figuras/                # Nueve gráficas de exploración
│   └── modelos/                # dolar.joblib, glucosa.joblib, energia.joblib
├── inicio.py                   # Interfaz web con tres pestañas
├── resultados.json             # Resumen de resultados, generado por el script
└── requirements.txt
```

Las fuentes se conservan sin modificar. Los valores extremos detectados se documentan en el análisis; se mantienen cuando no hay evidencia de que sean errores. Cada modelo se evalúa con un conjunto de prueba antes de generar el archivo final para la aplicación.

## Ejecutar en VS Code / PowerShell

Abre **la carpeta raíz del proyecto** en VS Code. Comprueba que la terminal esté situada en esa carpeta antes de ejecutar los comandos; `codigo/ejecutar_laboratorio.py` se invoca desde la raíz:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python codigo/ejecutar_laboratorio.py
python -m streamlit run inicio.py
```

Si PowerShell bloquea la activación, ejecuta una vez en esa terminal `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` y vuelve a activar el entorno.

El script de laboratorio espera `fuentes/dolar.csv`, `fuentes/glucosa.csv` y `fuentes/energia.csv`. Genera `preparados/*_validado.csv`, nueve imágenes en `salidas/figuras/`, los tres modelos en `salidas/modelos/` y `resultados.json`. La terminal muestra el número de filas y las métricas de prueba de cada caso. El detalle de la validación y los valores extremos está en `resultados.json`; revisa esos datos antes de interpretarlos o citarlos.

## Probar la interfaz

1. En la pestaña **Dólar**, prueba día `300`, inflación `0.025` y tasa de interés `5.2`.
2. En **Glucosa**, prueba edad `55`, IMC `27.5` y actividad física `4` horas por semana.
3. En **Energía**, prueba temperatura `28` °C, hora `18` y día de la semana `3`.

Pulsa el botón de cada formulario y comprueba que se muestra un número con la unidad correspondiente. La estimación de glucosa es un ejercicio académico y no constituye un diagnóstico médico. Ninguno de los tres modelos garantiza resultados fuera del contexto y los rangos de los datos de entrenamiento.

## Publicar en Streamlit Community Cloud

1. Sube el proyecto a un repositorio de GitHub. Comprueba que `inicio.py`, `requirements.txt` y `salidas/modelos/` estén incluidos; deja `.venv/` fuera del repositorio.
2. En Streamlit Community Cloud, crea una aplicación desde ese repositorio y selecciona la rama publicada.
3. Como **Main file path** indica `inicio.py` y como versión de Python selecciona **3.12**.
4. Tras el despliegue, prueba los tres formularios y abre el enlace en una ventana privada para comprobar el acceso de otra persona.

El informe del laboratorio se entrega junto con el enlace a la aplicación y el repositorio. Añade el nombre de la autora, el curso y las interpretaciones propias al informe antes de presentarlo.
