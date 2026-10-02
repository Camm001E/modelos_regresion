"""Interfaz académica para consultar tres modelos de regresión lineal."""

from math import isfinite
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE = Path(__file__).resolve().parent
MODELOS = BASE / "salidas" / "modelos"
FIGURAS = BASE / "salidas" / "figuras"

VARIABLES = {
    "dolar": ["Dia", "Inflacion", "Tasa_interes"],
    "glucosa": ["Edad", "IMC", "Actividad_Fisica"],
    "energia": ["Temperatura", "Hora", "Dia_Semana"],
}

GRAFICAS = {
    "dolar": [
        ("dolar_dia.png", "Precio del dólar por día"),
        ("dolar_inflacion.png", "Inflación y precio del dólar"),
        ("dolar_tasa.png", "Tasa de interés y precio del dólar"),
    ],
    "glucosa": [
        ("glucosa_edad.png", "Edad y nivel de glucosa"),
        ("glucosa_imc.png", "IMC y nivel de glucosa"),
        ("glucosa_actividad.png", "Actividad física y nivel de glucosa"),
    ],
    "energia": [
        ("energia_temperatura.png", "Temperatura y consumo"),
        ("energia_hora.png", "Hora y consumo"),
        ("energia_semana.png", "Día de la semana y consumo"),
    ],
}


st.set_page_config(
    page_title="Explorador de modelos | Laboratorio de datos",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 1120px; padding-top: 1.4rem; padding-bottom: 3rem;}
      .hero {
        background: linear-gradient(120deg, #103f47 0%, #136c75 55%, #3b947f 100%);
        border-radius: 1.2rem;
        color: #fff;
        padding: 2.1rem 2.3rem;
        margin-bottom: 1.25rem;
      }
      .hero .eyebrow {
        color: #d6ffec;
        letter-spacing: .11em;
        text-transform: uppercase;
        font-size: .79rem;
        font-weight: 700;
      }
      .hero h1 {color: #fff; font-size: clamp(1.8rem, 4vw, 2.7rem); margin: .3rem 0 .5rem;}
      .hero p {color: #effffc; font-size: 1.05rem; max-width: 48rem; margin: 0;}
      .case-note {border-left: 4px solid #29a98e; padding: .6rem 1rem; margin: .4rem 0 1.2rem;}
      .case-note p {margin: 0;}
      div[data-testid="stForm"] {border: 1px solid #a6d5cc; border-radius: 1rem; padding: 1.1rem;}
      div[data-testid="stTabs"] button {font-weight: 650;}
      @media (max-width: 680px) {.hero {padding: 1.5rem 1.25rem;}}
    </style>
    <section class="hero">
      <div class="eyebrow">Laboratorio de minería de datos · Regresión lineal</div>
      <h1>Explora tres predicciones</h1>
      <p>Elige un tema, ajusta sus variables y consulta el valor calculado por el modelo.</p>
    </section>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def cargar_modelo(caso: str):
    """Lee exclusivamente el modelo generado por este proyecto."""
    ruta = MODELOS / f"{caso}.joblib"
    if not ruta.is_file():
        raise FileNotFoundError(f"Falta el archivo {ruta.relative_to(BASE)}")

    paquete = joblib.load(ruta)
    if not isinstance(paquete, dict) or not {"modelo", "variables"} <= paquete.keys():
        raise ValueError(f"El paquete {ruta.name} no tiene modelo y variables.")

    variables = paquete["variables"]
    if list(variables) != VARIABLES[caso]:
        raise ValueError(f"El orden de variables guardado en {ruta.name} no coincide con este formulario.")

    return paquete["modelo"], variables


def mostrar_resultado(caso: str, valores: dict, titulo: str, unidad: str, decimales: int = 2):
    try:
        modelo, variables = cargar_modelo(caso)
        entrada = pd.DataFrame([valores], columns=variables)
        resultado = float(modelo.predict(entrada)[0])
        if not isfinite(resultado):
            raise ValueError("El modelo devolvió una predicción no válida.")
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        st.error(f"No se pudo calcular la predicción: {error}")
        return

    st.success(f"**{titulo}: {resultado:,.{decimales}f} {unidad}**")
    st.caption("Estimación del ejercicio académico; puede diferir de una observación real.")


def mostrar_graficas(caso: str):
    disponibles = [
        (FIGURAS / nombre, titulo)
        for nombre, titulo in GRAFICAS[caso]
        if (FIGURAS / nombre).is_file()
    ]
    if not disponibles:
        return

    with st.expander("Ver gráficas exploratorias"):
        columnas = st.columns(len(disponibles))
        for columna, (ruta, titulo) in zip(columnas, disponibles):
            with columna:
                st.image(str(ruta), caption=titulo, width="stretch")


tab_dolar, tab_glucosa, tab_energia = st.tabs(
    ["01 · Dólar", "02 · Glucosa", "03 · Energía"]
)

with tab_dolar:
    st.subheader("Precio estimado del dólar")
    st.markdown(
        '<div class="case-note"><p>Prueba cómo cambia la estimación al modificar el día, '
        'la inflación y la tasa de interés.</p></div>',
        unsafe_allow_html=True,
    )
    with st.form("formulario_dolar"):
        col1, col2, col3 = st.columns(3)
        with col1:
            dia = st.number_input("Día", min_value=1, max_value=500, value=300, step=1)
        with col2:
            inflacion = st.number_input(
                "Inflación (proporción)",
                min_value=0.0,
                max_value=0.05,
                value=0.025,
                step=0.001,
                format="%.4f",
                help="Por ejemplo, 0.025 equivale a 2.5 %.",
            )
        with col3:
            tasa = st.number_input(
                "Tasa de interés",
                min_value=3.0,
                max_value=7.0,
                value=5.2,
                step=0.1,
                help="Ingresa 5.2 para representar 5.2 % en este conjunto de datos.",
            )
        enviar_dolar = st.form_submit_button("Estimar precio", type="primary")
    if enviar_dolar:
        mostrar_resultado(
            "dolar",
            {"Dia": dia, "Inflacion": inflacion, "Tasa_interes": tasa},
            "Precio estimado", "COP",
        )
    mostrar_graficas("dolar")

with tab_glucosa:
    st.subheader("Nivel estimado de glucosa")
    st.markdown(
        '<div class="case-note"><p>Introduce edad, índice de masa corporal y horas '
        'semanales de actividad física.</p></div>',
        unsafe_allow_html=True,
    )
    with st.form("formulario_glucosa"):
        col1, col2, col3 = st.columns(3)
        with col1:
            edad = st.number_input("Edad (años)", min_value=20, max_value=79, value=55, step=1)
        with col2:
            imc = st.number_input("IMC", min_value=10.0, max_value=40.0, value=27.5, step=0.1)
        with col3:
            actividad = st.number_input(
                "Actividad física (h/semana)", min_value=0, max_value=9, value=4, step=1
            )
        enviar_glucosa = st.form_submit_button("Estimar glucosa", type="primary")
    if enviar_glucosa:
        mostrar_resultado(
            "glucosa",
            {"Edad": edad, "IMC": imc, "Actividad_Fisica": actividad},
            "Nivel estimado", "mg/dL",
        )
    st.info("Este resultado es solo una simulación académica y no sirve para diagnosticar enfermedades.")
    mostrar_graficas("glucosa")

with tab_energia:
    st.subheader("Consumo estimado de energía")
    st.markdown(
        '<div class="case-note"><p>Estima el consumo con la temperatura, la hora y '
        'el día de la semana del conjunto de datos.</p></div>',
        unsafe_allow_html=True,
    )
    with st.form("formulario_energia"):
        col1, col2, col3 = st.columns(3)
        with col1:
            temperatura = st.number_input(
                "Temperatura (°C)", min_value=5.0, max_value=45.0, value=28.0, step=0.1
            )
        with col2:
            hora = st.number_input("Hora (1 a 24)", min_value=1, max_value=24, value=18, step=1)
        with col3:
            semana = st.number_input(
                "Día de la semana (1 a 7)",
                min_value=1,
                max_value=7,
                value=3,
                step=1,
                help="1 = lunes; 7 = domingo.",
            )
        enviar_energia = st.form_submit_button("Estimar consumo", type="primary")
    if enviar_energia:
        mostrar_resultado(
            "energia",
            {"Temperatura": temperatura, "Hora": hora, "Dia_Semana": semana},
            "Consumo estimado", "kWh",
        )
    mostrar_graficas("energia")

st.divider()
st.caption("Proyecto educativo · Tres regresiones lineales entrenadas con los datos del laboratorio")
