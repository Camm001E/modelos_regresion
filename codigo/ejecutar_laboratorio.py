"""Preparar, evaluar y exportar los tres casos del laboratorio.

Ejecutar desde cualquier directorio:
    python codigo/ejecutar_laboratorio.py

Los archivos de entrada están en fuentes/. Los registros atípicos por IQR se
documentan en resultados.json y se conservan: una alerta estadística por sí
sola no demuestra que una observación sea errónea.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # Permite generar figuras sin una pantalla conectada.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


RAIZ = Path(__file__).resolve().parents[1]
FUENTES = RAIZ / "fuentes"
PREPARADOS = RAIZ / "preparados"
FIGURAS = RAIZ / "salidas" / "figuras"
MODELOS = RAIZ / "salidas" / "modelos"


@dataclass(frozen=True)
class Caso:
    nombre: str
    variables: tuple[str, str, str]
    respuesta: str
    filas: int
    metodo_division: str
    color: str
    etiquetas: tuple[str, str, str]
    archivos_figuras: tuple[str, str, str]
    etiqueta_respuesta: str


CASOS = (
    Caso(
        nombre="dolar",
        variables=("Dia", "Inflacion", "Tasa_interes"),
        respuesta="Precio_Dolar",
        filas=500,
        metodo_division="temporal",
        color="#6955B8",
        etiquetas=("Día", "Inflación (%)", "Tasa de interés"),
        archivos_figuras=("dolar_dia.png", "dolar_inflacion.png", "dolar_tasa.png"),
        etiqueta_respuesta="Precio del dólar",
    ),
    Caso(
        nombre="glucosa",
        variables=("Edad", "IMC", "Actividad_Fisica"),
        respuesta="Nivel_Glucosa",
        filas=2000,
        metodo_division="aleatoria",
        color="#168B87",
        etiquetas=("Edad (años)", "IMC", "Actividad física (horas/semana)"),
        archivos_figuras=("glucosa_edad.png", "glucosa_imc.png", "glucosa_actividad.png"),
        etiqueta_respuesta="Nivel de glucosa (mg/dL)",
    ),
    Caso(
        nombre="energia",
        variables=("Temperatura", "Hora", "Dia_Semana"),
        respuesta="Consumo_Energia",
        filas=10000,
        metodo_division="aleatoria",
        color="#D36B53",
        etiquetas=("Temperatura (°C)", "Hora (1 a 24)", "Día de la semana (1 a 7)"),
        archivos_figuras=("energia_temperatura.png", "energia_hora.png", "energia_semana.png"),
        etiqueta_respuesta="Consumo de energía (kWh)",
    ),
)


def contar_invalidos(df: pd.DataFrame, caso: Caso) -> dict[str, int]:
    """Reglas mínimas del dominio; las horas cero de actividad son válidas."""
    if caso.nombre == "dolar":
        return {
            "dias_repetidos": int(df["Dia"].duplicated().sum()),
            "dias_fuera_de_1_a_500": int((~df["Dia"].between(1, 500)).sum()),
            "precios_no_positivos": int((df["Precio_Dolar"] <= 0).sum()),
        }
    if caso.nombre == "glucosa":
        return {
            "edades_negativas": int((df["Edad"] < 0).sum()),
            "imc_no_positivos": int((df["IMC"] <= 0).sum()),
            "actividad_negativa": int((df["Actividad_Fisica"] < 0).sum()),
            "glucosa_no_positiva": int((df["Nivel_Glucosa"] <= 0).sum()),
        }
    return {
        "horas_fuera_de_1_a_24": int((~df["Hora"].between(1, 24)).sum()),
        "dias_semana_fuera_de_1_a_7": int((~df["Dia_Semana"].between(1, 7)).sum()),
        "temperaturas_fisicamente_imposibles": int((df["Temperatura"] < -273.15).sum()),
        "consumos_no_positivos": int((df["Consumo_Energia"] <= 0).sum()),
    }


def diagnostico_iqr(df: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    resumen = {}
    for columna in df.columns:
        q1, q3 = df[columna].quantile([0.25, 0.75])
        recorrido = q3 - q1
        inferior = q1 - 1.5 * recorrido
        superior = q3 + 1.5 * recorrido
        resumen[columna] = {
            "limite_inferior": float(inferior),
            "limite_superior": float(superior),
            "registros_fuera_del_rango": int(
                ((df[columna] < inferior) | (df[columna] > superior)).sum()
            ),
        }
    return resumen


def leer_y_validar(caso: Caso) -> tuple[pd.DataFrame, dict]:
    ruta = FUENTES / f"{caso.nombre}.csv"
    if not ruta.is_file():
        raise FileNotFoundError(f"Falta el archivo de entrada: {ruta}")

    df = pd.read_csv(ruta)
    columnas = [*caso.variables, caso.respuesta]
    if list(df.columns) != columnas:
        raise ValueError(f"Columnas inesperadas en {ruta.name}: {list(df.columns)}")
    if len(df) != caso.filas:
        raise ValueError(f"{ruta.name}: esperaba {caso.filas} filas y encontré {len(df)}")
    if not all(pd.api.types.is_numeric_dtype(df[columna]) for columna in columnas):
        raise ValueError(f"{ruta.name}: hay columnas que no son numéricas")

    vacios = {c: int(df[c].isna().sum()) for c in columnas}
    duplicados = int(df.duplicated().sum())
    if any(vacios.values()) or duplicados or not np.isfinite(df.to_numpy(dtype=float)).all():
        raise ValueError(f"{ruta.name}: hay faltantes, duplicados o números no finitos")
    invalidos = contar_invalidos(df, caso)
    if any(invalidos.values()):
        raise ValueError(f"{ruta.name}: reglas del dominio incumplidas: {invalidos}")
    if caso.nombre == "dolar":
        if set(df["Dia"].tolist()) != set(range(1, 501)):
            raise ValueError("dolar.csv: deben estar presentes todos los días 1 a 500")
        df = df.sort_values("Dia").reset_index(drop=True)

    iqr = diagnostico_iqr(df)
    PREPARADOS.mkdir(parents=True, exist_ok=True)
    df.to_csv(PREPARADOS / f"{caso.nombre}_validado.csv", index=False)
    diagnostico = {
        "filas_originales": len(df),
        "filas_conservadas": len(df),
        "columnas": columnas,
        "valores_faltantes": vacios,
        "filas_duplicadas": duplicados,
        "valores_invalidos": invalidos,
        "iqr": iqr,
        "criterio_atipicos": "Se registran y se conservan sin alterar los datos",
        "archivo_preparado": f"preparados/{caso.nombre}_validado.csv",
    }
    return df, diagnostico


def division(df: pd.DataFrame, caso: Caso) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict]:
    x = df[list(caso.variables)]
    y = df[caso.respuesta]
    if caso.metodo_division == "temporal":
        corte = int(len(df) * 0.8)
        x_train, x_test = x.iloc[:corte], x.iloc[corte:]
        y_train, y_test = y.iloc[:corte], y.iloc[corte:]
        detalle = {
            "tipo": "Primeros 80 % de los días para entrenar, últimos 20 % para probar",
            "dias_entrenamiento": [int(df["Dia"].iloc[0]), int(df["Dia"].iloc[corte - 1])],
            "dias_prueba": [int(df["Dia"].iloc[corte]), int(df["Dia"].iloc[-1])],
        }
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=17, shuffle=True
        )
        detalle = {"tipo": "Aleatoria 80/20", "semilla": 17}
    detalle["filas_entrenamiento"] = len(x_train)
    detalle["filas_prueba"] = len(x_test)
    return x_train, x_test, y_train, y_test, detalle


def graficar(df: pd.DataFrame, caso: Caso) -> None:
    FIGURAS.mkdir(parents=True, exist_ok=True)
    for variable, etiqueta, archivo in zip(
        caso.variables, caso.etiquetas, caso.archivos_figuras
    ):
        x = df[variable].to_numpy(dtype=float)
        y = df[caso.respuesta].to_numpy(dtype=float)
        x_dibujo = x * 100 if variable == "Inflacion" else x
        correlacion = float(np.corrcoef(x, y)[0, 1])
        fig, ax = plt.subplots(figsize=(8.3, 5))
        fig.patch.set_facecolor("#F9FAFC")
        ax.set_facecolor("#FFFFFF")
        ax.scatter(x_dibujo, y, s=12 if len(df) < 3000 else 8,
                   alpha=0.28 if len(df) > 3000 else 0.43,
                   c=caso.color, edgecolors="none", rasterized=True)

        # Línea de tendencia descriptiva de UNA variable; no es la predicción
        # del modelo multivariable, que utiliza las tres entradas juntas.
        pendiente, intercepto = np.polyfit(x_dibujo, y, deg=1)
        eje = np.linspace(x_dibujo.min(), x_dibujo.max(), 150)
        ax.plot(eje, pendiente * eje + intercepto, color="#18263C",
                linewidth=2.2, label="Tendencia descriptiva")
        ax.set_xlabel(etiqueta, fontsize=10)
        ax.set_ylabel(caso.etiqueta_respuesta, fontsize=10)
        ax.set_title(f"{caso.etiqueta_respuesta} y {etiqueta.lower()}",
                     loc="left", fontsize=13, fontweight="bold", pad=12)
        ax.text(0.99, 0.98, f"r = {correlacion:+.2f}  •  n = {len(df):,}",
                ha="right", va="top", transform=ax.transAxes,
                fontsize=9, color="#455366",
                bbox={"boxstyle": "round,pad=0.35", "facecolor": "#F3F6FA", "edgecolor": "none"})
        ax.grid(alpha=0.13)
        ax.spines[["right", "top"]].set_visible(False)
        ax.legend(frameon=False, loc="lower right", fontsize=9)
        fig.tight_layout()
        fig.savefig(FIGURAS / archivo, dpi=165, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)


def entrenar(caso: Caso, df: pd.DataFrame) -> dict:
    x_train, x_test, y_train, y_test, detalle = division(df, caso)
    evaluado = LinearRegression().fit(x_train, y_train)
    predicciones = evaluado.predict(x_test)
    mse = float(mean_squared_error(y_test, predicciones))
    r2 = float(r2_score(y_test, predicciones))
    metricas = {"R2": r2}
    if caso.nombre == "energia":
        metricas["RMSE"] = float(np.sqrt(mse))
    else:
        metricas["MSE"] = mse

    coeficientes = dict(zip(caso.variables, map(float, evaluado.coef_)))
    desviaciones = x_train.std(ddof=1).to_dict()
    impacto = {
        columna: float(coeficientes[columna] * desviaciones[columna])
        for columna in caso.variables
    }
    mayor_impacto = max(impacto, key=lambda columna: abs(impacto[columna]))

    # El objeto exportado se ajusta con TODOS los datos. Las métricas de prueba
    # anteriores pertenecen únicamente al ajuste hecho con entrenamiento.
    final = LinearRegression().fit(df[list(caso.variables)], df[caso.respuesta])
    MODELOS.mkdir(parents=True, exist_ok=True)
    ruta_modelo = MODELOS / f"{caso.nombre}.joblib"
    joblib.dump({"modelo": final, "variables": list(caso.variables)}, ruta_modelo)
    recargado = joblib.load(ruta_modelo)
    if recargado["variables"] != list(caso.variables):
        raise RuntimeError(f"Variables alteradas al cargar {ruta_modelo}")
    if not np.isclose(recargado["modelo"].predict(df[list(caso.variables)].iloc[:1])[0],
                      final.predict(df[list(caso.variables)].iloc[:1])[0]):
        raise RuntimeError(f"La predicción cambia al recargar {ruta_modelo}")

    graficar(df, caso)
    return {
        "division": detalle,
        "metricas_prueba": metricas,
        "modelo_evaluado": {"intercepto": float(evaluado.intercept_), "coeficientes": coeficientes},
        "impacto_una_desviacion_estandar_entrenamiento": impacto,
        "mayor_impacto_absoluto": mayor_impacto,
        "modelo_final": {
            "filas": len(df),
            "intercepto": float(final.intercept_),
            "coeficientes": dict(zip(caso.variables, map(float, final.coef_))),
            "archivo": f"salidas/modelos/{caso.nombre}.joblib",
            "variables": list(caso.variables),
        },
        "figuras": [f"salidas/figuras/{nombre}" for nombre in caso.archivos_figuras],
    }


def main() -> None:
    resultados = {
        "nota_metodologica": "Datos validados sin borrar observaciones IQR. Métricas en conjunto de prueba; artefactos finales ajustados con la totalidad.",
        "casos": {},
    }
    for caso in CASOS:
        datos, auditoria = leer_y_validar(caso)
        resultados["casos"][caso.nombre] = {
            "auditoria": auditoria,
            **entrenar(caso, datos),
        }
        metrica = resultados["casos"][caso.nombre]["metricas_prueba"]
        print(f"{caso.nombre.capitalize()}: {len(datos)} filas, {metrica}")

    salida = RAIZ / "resultados.json"
    salida.write_text(json.dumps(resultados, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Resultados completos: {salida}")


if __name__ == "__main__":
    main()
