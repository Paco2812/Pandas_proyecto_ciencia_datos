"""
Generador de la muestra simulada de la encuesta de satisfacción con el SII (ISC - ITVER).

Modelo de variable latente:
    Cada estudiante i tiene una satisfacción latente  theta_i ~ N(mu_estrato, SIGMA_THETA).
    Cada ítem Likert j se genera como
        y_ij = 3 + LAMBDA_j * theta_i + B_j + e_ij,   e_ij ~ N(0, SIGMA_ERROR)
    y se redondea y recorta a la escala 1–5. Al compartir theta, los ítems quedan
    correlacionados entre sí (consistencia interna medible con el alfa de Cronbach).
    El resto de las respuestas (caídas, inconsistencias, problemas, evolución, etc.)
    también dependen de theta o de los ítems relacionados, para que la encuesta sea coherente.

Se conservan del archivo original: id (Marca temporal), inscrito, uso_ciclo, bloque
(70 / 57), número de control y comentarios. Las preguntas (encabezados) no cambian.

Uso:
    python src/generar_muestra.py            # genera data/encuesta_sii_simulada.csv y valida
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
ORIGINAL = RAIZ / "data" / "encuesta_sii_original.csv"
SALIDA = RAIZ / "data" / "encuesta_sii_simulada.csv"

# --- Objetivos (documento del proyecto, secciones 4–5) -----------------------------------
ESTRATO_1, ESTRATO_2 = "1.º – 4.º semestre", "5.º – 9.º semestre"
OBJETIVO_SATISFECHOS = {ESTRATO_1: 52, ESTRATO_2: 27}   # 52/70 = 0.7429, 27/57 = 0.4737
ALFA_MINIMO = 0.80
CONCORDANCIA_BINARIA = (0.82, 0.95)                      # satisfecho_binario vs. índice Likert

# --- Parámetros del modelo -----------------------------------------------------------------
MU = {ESTRATO_1: 0.55, ESTRATO_2: -0.05}
SIGMA_THETA = 0.75
SIGMA_ERROR = 0.85

# Ítems Likert: (carga LAMBDA_j, desplazamiento B_j). B_j conserva el perfil de medias del
# documento (información y calificaciones más altas; rendimiento, reinscripción y diseño más bajas).
ITEMS = {
    "sat_general":        (1.10,  0.15),
    "sat_facilidad":      (0.90,  0.00),
    "sat_diseno":         (0.85, -0.35),
    "sat_rendimiento":    (1.00, -0.45),
    "sat_disponibilidad": (1.00,  0.00),
    "sat_info":           (0.80,  0.55),
    "sat_seguridad":      (0.75,  0.20),
    "sat_calificaciones": (0.85,  0.40),
    "sat_reinscripcion":  (1.05, -0.45),
    "sat_citas":          (0.90, -0.30),
}

# --- Categorías exactas del Forms ----------------------------------------------------------
FRECUENCIA = ["Solo en periodos clave (inscripción, calificaciones)", "Algunas veces al mes",
              "Varias veces por semana", "Todos los días"]
DISPOSITIVO = ["Computadora o laptop", "Computadora y celular por igual", "Celular", "Tablet"]
TRAMITES = ["Agendar citas en departamentos", "Consultar calificaciones", "Consultar datos personales",
            "Consultar horario o kárdex", "Consultar o actualizar datos socioeconómicos",
            "Reinscripción / selección de materias"]
EVOLUCION = ["Ha empeorado", "Sigue igual", "Ha mejorado"]
CAIDAS = ["Nunca", "Rara vez", "Algunas veces", "Frecuentemente", "Siempre"]
INCONSISTENCIAS = ["Nunca", "Una vez", "Varias veces"]
PROBLEMAS = ["Caídas o falta de conexión con el sistema", "Errores de visualización o de diseño",
             "Fallas al agendar citas", "Fallas al seleccionar materias",
             "Información incorrecta o desactualizada", "Lentitud al cargar las páginas",
             "Mala adaptación al celular", "Problemas de acceso (usuario o contraseña)"]
SIN_PROBLEMAS = "No he tenido problemas"
REINSCRIPCION = ["No, necesité apoyo presencial o de un tercero",
                 "Sí, pero con dificultades o varios intentos", "Sí, sin problemas"]
MEJORAS = ["Diseño y facilidad de uso", "Estabilidad y velocidad", "Proceso de reinscripción",
           "Exactitud de la información", "Agenda de citas", "Seguridad y acceso", "Versión para celular"]

# Probabilidad de haber realizado cada trámite, por estrato
P_TRAMITE = {
    ESTRATO_1: [0.45, 0.85, 0.50, 0.40, 0.40, 0.70],
    ESTRATO_2: [0.45, 0.85, 0.50, 0.50, 0.40, 0.85],
}

# Nombres cortos de las columnas (mismo orden que el notebook)
NOMBRES = ["id_respuesta", "inscrito", "uso_ciclo", "bloque", "num_control", "frecuencia_uso",
           "dispositivo", "tramites", *ITEMS, "satisfecho_binario", "evolucion", "freq_caidas",
           "freq_inconsistencias", "problemas_tipos", "reinscripcion_exito", "acudio_presencial",
           "mejora_prioritaria", "comentarios"]


def _sigmoide(x):
    return 1 / (1 + np.exp(-x))


def _ordinal(z, cortes, categorias):
    """Asigna categorías ordinales según en qué intervalo de `cortes` cae z."""
    return [categorias[i] for i in np.searchsorted(cortes, z)]


def alfa_cronbach(items: pd.DataFrame) -> float:
    datos = items.dropna().astype(float)
    k = datos.shape[1]
    return k / (k - 1) * (1 - datos.var(ddof=1).sum() / datos.sum(axis=1).var(ddof=1))


def generar(semilla: int, base: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    n = len(base)
    df = pd.DataFrame({c: base[c] for c in ["id_respuesta", "inscrito", "uso_ciclo", "bloque",
                                            "num_control", "comentarios"]})
    estrato = df["bloque"].to_numpy()

    # 1. Uso y dispositivo (independientes de la satisfacción)
    df["frecuencia_uso"] = rng.choice(FRECUENCIA, size=n, p=[0.32, 0.39, 0.24, 0.05])
    df["dispositivo"] = rng.choice(DISPOSITIVO, size=n, p=[0.55, 0.19, 0.19, 0.07])

    # 2. Trámites (al menos uno por estudiante)
    p = np.array([P_TRAMITE[e] for e in estrato])
    hizo = rng.random((n, len(TRAMITES))) < p
    vacios = ~hizo.any(axis=1)
    hizo[vacios, rng.integers(0, len(TRAMITES), vacios.sum())] = True
    tramites = pd.DataFrame(hizo, columns=TRAMITES)
    df["tramites"] = [", ".join(t for t, h in zip(TRAMITES, fila) if h) for fila in hizo]

    # 3. Satisfacción latente e ítems Likert
    theta = np.array([rng.normal(MU[e], SIGMA_THETA) for e in estrato])
    for item, (carga, desplazamiento) in ITEMS.items():
        y = 3 + carga * theta + desplazamiento + rng.normal(0, SIGMA_ERROR, n)
        df[item] = pd.array(np.clip(np.rint(y), 1, 5).astype(int), dtype="Int64")
    # Ítems opcionales: solo responde quien realizó el trámite
    for item, tramite in [("sat_calificaciones", "Consultar calificaciones"),
                          ("sat_reinscripcion", "Reinscripción / selección de materias"),
                          ("sat_citas", "Agendar citas en departamentos")]:
        df.loc[~tramites[tramite].to_numpy(), item] = pd.NA
    likert = df[list(ITEMS)].astype(float)
    promedio = likert.mean(axis=1).to_numpy()

    # 4. Pregunta directa de satisfacción (coherente con el índice, con algo de ruido)
    df["satisfecho_binario"] = np.where(promedio + rng.normal(0, 0.30, n) > 3, "Sí", "No")

    # 5. Variables que dependen de la satisfacción
    df["evolucion"] = _ordinal(0.8 * theta + rng.normal(0, 0.9, n), [-0.85, 1.10], EVOLUCION)
    disp = likert["sat_disponibilidad"].to_numpy() - 3
    df["freq_caidas"] = _ordinal(-(0.4 * theta + 0.5 * disp) + rng.normal(0, 0.7, n),
                                 [-1.30, -0.45, 0.45, 1.05], CAIDAS)
    info = likert["sat_info"].to_numpy() - 3
    df["freq_inconsistencias"] = _ordinal(-(0.3 * theta + 0.5 * info) + rng.normal(0, 0.7, n),
                                          [0.05, 0.95], INCONSISTENCIAS)

    # Problemas: cada uno depende del ítem relacionado (y del trámite / dispositivo cuando aplica)
    relacion = {
        "Caídas o falta de conexión con el sistema": likert["sat_disponibilidad"],
        "Errores de visualización o de diseño": likert["sat_diseno"],
        "Fallas al agendar citas": likert["sat_citas"],
        "Fallas al seleccionar materias": likert["sat_reinscripcion"],
        "Información incorrecta o desactualizada": likert["sat_info"],
        "Lentitud al cargar las páginas": likert["sat_rendimiento"],
        "Mala adaptación al celular": likert["sat_diseno"],
        "Problemas de acceso (usuario o contraseña)": likert["sat_seguridad"],
    }
    usa_celular = df["dispositivo"].isin(["Celular", "Computadora y celular por igual", "Tablet"]).to_numpy()
    problemas = {}
    for problema, item in relacion.items():
        prob = _sigmoide(-1.0 - 1.1 * (item.to_numpy() - 3))
        posible = ~np.isnan(item.to_numpy())          # sin trámite de citas/reinscripción no hay falla
        if problema == "Mala adaptación al celular":
            posible &= usa_celular
        problemas[problema] = posible & (rng.random(n) < np.nan_to_num(prob))
    problemas = pd.DataFrame(problemas)
    df["problemas_tipos"] = [", ".join(p for p, h in zip(PROBLEMAS, fila) if h) or SIN_PROBLEMAS
                             for fila in problemas[PROBLEMAS].to_numpy()]

    # Éxito en la reinscripción: solo quien realizó el trámite
    reins = likert["sat_reinscripcion"].to_numpy() - 3
    z = reins + 0.4 * theta - 0.6 * problemas["Fallas al seleccionar materias"].to_numpy() + rng.normal(0, 0.6, n)
    exito = np.array(_ordinal(z, [-1.20, 0.20], REINSCRIPCION), dtype=object)
    exito[np.isnan(reins)] = None
    df["reinscripcion_exito"] = exito

    # Acudir presencialmente: más probable con baja satisfacción, reinscripción fallida o datos erróneos
    p_presencial = _sigmoide(-0.9 - 0.9 * theta
                             + 1.8 * (exito == REINSCRIPCION[0])
                             + 0.6 * problemas["Información incorrecta o desactualizada"].to_numpy())
    df["acudio_presencial"] = np.where(rng.random(n) < p_presencial, "Sí", "No")

    # Mejora prioritaria: el aspecto peor evaluado por el estudiante (con algo de azar)
    puntajes = pd.DataFrame({
        "Diseño y facilidad de uso": likert[["sat_diseno", "sat_facilidad"]].mean(axis=1),
        "Estabilidad y velocidad": likert[["sat_rendimiento", "sat_disponibilidad"]].mean(axis=1) - 0.4,
        "Proceso de reinscripción": likert["sat_reinscripcion"] - 0.2,
        "Exactitud de la información": likert[["sat_info", "sat_calificaciones"]].mean(axis=1) + 0.3,
        "Agenda de citas": likert["sat_citas"] + 0.3,
        "Seguridad y acceso": likert["sat_seguridad"] + 0.6,
        "Versión para celular": np.where(problemas["Mala adaptación al celular"], 1.8, 6.0),
    })[MEJORAS].fillna(9.0)
    ruido = rng.gumbel(0, 0.6, puntajes.shape)
    df["mejora_prioritaria"] = np.array(MEJORAS)[np.argmin(puntajes.to_numpy() - ruido, axis=1)]

    return df[NOMBRES]


def indice_satisfecho(df: pd.DataFrame) -> pd.Series:
    """Índice Likert prorrateado (promedio de ítems respondidos) > 3."""
    return df[list(ITEMS)].astype(float).mean(axis=1) > 3


def cumple_objetivos(df: pd.DataFrame) -> bool:
    sat = indice_satisfecho(df)
    conteos = sat.groupby(df["bloque"]).sum()
    concordancia = (sat == (df["satisfecho_binario"] == "Sí")).mean()
    return (all(conteos[e] == v for e, v in OBJETIVO_SATISFECHOS.items())
            and alfa_cronbach(df[list(ITEMS)]) >= ALFA_MINIMO
            and CONCORDANCIA_BINARIA[0] <= concordancia <= CONCORDANCIA_BINARIA[1])


def validar(df: pd.DataFrame, original: pd.DataFrame) -> None:
    print("=== Validación de la muestra simulada ===")
    print("Filas:", len(df), "| por estrato:", df["bloque"].value_counts().to_dict())
    assert df["bloque"].value_counts().to_dict() == {ESTRATO_1: 70, ESTRATO_2: 57}
    assert (df["num_control"] == original["num_control"]).all()

    # Categorías válidas (las mismas del Forms)
    simples = {"frecuencia_uso": FRECUENCIA, "dispositivo": DISPOSITIVO, "satisfecho_binario": ["Sí", "No"],
               "evolucion": EVOLUCION, "freq_caidas": CAIDAS, "freq_inconsistencias": INCONSISTENCIAS,
               "reinscripcion_exito": REINSCRIPCION, "acudio_presencial": ["Sí", "No"],
               "mejora_prioritaria": MEJORAS}
    for col, cats in simples.items():
        assert set(df[col].dropna()) <= set(cats), col
    assert set(df["tramites"].str.get_dummies(sep=", ").columns) <= set(TRAMITES)
    assert set(df["problemas_tipos"].str.get_dummies(sep=", ").columns) <= set(PROBLEMAS + [SIN_PROBLEMAS])
    print("Categorías: todas pertenecen a las opciones del Forms ✔")

    # Coherencia de nulos con los trámites
    for item, tramite in [("sat_calificaciones", "Consultar calificaciones"),
                          ("sat_reinscripcion", "Reinscripción / selección de materias"),
                          ("sat_citas", "Agendar citas en departamentos")]:
        assert (df[item].notna() == df["tramites"].str.contains(tramite, regex=False)).all(), item
    assert (df["reinscripcion_exito"].notna() == df["sat_reinscripcion"].notna()).all()
    sin = df["problemas_tipos"].str.contains(SIN_PROBLEMAS)
    assert (df.loc[sin, "problemas_tipos"] == SIN_PROBLEMAS).all()
    print("Nulos coherentes con los trámites; «No he tenido problemas» es exclusiva ✔")
    print("Nulos por ítem:", df[list(ITEMS)].isna().sum()[lambda s: s > 0].to_dict())

    alfa = alfa_cronbach(df[list(ITEMS)])
    print(f"Alfa de Cronbach (10 ítems, casos completos n={len(df[list(ITEMS)].dropna())}): {alfa:.3f}")
    print(f"Alfa de Cronbach (7 ítems sin nulos, n=127): {alfa_cronbach(df[list(ITEMS)[:7]]):.3f}")

    sat = indice_satisfecho(df)
    resumen = sat.groupby(df["bloque"]).agg(["sum", "size", "mean"])
    print("Índice Likert (promedio > 3) por estrato:\n", resumen.round(4))
    print("satisfecho_binario por estrato:",
          (df["satisfecho_binario"] == "Sí").groupby(df["bloque"]).mean().round(4).to_dict())
    print(f"Concordancia satisfecho_binario vs índice: {(sat == (df['satisfecho_binario'] == 'Sí')).mean():.1%}")
    print("Medias por ítem:", df[list(ITEMS)].astype(float).mean().round(2).to_dict())


def main():
    original = pd.read_csv(ORIGINAL)
    encabezados = list(original.columns)
    base = original.copy()
    base.columns = NOMBRES

    for semilla in range(1, 20_000):
        simulada = generar(semilla, base)
        if cumple_objetivos(simulada):
            break
    else:
        raise RuntimeError("Ninguna semilla cumplió los objetivos; ajustar parámetros.")

    print(f"Semilla seleccionada: {semilla}")
    validar(simulada, base)

    salida = simulada.copy()
    salida.columns = encabezados
    salida.to_csv(SALIDA, index=False, encoding="utf-8")
    print(f"\nArchivo generado: {SALIDA}")


if __name__ == "__main__":
    main()
