# app.py
# Ejecuta con:
#   streamlit run app.py

import pandas as pd
import streamlit as st
from datetime import date

CSV_PATH = "Consulta_20260305-085312239.csv"


@st.cache_data(show_spinner=False)
def load_udi_csv(path: str) -> pd.DataFrame:
    """
    Carga el CSV de Banxico donde los datos reales inician en la línea 19 (0-indexed),
    con encoding latin1 y sin header en los datos.
    """
    df = pd.read_csv(
        path,
        encoding="latin1",
        sep=",",
        skiprows=19,
        header=None,
        names=["date", "udi"],
    )

    # Parseo robusto de fecha (DD/MM/YYYY)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["udi"] = pd.to_numeric(df["udi"], errors="coerce")

    df = df.dropna(subset=["date", "udi"]).copy()
    df = df.sort_values("date").reset_index(drop=True)

    # Normalizamos a "date" (sin hora) para comparar con inputs de Streamlit
    df["date"] = df["date"].dt.date

    return df


def get_udi(df: pd.DataFrame, fecha: date) -> float:
    """
    Regresa el UDI de la fecha exacta; si no existe, regresa el último valor anterior disponible
    (equivalente a "asof"/forward fill hacia atrás).

    Lanza ValueError si no hay UDI anterior disponible.
    """
    if df.empty:
        raise ValueError("El dataset UDI está vacío o no se pudo cargar.")

    # Asumimos df ordenado ascendente por date
    fechas = df["date"].tolist()

    # Búsqueda binaria manual con pandas (sin numpy explícito)
    # Convertimos a Series para usar searchsorted.
    s = pd.Series(fechas)
    pos = s.searchsorted(fecha, side="right") - 1  # índice del último <= fecha

    if pos < 0:
        raise ValueError(
            f"No hay UDI anterior disponible para {fecha.strftime('%d/%m/%Y')} "
            f"(la primera fecha disponible es {fechas[0].strftime('%d/%m/%Y')})."
        )

    return float(df.loc[int(pos), "udi"])


def money(x: float) -> str:
    return f"${x:,.2f} MXN"


st.set_page_config(
    page_title="Demo: Indemnización por mora (Seguros)",
    page_icon="🧮",
    layout="centered",
)

st.title("🧮 DEMO — Indemnización por mora (seguros)")
st.caption(
    "Calcula **obligación principal + actualización (UDI) + intereses moratorios**. "
    "No se guarda nada; todo se calcula al vuelo."
)

with st.sidebar:
    st.subheader("📄 Dataset UDI")
    st.write("Ruta fija del CSV (según requerimiento):")
    st.code(CSV_PATH, language="text")
    st.write(
        "El parseo asume encabezados Banxico y que los datos reales empiezan en la línea 19 (0-indexed)."
    )

# Inputs
col1, col2 = st.columns(2)
with col1:
    fecha_inicio = st.date_input("Fecha de inicio", value=date(2024, 1, 1))
with col2:
    fecha_pago = st.date_input("Fecha de pago", value=date(2024, 2, 1))

principal = st.number_input(
    "Monto principal (MXN)",
    min_value=0.0,
    value=10000.0,
    step=100.0,
    format="%.2f",
)

tasa_anual = st.number_input(
    "Tasa anual moratoria (ej. 0.30 = 30%)",
    min_value=0.0,
    value=0.30,
    step=0.01,
    format="%.4f",
)

calcular = st.button("Calcular", type="primary")

# Validaciones de entrada
errors = []
if fecha_pago < fecha_inicio:
    errors.append("❌ **fecha_pago** debe ser mayor o igual a **fecha_inicio**.")
if principal <= 0:
    errors.append("❌ **principal** debe ser mayor a 0.")
if tasa_anual < 0:
    errors.append("❌ **tasa_anual** debe ser mayor o igual a 0.")

if calcular:
    if errors:
        for e in errors:
            st.error(e)
        st.stop()

    # Cargar UDI
    try:
        df_udi = load_udi_csv(CSV_PATH)
    except FileNotFoundError:
        st.error(f"❌ No se encontró el archivo CSV en la ruta: {CSV_PATH}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error cargando el CSV UDI: {e}")
        st.stop()

    # Obtener UDI inicio y pago (exacto o último anterior)
    try:
        udi_inicio = get_udi(df_udi, fecha_inicio)
        udi_pago = get_udi(df_udi, fecha_pago)
    except ValueError as e:
        st.error(f"❌ {e}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error obteniendo UDI: {e}")
        st.stop()

    # Reglas de cálculo (demo)
    dias = (fecha_pago - fecha_inicio).days
    factor_udi = udi_pago / udi_inicio
    principal_actualizado = principal * factor_udi
    actualizacion = principal_actualizado - principal

    intereses = principal_actualizado * (tasa_anual / 365.0) * dias
    total = principal + actualizacion + intereses

    # Outputs: métricas
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Principal", money(principal))
    m2.metric("Actualización", money(actualizacion))
    m3.metric("Intereses", money(intereses))
    m4.metric("Total", money(total))

    # Tabla desglose
    st.subheader("📌 Desglose")
    df_breakdown = pd.DataFrame(
        [
            {"concepto": "Obligación principal", "monto": principal},
            {"concepto": "Actualización (UDI)", "monto": actualizacion},
            {"concepto": "Intereses moratorios", "monto": intereses},
            {"concepto": "TOTAL A PAGAR", "monto": total},
        ]
    )
    df_breakdown_display = df_breakdown.copy()
    df_breakdown_display["monto"] = df_breakdown_display["monto"].map(money)
    st.table(df_breakdown_display)

    # Detalles UDI
    st.subheader("📈 Detalles UDI")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("UDI inicio", f"{udi_inicio:.6f}")
    d2.metric("UDI pago", f"{udi_pago:.6f}")
    d3.metric("Factor UDI", f"{factor_udi:.8f}")
    d4.metric("Días", f"{dias}")

    with st.expander("Ver fórmula (demo)", expanded=False):
        st.markdown(
            """
- **factor_udi** = udi(fecha_pago) / udi(fecha_inicio)  
- **principal_actualizado** = principal × factor_udi  
- **actualización** = principal_actualizado − principal  
- **intereses** = principal_actualizado × (tasa_anual / 365) × días  
- **total** = principal + actualización + intereses
            """.strip()
        )
else:
    st.info("Ingresa los datos y presiona **Calcular**.")