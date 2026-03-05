from pathlib import Path
import pandas as pd
import streamlit as st
from datetime import date

st.set_page_config(page_title="Moratorios (UDI) - Demo", layout="centered")

CSV_NAME = "Consulta_20260305-085312239.csv"  # tu archivo exacto
CSV_PATH = Path(__file__).parent / CSV_NAME  # MISMA carpeta que app.py

@st.cache_data
def load_udis(csv_path: Path) -> pd.DataFrame:
    # Este CSV de Banxico trae metadata y luego datos
    df = pd.read_csv(
        csv_path,
        encoding="latin1",
        sep=",",
        skiprows=19,          # datos empiezan en 04/04/1995...
        header=None,
        names=["date", "udi"]
    )
    df["date"] = pd.to_datetime(df["date"], dayfirst=True).dt.date
    df = df.sort_values("date").reset_index(drop=True)
    return df

def get_udi(df: pd.DataFrame, d: date) -> float:
    eligible = df[df["date"] <= d]
    if eligible.empty:
        raise ValueError(f"No hay UDI disponible en o antes de {d}")
    return float(eligible.iloc[-1]["udi"])

def calculate(principal: float, start: date, end: date, annual_rate: float, udi_start: float, udi_end: float):
    days = (end - start).days
    if days < 0:
        raise ValueError("La fecha de pago no puede ser anterior a la fecha de inicio.")
    factor = udi_end / udi_start
    updated_principal = principal * factor
    actualizacion = updated_principal - principal
    intereses = updated_principal * (annual_rate / 365.0) * days
    total = principal + actualizacion + intereses
    return days, factor, actualizacion, intereses, total

st.title("Indemnización por Mora (Seguros) — Demo con UDI")
st.caption("Total = Principal + Actualización (UDI) + Intereses moratorios")

# Cargar CSV
if not CSV_PATH.exists():
    st.error(f"No encuentro el CSV en: {CSV_PATH}. Súbelo al repo en la carpeta 'moratorios/'.")
    st.stop()

df = load_udis(CSV_PATH)
st.success(f"UDIs cargadas: {len(df):,} filas")

c1, c2 = st.columns(2)
with c1:
    start = st.date_input("Fecha inicio", value=date(2025, 1, 1))
with c2:
    end = st.date_input("Fecha pago", value=date(2025, 3, 1))

principal = st.number_input("Obligación principal (MXN)", min_value=0.0, value=100000.0, step=1000.0)
annual_rate = st.number_input("Tasa anual (ej. 0.30 = 30%)", min_value=0.0, value=0.30, step=0.01)

if st.button("Calcular"):
    udi_start = get_udi(df, start)
    udi_end = get_udi(df, end)
    days, factor, actualizacion, intereses, total = calculate(principal, start, end, annual_rate, udi_start, udi_end)

    st.subheader("Resultado")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Principal", f"${principal:,.2f}")
    m2.metric("Actualización", f"${actualizacion:,.2f}")
    m3.metric("Intereses", f"${intereses:,.2f}")
    m4.metric("Total", f"${total:,.2f}")

    st.subheader("Detalles UDI")
    st.write(f"UDI inicio ({start}): **{udi_start}**")
    st.write(f"UDI pago ({end}): **{udi_end}**")
    st.write(f"Factor: **{factor:.8f}**  |  Días: **{days}**")