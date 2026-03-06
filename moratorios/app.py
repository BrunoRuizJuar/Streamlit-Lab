# streamlit run app.py

from pathlib import Path
from datetime import date
import calendar

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Moratorios con UDI y CCP",
    page_icon="💸",
    layout="wide"
)

BASE_DIR = Path(__file__).parent
UDI_FILE = BASE_DIR / "Diaria.csv"
CCP_FILE = BASE_DIR / "mensual.xlsx"


@st.cache_data
def load_udi_data(path: Path) -> pd.DataFrame:
    """
    Carga archivo de UDI diaria.
    Intenta detectar automáticamente columnas de fecha y valor.
    """
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, encoding="latin1")
    else:
        df = pd.read_excel(path)

    df.columns = [str(c).strip() for c in df.columns]

    date_col = None
    value_col = None

    for c in df.columns:
        c_low = c.lower()
        if date_col is None and ("fecha" in c_low or "date" in c_low):
            date_col = c
        if value_col is None and ("udi" in c_low or "valor" in c_low):
            value_col = c

    if date_col is None:
        date_col = df.columns[0]
    if value_col is None:
        value_col = df.columns[1]

    df = df[[date_col, value_col]].copy()
    df.columns = ["fecha", "udi"]

    df["fecha"] = pd.to_datetime(df["fecha"], dayfirst=True, errors="coerce")
    df["udi"] = pd.to_numeric(df["udi"], errors="coerce")

    df = df.dropna(subset=["fecha", "udi"]).sort_values("fecha").reset_index(drop=True)
    df["fecha"] = df["fecha"].dt.date
    return df


@st.cache_data
def load_ccp_data(path: Path) -> pd.DataFrame:
    """
    Carga archivo mensual de CCP.
    Intenta detectar automáticamente columnas de fecha/mes y valor CCP.
    """
    if path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, encoding="latin1")

    df.columns = [str(c).strip() for c in df.columns]

    date_col = None
    value_col = None

    for c in df.columns:
        c_low = c.lower()
        if date_col is None and ("fecha" in c_low or "mes" in c_low or "date" in c_low):
            date_col = c
        if value_col is None and ("ccp" in c_low or "valor" in c_low or "tasa" in c_low):
            value_col = c

    if date_col is None:
        date_col = df.columns[0]
    if value_col is None:
        value_col = df.columns[1]

    df = df[[date_col, value_col]].copy()
    df.columns = ["fecha", "ccp"]

    df["fecha"] = pd.to_datetime(df["fecha"], dayfirst=True, errors="coerce")
    df["ccp"] = pd.to_numeric(df["ccp"], errors="coerce")

    df = df.dropna(subset=["fecha", "ccp"]).sort_values("fecha").reset_index(drop=True)
    df["anio"] = df["fecha"].dt.year
    df["mes"] = df["fecha"].dt.month

    return df


def get_udi_value(udi_df: pd.DataFrame, target_date: date) -> float:
    """
    Busca UDI exacta; si no existe, usa el último valor anterior.
    """
    eligible = udi_df[udi_df["fecha"] <= target_date]
    if eligible.empty:
        raise ValueError(f"No hay UDI disponible en o antes de {target_date}")
    return float(eligible.iloc[-1]["udi"])


def month_range(start_date: date, end_date: date):
    """
    Genera (anio, mes) entre start_date y end_date, inclusive.
    """
    y, m = start_date.year, start_date.month
    while (y, m) <= (end_date.year, end_date.month):
        yield y, m
        if m == 12:
            y += 1
            m = 1
        else:
            m += 1


def get_days_in_month_segment(start_date: date, end_date: date, year: int, month: int) -> int:
    """
    Regresa cuántos días de [start_date, end_date] caen dentro de ese mes.
    """
    first_day = date(year, month, 1)
    last_day = date(year, month, calendar.monthrange(year, month)[1])

    segment_start = max(start_date, first_day)
    segment_end = min(end_date, last_day)

    if segment_start > segment_end:
        return 0

    return (segment_end - segment_start).days + 1


def get_ccp_for_month(ccp_df: pd.DataFrame, year: int, month: int) -> float:
    """
    Obtiene el CCP del mes indicado.
    Si no existe exacto, usa el último valor anterior disponible.
    """
    exact = ccp_df[(ccp_df["anio"] == year) & (ccp_df["mes"] == month)]
    if not exact.empty:
        return float(exact.iloc[-1]["ccp"])

    ref_date = pd.Timestamp(year=year, month=month, day=1)
    eligible = ccp_df[ccp_df["fecha"] <= ref_date]
    if eligible.empty:
        raise ValueError(f"No hay CCP disponible para {month:02d}/{year} ni meses anteriores.")
    return float(eligible.iloc[-1]["ccp"])


def calculate_pipeline(
    deuda_inicial: float,
    fecha_inicial: date,
    fecha_final: date,
    udi_df: pd.DataFrame,
    ccp_df: pd.DataFrame
):
    if fecha_final < fecha_inicial:
        raise ValueError("La fecha final no puede ser anterior a la fecha inicial.")
    if deuda_inicial <= 0:
        raise ValueError("La deuda inicial debe ser mayor a 0.")

    valor_udi_inicial = get_udi_value(udi_df, fecha_inicial)
    valor_udi_final = get_udi_value(udi_df, fecha_final)

    detalle_meses = []
    castigo_total = 0.0

    for anio, mes in month_range(fecha_inicial, fecha_final):
        ccp = get_ccp_for_month(ccp_df, anio, mes)
        dias_mes_tramo = get_days_in_month_segment(fecha_inicial, fecha_final, anio, mes)

        castigo_mes = ((((ccp / 100.0) * 1.25) / 365.0) * dias_mes_tramo) * valor_udi_final
        castigo_total += castigo_mes

        detalle_meses.append({
            "Año": anio,
            "Mes": mes,
            "CCP": ccp,
            "Días del tramo": dias_mes_tramo,
            "Factor diario": (((ccp / 100.0) * 1.25) / 365.0),
            "Valor UDI final": valor_udi_final,
            "Castigo del mes": castigo_mes,
        })

    # Conversión de pesos a UDIS
    udis_inicial = deuda_inicial / valor_udi_inicial

    # Actualización en pesos
    actualizacion = udis_inicial * valor_udi_final

    # Deuda final
    deuda_final = actualizacion + castigo_total

    return {
        "valor_udi_inicial": valor_udi_inicial,
        "valor_udi_final": valor_udi_final,
        "udis_inicial": udis_inicial,
        "actualizacion": actualizacion,
        "castigo_total": castigo_total,
        "deuda_final": deuda_final,
        "detalle_meses": pd.DataFrame(detalle_meses),
    }


st.title("💸 Calculadora de Moratorios")
st.caption("Demo en Streamlit usando UDI diaria + CCP mensual")

with st.sidebar:
    st.header("Archivos")
    st.write(f"**UDI:** `{UDI_FILE.name}`")
    st.write(f"**CCP:** `{CCP_FILE.name}`")

    if not UDI_FILE.exists():
        st.error(f"No se encontró {UDI_FILE.name}")
    if not CCP_FILE.exists():
        st.error(f"No se encontró {CCP_FILE.name}")

if not UDI_FILE.exists() or not CCP_FILE.exists():
    st.stop()

try:
    udi_df = load_udi_data(UDI_FILE)
    ccp_df = load_ccp_data(CCP_FILE)
except Exception as e:
    st.error(f"Error cargando archivos: {e}")
    st.stop()

with st.expander("Ver muestra de datos cargados"):
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("UDI diaria")
        st.dataframe(udi_df.head(10), use_container_width=True)
    with c2:
        st.subheader("CCP mensual")
        st.dataframe(ccp_df.head(10), use_container_width=True)

st.subheader("Entradas")

col1, col2, col3 = st.columns(3)

with col1:
    fecha_inicial = st.date_input("Fecha inicial", value=date(2025, 1, 1))

with col2:
    fecha_final = st.date_input("Fecha final", value=date(2025, 3, 15))

with col3:
    deuda_inicial = st.number_input(
        "Deuda inicial (MXN)",
        min_value=0.01,
        value=100000.00,
        step=1000.00,
        format="%.2f"
    )

if st.button("Calcular", type="primary"):
    try:
        result = calculate_pipeline(
            deuda_inicial=deuda_inicial,
            fecha_inicial=fecha_inicial,
            fecha_final=fecha_final,
            udi_df=udi_df,
            ccp_df=ccp_df
        )

        st.success("Cálculo realizado correctamente.")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("UDI inicial", f"{result['valor_udi_inicial']:,.6f}")
        m2.metric("UDI final", f"{result['valor_udi_final']:,.6f}")
        m3.metric("Castigo", f"${result['castigo_total']:,.2f}")
        m4.metric("Deuda final", f"${result['deuda_final']:,.2f}")

        st.subheader("Resumen del cálculo")

        resumen = pd.DataFrame([
            {"Concepto": "Deuda inicial", "Monto": deuda_inicial},
            {"Concepto": "UDIS inicial", "Monto": result["udis_inicial"]},
            {"Concepto": "Actualización", "Monto": result["actualizacion"]},
            {"Concepto": "Castigo", "Monto": result["castigo_total"]},
            {"Concepto": "Deuda final", "Monto": result["deuda_final"]},
        ])

        st.dataframe(
            resumen.style.format({"Monto": "{:,.6f}"}),
            use_container_width=True
        )

        st.subheader("Detalle mensual del castigo")
        st.dataframe(
            result["detalle_meses"].style.format({
                "CCP": "{:,.6f}",
                "Factor diario": "{:,.10f}",
                "Valor UDI final": "{:,.6f}",
                "Castigo del mes": "${:,.6f}",
            }),
            use_container_width=True
        )

        with st.expander("Fórmulas usadas"):
            st.markdown(
                    """
            **Castigo por mes**

                (((((CCP / 100) * 1.25) / 365) * días_del_tramo_en_ese_mes) * valor_UDI_final)

            **UDIS inicial**

                deuda_inicial / valor_UDI_inicial

            **Actualización**

                UDIS_inicial * valor_UDI_final

            **Deuda final**

                actualización + castigo
                            """
            )

    except Exception as e:
        st.error(f"Error en el cálculo: {e}")

    with st.expander("Fórmulas usadas"):
        st.markdown("**Castigo por mes**")
        st.code("(((((CCP / 100) * 1.25) / 365) * días_del_tramo_en_ese_mes) * valor_UDI_final)", language="text")

        st.markdown("**UDIS inicial**")
        st.code("deuda_inicial / valor_UDI_inicial", language="text")

        st.markdown("**Actualización**")
        st.code("UDIS_inicial * valor_UDI_final", language="text")

        st.markdown("**Deuda final**")
        st.code("actualización + castigo", language="text")