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
CCP_FILE = BASE_DIR / "mensual.csv"


def file_version(path: Path) -> float:
    """
    Regresa la fecha de modificación del archivo para invalidar caché cuando cambie.
    """
    return path.stat().st_mtime


@st.cache_data
def load_udi_data(path: Path, version: float) -> pd.DataFrame:
    """
    Carga archivo de UDI diaria exportado desde Banxico.
    """
    with open(path, "r", encoding="latin1") as f:
        lines = f.readlines()

    start_idx = None

    for i, line in enumerate(lines):
        clean = line.strip().replace('"', "")
        if clean.lower().startswith("fecha,"):
            start_idx = i + 1
            break

    if start_idx is None:
        raise ValueError("No se encontró el encabezado real de datos en el archivo UDI.")

    records = []
    for line in lines[start_idx:]:
        clean = line.strip().replace('"', "")

        if not clean:
            continue

        clean = clean.rstrip(";")
        parts = clean.split(",")

        if len(parts) < 2:
            continue

        fecha_txt = parts[0].strip()
        udi_txt = parts[1].strip()

        records.append([fecha_txt, udi_txt])

    if not records:
        raise ValueError("No se encontraron registros de UDI en el archivo.")

    df = pd.DataFrame(records, columns=["fecha", "udi"])

    df["fecha"] = pd.to_datetime(df["fecha"], format="%d/%m/%Y", errors="coerce")
    df["udi"] = pd.to_numeric(df["udi"], errors="coerce")

    df = df.dropna(subset=["fecha", "udi"]).sort_values("fecha").reset_index(drop=True)
    df["fecha"] = df["fecha"].dt.normalize()

    return df


@st.cache_data
def load_ccp_data(path: Path, version: float) -> pd.DataFrame:
    """
    Carga CCP mensual desde CSV Banxico.
    Busca la fila real de encabezado: "Fecha","SF286",...
    y extrae la columna SF286.
    """
    with open(path, "r", encoding="latin1") as f:
        lines = f.readlines()

    start_idx = None

    for i, line in enumerate(lines):
        clean = line.strip().replace('"', "")
        if clean.startswith("Fecha,") and "SF286" in clean:
            start_idx = i + 1
            break

    if start_idx is None:
        raise ValueError("No se encontró el encabezado real de datos en el archivo CCP.")

    records = []
    for line in lines[start_idx:]:
        clean = line.strip().replace('"', "")

        if not clean:
            continue

        parts = [p.strip() for p in clean.split(",")]

        if len(parts) < 2:
            continue

        fecha_txt = parts[0]
        ccp_txt = parts[1]

        records.append([fecha_txt, ccp_txt])

    if not records:
        raise ValueError("No se encontraron registros de CCP en el archivo.")

    df = pd.DataFrame(records, columns=["fecha", "ccp"])

    df["ccp"] = df["ccp"].replace(["N/E", "", "nan", "None"], pd.NA)
    df["ccp"] = pd.to_numeric(df["ccp"], errors="coerce")

    df["fecha"] = pd.to_datetime(df["fecha"], format="%d/%m/%Y", errors="coerce")

    df = df.dropna(subset=["fecha", "ccp"]).sort_values("fecha").reset_index(drop=True)
    df["fecha"] = df["fecha"].dt.normalize()

    df["anio"] = df["fecha"].dt.year
    df["mes"] = df["fecha"].dt.month

    return df


def get_udi_value(udi_df: pd.DataFrame, target_date: date) -> float:
    """
    Busca UDI exacta.
    """
    target_ts = pd.Timestamp(target_date).normalize()
    exact = udi_df.loc[udi_df["fecha"].eq(target_ts), "udi"]

    if exact.empty:
        cercanas = udi_df[
            (udi_df["fecha"] >= target_ts - pd.Timedelta(days=2)) &
            (udi_df["fecha"] <= target_ts + pd.Timedelta(days=2))
        ][["fecha", "udi"]]

        raise ValueError(
            f"No existe UDI exacta para la fecha {target_date}. "
            f"Fechas cercanas encontradas: {cercanas.to_dict(orient='records')}"
        )

    return float(exact.iloc[0])


def month_range(start_date: date, end_date: date):
    y, m = start_date.year, start_date.month
    while (y, m) <= (end_date.year, end_date.month):
        yield y, m
        if m == 12:
            y += 1
            m = 1
        else:
            m += 1


def get_days_in_month_segment(start_date: date, end_date: date, year: int, month: int) -> int:
    first_day = date(year, month, 1)
    last_day = date(year, month, calendar.monthrange(year, month)[1])

    segment_start = max(start_date, first_day)
    segment_end = min(end_date, last_day)

    if segment_start > segment_end:
        return 0

    return (segment_end - segment_start).days + 1


def get_ccp_for_month(ccp_df: pd.DataFrame, year: int, month: int) -> float:
    exact = ccp_df[(ccp_df["anio"] == year) & (ccp_df["mes"] == month)]
    if not exact.empty:
        return float(exact.iloc[-1]["ccp"])

    ref_date = pd.Timestamp(year=year, month=month, day=1).normalize()
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

    udis_inicial = deuda_inicial / valor_udi_inicial

    detalle_meses = []
    castigo_total = 0.0
    

    for anio, mes in month_range(fecha_inicial, fecha_final):
        ccp = get_ccp_for_month(ccp_df, anio, mes)
        dias_mes_tramo = get_days_in_month_segment(fecha_inicial, fecha_final, anio, mes)


        ## AQUI ESTA EL NMALDITO ERRORRR PERO NO SE QUE ESSSSSS
        factor_diario = (((ccp / 100.0) * 1.25) / 365.0)
        castigo_mes = (factor_diario * dias_mes_tramo * udis_inicial) * valor_udi_final
        castigo_total += castigo_mes

        detalle_meses.append({
            "Año": anio,
            "Mes": mes,
            "CCP": ccp,
            "Días del tramo": dias_mes_tramo,
            "Factor diario": factor_diario,
            "UDIS inicial": udis_inicial,
            "Valor UDI inicial": valor_udi_inicial,
            "Valor UDI final": valor_udi_final,
            "Castigo del mes": castigo_mes,
        })

    actualizacion = udis_inicial * valor_udi_final
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
    udi_df = load_udi_data(UDI_FILE, file_version(UDI_FILE))
    ccp_df = load_ccp_data(CCP_FILE, file_version(CCP_FILE))
except Exception as e:
    st.error(f"Error cargando archivos: {e}")
    st.stop()

with st.expander("Ver muestra de datos cargados"):
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("UDI diaria")
        st.dataframe(udi_df.head(10), width="stretch")
    with c2:
        st.subheader("CCP mensual")
        st.dataframe(ccp_df.head(10), width="stretch")

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
            width="stretch"
        )

        st.subheader("Detalle mensual del castigo")
        st.dataframe(
            result["detalle_meses"].style.format({
                "CCP": "{:,.6f}",
                "Factor diario": "{:,.10f}",
                "UDIS inicial": "{:,.6f}",
                "Valor UDI inicial": "{:,.6f}",
                "Valor UDI final": "{:,.6f}",
                "Castigo del mes": "${:,.6f}",
            }),
            width="stretch"
        )

    except Exception as e:
        st.error(f"Error en el cálculo: {e}")