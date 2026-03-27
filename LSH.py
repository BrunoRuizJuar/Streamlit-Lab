import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer


st.set_page_config(page_title="Demo LSH: clasificación de palabras", layout="wide")


# =============================
# Datos
# =============================

DATASET: Dict[str, List[str]] = {
    "Comida": [
        "pizza", "taco", "pan", "queso", "manzana", "pera", "uva", "sopa",
        "ensalada", "arroz", "fruta", "verdura", "sandwich", "hamburguesa",
        "pasta", "leche", "yogur", "galleta", "chocolate", "cafe"
    ],
    "Transporte": [
        "coche", "auto", "camion", "autobus", "metro", "tren", "tranvia",
        "bicicleta", "moto", "avion", "barco", "taxi", "camioneta",
        "patineta", "helicoptero", "subway", "ferrocarril", "lancha", "bus", "uber"
    ],
    "Animal": [
        "perro", "gato", "caballo", "vaca", "oveja", "cerdo", "leon", "tigre",
        "elefante", "jirafa", "mono", "conejo", "zorro", "lobo", "pez", "delfin",
        "aguila", "pajaro", "serpiente", "tortuga"
    ],
}


# =============================
# Utilidades de texto y vectores
# =============================

def normalize_text(text: str) -> str:
    return text.strip().lower()


def flatten_dataset(dataset: Dict[str, List[str]]) -> Tuple[List[str], List[str]]:
    words, labels = [], []
    for label, vocab in dataset.items():
        for word in vocab:
            words.append(normalize_text(word))
            labels.append(label)
    return words, labels


# =============================
# LSH con hyperplanes aleatorios
# =============================

@dataclass
class HyperplaneHash:
    normal: np.ndarray

    def bit(self, x: np.ndarray) -> int:
        return 1 if float(np.dot(self.normal, x)) >= 0 else 0


class ANDHash:
    def __init__(self, hyperplanes: List[HyperplaneHash]):
        self.hyperplanes = hyperplanes

    def signature(self, x: np.ndarray) -> Tuple[int, ...]:
        return tuple(h.bit(x) for h in self.hyperplanes)


class LSHIndex:
    def __init__(self, groups: List[ANDHash]):
        self.groups = groups
        self.tables = [dict() for _ in groups]

    def fit(self, vectors: np.ndarray):
        for table in self.tables:
            table.clear()
        for idx, x in enumerate(vectors):
            for table, group in zip(self.tables, self.groups):
                sig = group.signature(x)
                table.setdefault(sig, []).append(idx)

    def query(self, q: np.ndarray) -> Tuple[List[int], List[Tuple[int, ...]]]:
        candidates = set()
        signatures = []
        for table, group in zip(self.tables, self.groups):
            sig = group.signature(q)
            signatures.append(sig)
            candidates.update(table.get(sig, []))
        return sorted(candidates), signatures


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0:
        return 0.0
    return float(np.dot(x, y) / denom)


def random_hyperplane(dim: int, rng: np.random.Generator) -> HyperplaneHash:
    normal = rng.normal(size=dim)
    norm = np.linalg.norm(normal)
    if norm == 0:
        normal[0] = 1.0
        norm = 1.0
    return HyperplaneHash(normal=normal / norm)


def build_lsh(dim: int, k: int, L: int, seed: int) -> List[ANDHash]:
    rng = np.random.default_rng(seed)
    groups = []
    for _ in range(L):
        planes = [random_hyperplane(dim, rng) for _ in range(k)]
        groups.append(ANDHash(planes))
    return groups


# =============================
# Modelo de demostración
# =============================

def build_vectorizer(words: List[str]) -> TfidfVectorizer:
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))


def classify_with_lsh(
    query_word: str,
    words: List[str],
    labels: List[str],
    vectors: np.ndarray,
    vectorizer: TfidfVectorizer,
    index: LSHIndex,
    top_n: int = 5,
):
    q_vec = vectorizer.transform([normalize_text(query_word)]).toarray()[0]
    candidates, signatures = index.query(q_vec)

    if len(candidates) == 0:
        # fallback: si no hay candidatos, comparamos con todo el dataset
        candidate_ids = list(range(len(words)))
        used_fallback = True
    else:
        candidate_ids = candidates
        used_fallback = False

    rows = []
    for idx in candidate_ids:
        sim = cosine_similarity(q_vec, vectors[idx])
        rows.append((idx, words[idx], labels[idx], sim))

    df = pd.DataFrame(rows, columns=["idx", "palabra", "clase", "sim_coseno"])
    df = df.sort_values("sim_coseno", ascending=False).reset_index(drop=True)
    top_df = df.head(top_n).copy()

    if len(top_df) == 0:
        predicted_label = "Sin clasificación"
        score_by_class = pd.DataFrame(columns=["clase", "score_total", "vecinos"])
    else:
        score_by_class = (
            top_df.groupby("clase", as_index=False)
            .agg(score_total=("sim_coseno", "sum"), vecinos=("palabra", "count"))
            .sort_values(["score_total", "vecinos"], ascending=False)
            .reset_index(drop=True)
        )
        predicted_label = str(score_by_class.iloc[0]["clase"])

    return q_vec, candidates, signatures, top_df, score_by_class, predicted_label, used_fallback


# =============================
# Visualización
# =============================

def plot_class_distribution(labels: List[str]):
    counts = pd.Series(labels).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index, counts.values)
    ax.set_title("Número de palabras por clase")
    ax.set_xlabel("Clase")
    ax.set_ylabel("Cantidad")
    ax.grid(axis="y", alpha=0.3)
    return fig


def plot_probability_curves(k: int, L: int):
    p = np.linspace(0, 1, 300)
    and_curve = p ** k
    or_curve = 1 - (1 - p ** k) ** L
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(p, and_curve, label=rf"AND: $p^k$, k={k}")
    ax.plot(p, or_curve, label=rf"AND+OR: $1-(1-p^k)^L$, L={L}")
    ax.set_title("Amplificación de probabilidades")
    ax.set_xlabel("Probabilidad base p")
    ax.set_ylabel("Probabilidad amplificada")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


def plot_similarity_scores(top_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    if len(top_df) == 0:
        ax.text(0.5, 0.5, "No hay vecinos", ha="center", va="center")
        ax.set_axis_off()
        return fig
    labels_plot = [f"{w}\n({c})" for w, c in zip(top_df["palabra"], top_df["clase"])]
    ax.barh(labels_plot[::-1], top_df["sim_coseno"].values[::-1])
    ax.set_title("Vecinos más similares dentro de candidatos LSH")
    ax.set_xlabel("Similitud coseno")
    ax.grid(axis="x", alpha=0.3)
    return fig


# =============================
# App
# =============================

st.title("Clasificar una palabra con LSH")
st.markdown(
    """
La idea de esta demo es usar **Locality-Sensitive Hashing** para reducir el espacio de búsqueda.
Tenemos tres clases:

- **Comida**
- **Transporte**
- **Animal**

El usuario escribe una palabra, la convertimos en un vector de caracteres con **TF-IDF de n-gramas**,
la pasamos por varias tablas LSH, recuperamos candidatos y luego decidimos la clase usando los vecinos más similares.

> Importante: esto es una **demostración didáctica**. Para un clasificador semántico real,
> normalmente convendría usar embeddings más potentes.
"""
)

with st.sidebar:
    st.header("Parámetros")
    k = st.slider("k (hashes por AND)", min_value=1, max_value=10, value=4, step=1)
    L = st.slider("L (tablas OR)", min_value=1, max_value=12, value=5, step=1)
    seed_hash = st.number_input("Semilla LSH", min_value=0, max_value=9999, value=17, step=1)
    top_n = st.slider("Vecinos usados para decidir clase", min_value=1, max_value=10, value=5, step=1)
    st.divider()
    query_word = st.text_input("Palabra a clasificar", value="bicicleta")

words, labels = flatten_dataset(DATASET)
vectorizer = build_vectorizer(words)
X = vectorizer.fit_transform(words).toarray()
dim = X.shape[1]

groups = build_lsh(dim=dim, k=k, L=L, seed=int(seed_hash))
index = LSHIndex(groups)
index.fit(X)

q_vec, candidates, signatures, top_df, score_by_class, predicted_label, used_fallback = classify_with_lsh(
    query_word=query_word,
    words=words,
    labels=labels,
    vectors=X,
    vectorizer=vectorizer,
    index=index,
    top_n=top_n,
)

col1, col2 = st.columns([1.15, 1])

with col1:
    st.subheader("Resultado")
    st.markdown(f"### La palabra **{normalize_text(query_word)}** cae en: **{predicted_label}**")

    if used_fallback:
        st.warning("No hubo colisiones LSH para esa palabra. Se usó comparación contra todo el dataset como respaldo.")
    else:
        st.success("Sí hubo candidatos recuperados mediante LSH.")

    st.write("**Firmas de la query por tabla OR:**")
    sig_df = pd.DataFrame({
        "tabla": list(range(1, L + 1)),
        "firma": [str(sig) for sig in signatures],
    })
    st.dataframe(sig_df, use_container_width=True, hide_index=True)

    st.write("**Puntaje agregado por clase:**")
    st.dataframe(score_by_class, use_container_width=True, hide_index=True)

    st.metric("Candidatos LSH", len(candidates))
    st.metric("Tamaño del vocabulario", len(words))
    if len(words) > 0:
        reduction = 1.0 - len(candidates) / len(words) if len(candidates) > 0 else 0.0
        st.metric("Reducción de búsqueda", f"{100 * reduction:.1f}%")

with col2:
    st.subheader("Cómo se decidió")
    st.markdown(
        f"""
1. La palabra se vectoriza con n-gramas de caracteres.
2. Cada tabla OR usa una firma AND de longitud **k = {k}**.
3. Recuperamos palabras que colisionan en al menos una de las **L = {L}** tablas.
4. Entre esos candidatos, calculamos similitud coseno.
5. La clase final se decide agregando los **top {top_n} vecinos** más similares.
"""
    )
    fig_scores = plot_similarity_scores(top_df)
    st.pyplot(fig_scores)

st.divider()

left, right = st.columns(2)
with left:
    st.subheader("Vecinos más cercanos dentro de los candidatos")
    st.dataframe(top_df, use_container_width=True, hide_index=True)

with right:
    st.subheader("Vocabulario de entrenamiento")
    vocab_df = pd.DataFrame({"palabra": words, "clase": labels})
    st.dataframe(vocab_df, use_container_width=True, hide_index=True, height=320)

st.divider()

c1, c2 = st.columns(2)
with c1:
    st.subheader("Distribución de clases")
    fig_dist = plot_class_distribution(labels)
    st.pyplot(fig_dist)

with c2:
    st.subheader("Efecto de AND y OR")
    fig_prob = plot_probability_curves(k=k, L=L)
    st.pyplot(fig_prob)

st.divider()
st.subheader("Pruebas sugeridas")
st.markdown(
    """
Prueba palabras como:

- **Comida:** `pizza`, `manzana`, `chocolate`
- **Transporte:** `tren`, `moto`, `barco`
- **Animal:** `perro`, `tigre`, `delfin`

También puedes probar palabras no exactas como `camioncito`, `gatito` o `quesadilla`
para ver cómo responde el sistema por similitud de forma.
"""
)

st.caption("Demo didáctica: LSH reduce candidatos; la clase se decide usando similitud sobre esos candidatos.")