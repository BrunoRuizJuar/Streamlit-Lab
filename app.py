import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from buffon import buffon_simulation
from ga_basic import GA

# ======================
#  TÍTULO PRINCIPAL
# ======================
st.title("🎯 Monte Carlo & Algoritmos Genéticos")
st.write("Esta app incluye: Monte Carlo para π, el método de Buffon y un mini Algoritmo Genético.")

# ======================
#  MENÚ LATERAL
# ======================
st.sidebar.header("Navegación")
section = st.sidebar.radio("Elegir demo:", ["Monte Carlo π", "Buffon", "Algoritmo Genético"])

# ============================================================
#  SECCIÓN 1: MONTE CARLO π
# ============================================================
if section == "Monte Carlo π":
    st.header("🔵 Aproximación de π con Monte Carlo")

    st.markdown("### 📘 Fórmula utilizada")
    st.latex(r"""
    \pi \approx 4 \cdot 
    \frac{\text{puntos dentro}}{\text{total de puntos}}
    """)

    st.write("Interpretación:")
    st.write("- El cuadrado tiene área 4.")
    st.write("- El círculo unitario tiene área π.")
    st.write("- La proporción entre ambas sirve para estimar π usando puntos aleatorios.")

    num_points = st.sidebar.slider("Número de puntos", 100, 50000, 3000)

    x = np.random.uniform(-1, 1, num_points)
    y = np.random.uniform(-1, 1, num_points)
    inside = x**2 + y**2 <= 1

    pi_est = 4 * np.mean(inside)
    st.metric("Estimación de π", f"{pi_est:.6f}")

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.scatter(x[inside], y[inside], s=5, color="dodgerblue")
    ax.scatter(x[~inside], y[~inside], s=5, color="orange")
    circle = plt.Circle((0, 0), 1, fill=False, linewidth=2)
    ax.add_patch(circle)
    st.pyplot(fig)

# ============================================================
#  SECCIÓN 2: MÉTODO DE BUFFON
# ============================================================
elif section == "Buffon":
    st.header("📏 Método de la Aguja de Buffon")

    st.markdown("### 📘 Fórmula de Buffon")
    st.latex(r"""
    P(\text{tocar línea}) = \frac{2L}{\pi D}
    """)
    st.markdown("Despejando π:")
    st.latex(r"""
    \pi \approx \frac{2L}{D \cdot P}
    """)

    st.write("Donde:")
    st.write("- **L** = longitud de la aguja")
    st.write("- **D** = distancia entre líneas paralelas (D > L)")

    num_needles = st.sidebar.slider("Agujas", 1000, 50000, 10000)
    L = st.sidebar.slider("Longitud L", 0.1, 2.5, 1.0)
    D = st.sidebar.slider("Distancia entre líneas D", L + 0.01, 4.0, 2.0)

    # Cálculo numérico
    pi_est, hits = buffon_simulation(num_needles=num_needles, L=L, D=D)
    st.metric("Estimación de π", f"{pi_est:.6f}")
    st.write(f"Aciertos: {hits}/{num_needles}")

    # ---- GRÁFICO DE BUFFON ----
    st.markdown("### 🎨 Visualización del experimento")

    from buffon import buffon_visual
    x1, y1, x2, y2, hit_mask, max_x, max_y = buffon_visual(
        num_needles=200, 
        L=L, 
        D=D, 
        seed=0
    )

    fig, ax = plt.subplots(figsize=(6, 6))

    # Dibujar líneas horizontales bien visibles
    for y in np.arange(0, max_y + D, D):
        ax.axhline(y=y, color="black", linewidth=1.2)

    # Agujas que NO tocan (color oro)
    for i in np.where(~hit_mask)[0]:
        ax.plot([x1[i], x2[i]], [y1[i], y2[i]], color="goldenrod", alpha=0.8)

    # Agujas que SÍ tocan (color rojo)
    for i in np.where(hit_mask)[0]:
        ax.plot([x1[i], x2[i]], [y1[i], y2[i]], color="red", linewidth=2)

    ax.set_title("Simulación visual de la aguja de Buffon")
    ax.set_xlim(0, max_x)
    ax.set_ylim(0, max_y)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    st.pyplot(fig)

# ============================================================
#  SECCIÓN 3: MINI ALGORITMO GENÉTICO
# ============================================================
elif section == "Algoritmo Genético":
    st.header("🧬 Mini Algoritmo Genético")

    st.markdown("### 📘 Función objetivo a maximizar")
    st.latex(r"""
    f(x) = - \sum_{i=1}^{n} (x_i - 0.5)^2
    """)
    st.write("La función alcanza su máximo cuando todos los genes son iguales a **0.5**.")

    pop_size = st.sidebar.slider("Población", 10, 200, 30)
    num_genes = st.sidebar.slider("Genes", 2, 50, 10)
    generations = st.sidebar.slider("Generaciones", 10, 200, 50)

    def fitness(ind):
        return -np.sum((ind - 0.5) ** 2)

    ga = GA(
        fitness_fn=fitness,
        pop_size=pop_size,
        num_genes=num_genes,
        generations=generations
    )

    best, history = ga.run()
    st.metric("Mejor fitness", f"{fitness(best):.4f}")

    fig, ax = plt.subplots()
    ax.plot(history)
    ax.set_title("Evolución del Fitness")
    ax.set_xlabel("Generación")
    ax.set_ylabel("Fitness")
    st.pyplot(fig)
