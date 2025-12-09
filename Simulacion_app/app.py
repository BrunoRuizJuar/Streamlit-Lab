# app.py
import math
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ==========================
# Utilidades generales
# ==========================

def gamma_pdf(x, alpha, lam):
    """Densidad Gamma(alpha, lam) en x > 0."""
    return (lam**alpha) * (x**(alpha - 1)) * np.exp(-lam * x) / math.gamma(alpha)

def pareto_pdf(x, alpha, xm):
    """Densidad Pareto(alpha, xm) en x >= xm."""
    out = np.zeros_like(x, dtype=float)
    mask = x >= xm
    out[mask] = alpha * xm**alpha / (x[mask] ** (alpha + 1))
    return out

# =========================================================
# Ejercicio 1: Gamma(alpha, lambda) con alpha entero
# =========================================================

def ejercicio_1():
    st.header("Ejercicio 1: Distribución Gamma con α entero")

    st.markdown(r"""
**Enunciado resumido**

Sea una distribución Gamma con parámetros \((\alpha,\lambda)\) y densidad

\[
f(x)=\frac{\lambda^\alpha x^{\alpha-1} e^{-\lambda x}}{\Gamma(\alpha)},\quad x>0.
\]

1. Explique por qué esta distribución no admite transformada inversa cerrada para todo \(\alpha\).
2. Simule \(10^5\) observaciones de una Gamma con \(\alpha\) entero usando suma de exponenciales.
3. Grafique el histograma y compárelo con la densidad teórica.
""")

    st.subheader("1(a) Comentario teórico: transformada inversa")

    st.markdown(r"""
La función de distribución acumulada de una Gamma es

\[
F(x)=\int_0^x \frac{\lambda^\alpha t^{\alpha-1} e^{-\lambda t}}{\Gamma(\alpha)}\,dt.
\]

Para \(\alpha\) no entero, esta integral se expresa en términos de la **función gamma incompleta**, la cual no tiene forma cerrada elemental.  

Para usar transformada inversa requerimos \(F^{-1}(u)\), pero esta no existe de manera cerrada en general.  
""")

    st.subheader("1(b) Simulación con suma de exponenciales")

    st.markdown(r"""
Si \(\alpha=k\) es entero positivo:

\[
X = \sum_{i=1}^k Y_i, \qquad Y_i\sim \text{Exp}(\lambda).
\]

Esto se obtiene porque la convolución de exponenciales iid genera una Gamma con parámetro entero.
""")

    alpha_int = st.number_input("α (entero positivo)", min_value=1, value=3, step=1)
    lam = st.number_input("λ (tasa > 0)", min_value=0.0001, value=1.0, step=0.1, format="%.4f")
    n = st.number_input("Tamaño de la muestra", min_value=1000, max_value=200000,
                        value=100000, step=1000)
    seed = st.number_input("Semilla aleatoria", min_value=0, value=123, step=1)

    if st.button("Simular Gamma por suma de exponenciales"):
        np.random.seed(seed)

        k = int(alpha_int)
        U = np.random.rand(int(n), k)
        exp_samples = -np.log(U) / lam
        samples = exp_samples.sum(axis=1)

        st.write(f"Muestra generada: {len(samples)} valores")

        emp_mean = samples.mean()
        emp_var = samples.var(ddof=1)

        th_mean = alpha_int / lam
        th_var = alpha_int / lam**2

        st.markdown(r"### **Media y varianza**")
        st.write(f"Media empírica: {emp_mean:.4f} — Teórica: {th_mean:.4f}")
        st.write(f"Var empírica: {emp_var:.4f} — Teórica: {th_var:.4f}")

        fig, ax = plt.subplots()
        ax.hist(samples, bins=60, density=True, alpha=0.5)

        x = np.linspace(0, np.percentile(samples, 99.5), 400)
        y = gamma_pdf(x, alpha_int, lam)
        ax.plot(x, y, linewidth=2)

        ax.set_title(f"Gamma(α={alpha_int}, λ={lam})")
        st.pyplot(fig)

# =========================================================
# Ejercicio 2: Gamma(α=2.7, λ=1) con Aceptación-Rechazo
# =========================================================

def ejercicio_2():
    st.header("Ejercicio 2: Gamma(α=2.7, λ=1) por Aceptación–Rechazo")

    st.markdown(r"""
Queremos simular \(X\sim\mathrm{Gamma}(\alpha=2.7,\lambda=1)\) usando la propuesta  
\(Y\sim\mathrm{Exp}(1)\).

La densidad objetivo es:

\[
f(x)=\frac{x^{\alpha-1} e^{-x}}{\Gamma(\alpha)},\qquad x>0.
\]

La propuesta es:

\[
g(x)=e^{-x},\qquad x>0.
\]

Entonces:

\[
\frac{f(x)}{g(x)}=\frac{x^{\alpha-1}}{\Gamma(\alpha)}.
\]

Buscamos \(c\) tal que:

\[
f(x) \le c\, g(x)\qquad \forall x>0.
\]

Esto equivale a:

\[
\frac{x^{\alpha-1}}{\Gamma(\alpha)} \le c.
\]

La función \(x^{\alpha-1}\) se maximiza en:

\[
x^\ast = \alpha - 1.
\]

Por lo tanto:

\[
c = \frac{(\alpha-1)^{\alpha-1}}{\Gamma(\alpha)}.
\]
""")

    alpha = 2.7
    lam = 1.0

    c = (alpha - 1)**(alpha - 1) / math.gamma(alpha)

    st.markdown(rf"""
### **Constante teórica**

\[
c = \frac{{(\alpha - 1)^{{\alpha - 1}}}}{{\Gamma(\alpha)}} \approx {c:.4f}
\]
""")


    st.subheader("Simulación por Aceptación–Rechazo")

    n = st.number_input("Tamaño de la muestra", value=20000, min_value=1000, max_value=100000)
    seed = st.number_input("Semilla", value=123, min_value=0)

    if st.button("Simular Gamma(2.7,1)"):
        np.random.seed(seed)

        samples = []
        n_target = int(n)
        proposals = 0

        while len(samples) < n_target:
            proposals += 1
            y = -np.log(np.random.rand())  # Exp(1)
            ratio = (y**(alpha - 1)) / (math.gamma(alpha) * c)

            if np.random.rand() < ratio:
                samples.append(y)

        samples = np.array(samples)
        acc_rate = len(samples) / proposals

        st.write(f"**Tasa de aceptación ≈ {acc_rate:.3f}**")

        emp_mean = samples.mean()
        emp_var = samples.var(ddof=1)

        th_mean = alpha / lam
        th_var = alpha / lam**2

        st.write(f"Media empírica: {emp_mean:.4f} — Teórica: {th_mean:.4f}")
        st.write(f"Var empírica: {emp_var:.4f} — Teórica: {th_var:.4f}")

        fig, ax = plt.subplots()
        ax.hist(samples, bins=60, density=True, alpha=0.5)

        x = np.linspace(0, np.percentile(samples, 99.5), 400)
        ax.plot(x, gamma_pdf(x, alpha, lam), linewidth=2)

        ax.set_title("AR → Gamma(2.7,1)")
        st.pyplot(fig)

# =========================================================
# Ejercicio 3: Pareto(α=3, x_m=2)
# =========================================================

def ejercicio_3():
    st.header("Ejercicio 3: Distribución Pareto(α=3, x_m=2)")

    st.markdown(r"""
La distribución Pareto tiene F.C.D.A:

\[
F(x)=1 - \left(\frac{x_m}{x}\right)^\alpha,\qquad x\ge x_m.
\]

Usamos la transformada inversa:

\[
X = x_m (1-U)^{-1/\alpha}.
\]
""")

    alpha = 3
    xm = 2

    n = st.number_input("Tamaño de muestra", min_value=1000, max_value=100000,
                        value=5000)
    seed = st.number_input("Semilla", min_value=0, value=42)

    if st.button("Simular Pareto(3,2)"):
        np.random.seed(seed)

        U = np.random.rand(int(n))
        X = xm * (1 - U)**(-1/alpha)

        count_20 = np.sum(X > 20)
        p_over20 = (xm / 20)**alpha
        expected = p_over20 * n

        st.write(f"Observaciones con X>20: {count_20}")
        st.write(f"Esperado teórico: {expected:.2f}")

        fig, ax = plt.subplots()
        xmax = np.percentile(X, 99)
        grid = np.linspace(xm, xmax, 400)

        ax.hist(X[X <= xmax], bins=40, density=True, alpha=0.5)
        ax.plot(grid, pareto_pdf(grid, alpha, xm), linewidth=2)

        ax.set_title("Pareto(3,2)")
        st.pyplot(fig)
# =========================================================
# Ejercicio 4: Pareto con muestreo estratificado
# Estratos: (2,5], (5,15], (15,∞)
# =========================================================

def ejercicio_4():
    st.header("Ejercicio 4: Pareto con reducción de varianza (muestreo estratificado)")

    st.markdown(r"""
Consideramos \(X\sim\text{Pareto}(\alpha=3, x_m=2)\).

Estratos:

1. \(E_1 = (2,5]\)
2. \(E_2 = (5,15]\)
3. \(E_3 = (15,\infty)\)

Probabilidades teóricas:

\[
P(X>a) = \left(\frac{x_m}{a}\right)^\alpha.
\]
""")

    alpha = 3
    xm = 2

    def p_tail(x):
        return (xm / x)**alpha if x > xm else 1.0

    P1 = p_tail(2) - p_tail(5)
    P2 = p_tail(5) - p_tail(15)
    P3 = p_tail(15)

    st.write(f"P(E1) ≈ {P1:.4f}")
    st.write(f"P(E2) ≈ {P2:.4f}")
    st.write(f"P(E3) ≈ {P3:.4f}")

    n = st.number_input("Tamaño de muestra", min_value=1000, max_value=50000,
                        value=10000)
    seed = st.number_input("Semilla", min_value=0, value=123, key="seed4")

    if st.button("Simular estratificado"):
        np.random.seed(seed)

        n1, n2, n3 = int(n * P1), int(n * P2), int(n * P3)

        def pareto_inv(u):
            return xm * (1 - u)**(-1/alpha)

        def simula_estrato(a, b, n):
            vals = []
            while len(vals) < n:
                x = pareto_inv(np.random.rand())
                if a < x <= b:
                    vals.append(x)
            return np.array(vals)

        E1 = simula_estrato(2, 5, n1)
        E2 = simula_estrato(5, 15, n2)
        E3 = simula_estrato(15, np.inf, n3)

        mean_est = P1 * E1.mean() + P2 * E2.mean() + P3 * E3.mean()

        # estimador simple
        U = np.random.rand(n)
        Xsimple = pareto_inv(U)
        mean_simple = Xsimple.mean()

        true_mean = alpha * xm / (alpha - 1)

        strat_var = (P1**2 * np.var(E1, ddof=1)/n1
                     + P2**2 * np.var(E2, ddof=1)/n2
                     + P3**2 * np.var(E3, ddof=1)/n3)

        simple_var = np.var(Xsimple, ddof=1) / n

        st.subheader("Resultados")
        st.write(f"Media estratificada: {mean_est:.4f}")
        st.write(f"Media simple:       {mean_simple:.4f}")
        st.write(f"Media teórica:      {true_mean:.4f}")

        st.subheader("Varianzas")
        st.write(f"Var estratificada: {strat_var:.6f}")
        st.write(f"Var simple:        {simple_var:.6f}")

# =========================================================
# Ejercicio 5: Simular Beta(3,5) usando Z = X/(X+Y)
# =========================================================

def ejercicio_5():
    st.header("Ejercicio 5: Simulación de Beta(3,5) usando cociente de Gammas")

    st.markdown(r"""
Si  

\[
X\sim\Gamma(3,1),\qquad Y\sim\Gamma(5,1),
\]

entonces:

\[
Z=\frac{X}{X+Y}\sim\text{Beta}(3,5).
\]
""")

    n = st.number_input("Tamaño de la muestra", min_value=1000, max_value=100000,
                        value=20000, key="n5")
    seed = st.number_input("Semilla", min_value=0, value=99, key="seed5")

    if st.button("Simular Beta(3,5)"):
        np.random.seed(seed)

        Ux = np.random.rand(n, 3)
        X = -np.log(Ux).sum(axis=1)

        Uy = np.random.rand(n, 5)
        Y = -np.log(Uy).sum(axis=1)

        Z = X / (X + Y)

        from scipy.stats import beta

        fig, ax = plt.subplots()
        ax.hist(Z, bins=50, density=True, alpha=0.5)

        grid = np.linspace(0, 1, 400)
        ax.plot(grid, beta.pdf(grid, 3, 5), linewidth=2)

        st.pyplot(fig)

        st.write(f"Media empírica: {Z.mean():.4f} — Teórica: {3/(3+5):.4f}")
        var_th = (3*5)/((3+5)**2 * (3+5+1))
        st.write(f"Var empírica: {Z.var(ddof=1):.4f} — Teórica: {var_th:.4f}")

# =========================================================
# Ejercicio 6: Beta(0.5, 0.5) por Aceptación–Rechazo
# =========================================================

def ejercicio_6():
    st.header("Ejercicio 6: Beta(0.5, 0.5) con propuesta Uniforme(0,1)")

    st.markdown(r"""
La densidad es:

\[
f(x)=\frac{1}{\pi\sqrt{x(1-x)}},\qquad x\in(0,1).
\]

La propuesta es uniforme: \(g(x)=1\).  

El máximo ocurre en \(x=1/2\):

\[
c = f(1/2) = \frac{2}{\pi}.
\]
""")

    c = 2 / math.pi
    st.write(f"Constante óptima: c = {c:.4f}")

    n = st.number_input("Tamaño objetivo", min_value=1000, max_value=100000,
                        value=20000, key="n6")
    seed = st.number_input("Semilla", min_value=0, value=777, key="seed6")

    if st.button("Simular Beta(0.5,0.5)"):
        np.random.seed(seed)

        samples = []
        proposals = 0

        while len(samples) < n:
            x = np.random.rand()
            proposals += 1
            u = np.random.rand()
            ratio = (1/(math.pi * math.sqrt(x*(1-x)))) / c
            if u < ratio:
                samples.append(x)

        samples = np.array(samples)
        acc = n / proposals

        st.write(f"Tasa de aceptación ≈ {acc:.4f}")

        from scipy.stats import beta

        fig, ax = plt.subplots()
        ax.hist(samples, bins=60, density=True, alpha=0.5)

        grid = np.linspace(0.001, 0.999, 400)
        ax.plot(grid, beta.pdf(grid, 0.5, 0.5), linewidth=2)

        st.pyplot(fig)

# =========================================================
# Ejercicio 7: Distribución Gumbel
# =========================================================

def ejercicio_7():
    st.header("Ejercicio 7: Distribución Gumbel")

    st.markdown(r"""
La CDF es:

\[
F(x)=\exp\{-e^{-(x-\mu)/\beta}\}.
\]

Transformada inversa:

\[
X=\mu - \beta \ln(-\ln U).
\]
""")

    mu = st.number_input("μ", value=0.0, key="mu7")
    beta = st.number_input("β", value=1.0, key="beta7")
    n = st.number_input("Tamaño", min_value=1000, max_value=50000, value=10000, key="n7")
    seed = st.number_input("Semilla", min_value=0, value=123, key="seed7")

    if st.button("Simular Gumbel"):
        np.random.seed(seed)
        U = np.random.rand(n)
        X = mu - beta * np.log(-np.log(U))

        emp_p95 = np.percentile(X, 95)
        th_p95 = mu - beta*np.log(-np.log(0.95))

        st.write(f"P95 empírico:  {emp_p95:.4f}")
        st.write(f"P95 teórico:   {th_p95:.4f}")

        fig, ax = plt.subplots()
        ax.hist(X, bins=50, density=True, alpha=0.5)
        ax.axvline(emp_p95, color="red", linestyle="--")
        ax.axvline(th_p95, color="green", linestyle="--")
        st.pyplot(fig)

# =========================================================
# Ejercicio 8: Gumbel con variable de control
# =========================================================

def ejercicio_8():
    st.header("Ejercicio 8: Gumbel con variable de control")

    st.markdown(r"""
Sea:

\[
X=\mu - \beta \ln(-\ln U), \qquad Y = -\ln U \sim \text{Exp}(1).
\]

Estimador corregido:

\[
\hat{\theta}_c = \bar X - b (\bar Y - 1),
\qquad b = \frac{\mathrm{Cov}(X,Y)}{\mathrm{Var}(Y)}.
\]
""")

    mu = st.number_input("μ", value=0.0, key="mu8b")
    beta = st.number_input("β", value=1.0, key="beta8b")
    n = st.number_input("Tamaño", min_value=1000, max_value=50000, value=20000, key="n8")
    seed = st.number_input("Semilla", min_value=0, value=123, key="seed8b")

    if st.button("Aplicar variable de control"):
        np.random.seed(seed)
        U = np.random.rand(n)
        X = mu - beta*np.log(-np.log(U))
        Y = -np.log(U)

        Xbar = X.mean()
        Ybar = Y.mean()

        cov = np.cov(X, Y, ddof=1)[0,1]
        varY = np.var(Y, ddof=1)
        b = cov / varY

        theta_c = Xbar - b*(Ybar - 1)

        var_simple = np.var(X, ddof=1)/n
        adjusted = X - b*(Y-1)
        var_corr = np.var(adjusted, ddof=1)/n

        gamma = 0.5772156649
        th_mean = mu + beta*gamma

        st.write(f"Estimador simple:   {Xbar:.4f}")
        st.write(f"Estimador corregido:{theta_c:.4f}")
        st.write(f"Teórico:            {th_mean:.4f}")

        st.write(f"Var simple:   {var_simple:.6f}")
        st.write(f"Var corregida:{var_corr:.6f}")
# =========================================================
# Ejercicio 9: Simular Beta(2,6) por 3 métodos
# =========================================================

def ejercicio_9():
    st.header("Ejercicio 9: Comparación de métodos para Beta(2,6)")

    st.markdown(r"""
Simulamos \(X\sim\text{Beta}(2,6)\) mediante:

1. **Transformación con Gamma:**
   \[
   X = \frac{G_1}{G_1 + G_2},\quad
   G_1\sim\Gamma(2,1),\ G_2\sim\Gamma(6,1).
   \]

2. **Aceptación–Rechazo** con propuesta uniforme.

3. **Transformada aproximada** (usando la \(F^{-1}\) de la Beta vía librería).
""")

    n = st.number_input("Tamaño de la muestra", min_value=2000, max_value=50000,
                        value=10000, step=2000, key="n9")
    seed = st.number_input("Semilla", min_value=0, value=777, key="seed9")

    if st.button("Simular Beta(2,6) por los 3 métodos"):
        np.random.seed(seed)

        # -------- 1. Método Gamma --------
        U1 = np.random.rand(n, 2)
        G1 = -np.log(U1).sum(axis=1)

        U2 = np.random.rand(n, 6)
        G2 = -np.log(U2).sum(axis=1)

        X1 = G1 / (G1 + G2)

        # -------- 2. Aceptación–Rechazo --------
        # f(x) = Beta(2,6) = 7 x^{1}(1-x)^5 en (0,1)
        # propuesta g(x)=1, c = 7

        samples_ar = []
        props_ar = 0
        while len(samples_ar) < n:
            x = np.random.rand()
            props_ar += 1
            u = np.random.rand()
            fx = 7 * x * (1 - x)**5
            if u < fx / 7:
                samples_ar.append(x)

        X2 = np.array(samples_ar)
        acc_ar = n / props_ar

        # -------- 3. Transformada aproximada: usar F^{-1} de Beta --------
        from scipy.stats import beta
        U = np.random.rand(n)
        X3 = beta.ppf(U, 2, 6)

        # -------- Histogramas comparados --------
        fig, ax = plt.subplots()
        ax.hist(X1, bins=50, alpha=0.4, density=True, label="Gamma")
        ax.hist(X2, bins=50, alpha=0.4, density=True, label="A-R")
        ax.hist(X3, bins=50, alpha=0.4, density=True, label="F^{-1} Beta")

        grid = np.linspace(0, 1, 300)
        ax.plot(grid, beta.pdf(grid, 2, 6), linewidth=2, label="Beta(2,6) teórica")

        ax.legend()
        ax.set_title("Comparación de métodos para Beta(2,6)")
        st.pyplot(fig)

        st.subheader("Eficiencias y medias")
        st.write(f"Tasa de aceptación A-R ≈ {acc_ar:.3f}")
        st.write(f"Media Gamma: {X1.mean():.4f}")
        st.write(f"Media A-R:   {X2.mean():.4f}")
        st.write(f"Media F^{-1}:{X3.mean():.4f}")
        st.write(f"Media teórica: {2/(2+6):.4f}")

# =========================================================
# Ejercicio 10: Mezcla 50%-50% Gumbel y Gamma
# =========================================================

def ejercicio_10():
    st.header("Ejercicio 10: Mezcla de Gumbel y Gamma")

    st.markdown(r"""
Sea una mezcla:

\[
X = \begin{cases}
X_G & \text{con prob. } 0.5,\\[4pt]
X_\Gamma & \text{con prob. } 0.5,
\end{cases}
\]

donde:

- \(X_G \sim \text{Gumbel}(\mu,\beta)\)
- \(X_\Gamma \sim \text{Gamma}(\alpha,\lambda)\)

### **Valores teóricos**

\[
E[X] = 0.5\,E[X_G] + 0.5\,E[X_\Gamma],
\]

\[
\mathrm{Var}(X) = 0.5\,\mathrm{Var}(X_G) + 0.5\,\mathrm{Var}(X_\Gamma)
+ 0.5(1)\,(E[X_G] - E[X_\Gamma])^2 .
\]

Usando:

\[
E[X_G] = \mu + \beta\gamma,\qquad
\mathrm{Var}(X_G)=\frac{\pi^2}{6}\beta^2,
\]

\[
E[X_\Gamma] = \frac{\alpha}{\lambda},\qquad
\mathrm{Var}(X_\Gamma]=\frac{\alpha}{\lambda^2}.
\]
""")

    mu = st.number_input("μ (Gumbel)", value=0.0)
    beta = st.number_input("β (Gumbel)", value=1.0)
    alpha = st.number_input("α (Gamma)", value=2.0)
    lam = st.number_input("λ (Gamma)", value=1.0)
    n = st.number_input("Tamaño de muestra", min_value=2000, max_value=50000, value=10000)
    seed = st.number_input("Semilla", value=987)

    if st.button("Simular mezcla"):
        np.random.seed(seed)

        choice = np.random.rand(int(n)) < 0.5

        # Gumbel
        U = np.random.rand(int(n))
        Xg = mu - beta * np.log(-np.log(U))

        # Gamma
        if float(alpha).is_integer():
            k = int(alpha)
            Ug = np.random.rand(int(n), k)
            Xga = -np.log(Ug).sum(axis=1) / lam
        else:
            from scipy.stats import gamma
            Xga = gamma.rvs(alpha, scale=1/lam, size=int(n))

        X = np.where(choice, Xg, Xga)

        emp_mean = X.mean()
        emp_var = X.var(ddof=1)

        gamma_const = 0.5772156649
        EG = mu + beta * gamma_const
        VG = (np.pi**2 / 6) * beta**2
        EGa = alpha / lam
        VGa = alpha / lam**2

        true_mean = 0.5*(EG + EGa)
        true_var = 0.5*VG + 0.5*VGa + 0.5*(EG - EGa)**2

        st.subheader("Resultados")
        st.write(f"Media empírica = {emp_mean:.4f}")
        st.write(f"Media teórica = {true_mean:.4f}")
        st.write(f"Varianza empírica = {emp_var:.4f}")
        st.write(f"Varianza teórica = {true_var:.4f}")

        fig, ax = plt.subplots()
        ax.hist(X, bins=60, density=True, alpha=0.6)
        ax.set_title("Mezcla Gumbel–Gamma")
        st.pyplot(fig)

# =========================================================
# Ejercicio 11: Metropolis–Hastings para Beta(a,b)
# =========================================================

def ejercicio_11():
    st.header("Ejercicio 11: Metropolis–Hastings con propuesta Normal truncada")

    st.markdown(r"""
Queremos simular de la distribución Beta:

\[
\pi(x) \propto x^{a-1}(1-x)^{b-1}, \qquad x\in(0,1).
\]

Usaremos Metropolis–Hastings con propuesta:

\[
Y \sim N(x_t,\sigma^2)\text{ truncada a }(0,1).
\]

El paso de aceptación es:

\[
\alpha(x_t,y)=\min\left\{1,
\frac{\pi(y)\,q(x_t|y)}{\pi(x_t)\,q(y|x_t)}
\right\}.
\]
""")

    a = st.number_input("a (>0)", value=2.0)
    b = st.number_input("b (>0)", value=5.0)
    sigma = st.number_input("σ propuesta", value=0.15)
    n = st.number_input("Iteraciones MH", min_value=2000, max_value=50000,
                        value=15000)
    seed = st.number_input("Semilla", value=2024)

    if st.button("Ejecutar Metropolis-Hastings"):
        np.random.seed(seed)

        def log_pi(x):
            return (a - 1)*np.log(x) + (b - 1)*np.log(1 - x)

        def q_sample(x):
            y = np.random.normal(x, sigma)
            return min(max(y, 0), 1)

        def log_q(x_from, x_to):
            return -0.5*((x_from - x_to)**2) / sigma**2

        X = np.zeros(int(n))
        X[0] = 0.5
        accepts = 0

        for t in range(1, int(n)):
            xt = X[t-1]
            y = q_sample(xt)

            log_alpha = (log_pi(y) - log_pi(xt)) \
                        + (log_q(xt, y) - log_q(y, xt))

            if np.log(np.random.rand()) < log_alpha:
                X[t] = y
                accepts += 1
            else:
                X[t] = xt

        st.write(f"Tasa de aceptación ≈ {accepts/n:.3f}")

        fig, ax = plt.subplots(2,1)
        ax[0].plot(X[:500])
        ax[0].set_title("Primeros 500 valores")

        from scipy.stats import beta
        ax[1].hist(X[int(n/2):], bins=50, density=True)
        grid = np.linspace(0.001, 0.999, 300)
        ax[1].plot(grid, beta.pdf(grid, a, b), label="Beta(a,b)")
        ax[1].legend()

        st.pyplot(fig)

# =========================================================
# Ejercicio 12: Laplace(0,b) — Transformada y mezcla Exp–Bernoulli
# =========================================================

def ejercicio_12():
    st.header("Ejercicio 12: Distribución Laplace — Transformada y Mezcla")

    st.markdown(r"""
La densidad Laplace(0,b) es:

\[
f(x)=\frac{1}{2b}e^{-|x|/b}.
\]

### **Transformada inversa**

\[
X = \begin{cases}
b\ln(2U), & U<1/2,\\[4pt]
-b\ln(2(1-U)), & U\ge 1/2.
\end{cases}
\]

### **Mezcla exponencial**

\[
X = 
\begin{cases}
E, & B=1,\\
-E,& B=0,
\end{cases}
\]

con \(B\sim\text{Bernoulli}(0.5)\) y \(E\sim\text{Exp}(1/b)\).
""")

    b = st.number_input("Parámetro b", value=1.0)
    n = st.number_input("Tamaño de muestra", value=20000)
    seed = st.number_input("Semilla", value=321)

    if st.button("Simular Laplace"):
        np.random.seed(seed)

        U = np.random.rand(int(n))
        X1 = np.where(U < 0.5, b*np.log(2*U), -b*np.log(2*(1-U)))

        B = np.random.rand(int(n)) < 0.5
        E = np.random.exponential(scale=b, size=int(n))
        X2 = np.where(B, E, -E)

        fig, ax = plt.subplots(1,2, figsize=(12,5))
        ax[0].hist(X1, bins=60, density=True)
        ax[0].set_title("Transformada")

        ax[1].hist(X2, bins=60, density=True)
        ax[1].set_title("Mezcla Exp–Bernoulli")

        grid = np.linspace(-7*b, 7*b, 300)
        pdf = (1/(2*b))*np.exp(-np.abs(grid)/b)
        ax[0].plot(grid, pdf)
        ax[1].plot(grid, pdf)

        st.pyplot(fig)

# =========================================================
# Ejercicio 13: Distribución Cauchy(0,1) vía transformada
# =========================================================

def ejercicio_13():
    st.header("Ejercicio 13: Distribución Cauchy por Transformada")

    st.markdown(r"""
La Cauchy(0,1) tiene CDF:

\[
F(x)=\frac1\pi\arctan(x)+\frac12.
\]

**Transformada inversa:**

\[
X = \tan\big(\pi(U-\tfrac12)\big).
\]
""")

    n = st.number_input("Tamaño de muestra", value=20000)
    seed = st.number_input("Semilla", value=31415)

    if st.button("Simular Cauchy"):
        np.random.seed(seed)
        U = np.random.rand(int(n))
        X = np.tan(np.pi*(U - 0.5))

        fig, ax = plt.subplots()
        Xplot = X[np.abs(X) < np.percentile(X, 97)]
        ax.hist(Xplot, bins=60, density=True)

        grid = np.linspace(-10, 10, 300)
        pdf = 1/(np.pi*(1+grid**2))
        ax.plot(grid, pdf)

        st.pyplot(fig)

# =========================================================
# Ejercicio 14: Normal Multivariada usando Cholesky
# =========================================================
def ejercicio_14():
    st.header("Ejercicio 14: Normal Multivariada usando Cholesky")

    st.markdown(r"""
Simulamos:

\[
X\sim N(\mu,\Sigma).
\]

Método:

1. Obtener \(L\) tal que \(LL^\top=\Sigma\).
2. Generar \(Z\sim N(0,I)\).
3. \(X=\mu+LZ\).
""")

    rho = st.slider("Correlación ρ", -0.95, 0.95, 0.6)
    n = st.number_input("Tamaño muestra", value=20000)
    seed = st.number_input("Semilla", value=2025)

    if st.button("Simular Normal Multivariada"):
        np.random.seed(seed)

        mu = np.array([0,0])
        Sigma = np.array([[1, rho],[rho,1]])

        L = np.linalg.cholesky(Sigma)

        Z = np.random.randn(2, int(n))
        X = mu.reshape(2,1) + L @ Z

        fig, ax = plt.subplots()
        ax.scatter(X[0,:2000], X[1,:2000], alpha=0.3, s=8)
        st.pyplot(fig)

        st.write("Covarianza estimada:")
        st.write(np.cov(X))

# =========================================================
# Ejercicio 15: Aceptación–Rechazo para densidad complicada
# =========================================================

def ejercicio_15():
    st.header("Ejercicio 15: Aceptación–Rechazo para densidad complicada")

    st.markdown(r"""
Queremos simular una variable con densidad:

\[
f(x) \propto x^2 e^{-x^2/2}, \qquad x>0.
\]

Usaremos como propuesta:

\[
g(x)=e^{-x},\quad x>0.
\]

Buscamos:

\[
c=\max_x \frac{f(x)}{g(x)} = x^2 e^{-x^2/2+x}.
\]
""")

    n = st.number_input("Tamaño de muestra", value=10000)
    seed = st.number_input("Semilla", value=13579)

    if st.button("Simular densidad complicada"):
        np.random.seed(seed)

        xs = np.linspace(0.001, 10, 20000)
        c = np.max(xs**2 * np.exp(-xs**2/2 + xs))
        st.write(f"Constante mayorante c ≈ {c:.4f}")

        samples = []
        proposals = 0

        while len(samples) < n:
            proposals += 1
            y = np.random.exponential()
            u = np.random.rand()

            if u < (y**2 * np.exp(-y**2/2 + y)) / c:
                samples.append(y)

        X = np.array(samples)
        acc = n/proposals
        st.write(f"Aceptación ≈ {acc:.3f}")

        fig, ax = plt.subplots()
        ax.hist(X, bins=60, density=True, alpha=0.6)

        grid = np.linspace(0, max(X), 300)
        f_grid = grid**2 * np.exp(-grid**2/2)
        f_grid /= np.trapz(f_grid, grid)
        ax.plot(grid, f_grid)

        st.pyplot(fig)

# =========================================================
# Ejercicio 16: Algoritmo genético para minimizar f(x) = x² + 3 sin(5x)
# =========================================================

def ejercicio_16():
    st.header("Ejercicio 16: Minimización con Algoritmo Genético (GA)")

    st.markdown(r"""
Queremos minimizar la función:

\[
f(x)=x^2+3\sin(5x),\qquad x\in[-4,4].
\]

Un Algoritmo Genético típico sigue estos pasos:

1. **Inicialización:** generar una población aleatoria.
2. **Evaluación:** computar \(f(x)\) para cada individuo.
3. **Selección:** elegir padres proporcionalmente a su aptitud.
4. **Cruza:** combinar dos padres para formar nuevos hijos.
5. **Mutación:** perturbar ligeramente los hijos para mantener variabilidad.
6. **Reemplazo:** nueva generación reemplaza la anterior.
7. **Registrar el mejor individuo por generación**.
""")

    pop_size = st.number_input("Tamaño de la población", value=40)
    gens = st.number_input("Número de generaciones", value=60)
    mutation_rate = st.number_input("Tasa de mutación", value=0.15)
    seed = st.number_input("Semilla", value=42)

    if st.button("Ejecutar GA"):
        np.random.seed(seed)

        def f(x):
            return x**2 + 3*np.sin(5*x)

        pop = np.random.uniform(-4, 4, int(pop_size))

        best_vals = []
        best_pos = []

        for g in range(int(gens)):
            fitness = -f(pop)  # minimizar → maximizamos -f
            probs = (fitness - fitness.min()) + 1e-6
            probs /= probs.sum()

            idx = np.random.choice(len(pop), size=int(pop_size), p=probs)
            parents = pop[idx]

            children = []
            for i in range(0, len(parents), 2):
                p1, p2 = parents[i], parents[(i+1) % len(parents)]
                w = np.random.rand()
                child = w*p1 + (1-w)*p2
                children.append(child)

            children = np.array(children)
            mut_mask = np.random.rand(len(children)) < mutation_rate
            children[mut_mask] += np.random.normal(0, 0.3, mut_mask.sum())
            children = np.clip(children, -4, 4)

            pop = children.copy()

            best_idx = np.argmin(f(pop))
            best_vals.append(f(pop)[best_idx])
            best_pos.append(pop[best_idx])

        st.subheader("Evolución del mejor individuo")

        fig, ax = plt.subplots(1, 2, figsize=(12,5))

        ax[0].plot(best_vals)
        ax[0].set_title("Mejor valor por generación")
        ax[0].set_xlabel("Generación")
        ax[0].set_ylabel("f(x)")

        x_grid = np.linspace(-4,4,400)
        ax[1].plot(x_grid, f(x_grid))
        ax[1].scatter(best_pos[-1], best_vals[-1], color='red')
        ax[1].set_title("Mejor valor encontrado")
        ax[1].set_xlabel("x")

        st.pyplot(fig)

        st.write(f"Mejor x encontrado ≈ {best_pos[-1]:.4f}")
        st.write(f"f(x) ≈ {best_vals[-1]:.4f}")

# =========================================================
# Ejercicio 17: GA para maximizar Sharpe de un portafolio de 5 activos
# =========================================================

def ejercicio_17():
    st.header("Ejercicio 17: Optimización del portafolio (Sharpe) con Algoritmo Genético")

    st.markdown(r"""
Maximizamos el índice de Sharpe:

\[
\text{Sharpe}(w)= \frac{\mu^\top w - r_f}{\sqrt{w^\top\Sigma w}},
\qquad w\ge0,\quad \sum_i w_i=1.
\]

Usamos un GA para encontrar el vector de pesos \(w\in\mathbb{R}^5\).
""")

    mu1 = st.text_input("Media de los activos (5 valores)", "0.12,0.08,0.10,0.07,0.15")
    mu = np.array([float(x) for x in mu1.split(",")])

    st.markdown("### Matriz de covarianzas (5×5)")
    Sigma_input = st.text_area("Ingresa Sigma fila por fila",
                               "0.10,0.02,0.03,0.00,0.01\n"
                               "0.02,0.09,0.01,0.02,0.00\n"
                               "0.03,0.01,0.20,0.01,0.03\n"
                               "0.00,0.02,0.01,0.07,0.02\n"
                               "0.01,0.00,0.03,0.02,0.25")

    Sigma = np.array([[float(x) for x in row.split(",")]
                      for row in Sigma_input.split("\n")])

    rf = st.number_input("Tasa libre de riesgo r_f", value=0.03)
    pop_size = st.number_input("Tamaño población", value=50)
    gens = st.number_input("Generaciones", value=80)
    mutation = st.number_input("Mutación", value=0.10)

    seed = st.number_input("Semilla", value=11)

    if st.button("Optimizar Sharpe"):
        np.random.seed(seed)

        def normalize(w):
            w = np.maximum(w, 0)
            return w / w.sum()

        def sharpe(w):
            ret = mu @ w
            vol = np.sqrt(w @ Sigma @ w)
            return (ret - rf) / vol

        pop = np.random.dirichlet(np.ones(5), size=int(pop_size))

        best_vals = []
        best_ws = []

        for g in range(int(gens)):
            fit = np.array([sharpe(w) for w in pop])
            probs = fit - fit.min() + 1e-6
            probs /= probs.sum()

            idx = np.random.choice(len(pop), size=len(pop), p=probs)
            parents = pop[idx]

            children = []
            for i in range(0, len(parents), 2):
                p1 = parents[i]
                p2 = parents[(i+1) % len(parents)]
                alpha = np.random.rand()
                c = alpha*p1 + (1-alpha)*p2
                children.append(c)

            children = np.array(children)

            mut_mask = np.random.rand(*children.shape) < mutation
            children = children + mut_mask * np.random.normal(0, 0.05, children.shape)

            children = np.array([normalize(w) for w in children])

            pop = children.copy()

            best_idx = np.argmax([sharpe(w) for w in pop])
            best_w = pop[best_idx]
            best_ws.append(best_w)
            best_vals.append(sharpe(best_w))

        st.subheader("Sharpe por generación")

        fig, ax = plt.subplots()
        ax.plot(best_vals)
        ax.set_title("Mejor Sharpe")
        ax.set_xlabel("Generación")
        ax.set_ylabel("Sharpe")
        st.pyplot(fig)

        st.subheader("Mejor portafolio encontrado")
        st.write(best_ws[-1])
        st.write(f"Sharpe ≈ {best_vals[-1]:.4f}")

# =========================================================
# Ejercicio 18: EM para mezcla 0.6 N(0,1) + 0.4 N(4,1.5²)
# =========================================================

def ejercicio_18():
    st.header("Ejercicio 18: EM para mezcla 0.6 N(0,1) + 0.4 N(4,1.5^2)")

    st.markdown(r"""
Mezcla verdadera:

\[
0.6\,\mathcal{N}(0,1) + 0.4\,\mathcal{N}(4,1.5^2).
\]

### **Parámetros a estimar**  
\[
(\pi_1,\mu_1,\sigma_1^2,\mu_2,\sigma_2^2).
\]

---

### **E–paso**

\[
\gamma_{i1} =
\frac{\pi_1\phi(x_i;\mu_1,\sigma_1^2)}
{\pi_1\phi(x_i;\mu_1,\sigma_1^2) + (1-\pi_1)\phi(x_i;\mu_2,\sigma_2^2)}.
\]

\[
\gamma_{i2}=1-\gamma_{i1}.
\]

---

### **M–paso**

\[
\pi_1^{(t+1)} = \frac{1}{n}\sum_i \gamma_{i1},
\]

\[
\mu_1^{(t+1)} = \frac{\sum_i \gamma_{i1}x_i}{\sum_i \gamma_{i1}},
\qquad
\sigma_1^{2(t+1)} = \frac{\sum_i \gamma_{i1}(x_i-\mu_1)^2}{\sum_i \gamma_{i1}},
\]

y análogo para el componente 2.
""")

    n = st.number_input("Tamaño muestra simulada", value=2000)
    it = st.number_input("Iteraciones EM", value=20)
    seed = st.number_input("Semilla", value=999)

    def normal_pdf(x, mu, s2):
        return (1/np.sqrt(2*np.pi*s2))*np.exp(-(x-mu)**2/(2*s2))

    if st.button("Ejecutar EM"):
        np.random.seed(seed)
        X = np.where(
            np.random.rand(int(n)) < 0.6,
            np.random.normal(0,1,int(n)),
            np.random.normal(4,1.5,int(n))
        )

        pi1 = 0.5
        mu1, mu2 = -1, 3
        s1, s2 = 1.2, 1.8

        loglik = []

        for t in range(int(it)):
            num1 = pi1 * normal_pdf(X, mu1, s1)
            num2 = (1 - pi1) * normal_pdf(X, mu2, s2)
            gamma1 = num1 / (num1 + num2)
            gamma2 = 1 - gamma1

            N1 = gamma1.sum()
            N2 = gamma2.sum()

            pi1 = N1 / n
            mu1 = (gamma1 * X).sum() / N1
            mu2 = (gamma2 * X).sum() / N2
            s1 = (gamma1 * (X - mu1)**2).sum() / N1
            s2 = (gamma2 * (X - mu2)**2).sum() / N2

            loglik.append(
                np.sum(np.log(
                    pi1*normal_pdf(X,mu1,s1) + (1-pi1)*normal_pdf(X,mu2,s2)
                ))
            )

        st.write(f"π1 ≈ {pi1:.3f}")
        st.write(f"μ1 ≈ {mu1:.3f}, σ1² ≈ {s1:.3f}")
        st.write(f"μ2 ≈ {mu2:.3f}, σ2² ≈ {s2:.3f}")

        fig, ax = plt.subplots()
        ax.plot(loglik)
        ax.set_title("Convergencia log-verosimilitud")
        st.pyplot(fig)

# =========================================================
# Aplicación principal
# =========================================================

def main():
    st.title("Tarea de Simulación – Ejercicios 1–18")

    st.sidebar.title("Navegación")
    choice = st.sidebar.radio(
        "Selecciona el ejercicio",
        (
            "Ejercicio 1: Gamma α entero",
            "Ejercicio 2: Gamma(2.7,1) A-R",
            "Ejercicio 3: Pareto(3,2)",
            "Ejercicio 4: Pareto estratificado",
            "Ejercicio 5: Beta(3,5) vía Gamma",
            "Ejercicio 6: Beta(0.5,0.5)",
            "Ejercicio 7: Gumbel",
            "Ejercicio 8: Gumbel con variable de control",
            "Ejercicio 9: Beta(2,6) — comparación",
            "Ejercicio 10: Mezcla Gumbel/Gamma",
            "Ejercicio 11: Metropolis–Hastings Beta",
            "Ejercicio 12: Laplace — dos métodos",
            "Ejercicio 13: Cauchy por transformada",
            "Ejercicio 14: Normal multivariada (Cholesky)",
            "Ejercicio 15: A-R densidad complicada",
            "Ejercicio 16: GA para f(x)",
            "Ejercicio 17: GA para portafolio",
            "Ejercicio 18: EM para mezcla normal"
        )
    )

    if choice.startswith("Ejercicio 1"):
        ejercicio_1()
    elif choice.startswith("Ejercicio 2"):
        ejercicio_2()
    elif choice.startswith("Ejercicio 3"):
        ejercicio_3()
    elif choice.startswith("Ejercicio 4"):
        ejercicio_4()
    elif choice.startswith("Ejercicio 5"):
        ejercicio_5()
    elif choice.startswith("Ejercicio 6"):
        ejercicio_6()
    elif choice.startswith("Ejercicio 7"):
        ejercicio_7()
    elif choice.startswith("Ejercicio 8"):
        ejercicio_8()
    elif choice.startswith("Ejercicio 9"):
        ejercicio_9()
    elif choice.startswith("Ejercicio 10"):
        ejercicio_10()
    elif choice.startswith("Ejercicio 11"):
        ejercicio_11()
    elif choice.startswith("Ejercicio 12"):
        ejercicio_12()
    elif choice.startswith("Ejercicio 13"):
        ejercicio_13()
    elif choice.startswith("Ejercicio 14"):
        ejercicio_14()
    elif choice.startswith("Ejercicio 15"):
        ejercicio_15()
    elif choice.startswith("Ejercicio 16"):
        ejercicio_16()
    elif choice.startswith("Ejercicio 17"):
        ejercicio_17()
    elif choice.startswith("Ejercicio 18"):
        ejercicio_18()

if __name__ == "__main__":
    main()
