# 🎯 Monte Carlo & Algoritmos Genéticos — Librería + Web App

Este repositorio contiene una colección introductoria de métodos probabilísticos y evolutivos:

- Aproximación de π con **método de Monte Carlo**
- Estimación de π con el **método de la aguja de Buffon**
- Un **mini Algoritmo Genético modular**, diseñado para expandirse
- Una **web app en Streamlit** que integra todas las simulaciones

Este proyecto está pensado tanto como material didáctico como base para construir una librería personal de computación estocástica y metaheurísticas.

---

## 📦 Contenido del repositorio

```

mi_libreria_pi_ga/
│
├── app.py               # Web App de Streamlit
├── buffon.py            # Simulación del método de Buffon
├── ga_basic.py          # Mini Algoritmo Genético modular
├── utils.py             # (opcional) Funciones auxiliares
└── README.md            # Este archivo


---

## 🧠 Métodos incluidos

### 1. Aproximación de π por Monte Carlo
Usamos puntos aleatorios en un cuadrado y verificamos cuántos caen dentro de un círculo de radio 1.

\[
\pi \approx 4 \cdot \frac{\text{puntos dentro}}{\text{total}}
\]

La web app permite elegir:
- cuadrante (0,1)×(0,1)
- círculo completo (−1,1)×(−1,1)
- número de puntos interactivo

---

### 2. Método de la Aguja de Buffon

Simulación del experimento clásico:

> Se lanza una aguja sobre un piso con líneas paralelas.  
> La probabilidad de que toque una línea está relacionada con π.

Fórmula:

\[
P(\text{tocar línea}) = \frac{2L}{\pi D}
\]

La app permite ajustar:
- longitud de la aguja  
- distancia entre líneas  
- número de lanzamientos  

---

### 3. Mini Algoritmo Genético

Algoritmo básico pero completamente funcional con:

- Población inicial  
- Selección por torneo  
- Crossover de un punto  
- Mutación por reinicio aleatorio  
- Registro del mejor fitness por generación  

El GA optimiza:

\[
f(x) = -\sum_{i=1}^n (x_i - 0.5)^2
\]

Es decir, empuja todos los genes hacia 0.5.

---

## 🖥️ Demo en Streamlit

Para correr la app localmente:

```bash
pip install streamlit numpy matplotlib
streamlit run app.py
````

La aplicación incluye:

* Visualización interactiva de Monte Carlo
* Simulación del método de Buffon
* Entrenamiento en vivo del Algoritmo Genético
* Gráficas actualizadas dinámicamente

---

## 🚀 Cómo usar la librería

### En Python:

```python
from buffon import buffon_simulation
from ga_basic import GA

# Buffon
pi_est, hits = buffon_simulation(num_needles=50000)

# Algoritmo Genético
def fitness(x):
    return -((x - 0.5)**2).sum()

ga = GA(fitness_fn=fitness)
best, history = ga.run()
```

---

## 📚 Expansión futura

Este proyecto está diseñado para crecer. Algunas mejoras sugeridas:

* Añadir más operadores evolutivos (uniform crossover, elitismo, mutación gaussiana)
* Implementar selección por ruleta
* GA para problemas reales (TSP, regresión, optimización)
* Métodos Monte Carlo para integrales multidimensionales
* Web app ampliada con paneles y visualizaciones más avanzadas
* Integración con Manim para animaciones educativas

---

## 📘 Requisitos

```
Python 3.8+
numpy
matplotlib
streamlit
```

---

## 💡 Autor

Proyecto creado por **Bruno Ruiz Juarez**
Facultad de Estudios Superiores Acatlán — UNAM
Matemáticas Aplicadas y Computación / Ciencias de Datos

---

## 📝 Licencia

Este proyecto se distribuye bajo la **Licencia MIT**.
Puedes usar, modificar y redistribuir libremente el código.

```
MIT License
Copyright (...)
```

---

## ⭐ Si te sirve este repo…

¡No olvides dejar una estrella en GitHub ⭐ y compartirlo con otros estudiantes de MAC!
