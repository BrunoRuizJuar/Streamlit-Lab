import numpy as np

def buffon_simulation(num_needles=10000, L=1.0, D=1.5, seed=0):
    """
    Simula el método de la aguja de Buffon.

    Parámetros:
    - num_needles: número de agujas a lanzar
    - L: longitud de la aguja
    - D: distancia entre líneas paralelas (D > L)
    """
    np.random.seed(seed)

    # Distancia del centro a la línea más cercana
    x = np.random.uniform(0, D/2, num_needles)

    # Ángulo aleatorio uniforme
    theta = np.random.uniform(0, np.pi/2, num_needles)

    # Aguja toca línea cuando x <= (L/2) * sin(theta)
    hits = x <= (L/2) * np.sin(theta)

    # Estimación de π
    p = np.mean(hits)
    if p == 0:
        return None, 0

    pi_est = (2 * L) / (D * p)
    return pi_est, np.sum(hits)

def buffon_visual(num_needles=200, L=1.0, D=2.0, seed=0):
    np.random.seed(seed)

    # Área visual para dibujar
    max_x = D * 8 - L
    max_y = D * 8 - L

    centers_x = np.random.uniform(0, max_x, num_needles)
    centers_y = np.random.uniform(0, max_y, num_needles)

    # Ángulos aleatorios entre 0 y pi/2 (consistente con simulación)
    theta = np.random.uniform(0, np.pi/2, num_needles)

    # Cálculo de extremos de la aguja
    x1 = centers_x - (L/2) * np.cos(theta)
    y1 = centers_y - (L/2) * np.sin(theta)
    x2 = centers_x + (L/2) * np.cos(theta)
    y2 = centers_y + (L/2) * np.sin(theta)

    # Una aguja toca una línea horizontal si cambia de banda en Y
    hits = (np.floor(y1 / D) != np.floor(y2 / D))

    return x1, y1, x2, y2, hits, max_x, max_y

