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
