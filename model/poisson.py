import numpy as np
from scipy.stats import poisson

def get_poisson_1x2(lh, la, max_goals=8): # Incrementamos a 8 para mayor precisión
    # 1. Crear matriz de probabilidades conjuntas
    goals = np.arange(max_goals + 1)
    prob_h = poisson.pmf(goals, lh)
    prob_a = poisson.pmf(goals, la)
    
    # Producto externo para crear la matriz (más rápido que bucles anidados)
    matrix = np.outer(prob_h, prob_a)
    
    # 2. Normalización (Fase de precisión ACBE+)
    matrix /= matrix.sum() 
    
    # 3. Extracción de mercados
    p1 = np.tril(matrix, -1).sum()
    px = np.trace(matrix)
    p2 = np.triu(matrix, 1).sum()
    
    return {"1": p1, "X": px, "2": p2}