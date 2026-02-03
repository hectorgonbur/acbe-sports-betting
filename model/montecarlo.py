import numpy as np

def run_monte_carlo(l_local, l_visita, iter=10000):
    """
    Simulación de 10,000 iteraciones basada en Poisson.
    """
    g_local = np.random.poisson(l_local, iter)
    g_visita = np.random.poisson(l_visita, iter)
    
    p1 = np.mean(g_local > g_visita)
    px = np.mean(g_local == g_visita)
    p2 = np.mean(g_local < g_visita)
    
    # Cálculo de skewness (asimetría) para Fase 5
    skew = (np.mean((g_local - l_local)**3)) / (np.std(g_local)**3)
    
    return {"p1": p1, "px": px, "p2": p2, "skew": skew}