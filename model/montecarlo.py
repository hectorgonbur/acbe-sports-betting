import numpy as np

def run_monte_carlo(l_h, l_a, iterations=10000):
    """
    Fase 2: Simulación de Monte Carlo vectorizada.
    Genera 10,000 escenarios posibles basados en los Lambdas ajustados.
    """
    # Generamos los goles aleatorios siguiendo una distribución de Poisson
    # para ambos equipos en todas las iteraciones simultáneamente.
    home_goals = np.random.poisson(l_h, iterations)
    away_goals = np.random.poisson(l_a, iterations)
    
    # Comparamos resultados
    wins = np.sum(home_goals > away_goals)
    draws = np.sum(home_goals == away_goals)
    losses = np.sum(home_goals < away_goals)
    
    # Calculamos las probabilidades finales del modelo
    return {
        "1": wins / iterations,
        "X": draws / iterations,
        "2": losses / iterations
    }