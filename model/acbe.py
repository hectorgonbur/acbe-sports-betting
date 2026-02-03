def run_acbe_convergence(p_poisson, p_mc, p_alt=None):
    """
    Fase 2.3: Algoritmo de Convergencia Bayesiana Estructural (ACBE+).
    Encuentra el punto de equilibrio entre la teoría (Poisson) y la simulación (Monte Carlo).
    """
    p_final = {}
    labels = ["1", "X", "2"]
    
    # Ponderación basada en la estabilidad del modelo (Varianza Inversa)
    # Poisson (Teórico) tiene un peso del 45%
    # Monte Carlo (Simulación Estocástica) tiene un peso del 55%
    w_poisson = 0.45
    w_mc = 0.55
    
    for label in labels:
        # Calculamos el promedio ponderado para cada resultado
        prob_local = (p_poisson[label] * w_poisson) + (p_mc[label] * w_mc)
        
        # Si existe un tercer modelo (p_alt), lo integramos para mayor precisión
        if p_alt:
            p_final[label] = (prob_local * 0.9) + (p_alt[label] * 0.1)
        else:
            p_final[label] = prob_local
            
    # Normalización final para asegurar que la suma de probabilidades sea exactamente 100%
    total = sum(p_final.values())
    return {k: v / total for k, v in p_final.items()}