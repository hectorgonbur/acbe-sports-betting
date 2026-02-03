def calculate_kelly_bayesian(cuota, prob_final, entropy):
    """
    Calcula el Stake Final usando Kelly Bayesiano Fraccional.
    """
    b = cuota - 1
    p = prob_final
    q = 1 - p
    
    # Kelly Base
    f_star = (b * p - q) / b
    
    # Ajuste por Entropía (k = 1 / (1 + H))
    k = 1 / (1 + entropy)
    
    # Stake Final con Half-Kelly (0.5) para control de drawdown
    stake_final = f_star * k * 0.5
    
    return max(0, stake_final) # No permite apuestas negativas