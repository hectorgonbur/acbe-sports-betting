import math

def calculate_normalized_entropy(probs):
    """
    Fase 1.2: Calcula la Entropía de Shannon Normalizada.
    H = 0: Máxima certeza | H = 1: Máximo desorden.
    """
    n = len(probs) # Para fútbol 1X2, n = 3
    if n <= 1:
        return 0
    
    # Cálculo de Entropía Base
    h = -sum(p * math.log(p) for p in probs if p > 0)
    
    # Normalización (H / log(n))
    h_norm = h / math.log(n)
    
    return h_norm

def classify_market_entropy(h_norm):
    """Clasifica el mercado según el umbral del prompt."""
    if h_norm < 0.55: return "Baja (Predecible)"
    if h_norm <= 0.70: return "Media"
    return "Alta (Ruidoso)" # Bloqueo automático si > 0.72 en Fase 0