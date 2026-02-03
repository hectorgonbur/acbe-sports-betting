def apply_market_filters(cuota_min, overround, entropy, var_goles):
    """
    Filtro de Seguridad de Fase 0.
    Retorna True si el mercado es elegible, False si debe evadirse.
    """
    if cuota_min < 1.60: return False
    if overround > 0.07: return False # Umbral del 7%
    if entropy > 0.72: return False
    # Umbral tau sugerido para varianza: 2.5
    if var_goles > 2.5: return False
    
    return True