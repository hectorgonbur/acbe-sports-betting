def run_acbe_convergence(p_poisson, p_h2h, p_mc, omega=(0.4, 0.3, 0.3), sigma_vake=0.05, sigma_sharp=0.02):
    """
    Fase 2.3: Convergencia Bayesiana Estructural ACBE+.
    Integra modelos y penaliza riesgos sistémicos.
    """
    # 1. Validación de pesos (Suma debe ser 1)
    if not np.isclose(sum(omega), 1.0):
        omega = [w/sum(omega) for w in omega]

    out = {}
    for k in p_poisson.keys():
        # Aplicación de la fórmula estructural
        val = (omega[0] * p_poisson[k] + 
               omega[1] * p_h2h[k] + 
               omega[2] * p_mc[k]) - (sigma_vake + sigma_sharp)
        
        out[k] = max(val, 0) # Control de drawdown (no valores negativos)

    # 2. Re-normalización (Asegura que el total sea 1.0 tras penalizaciones)
    total = sum(out.values())
    if total > 0:
        for k in out:
            out[k] /= total
            
    return out