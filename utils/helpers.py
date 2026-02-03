import pandas as pd
import numpy as np

def calculate_dynamic_lambdas(match_data_df, team_name, window=10):
    """
    Fase 1.1: Extrae métricas ofensivas/defensivas y calcula λ.
    Ajusta por xG y goles reales de los últimos 'window' partidos.
    """
    # Filtrar últimos n partidos del equipo
    recent_matches = match_data_df[
        (match_data_df['home_team'] == team_name) | 
        (match_data_df['away_team'] == team_name)
    ].tail(window)
    
    # Calcular promedio de goles anotados y recibidos
    goles_anotados = []
    for _, row in recent_matches.iterrows():
        if row['home_team'] == team_name:
            goles_anotados.append(row['home_score'])
        else:
            goles_anotados.append(row['away_score'])
            
    # λ base = Promedio simple de goles
    lambda_base = np.mean(goles_anotados)
    
    # Ajuste estructural por xG (si la API lo provee)
    # Ratio xG/G: Si xG > Goles, el equipo está 'debido' (sube λ)
    if 'xg' in recent_matches.columns:
        xg_avg = recent_matches['xg'].mean()
        factor_ajuste = xg_avg / (lambda_base if lambda_base > 0 else 1)
        lambda_final = lambda_base * factor_ajuste
    else:
        lambda_final = lambda_base
        
    return lambda_final

def get_strength_index_adjustment(base_si, injuries_list):
    """
    Fase 1.3: Ajuste al Strength Index según vector de lesiones.
    """
    # Vector de impacto por posición (según prompt v2.0)
    impact_map = {
        'arquero_titular': 0.05,
        'defensa_clave': 0.04,
        'mediocentro_creativo': 0.06,
        'goleador': 0.07
    }
    
    total_delta = sum(impact_map.get(inj, 0.02) for inj in injuries_list)
    return base_si * (1 - total_delta)