import numpy as np

def calculate_dynamic_lambdas(api_response, team_name):
    """
    Fase 1.1: Extrae goles y xG para calcular el λ (Lambda) real.
    """
    goles_anotados = []
    xg_data = []

    for match in api_response:
        # Lógica para identificar si el equipo fue local o visitante
        stats = match.get('statistics', [])
        for s in stats:
            if s['team']['name'] == team_name:
                # Extraemos goles reales
                goals = match['goals']['home' if match['teams']['home']['name'] == team_name else 'away']
                goles_anotados.append(goals)
                
                # Extraemos xG (si está disponible en la API)
                for stat in s['statistics']:
                    if stat['type'] == 'expected_goals':
                        xg_data.append(float(stat['value']) if stat['value'] else goals)

    # Cálculo Bayesiano: λ ajustado por xG
    lambda_base = np.mean(goles_anotados) if goles_anotados else 1.0
    lambda_xg = np.mean(xg_data) if xg_data else lambda_base
    
    # λ Final = Promedio ponderado entre realidad y expectativa
    return (lambda_base * 0.4) + (lambda_xg * 0.6)

def get_structural_adjustment(base_si, injuries_list):
    """
    Fase 1.3: Ajuste Estructural al Strength Index (SI).
    """
    # Vector de impacto basado en el prompt v2.0
    impact_map = {
        'arquero_titular': 0.05,
        'defensa_clave': 0.04,
        'mediocentro_creativo': 0.06,
        'goleador': 0.07
    }
    
    total_delta = sum(impact_map.get(inj, 0.02) for inj in injuries_list)
    return base_si * (1 - total_delta)