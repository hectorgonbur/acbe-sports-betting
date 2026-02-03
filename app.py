import streamlit as st
import pandas as pd
import numpy as np
import datetime
from utils.api_client import FootballDataClient
from utils.helpers import calculate_dynamic_lambdas, get_structural_adjustment
from model.poisson import get_poisson_1x2
from model.acbe import run_acbe_convergence
from risk.entropy import calculate_normalized_entropy
from risk.kelly import calculate_kelly_bayesian

# 1. Configuración de la Interfaz
st.set_page_config(page_title="ACBE Quantum Terminal", layout="wide")
st.title("🏛️ Sistema de Arbitraje Estadístico ACBE-Kelly v2.0")
st.markdown("---")

# 2. Diccionario Maestro de Ligas
ligas_disponibles = {
    "Premier League 🏴󠁧󠁢󠁥󠁮󠁧󠁿": 39,
    "La Liga 🇪🇸": 140,
    "Serie A 🇮🇹": 135,
    "Bundesliga 🇩🇪": 78,
    "Ligue 1 🇫🇷": 61,
    "UEFA Champions League 🇪🇺": 2,
    "Liga Profesional 🇦🇷": 128,
    "Saudi Pro League 🇸🇦": 307,
    "Brasileirão 🇧🇷": 71
}

# --- BARRA LATERAL: DATA MINING AUTOMATIZADO ---
st.sidebar.header("📥 Selección Inteligente")

# A. Selector de Fecha y Liga
fecha_busqueda = st.sidebar.date_input("Fecha de Análisis", datetime.date.today())
nombre_liga = st.sidebar.selectbox("Selecciona la Liga", list(ligas_disponibles.keys()))
id_liga = ligas_disponibles[nombre_liga]

# B. Conexión y Reconocimiento Automático de Partidos
client = FootballDataClient()
partidos_crudos = client.get_fixtures_by_date(id_liga, fecha_busqueda)

if partidos_crudos:
    mapa_partidos = {
        f"{p['teams']['home']['name']} vs {p['teams']['away']['name']}": p 
        for p in partidos_crudos
    }
    seleccion = st.sidebar.selectbox("Elige el partido", list(mapa_partidos.keys()))
    
    partido_seleccionado = mapa_partidos[seleccion]
    fixture_id = partido_seleccionado['fixture']['id']
    team_h = partido_seleccionado['teams']['home']['name']
    team_a = partido_seleccionado['teams']['away']['name']
    
    st.sidebar.success(f"ID: {fixture_id} | Reconocido ✅")
else:
    st.sidebar.warning("No hay partidos para esta fecha/liga.")
    fixture_id = None

# C. Ajustes Estructurales y Cuotas
st.sidebar.markdown("---")
if fixture_id:
    st.sidebar.header("🏥 Ajustes Estructurales")
    bajas_h = st.sidebar.multiselect(f"Bajas {team_h}", ['arquero_titular', 'defensa_clave', 'mediocentro_creativo', 'goleador'])
    bajas_a = st.sidebar.multiselect(f"Bajas {team_a}", ['arquero_titular', 'defensa_clave', 'mediocentro_creativo', 'goleador'])
    
    st.sidebar.header("💰 Cuotas del Mercado")
    c1 = st.sidebar.number_input("Cuota Local (1)", value=2.00, step=0.01)
    cx = st.sidebar.number_input("Cuota Empate (X)", value=3.40, step=0.01)
    c2 = st.sidebar.number_input("Cuota Visita (2)", value=3.20, step=0.01)

# --- MOTOR DE CÁLCULO CORREGIDO (REEMPLAZO TOTAL) ---
if st.sidebar.button("🚀 EJECUTAR ANÁLISIS") and fixture_id:
    with st.spinner("Ejecutando Convergencia ACBE+ (Poisson vs Monte Carlo)..."):
        try:
            # 1. DATA MINING (Fase 1)
            raw_stats = client.get_match_stats(fixture_id)
            l_h_base = calculate_dynamic_lambdas(raw_stats, team_h)
            l_a_base = calculate_dynamic_lambdas(raw_stats, team_a)
            
            # 2. AJUSTE ESTRUCTURAL (Fase 1.3)
            si_h = get_structural_adjustment(1.0, bajas_h)
            si_a = get_structural_adjustment(1.0, bajas_a)
            l_h, l_a = l_h_base * si_h, l_a_base * si_a

            # 3. MODELADO MULTI-CAPA (Fase 2)
            # Capa A: Distribución de Poisson
            p_poisson = get_poisson_1x2(l_h, l_a)
            
            # Capa B: Simulación de Monte Carlo (Simulamos 10,000 escenarios)
            from model.montecarlo import run_monte_carlo
            p_mc = run_monte_carlo(l_h, l_a, iterations=10000)
            
            # Capa C: Convergencia Bayesiana (ACBE+)
            # Aquí comparamos ambos modelos para mayor precisión
            p_final = run_acbe_convergence(p_poisson, p_mc, p_poisson) 
            
            # 4. RIESGO Y DECISIÓN (Fase 3 y 4)
            h_entropy = calculate_normalized_entropy(list(p_final.values()))
            
            # 5. RENDERIZADO DE RESULTADOS (Fase 6)
            st.subheader(f"📊 Análisis Quántico: {team_h} vs {team_a}")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("λ Local (Ajustado)", f"{l_h:.2f}")
            col2.metric("λ Visita (Ajustado)", f"{l_a:.2f}")
            col3.metric("Entropía del Mercado (H)", f"{h_entropy:.4f}")

            # Matriz de Decisión Profesional
            res_df = []
            cuotas = [c1, cx, c2]
            labels = ["1", "X", "2"]
            
            for i, label in enumerate(labels):
                prob = p_final[label]
                q_casa = cuotas[i]
                q_justa = 1/prob if prob > 0 else 0
                val = (prob * q_casa) - 1
                stake = calculate_kelly_bayesian(q_casa, prob, h_entropy)
                
                res_df.append({
                    "Resultado": label,
                    "Probabilidad ACBE": f"{prob:.2%}",
                    "Cuota Justa": f"{q_justa:.2f}",
                    "Cuota Casa": f"{q_casa:.2f}",
                    "Value (%)": f"{val:.2%}",
                    "Stake Sugerido": f"{stake:.2%}" if val > 0 else "0.00%"
                })
            
            st.table(pd.DataFrame(res_df))

            # Alertas de Valor
            if any(((p_final[labels[i]] * cuotas[i]) - 1) > 0.05 for i in range(3)):
                st.success("✅ ¡Oportunidad de Value Alto detectada!")
            else:
                st.info("ℹ️ No hay ventajas claras sobre el mercado actualmente.")

        except Exception as e:
            st.error(f"Error técnico en el motor: {e}")