import streamlit as st
import pandas as pd
import datetime
from utils.api_client import FootballDataClient
from utils.helpers import calculate_dynamic_lambdas, get_structural_adjustment
from model.poisson import get_poisson_1x2
from model.montecarlo import run_monte_carlo
from model.acbe import run_acbe_convergence
from risk.entropy import calculate_normalized_entropy
from risk.kelly import calculate_kelly_bayesian

st.set_page_config(page_title="ACBE Quantum Terminal", layout="wide")
st.title("🏛️ Sistema de Arbitraje Estadístico ACBE-Kelly v2.0")

# --- BARRA LATERAL: ENTRADA DE DATOS MANUAL ---
st.sidebar.header("📥 Datos del Evento")

# Tú colocas los nombres para el reporte
team_h = st.sidebar.text_input("Local", value="Real Madrid")
team_a = st.sidebar.text_input("Visitante", value="Manchester City")

# El ID sigue siendo necesario para las stats, pero lo pones tú una vez
fixture_id = st.sidebar.text_input("Fixture ID (API-Football)", value="1133575")

st.sidebar.header("💰 Cuotas Actuales")
c1 = st.sidebar.number_input(f"Cuota {team_h} (1)", value=2.10, step=0.01)
cx = st.sidebar.number_input("Cuota Empate (X)", value=3.50, step=0.01)
c2 = st.sidebar.number_input(f"Cuota {team_a} (2)", value=3.40, step=0.01)

st.sidebar.header("🏥 Ajustes de Bajas")
bajas_h = st.sidebar.multiselect(f"Bajas {team_h}", ['arquero_titular', 'defensa_clave', 'goleador'])
bajas_a = st.sidebar.multiselect(f"Bajas {team_a}", ['arquero_titular', 'defensa_clave', 'goleador'])

# --- MOTOR DE ANÁLISIS ---
if st.sidebar.button("🚀 EJECUTAR ANÁLISIS") and fixture_id:
    with st.spinner("Procesando estadísticas y convergencia bayesiana..."):
        try:
            client = FootballDataClient()
            # 1. Traemos solo las estadísticas de este ID específico (Eficiencia de API)
            raw_stats = client.get_match_stats(fixture_id)
            
            # 2. Cálculo de Lambdas y Ajustes
            l_h_base = calculate_dynamic_lambdas(raw_stats, team_h)
            l_a_base = calculate_dynamic_lambdas(raw_stats, team_a)
            l_h = l_h_base * get_structural_adjustment(1.0, bajas_h)
            l_a = l_a_base * get_structural_adjustment(1.0, bajas_a)

            # 3. Convergencia de Modelos
            p_poisson = get_poisson_1x2(l_h, l_a)
            p_mc = run_monte_carlo(l_h, l_a, iterations=10000)
            p_final = run_acbe_convergence(p_poisson, p_mc)
            
            # 4. Cálculo de Entropía y Stake
            h_entropy = calculate_normalized_entropy(list(p_final.values()))

            # --- RENDERIZADO ---
            st.subheader(f"📊 Análisis Quántico: {team_h} vs {team_a}")
            # ... (Aquí va el mismo código de tablas y métricas que ya tienes)
            # Esto asegura que los resultados se vean profesionales como en image_aa6cd8.png
            
        except Exception as e:
            st.error(f"Falla en el motor: {e}")