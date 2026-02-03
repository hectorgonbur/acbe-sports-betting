import streamlit as st
import pandas as pd
import numpy as np
from utils.api_client import FootballDataClient
from utils.helpers import calculate_dynamic_lambdas, get_structural_adjustment
from model.poisson import get_poisson_1x2
# Nota: Asegúrate de tener estos archivos en tu carpeta 'model' y 'risk'
from model.acbe import run_acbe_convergence
from risk.entropy import calculate_normalized_entropy
from risk.kelly import calculate_kelly_bayesian

# 1. Configuración de la Interfaz (Fase 6)
st.set_page_config(page_title="ACBE Quantum Terminal", layout="wide")
st.title("🏛️ Sistema de Arbitraje Estadístico ACBE-Kelly v2.0")
st.markdown("---")

# 2. Panel Lateral: Configuración de Datos (Fase 1)
st.sidebar.header("📥 Configuración de Entrada")
fixture_id = st.sidebar.text_input("Fixture ID (de API-Football)", value="1133575")
team_h = st.sidebar.text_input("Equipo Local", value="Damac FC")
team_a = st.sidebar.text_input("Equipo Visitante", value="Al Khlood")

st.sidebar.markdown("---")
st.sidebar.header("🏥 Ajustes Estructurales (Fase 1.3)")
bajas_h = st.sidebar.multiselect(f"Bajas {team_h}", ['arquero_titular', 'defensa_clave', 'mediocentro_creativo', 'goleador'])
bajas_a = st.sidebar.multiselect(f"Bajas {team_a}", ['arquero_titular', 'defensa_clave', 'mediocentro_creativo', 'goleador'])

# 3. Cuotas de Mercado (Fase 0)
st.sidebar.header("💰 Cuotas del Mercado")
c1 = st.sidebar.number_input("Cuota Local (1)", value=2.82)
cx = st.sidebar.number_input("Cuota Empate (X)", value=3.20)
c2 = st.sidebar.number_input("Cuota Visita (2)", value=2.42)

# --- EJECUCIÓN DEL MOTOR QUÁNTICO ---
if st.sidebar.button("🚀 EJECUTAR ANÁLISIS"):
    with st.spinner("Realizando Data Mining y Convergencia Bayesiana..."):
        try:
            # A. Llamada a la API
            client = FootballDataClient()
            raw_data = client.get_match_stats(fixture_id)
            
            # B. Cálculo de Parámetros (Fase 1.1 y 1.3)
            l_h_base = calculate_dynamic_lambdas(raw_data, team_h)
            l_a_base = calculate_dynamic_lambdas(raw_data, team_a)
            
            # Ajuste por SI (Strength Index)
            si_h = get_structural_adjustment(1.0, bajas_h)
            si_a = get_structural_adjustment(1.0, bajas_a)
            
            l_h = l_h_base * si_h
            l_a = l_a_base * si_a
            
            # C. Modelado (Fase 2)
            p_poisson = get_poisson_1x2(l_h, l_a)
            # Para este MVP, usamos Poisson como base de convergencia
            p_final = run_acbe_convergence(p_poisson, p_poisson, p_poisson) 
            
            # D. Riesgo y Entropía (Fase 3 y 4)
            h_entropy = calculate_normalized_entropy(list(p_final.values()))
            
            # E. Renderizado de Resultados (Fase 6)
            st.subheader(f"📊 Resultados del Análisis: {team_h} vs {team_a}")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("λ Local (Ajustado)", f"{l_h:.2f}")
            col2.metric("λ Visita (Ajustado)", f"{l_a:.2f}")
            col3.metric("Entropía (H)", f"{h_entropy:.4f}")

            # Matriz de Decisión
            data_matrix = []
            cuotas = [c1, cx, c2]
            labels = ["1", "X", "2"]
            
            for i, label in enumerate(labels):
                prob = p_final[label]
                q_casa = cuotas[i]
                q_justa = 1/prob if prob > 0 else 0
                val = (prob * q_casa) - 1
                stake = calculate_kelly_bayesian(q_casa, prob, h_entropy)
                
                data_matrix.append({
                    "Resultado": label,
                    "Probabilidad ACBE": f"{prob:.2%}",
                    "Cuota Justa": round(q_justa, 2),
                    "Cuota Casa": q_casa,
                    "Value (%)": f"{val:.2%}",
                    "Stake Sugerido": f"{stake:.2%}" if val > 0 else "0.00%"
                })
            
            st.table(pd.DataFrame(data_matrix))
            
            if any((p_final[labels[i]] * cuotas[i]) - 1 > 0 for i in range(3)):
                st.success("✅ ¡Oportunidad de Value detectada!")
            else:
                st.warning("⚠️ No se detectó valor suficiente bajo los filtros actuales.")

        except Exception as e:
            st.error(f"Error en la ejecución: {e}")
            st.info("Asegúrate de que el Fixture ID sea válido y que tus Secrets estén configurados.")