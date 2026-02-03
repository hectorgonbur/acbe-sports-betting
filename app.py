import streamlit as st
import pandas as pd
import numpy as np

# Configuración de Terminal Profesional
st.set_page_config(page_title="ACBE Quantum Terminal", layout="wide")
st.title("🏛️ Sistema de Inteligencia Predictiva ACBE-Kelly (Versión Cuántica)")
st.markdown("---")

# --- FASE 1: INGENIERÍA DE DATOS (INPUT MANUAL DE ALTA FIDELIDAD) ---
st.sidebar.header("📥 Fase 1: Data Mining")
team_h = st.sidebar.text_input("Local", value="Bologna")
team_a = st.sidebar.text_input("Visitante", value="AC Milan")

col1, col2 = st.sidebar.columns(2)
with col1:
    st.markdown(f"**{team_h}**")
    g_h = st.number_input("Goles (10p)", value=1.5, step=0.1)
    xg_h = st.number_input("xG (10p)", value=1.65, step=0.05)
    tiros_h = st.number_input("Tiros al arco", value=5.0, step=0.1)
    # δ_posicion para ajuste estructural
    delta_h = st.sidebar.slider(f"Σ δ Bajas {team_h}", 0.0, 0.25, 0.0, step=0.01)

with col2:
    st.markdown(f"**{team_a}**")
    g_a = st.number_input("Goles (10p)", value=1.2, step=0.1)
    xg_a = st.number_input("xG (10p)", value=1.40, step=0.05)
    tiros_a = st.number_input("Tiros al arco", value=4.5, step=0.1)
    delta_a = st.sidebar.slider(f"Σ δ Bajas {team_a}", 0.0, 0.25, 0.0, step=0.01)

st.sidebar.markdown("---")
st.sidebar.header("💰 Fase 3: Mercado")
c1 = st.sidebar.number_input("Cuota 1", value=2.90, min_value=1.01)
cx = st.sidebar.number_input("Cuota X", value=3.25, min_value=1.01)
c2 = st.sidebar.number_input("Cuota 2", value=2.45, min_value=1.01)

# Parámetros de Fase 0 y Fase 2.3
entropy = st.sidebar.slider("Entropía de Liga (H)", 0.30, 0.90, 0.62)
steam_move = st.sidebar.slider("Steam / Sharp Move (σ)", 0.0, 0.05, 0.0, step=0.01)

# --- FASE 0: FILTRO DE MERCADO (PRE-ANÁLISIS DE SEGURIDAD) ---
or_val = (1/c1 + 1/cx + 1/c2) - 1
evasion = False

if c1 < 1.60 or c2 < 1.60: evasion = "Cuota mínima < 1.60"
elif or_val > 0.07: evasion = f"Overround prohibitivo ({or_val:.2%})"
elif entropy > 0.72: evasion = f"Entropía excesiva (H={entropy:.2f})"

# --- MOTOR MATEMÁTICO (FASE 2) ---
if st.sidebar.button("🚀 EJECUTAR ANÁLISIS") and not evasion:
    with st.spinner("Ejecutando Convergencia Bayesiana..."):
        
        # 2.1 Cálculo de λ (Poisson Ajustado)
        f_forma_h = xg_h / g_h if g_h > 0 else 1.0
        f_forma_a = xg_a / g_a if g_a > 0 else 1.0
        
        # Ajuste estructural SI_adj = SI_base * (1 - Σ δ)
        l_h = g_h * f_forma_h * 1.10 * (1 - delta_h) 
        l_a = g_a * f_forma_a * 0.90 * (1 - delta_a)

        # 2.2 Simulación Monte Carlo (10,000 iteraciones)
        mc_h = np.random.poisson(l_h, 10000)
        mc_a = np.random.poisson(l_a, 10000)
        
        p1_mc = np.mean(mc_h > mc_a)
        px_mc = np.mean(mc_h == mc_a)
        p2_mc = np.mean(mc_h < mc_a)
        
        # 2.3 Convergencia Bayesiana (Simplificada para manual)
        p_final = {"1": p1_mc - steam_move, "X": px_mc, "2": p2_mc - steam_move}

        # --- FASE 4: GESTIÓN DE CAPITAL (KELLY BAYESIANO FRACCIONAL) ---
        k_adj = 1 / (1 + entropy)
        
        res_df = []
        for label, prob, cuota in zip(["1", "X", "2"], p_final.values(), [c1, cx, c2]):
            # FASE 3: Análisis de Valor
            value = (prob * cuota) - 1
            
            # Cálculo de Stake Kelly Fraccional (Half-Kelly obligatorio)
            b = cuota - 1
            q = 1 - prob
            f_star = (b * prob - q) / b if b > 0 else 0
            
            # Condición Crítica: Value >= 3% y f* > 0
            stake_final = max(0, f_star * k_adj * 0.5) if value >= 0.03 else 0
            
            res_df.append({
                "Resultado": label,
                "Probabilidad %": f"{prob:.2%}",
                "Cuota Casa": f"{cuota:.2f}",
                "Cuota Justa": f"{1/prob:.2f}" if prob > 0 else "N/A",
                "Value %": f"{value:.2%}",
                "Stake Kelly %": f"{stake_final:.2%}"
            })

        # --- FASE 5: MATRIZ FINAL DE RESULTADOS ---
        st.subheader(f"📊 Matriz Final: {team_h} vs {team_a}")
        st.table(pd.DataFrame(res_df))
        
        # Validación de Retorno Esperado
        if any(float(r["Value %"].strip('%'))/100 >= 0.03 for r in res_df):
            st.success("✅ Oportunidad detectada con Esperanza Matemática Positiva (EV+).")
        else:
            st.info("⚖️ Mercado Eficiente: No se detecta ineficiencia de mercado (Value < 3%).")

elif evasion:
    st.error(f"🚫 Evasión de Riesgo: {evasion}")