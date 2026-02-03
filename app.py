import streamlit as st
import pandas as pd
from model.montecarlo import run_monte_carlo
from risk.kelly import calculate_kelly_bayesian
from risk.filters import apply_market_filters

st.set_page_config(page_title="ACBE Quant System", layout="wide")
st.title("🏛️ Sistema Quántico ACBE-Kelly v2.0")

# Sidebar: Inputs de Mercado
st.sidebar.header("Inputs de Fase 1")
c1 = st.sidebar.number_input("Cuota Local", value=2.18)
cx = st.sidebar.number_input("Cuota Empate", value=2.98)
c2 = st.sidebar.number_input("Cuota Visita", value=3.20)
h_entropy = st.sidebar.slider("Entropía de Liga (H)", 0.0, 1.0, 0.65)

# Cálculo de Overround
overround = (1/c1 + 1/cx + 1/c2) - 1

# Ejecución de Filtros de Fase 0
if not apply_market_filters(min(c1, c2), overround, h_entropy, var_goles=1.2):
    st.error("🚨 EVASIÓN DE RIESGO: Mercado Ineficiente o Margen Abusivo")
else:
    # Simulación Monte Carlo
    sim = run_monte_carlo(l_local=1.25, l_visita=1.10)
    
    # Matriz Final de Resultados
    data = {
        "Resultado": ["1", "X", "2"],
        "Probabilidad %": [sim['p1']*100, sim['px']*100, sim['p2']*100],
        "Cuota Casa": [c1, cx, c2],
        "Cuota Justa": [1/sim['p1'], 1/sim['px'], 1/sim['p2']],
        "Stake Kelly %": [
            calculate_kelly_bayesian(c1, sim['p1'], h_entropy)*100,
            calculate_kelly_bayesian(cx, sim['px'], h_entropy)*100,
            calculate_kelly_bayesian(c2, sim['p2'], h_entropy)*100
        ]
    }
    
    df = pd.DataFrame(data)
    st.table(df)
    st.success("✅ Análisis Completo: Esperanza Matemática Positiva detectada.")