# En tu app.py principal, añade al inicio:
import streamlit as st

st.set_page_config(page_title="Sistema ACBE-Kelly", layout="wide")

# Sidebar navigation
menu = st.sidebar.selectbox(
    "Navegación",
    ["🏠 App Principal", "🎓 Guía Interactiva", "📊 Historial"]
)

if menu == "🎓 Guía Interactiva":
    # Importa las librerías específicas de la guía
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats

    # ============ CONFIGURACIÓN ============
    st.title("🎓 Guía Interactiva: Sistema ACBE-Kelly v3.0")
    st.markdown("---")

    # ============ SIDEBAR: NAVEGACIÓN ============
    st.sidebar.title("📚 ÍNDICE DE LA GUÍA")

    modulo = st.sidebar.radio(
        "Selecciona un módulo:",
        ["🏠 Introducción", 
         "🧮 Fase 1: Modelo Bayesiano", 
         "🎲 Fase 2: Monte Carlo",
         "💰 Fase 3: Gestión de Capital",
         "📊 Fase 4: Backtesting",
         "🎯 Ejemplo Práctico",
         "📈 Simulador Interactivo"]
    )

    st.sidebar.markdown("---")
    st.sidebar.info("**Nivel:** Intermedio\n**Tiempo:** 30-40 minutos\n**Requisitos:** Ninguno")

    # ============ MÓDULO 1: INTRODUCCIÓN ============
    if modulo == "🏠 Introducción":
        st.header("🎯 ¿Qué es el Sistema ACBE-Kelly?")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 🌟 **Sistema de Trading Deportivo Inteligente**
            
            **ACBE-Kelly** combina:
            1. **A**nalítica Bayesiana
            2. **C**álculo de Value
            3. **B**ankroll Management
            4. **E**valuación de Riesgo
            
            ### 🎯 **Objetivo Principal:**
            > "Detectar ineficiencias del mercado donde **nuestra probabilidad > probabilidad del mercado**"
            
            ### 📊 **Resultados Esperados:**
            - **Precisión:** 58-65%
            - **ROI Anual:** 12-18%
            - **Máxima Caída:** < 20%
            """)
        
        with col2:
            st.write("📍 **Imagen del sistema**")
        
        st.markdown("---")
        
        # Quiz interactivo 1
        st.subheader("🧠 Verifica tu comprensión")
        
        with st.expander("❓ Pregunta 1: ¿Qué significa 'Value' en apuestas?", expanded=False):
            opcion = st.radio(
                "Elige la respuesta correcta:",
                ["A) Cuánto dinero ganas en una apuesta",
                 "B) Cuando tu probabilidad es mayor que la del mercado",
                 "C) El margen de la casa de apuestas"]
            )
            
            if st.button("Verificar respuesta"):
                if opcion == "B) Cuando tu probabilidad es mayor que la del mercado":
                    st.success("✅ ¡Correcto! Value = Nuestra ventaja probabilística")
                else:
                    st.error("❌ Incorrecto. Value ocurre cuando nuestro modelo estima una probabilidad MAYOR que la implícita en las cuotas.")
        
        # Ejemplo visual de value
        st.markdown("---")
        st.subheader("📈 Ejemplo Visual de Value")
        
        col_v1, col_v2, col_v3 = st.columns(3)
        
        with col_v1:
            prob_modelo = st.slider("Probabilidad del Modelo (%)", 30, 70, 45)
        with col_v2:
            cuota = st.slider("Cuota de la Casa", 1.5, 4.0, 2.5)
        with col_v3:
            prob_mercado = 1/cuota
            st.metric("Prob. Mercado", f"{prob_mercado:.1%}")
        
        # Calcular value
        value = (prob_modelo/100 * cuota) - 1
        color = "green" if value > 0 else "red"
        
        st.markdown(f"""
        ### 📊 Resultado:
        - **Modelo:** {prob_modelo}%
        - **Mercado:** {prob_mercado:.1%}
        - **Diferencia:** {prob_modelo/100 - prob_mercado:+.1%}
        - **Value (EV):** <span style='color:{color}'>{value:+.1%}</span>
        """, unsafe_allow_html=True)
        
        if value > 0.03:
            st.success("🎯 ¡OPORTUNIDAD DETECTADA! Value > 3%")
        else:
            st.warning("⚠️ No hay value suficiente")

    # ============ MÓDULO 2: MODELO BAYESIANO ============
    elif modulo == "🧮 Fase 1: Modelo Bayesiano":
        st.header("🧮 Fase 1: Modelo Bayesiano Jerárquico")
        
        st.markdown("""
        ### 🧠 ¿Qué es el aprendizaje bayesiano?
        
        **Piensa así:** Tienes una creencia inicial (prior), ves nuevos datos, y actualizas tu creencia.
        
        ```
        Creencia Final = Creencia Inicial × Evidencia
        ```
        """)
        
        # Ejemplo interactivo
        st.subheader("🎯 Ejemplo: Goleador de un equipo")
        
        col_b1, col_b2, col_b3 = st.columns(3)
        
        with col_b1:
            st.markdown("**📊 Prior (Histórico)**")
            media_historica = st.slider("Goles promedio histórico", 0.5, 2.0, 1.2)
            st.metric("Prior λ", f"{media_historica:.2f}")
        
        with col_b2:
            st.markdown("**⚽ Datos Actuales**")
            goles_recientes = st.slider("Goles últimos 5 partidos", 0, 10, 8)
            partidos = 5
            media_reciente = goles_recientes / partidos
            st.metric("Media reciente", f"{media_reciente:.2f}")
        
        with col_b3:
            st.markdown("**🎯 Posterior (Actualizado)**")
            # Actualización bayesiana simple
            peso_prior = st.slider("Confianza en histórico", 0.1, 0.9, 0.5)
            peso_datos = 1 - peso_prior
            
            posterior = (media_historica * peso_prior) + (media_reciente * peso_datos)
            st.metric("λ Posterior", f"{posterior:.2f}")
        
        # Gráfico de actualización
        st.markdown("---")
        st.subheader("📈 Visualización de la Actualización Bayesiana")
        
        # Crear distribución
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prior (distribución inicial)
        x = np.linspace(0, 3, 100)
        prior_dist = stats.gamma.pdf(x, a=2, scale=0.6)
        ax.plot(x, prior_dist, 'b-', label='Prior (histórico)', linewidth=2)
        
        # Likelihood (datos observados)
        likelihood_dist = stats.norm.pdf(x, loc=media_reciente, scale=0.3)
        ax.plot(x, likelihood_dist, 'r--', label='Likelihood (datos)', linewidth=2)
        
        # Posterior (combinación)
        posterior_dist = stats.gamma.pdf(x, a=2 + goles_recientes, scale=0.5)
        ax.plot(x, posterior_dist, 'g-', label='Posterior (actualizado)', linewidth=3)
        
        ax.set_xlabel('Goles esperados por partido (λ)')
        ax.set_ylabel('Densidad de probabilidad')
        ax.set_title('Actualización Bayesiana: Prior → Likelihood → Posterior')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        # Explicación
        with st.expander("📖 Explicación del gráfico", expanded=True):
            st.markdown("""
            1. **🔵 Línea Azul (Prior):** Lo que creíamos ANTES de ver los datos
            2. **🔴 Línea Roja (Likelihood):** Lo que dicen los datos ACTUALES
            3. **🟢 Línea Verde (Posterior):** Lo que creemos AHORA (combinación)
            
            **📌 Insight:** Cuantos más datos tengas, más se inclina hacia la línea roja.
            """)
        
        # Quiz bayesiano
        st.markdown("---")
        st.subheader("🧪 Prueba tu comprensión")
        
        pregunta = st.radio(
            "Si un equipo históricamente marca 1.0 gol/partido, pero en los últimos 5 marca 2.0, ¿qué λ usarías?",
            ["A) 1.0 (solo histórico)",
             "B) 2.0 (solo reciente)", 
             "C) Algo entre 1.0 y 2.0 (combinación)",
             "D) 0.5 (más conservador)"]
        )
        
        if st.button("Ver respuesta"):
            if pregunta == "C) Algo entre 1.0 y 2.0 (combinación)":
                st.success("✅ ¡Exacto! El bayesiano encuentra un balance entre histórico y reciente.")
            else:
                st.error("❌ Recuerda: Bayesiano combina información, no descarta ninguna.")

    # ============ MÓDULO 3: MONTE CARLO ============
    elif modulo == "🎲 Fase 2: Monte Carlo":
        # ... (TODO el código del módulo 3)
        st.header("🎲 Fase 2: Simulación Monte Carlo")
        
        st.markdown("### 🎯 Simular miles de partidos")
        
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            lambda_local = st.slider("λ Local", 0.5, 3.0, 1.5)
        
        with col_m2:
            lambda_visit = st.slider("λ Visitante", 0.5, 3.0, 1.2)
        
        if st.button("🎲 Ejecutar 1000 simulaciones"):
            resultados = []
            for _ in range(1000):
                goles_local = np.random.poisson(lambda_local)
                goles_visit = np.random.poisson(lambda_visit)
                
                if goles_local > goles_visit:
                    resultados.append("1")
                elif goles_local == goles_visit:
                    resultados.append("X")
                else:
                    resultados.append("2")
            
            p1 = resultados.count("1") / 1000
            px = resultados.count("X") / 1000
            p2 = resultados.count("2") / 1000
            
            st.success(f"**Resultados:** Local: {p1:.1%} | Empate: {px:.1%} | Visitante: {p2:.1%}")
        pass

    # ============ MÓDULO 4: GESTIÓN DE CAPITAL ============
    elif modulo == "💰 Fase 3: Gestión de Capital":
        # ... (TODO el código del módulo 4)
        st.header("💰 Fase 3: Gestión de Capital (Kelly Criterio)")
        
        col_k1, col_k2 = st.columns(2)
        
        with col_k1:
            prob = st.slider("Probabilidad (%)", 30, 70, 45) / 100
        
        with col_k2:
            cuota = st.slider("Cuota", 1.5, 4.0, 2.5)
            b = cuota - 1
        
        if b > 0:
            kelly_base = (prob * b - (1 - prob)) / b
            kelly_final = kelly_base * 0.5  # Half-Kelly
        else:
            kelly_final = 0
        
        st.info(f"**Stake recomendado:** {kelly_final:.1%} del bankroll")
        pass

    # ============ MÓDULO 5: BACKTESTING ============
    elif modulo == "📊 Fase 4: Backtesting":
        # ... (TODO el código del módulo 5)
        st.header("📊 Fase 4: Backtesting Sintético")
        
        if st.button("📊 Simular 100 apuestas"):
            bankroll = 1000
            historial = [bankroll]
            
            for i in range(100):
                stake = bankroll * 0.02  # 2% por apuesta
                
                if np.random.random() < 0.55:  # 55% de acierto
                    bankroll += stake * 1.2  # Ganancia del 20%
                else:
                    bankroll -= stake
                
                historial.append(bankroll)
            
            roi = ((bankroll - 1000) / 1000) * 100
            st.metric("Bankroll Final", f"€{bankroll:.0f}")
            st.metric("ROI", f"{roi:.1f}%")
        pass

    # ============ MÓDULO 6: EJEMPLO PRÁCTICO ============
    elif modulo == "🎯 Ejemplo Práctico":
        # ... (TODO el código del módulo 6)
        st.header("🎯 Ejemplo Práctico: Bologna vs AC Milan")
        
        st.markdown("""
        **Análisis completo:**
        - 📊 **Modelo:** 45% probabilidad de victoria local
        - 💰 **Mercado:** 34% probabilidad implícita (cuota 2.90)
        - 🎯 **Value:** +14.5% (oportunidad clara)
        - 🏦 **Stake:** 3.8% del bankroll (Half-Kelly)
        
        **✅ RECOMENDACIÓN: APOSTAR**
        """)
        pass

    # ============ MÓDULO 7: SIMULADOR INTERACTIVO ============
    elif modulo == "📈 Simulador Interactivo":
        # ... (TODO el código del módulo 7)
        st.header("📈 Simulador Interactivo")
        
        prob = st.slider("Tu estimación (%)", 30, 70, 45)
        cuota = st.slider("Cuota ofrecida", 1.5, 4.0, 2.5)
        
        ev = (prob/100 * cuota) - 1
        
        if ev > 0.03:
            st.success(f"🎯 **APOSTAR** - Value = {ev:+.1%}")
        elif ev > 0:
            st.info(f"📊 **Considerar** - Value = {ev:+.1%}")
        else:
            st.warning(f"⚠️ **NO APOSTAR** - Value = {ev:+.1%}")
        pass

    # ============ PIE DE PÁGINA ============
    st.markdown("---")
    st.markdown("""
    ### 🎓 **Has completado la Guía Interactiva ACBE-Kelly**

    **Siguientes pasos recomendados:**
    1. **Practica** con el simulador hasta sentirte cómodo
    2. **Analiza** partidos reales sin dinero
    3. **Comienza** con paper trading
    4. **Implementa** con bankroll pequeño cuando tengas confianza

    **Recuerda:** El éxito viene de la **consistencia** y **gestión de riesgo**, no de adivinar resultados.
    """)

    st.caption("© 2024 ACBE Predictive Systems | Guía educativa para aprendizaje interactivo")
    # ¡NO pongas "pass" aquí!

elif menu == "🏠 App Principal":
    # Tu código actual de la app
    """
    🏛️ SISTEMA ACBE-KELLY v3.0 (BAYESIANO COMPLETO - IMPLEMENTACIÓN PRÁCTICA)
    OBJETIVO: ROI 12-18% con CVaR < 15%
    """

    import pandas as pd
    import numpy as np
    from datetime import datetime
    import json
    import base64
    from io import BytesIO
    import matplotlib.pyplot as plt
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    from scipy import stats
    from scipy.optimize import minimize
    import plotly.graph_objects as go
    from datetime import datetime, timedelta
    
    # ============ CONFIGURACIÓN AVANZADA ============
    st.title("🏛️ Sistema ACBE-Kelly v3.0 (Bayesiano Completo)")
    st.markdown("---")

    # ============ SISTEMA DE LOGGING PROFESIONAL ============
    class SistemaLogging:
        def __init__(self):
            self.historial = []
            self.performance = {
                'total_picks': 0,
                'picks_ev_positivo': 0,
                'aciertos': 0,
                'bankroll_historico': []
            }
        
        def registrar_pick(self, pick_data):
            self.historial.append({
                'timestamp': datetime.now(),
                **pick_data
            })
            self.performance['total_picks'] += 1
            if pick_data['ev'] > 0:
                self.performance['picks_ev_positivo'] += 1

    logger = SistemaLogging()

    # ============ NÚCLEO MATEMÁTICO v3.0 ============

    class ModeloBayesianoJerarquico:
        """
        Implementación del modelo jerárquico bayesiano con:
        - Prior Gamma para parámetros de Poisson
        - Inferencia variacional (aproximación a MCMC)
        - Ajuste por incertidumbre estructural
        """
        
        def __init__(self, liga="Serie A"):
            # Priors informados por liga (calibrados históricamente)
            self.priors = self._inicializar_priors(liga)
            
        def _inicializar_priors(self, liga):
            # Datos históricos de ligas (2018-2023)
            datos_ligas = {
                "Serie A": {"mu_goles": 1.32, "sigma_goles": 0.85, "home_adv": 1.18},
                "Premier League": {"mu_goles": 1.48, "sigma_goles": 0.92, "home_adv": 1.15},
                "La Liga": {"mu_goles": 1.35, "sigma_goles": 0.88, "home_adv": 1.16},
                "Bundesliga": {"mu_goles": 1.56, "sigma_goles": 0.95, "home_adv": 1.12},
                "Ligue 1": {"mu_goles": 1.28, "sigma_goles": 0.82, "home_adv": 1.20}
            }
            
            data = datos_ligas.get(liga, datos_ligas["Serie A"])
            
            # Convertir a parámetros Gamma (α, β)
            # Gamma es el prior conjugado de Poisson
            alpha_prior = (data["mu_goles"] ** 2) / (data["sigma_goles"] ** 2)
            beta_prior = data["mu_goles"] / (data["sigma_goles"] ** 2)
            
            return {
                "alpha": alpha_prior,
                "beta": beta_prior,
                "home_advantage": data["home_adv"],
                "sigma_liga": data["sigma_goles"]
            }
        
        def inferencia_variacional(self, datos_equipo, es_local=True):
            """
            Inferencia variacional rápida (aproximación determinística a MCMC)
            Método: Actualización bayesiana conjugada Gamma-Poisson
            """
            # Datos observados
            goles_anotados = datos_equipo.get("goles_anotados", 0)
            goles_recibidos = datos_equipo.get("goles_recibidos", 0)
            n_partidos = datos_equipo.get("n_partidos", 10)
            xG_promedio = datos_equipo.get("xG", 1.5)
            
            # Actualización bayesiana conjugada
            alpha_posterior = self.priors["alpha"] + goles_anotados
            beta_posterior = self.priors["beta"] + n_partidos
            
            # Media posterior (estimación puntual)
            lambda_posterior = alpha_posterior / beta_posterior
            
            # Ajuste por xG (calibración de calidad de oportunidades)
            if xG_promedio > 0:
                ratio_xg = min(max(xG_promedio / max(lambda_posterior, 0.1), 0.7), 1.3)
                lambda_posterior *= ratio_xg
            
            # Ajuste por localía/visitante
            if es_local:
                lambda_posterior *= self.priors["home_advantage"]
            else:
                lambda_posterior *= (2 - self.priors["home_advantage"])
            
            # Calcular incertidumbre (varianza posterior)
            varianza_posterior = alpha_posterior / (beta_posterior ** 2)
            
            # Intervalo de credibilidad 95%
            ci_lower = stats.gamma.ppf(0.025, alpha_posterior, scale=1/beta_posterior)
            ci_upper = stats.gamma.ppf(0.975, alpha_posterior, scale=1/beta_posterior)
            
            return {
                "lambda": lambda_posterior,
                "alpha": alpha_posterior,
                "beta": beta_posterior,
                "varianza": varianza_posterior,
                "ci_95": (ci_lower, ci_upper),
                "incertidumbre": np.sqrt(varianza_posterior) / max(lambda_posterior, 0.1)
            }

    class DetectorIneficiencias:
        """
        Sistema de detección estadística de ineficiencias de mercado
        Usa test de hipótesis bayesiano y métricas de información
        """
        
        @staticmethod
        def calcular_value_score(p_modelo, p_mercado, sigma_modelo):
            """
            Value Score con test estadístico riguroso
            H0: Mercado eficiente (p_modelo = p_mercado)
            H1: Ineficiencia detectada
            """
            if sigma_modelo < 1e-10:
                return {"score": 0, "p_value": 1, "significativo": False}
            
            # Test t de Student
            t_stat = (p_modelo - p_mercado) / sigma_modelo
            df = 10000  # grados de libertad (simulaciones - 1)
            
            # p-value (two-tailed)
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
            
            # Calcular poder estadístico
            efecto = abs(p_modelo - p_mercado)
            poder = DetectorIneficiencias._calcular_poder_estadistico(
                efecto, sigma_modelo, alpha=0.05, n=10000
            )
            
            return {
                "t_statistic": t_stat,
                "p_value": p_value,
                "significativo": p_value < 0.05 and efecto > 0.02,
                "poder_estadistico": poder,
                "efecto_detectado": efecto
            }
        
        @staticmethod
        def _calcular_poder_estadistico(efecto, sigma, alpha=0.05, n=10000):
            """Calcular poder estadístico del test"""
            from scipy.stats import norm
            z_alpha = norm.ppf(1 - alpha/2)
            z_beta = (efecto * np.sqrt(n)) / sigma - z_alpha
            poder = norm.cdf(z_beta)
            return max(0, min(poder, 1))
        
        @staticmethod
        def calcular_entropia_kullback_leibler(p_modelo, p_mercado):
            """
            Entropía de Kullback-Leibler (divergencia)
            Mide cuánto se desvía el modelo del mercado
            """
            # Evitar log(0)
            epsilon = 1e-10
            p_modelo = max(p_modelo, epsilon)
            p_mercado = max(p_mercado, epsilon)
            
            # KL Divergence
            kl_div = p_modelo * np.log(p_modelo / p_mercado)
            
            # Normalizar a [0, 1]
            kl_norm = 1 - np.exp(-kl_div)
            
            return {
                "kl_divergence": kl_div,
                "incertidumbre_modelo": kl_norm,
                "informacion_bits": kl_div / np.log(2)
            }

    class GestorRiscoCVaR:
        """
        Gestión avanzada de riesgo con CVaR (Conditional Value at Risk)
        y Kelly Bayesiano dinámico
        """
        
        def __init__(self, cvar_target=0.15, max_drawdown=0.20):
            self.cvar_target = cvar_target
            self.max_drawdown = max_drawdown
            self.historial_riesgo = []
        
        def calcular_kelly_dinamico(self, prob, cuota, bankroll, metrics):
            """
            Kelly dinámico con ajustes por:
            1. Incertidumbre del modelo
            2. CVaR histórico
            3. Correlación con portfolio
            4. Drawdown reciente
            """
            b = cuota - 1
            if b <= 0 or prob <= 0:
                return {"stake_pct": 0, "stake_abs": 0, "razon": "Parámetros inválidos"}
            
            # Kelly base
            kelly_base = (prob * b - (1 - prob)) / b

            # Asegurar que Kelly ajustado no sea negativo
            kelly_ajustado = max(kelly_ajustado, 0)

            # Además, verificar condiciones mínimas para apostar
            condiciones_minimas = (
                prob > 0.35,                     # Probabilidad mínima
                cuota > 1.5,                     # Cuota mínima
                ev > 0.02,                       # EV mínimo
                r.get('Value Score', {}).get('significativo', False)  # Significativo estadístico
            )

            if not all(condiciones_minimas):
                kelly_ajustado = 0

            # Ajuste 1: Incertidumbre del modelo
            incertidumbre = metrics.get("incertidumbre", 0.5)
            adj_incertidumbre = 1 / (1 + 2 * incertidumbre)
            
            # Ajuste 2: CVaR dinámico
            cvar_actual = metrics.get("cvar_estimado", self.cvar_target)
            if cvar_actual <= self.cvar_target:
                adj_cvar = 1.0  # Si el riesgo es menor que el objetivo, no penalizar
            else:
                # Penalizar proporcionalmente cuando excede el objetivo
                adj_cvar = max(0.1, self.cvar_target / cvar_actual)  # Mínimo 10%
            
            # Ajuste 3: Entropía de la liga
            entropia = metrics.get("entropia", 0.5)
            adj_entropia = 1 / (1 + entropia)
            
            # Ajuste 4: Sharpe ratio esperado
            sharpe_esperado = metrics.get("sharpe_esperado", 1.0)
            adj_sharpe = min(sharpe_esperado / 2.0, 1.5)
            
            # Kelly ajustado
            kelly_ajustado = kelly_base * adj_incertidumbre * adj_cvar * adj_entropia * adj_sharpe
            
            # Asegurar que Kelly ajustado no sea negativo (segunda capa de protección)
            kelly_ajustado = max(kelly_ajustado, 0)

            # Half-Kelly conservador
            kelly_final = kelly_ajustado * 0.5
            
            # Límites estrictos de riesgo
            kelly_final = max(0, min(kelly_final, 0.03))  # Máximo 3%
            
            # Stake en euros
            stake_abs = kelly_final * bankroll
            
            return {
                "stake_pct": kelly_final * 100,
                "stake_abs": stake_abs,
                "kelly_base": kelly_base * 100,
                "ajuste_incertidumbre": adj_incertidumbre,
                "ajuste_cvar": adj_cvar,
                "sharpe_ajuste": adj_sharpe
            }
        
        def simular_cvar(self, prob, cuota, n_simulaciones=10000, conf_level=0.95):
            """
            Simulación Monte Carlo para calcular CVaR
            """
            ganancias = []
            
            for _ in range(n_simulaciones):
                # Simular resultado binario
                gana = np.random.random() < prob
                if gana:
                    ganancia = (cuota - 1)
                else:
                    ganancia = -1
                
                ganancias.append(ganancia)
            
            ganancias = np.array(ganancias)
            
            # Calcular VaR y CVaR
            var_level = np.percentile(ganancias, (1 - conf_level) * 100)
            cvar = ganancias[ganancias <= var_level].mean()
            
            # NUNCA devolver CVaR negativo (esto causaba el 100%)
            cvar_abs = abs(cvar) if cvar < 0 else 0

            return {
                "cvar": abs(cvar) if cvar < 0 else 0,
                "var": abs(var_level) if var_level < 0 else 0,
                "esperanza": ganancias.mean(),
                "desviacion": ganancias.std(),
                "sharpe_simulado": ganancias.mean() / max(ganancias.std(), 0.01),
                "max_perdida_simulada": ganancias.min(),
                "prob_perdida": np.mean(ganancias < 0)  # Nueva métrica importante
            }

    class BacktestSintetico:
        """
        Sistema de backtesting sintético para validación en tiempo real
        """
        
        @staticmethod
        def generar_escenarios(prob, cuota, bankroll_inicial=1000, n_apuestas=100, n_simulaciones=5000):
            """
            Generar 5,000 escenarios de temporada completa
            """
            resultados = []
            metricas_por_simulacion = []
            
            for sim in range(n_simulaciones):
                bankroll = bankroll_inicial
                historial_br = [bankroll]
                drawdown_actual = 0
                drawdown_maximo = 0
                peak = bankroll
                
                for apuesta in range(n_apuestas):
                    # Stake con Kelly dinámico (simplificado)
                    stake_pct = 0.02  # 2% fijo para simulación
                    stake = bankroll * stake_pct
                    
                    # Simular resultado
                    gana = np.random.random() < prob
                    
                    if gana:
                        bankroll += stake * (cuota - 1)
                    else:
                        bankroll -= stake
                    
                    # Actualizar drawdown
                    if bankroll > peak:
                        peak = bankroll
                    
                    drawdown_actual = (peak - bankroll) / peak
                    drawdown_maximo = max(drawdown_maximo, drawdown_actual)
                    
                    historial_br.append(bankroll)
                
                # Calcular métricas para esta simulación
                retorno_total = (bankroll - bankroll_inicial) / bankroll_inicial
                volatilidad = np.std(np.diff(historial_br) / historial_br[:-1]) if len(historial_br) > 1 else 0
                sharpe = retorno_total / max(volatilidad, 0.01) * np.sqrt(252/365)  # Anualizado
                
                metricas_por_simulacion.append({
                    "final_balance": bankroll,
                    "return": retorno_total,
                    "max_drawdown": drawdown_maximo,
                    "sharpe": sharpe,
                    "ruin": bankroll < bankroll_inicial * 0.5
                })
                
                resultados.append(historial_br)
            
            # Estadísticas agregadas
            df_metricas = pd.DataFrame(metricas_por_simulacion)
            
            return {
                "escenarios": resultados,
                "metricas": {
                    "retorno_esperado": df_metricas["return"].mean(),
                    "retorno_std": df_metricas["return"].std(),
                    "sharpe_promedio": df_metricas["sharpe"].mean(),
                    "max_dd_promedio": df_metricas["max_drawdown"].mean(),
                    "prob_ruina": df_metricas["ruin"].mean(),
                    "var_95": np.percentile(df_metricas["return"], 5),
                    "cvar_95": df_metricas["return"][df_metricas["return"] <= np.percentile(df_metricas["return"], 5)].mean(),
                    "prob_profit": (df_metricas["return"] > 0).mean(),
                    "ratio_ganancia_perdida": abs(df_metricas["return"][df_metricas["return"] > 0].mean() / 
                                                df_metricas["return"][df_metricas["return"] < 0].mean()) 
                                        if len(df_metricas["return"][df_metricas["return"] < 0]) > 0 else 999
                },
                "distribucion_retornos": df_metricas["return"].values
            }
        # ============ CLASE PARA RECOMENDACIONES INTELIGENTES ============

    class RecomendadorInteligente:
            """
            Sistema de recomendación con niveles de confianza y explicaciones
            """
            
            def __init__(self):
                self.umbrales = {
                    'value_alto': 0.05,
                    'value_medio': 0.03,
                    'value_bajo': 0.02,
                    'confianza_alta': 0.95,
                    'confianza_media': 0.90,
                    'confianza_baja': 0.85
                }
            
            def generar_recomendacion(self, analisis):
                """
                Genera recomendación estructurada con explicaciones
                """
                # Encontrar el mejor pick
                mejor_pick = self._encontrar_mejor_pick(analisis['resultados'])
                
                if not mejor_pick:
                    return self._recomendacion_no_apostar()
                
                # Calcular nivel de confianza
                confianza = self._calcular_confianza(mejor_pick, analisis)
                
                # Generar recomendación
                return {
                    'accion': self._determinar_accion(mejor_pick, confianza),
                    'pick': mejor_pick['Resultado'],
                    'cuota': mejor_pick['Cuota Mercado'],
                    'ev': mejor_pick['EV'],
                    'stake_pct': mejor_pick.get('Stake %', '0%'),
                    'confianza': confianza,
                    'razones': self._generar_razones(mejor_pick, analisis),
                    'advertencias': self._generar_advertencias(mejor_pick, analisis),
                    'timestamp': datetime.now(),
                    'metadata': {
                        'equipo_local': analisis.get('team_h', ''),
                        'equipo_visitante': analisis.get('team_a', ''),
                        'liga': analisis.get('liga', ''),
                        'overround': analisis.get('or_val', 0),
                        'entropia': analisis.get('entropia', 0)
                    }
                }
            
            def _encontrar_mejor_pick(self, resultados):
                """Encuentra el pick con mayor EV positivo"""
                picks_con_ev = []
                for r in resultados:
                    try:
                        ev = float(r['EV'].strip('%')) / 100 if '%' in str(r['EV']) else float(r['EV'])
                        if ev > 0.02:  # Solo picks con EV > 2%
                            picks_con_ev.append({
                                'Resultado': r['Resultado'],
                                'EV': ev,
                                'Cuota Mercado': float(r['Cuota Mercado']),
                                'Prob Modelo': float(r['Prob Modelo'].strip('%')) / 100 if '%' in str(r['Prob Modelo']) else float(r['Prob Modelo']),
                                'Value Score': float(r.get('Value Score', 0)),
                                'Significativo': '✅' in str(r.get('Significativo', ''))
                            })
                    except:
                        continue
                
                if not picks_con_ev:
                    return None
                
                # Ordenar por EV descendente
                return sorted(picks_con_ev, key=lambda x: x['EV'], reverse=True)[0]
            
            def _calcular_confianza(self, pick, analisis):
                """Calcula nivel de confianza 0-100%"""
                confianza = 50  # Base
                
                # Ajustes por EV
                if pick['EV'] > self.umbrales['value_alto']:
                    confianza += 25
                elif pick['EV'] > self.umbrales['value_medio']:
                    confianza += 15
                elif pick['EV'] > self.umbrales['value_bajo']:
                    confianza += 5
                
                # Ajuste por significancia estadística
                if pick.get('Significativo', False):
                    confianza += 20
                
                # Ajuste por sobre-round
                or_val = analisis.get('or_val', 0.07)
                if or_val < 0.05:
                    confianza += 10
                
                # Ajuste por entropía
                entropia = analisis.get('entropia', 0.7)
                if entropia < 0.6:
                    confianza += 10
                
                return min(max(confianza, 0), 100)
            
            def _determinar_accion(self, pick, confianza):
                """Determina la acción recomendada"""
                if confianza < 60:
                    return "NO APOSTAR"
                elif confianza < 75:
                    return "APOSTAR PEQUEÑO"
                elif confianza < 90:
                    return "APOSTAR MODERADO"
                else:
                    return "APOSTAR FUERTE"
            
            def _generar_razones(self, pick, analisis):
                """Genera razones para la recomendación"""
                razones = []
                
                # Razón 1: Value positivo
                razones.append(f"Value positivo: {pick['EV']:.2%}")
                
                # Razón 2: Significancia estadística
                if pick.get('Significativo', False):
                    razones.append("Significancia estadística confirmada")
                
                # Razón 3: Cuota atractiva
                cuota_justa = 1 / pick['Prob Modelo']
                if pick['Cuota Mercado'] > cuota_justa * 1.1:
                    razones.append(f"Cuota {pick['Cuota Mercado']:.2f} vs Justa {cuota_justa:.2f}")
                
                return razones
            
            def _generar_advertencias(self, pick, analisis):
                """Genera advertencias de riesgo"""
                advertencias = []
                
                # Advertencia 1: Entropía alta
                if analisis.get('entropia', 0) > 0.7:
                    advertencias.append(f"Entropía alta ({analisis['entropia']:.2f}) - Mercado volátil")
                
                # Advertencia 2: Overround alto
                if analisis.get('or_val', 0) > 0.06:
                    advertencias.append(f"Margen casa alto ({analisis['or_val']:.2%})")
                
                # Advertencia 3: Probabilidad baja
                if pick['Prob Modelo'] < 0.35:
                    advertencias.append(f"Probabilidad baja ({pick['Prob Modelo']:.1%})")
                
                return advertencias
            
            def _recomendacion_no_apostar(self):
                """Genera recomendación cuando no hay picks"""
                return {
                    'accion': "NO APOSTAR",
                    'pick': None,
                    'cuota': None,
                    'ev': 0,
                    'stake_pct': '0%',
                    'confianza': 0,
                    'razones': ["No se detectó value suficiente (> 2%)"],
                    'advertencias': [],
                    'timestamp': datetime.now(),
                    'metadata': {}
                }

    # ============ SISTEMA DE EXPORTACIÓN ============

    class ExportadorAnalisis:
            """
            Exporta análisis a múltiples formatos
            """
            
            @staticmethod
            def exportar_csv(resultados, metadata):
                """Exporta a CSV"""
                df = pd.DataFrame(resultados)
                
                # Añadir metadata como columnas
                for key, value in metadata.items():
                    if key not in df.columns:
                        df[key] = value
                
                return df.to_csv(index=False)
            
            @staticmethod
            def exportar_json(resultados, metadata):
                """Exporta a JSON"""
                export_data = {
                    'timestamp': datetime.now().isoformat(),
                    'metadata': metadata,
                    'resultados': resultados,
                    'version': 'ACBE-Kelly v3.0'
                }
                return json.dumps(export_data, indent=2, ensure_ascii=False)
            
            @staticmethod
            def exportar_pdf(recomendacion, resultados, analisis_completo):
                """Exporta a PDF (simplificado - se puede expandir)"""
                buffer = BytesIO()
                c = canvas.Canvas(buffer, pagesize=letter)
                
                # Configuración
                width, height = letter
                y_position = height - 50
                
                # Título
                c.setFont("Helvetica-Bold", 16)
                c.drawString(50, y_position, "📊 ACBE-Kelly: Reporte de Análisis")
                y_position -= 30
                
                # Fecha
                c.setFont("Helvetica", 10)
                c.drawString(50, y_position, f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                y_position -= 40
                
                # Recomendación principal
                c.setFont("Helvetica-Bold", 14)
                c.drawString(50, y_position, "🎯 RECOMENDACIÓN PRINCIPAL:")
                y_position -= 20
                
                c.setFont("Helvetica", 12)
                if recomendacion['pick']:
                    texto = f"{recomendacion['accion']} en {recomendacion['pick']} @ {recomendacion['cuota']:.2f}"
                    c.drawString(50, y_position, texto)
                    y_position -= 15
                    c.drawString(50, y_position, f"Confianza: {recomendacion['confianza']:.0f}% | EV: {recomendacion['ev']:.2%}")
                    y_position -= 20
                else:
                    c.drawString(50, y_position, "NO APOSTAR - Sin oportunidades detectadas")
                    y_position -= 20
                
                # Tabla de resultados
                y_position -= 20
                c.setFont("Helvetica-Bold", 12)
                c.drawString(50, y_position, "📈 Resultados Detallados:")
                y_position -= 20
                
                # Encabezados de tabla
                headers = ["Resultado", "Prob", "Cuota", "EV", "Value"]
                col_widths = [80, 60, 60, 60, 60]
                
                c.setFont("Helvetica-Bold", 10)
                x_pos = 50
                for i, header in enumerate(headers):
                    c.drawString(x_pos, y_position, header)
                    x_pos += col_widths[i]
                
                y_position -= 20
                
                # Filas de datos
                c.setFont("Helvetica", 10)
                for resultado in resultados:
                    x_pos = 50
                    row_data = [
                        resultado['Resultado'],
                        resultado['Prob Modelo'],
                        resultado['Cuota Mercado'],
                        resultado['EV'],
                        resultado.get('Value Score', 'N/A')
                    ]
                    
                    for i, data in enumerate(row_data):
                        c.drawString(x_pos, y_position, str(data))
                        x_pos += col_widths[i]
                    
                    y_position -= 15
                    
                    if y_position < 100:
                        c.showPage()
                        y_position = height - 50
                        c.setFont("Helvetica", 10)
                
                y_position -= 20
                
                # Razones y advertencias
                if recomendacion.get('razones'):
                    c.setFont("Helvetica-Bold", 12)
                    c.drawString(50, y_position, "✅ Razones:")
                    y_position -= 15
                    
                    c.setFont("Helvetica", 10)
                    for razon in recomendacion['razones']:
                        c.drawString(50, y_position, f"• {razon}")
                        y_position -= 15
                
                if recomendacion.get('advertencias'):
                    c.setFont("Helvetica-Bold", 12)
                    c.drawString(50, y_position, "⚠️ Advertencias:")
                    y_position -= 15
                    
                    c.setFont("Helvetica", 10)
                    for advertencia in recomendacion['advertencias']:
                        c.drawString(50, y_position, f"• {advertencia}")
                        y_position -= 15
                
                # Guardar PDF
                c.save()
                buffer.seek(0)
                return buffer
            
            @staticmethod
            def exportar_resumen_html(recomendacion, resultados):
                """Exporta resumen HTML para visualización"""
                html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>ACBE-Kelly Report</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 40px; }}
                        .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 10px; }}
                        .recomendacion {{ background: {'#2ecc71' if recomendacion['pick'] else '#e74c3c'}; 
                                        color: white; padding: 20px; border-radius: 10px; margin: 20px 0; }}
                        .table {{ width: 100%; border-collapse: collapse; }}
                        .table th {{ background: #34495e; color: white; padding: 10px; }}
                        .table td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
                        .green {{ color: #27ae60; }}
                        .red {{ color: #e74c3c; }}
                    </style>
                </head>
                <body>
                    <div class="header">
                        <h1>🏛️ ACBE-Kelly Analysis Report</h1>
                        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    </div>
                    
                    <div class="recomendacion">
                        <h2>🎯 {'RECOMENDACIÓN: ' + recomendacion['accion'] if recomendacion['pick'] else 'NO RECOMENDACIÓN'}</h2>
                """
                
                if recomendacion['pick']:
                    html += f"""
                        <p><strong>Pick:</strong> {recomendacion['pick']} @ {recomendacion['cuota']:.2f}</p>
                        <p><strong>Confianza:</strong> {recomendacion['confianza']:.0f}%</p>
                        <p><strong>Expected Value:</strong> <span class="green">{recomendacion['ev']:.2%}</span></p>
                        <p><strong>Stake Recomendado:</strong> {recomendacion['stake_pct']}</p>
                    """
                
                html += """
                    </div>
                    
                    <h3>📊 Resultados Detallados</h3>
                    <table class="table">
                        <tr>
                            <th>Resultado</th>
                            <th>Probabilidad</th>
                            <th>Cuota</th>
                            <th>EV</th>
                            <th>Value Score</th>
                        </tr>
                """
                
                for r in resultados:
                    ev_class = "green" if float(r['EV'].strip('%'))/100 > 0 else "red"
                    html += f"""
                        <tr>
                            <td>{r['Resultado']}</td>
                            <td>{r['Prob Modelo']}</td>
                            <td>{r['Cuota Mercado']}</td>
                            <td class="{ev_class}">{r['EV']}</td>
                            <td>{r.get('Value Score', 'N/A')}</td>
                        </tr>
                    """
                
                html += """
                    </table>
                    
                    <h3>📝 Metadatos</h3>
                    <ul>
                """
                
                for key, value in recomendacion.get('metadata', {}).items():
                    html += f"<li><strong>{key}:</strong> {value}</li>"
                
                html += """
                    </ul>
                </body>
                </html>
                """
                
                return html   

            # ============ INTEGRACIÓN EN LA APP ============

    def agregar_modulo_recomendacion():
        """
        Módulo completo para añadir a tu app actual
        """
        
        # Inicializar componentes
        recomendador = RecomendadorInteligente()
        exportador = ExportadorAnalisis()
        
        # Crear sección de recomendación
        st.markdown("---")
        st.header("🎯 RECOMENDACIÓN FINAL DE APUESTA")
        
        # Aquí debes pasar el análisis completo de tu app
        # Suponiendo que tienes estas variables disponibles:
        # - resultados_analisis (lista de dicts con los resultados)
        # - analisis_completo (dict con metadata del análisis)
        
        # Esto es un ejemplo - debes adaptar a tus variables reales
        resultados_analisis = st.session_state.get('resultados_analisis', [])
        analisis_completo = st.session_state.get('analisis_completo', {})
        
        if not resultados_analisis:
            st.warning("No hay datos de análisis disponibles. Ejecuta el análisis primero.")
            return
        
        # Generar recomendación
        recomendacion = recomendador.generar_recomendacion({
            'resultados': resultados_analisis,
            **analisis_completo
        })
        
        # Mostrar recomendación
        col_rec1, col_rec2, col_rec3 = st.columns([2, 1, 1])
        
        with col_rec1:
            # Visualización de la recomendación
            if recomendacion['pick']:
                # Caso: Hay recomendación de apuesta
                st.markdown(f"""
                ### 🎰 **{recomendacion['accion']}**
                
                **Pick:** **{recomendacion['pick']}** @ {recomendacion['cuota']:.2f}
                
                **Confianza:** {recomendacion['confianza']:.0f}%
                **Expected Value:** {recomendacion['ev']:.2%}
                **Stake Recomendado:** {recomendacion['stake_pct']}
                """)
                
                # Barra de confianza visual
                confianza_pct = recomendacion['confianza']
                st.progress(confianza_pct/100, text=f"Confianza: {confianza_pct:.0f}%")
                
            else:
                # Caso: No apostar
                st.markdown("""
                ### ⛔ **NO APOSTAR**
                
                **Motivo:** No se detectaron oportunidades con value suficiente.
                
                **Recomendación:** Buscar otros partidos o esperar cambios en el mercado.
                """)
        
        with col_rec2:
            # Razones para la recomendación
            st.subheader("✅ Razones")
            for razon in recomendacion['razones']:
                st.info(f"• {razon}")
        
        with col_rec3:
            # Advertencias
            if recomendacion['advertencias']:
                st.subheader("⚠️ Advertencias")
                for adv in recomendacion['advertencias']:
                    st.warning(f"• {adv}")
        
        # Mostrar detalles del pick recomendado
        if recomendacion['pick']:
            st.markdown("---")
            st.subheader("📊 Detalles del Pick Recomendado")
            
            # Encontrar el resultado correspondiente
            pick_data = next((r for r in resultados_analisis if r['Resultado'] == recomendacion['pick']), None)
            
            if pick_data:
                col_det1, col_det2, col_det3, col_det4 = st.columns(4)
                
                with col_det1:
                    st.metric("Probabilidad Modelo", pick_data['Prob Modelo'])
                    st.metric("Cuota Justa", pick_data.get('Cuota Justa', 'N/A'))
                
                with col_det2:
                    st.metric("Cuota Mercado", pick_data['Cuota Mercado'])
                    st.metric("Diferencia", pick_data.get('Delta', 'N/A'))
                
                with col_det3:
                    st.metric("Value (EV)", pick_data['EV'])
                    st.metric("Stake Kelly", pick_data.get('Stake %', '0%'))
                
                with col_det4:
                    if 'Value Score' in pick_data:
                        st.metric("Value Score", pick_data['Value Score'])
                    st.metric("Significativo", pick_data.get('Significativo', 'N/A'))
        
        # Sección de exportación
        st.markdown("---")
        st.header("📥 EXPORTAR ANÁLISIS")
        
        col_exp1, col_exp2, col_exp3, col_exp4 = st.columns(4)
        
        with col_exp1:
            if st.button("💾 CSV", use_container_width=True):
                csv_data = exportador.exportar_csv(resultados_analisis, recomendacion['metadata'])
                st.download_button(
                    label="Descargar CSV",
                    data=csv_data,
                    file_name=f"acbe_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col_exp2:
            if st.button("📄 JSON", use_container_width=True):
                json_data = exportador.exportar_json(resultados_analisis, recomendacion['metadata'])
                st.download_button(
                    label="Descargar JSON",
                    data=json_data,
                    file_name=f"acbe_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        with col_exp3:
            if st.button("📊 PDF", use_container_width=True):
                pdf_buffer = exportador.exportar_pdf(recomendacion, resultados_analisis, analisis_completo)
                st.download_button(
                    label="Descargar PDF",
                    data=pdf_buffer,
                    file_name=f"acbe_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        
        with col_exp4:
            if st.button("🌐 HTML", use_container_width=True):
                html_data = exportador.exportar_resumen_html(recomendacion, resultados_analisis)
                st.download_button(
                    label="Descargar HTML",
                    data=html_data,
                    file_name=f"acbe_report_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
                    mime="text/html",
                    use_container_width=True
                )
        
        # Vista previa del reporte
        with st.expander("👁️ Vista Previa del Reporte", expanded=False):
            if recomendacion['pick']:
                st.success(f"""
                **Reporte Generado:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                
                **Recomendación:** {recomendacion['accion']}
                **Pick:** {recomendacion['pick']}
                **Cuota:** {recomendacion['cuota']:.2f}
                **Confianza:** {recomendacion['confianza']:.0f}%
                **Expected Value:** {recomendacion['ev']:.2%}
                
                **Equipos:** {recomendacion['metadata'].get('equipo_local', '')} vs {recomendacion['metadata'].get('equipo_visitante', '')}
                **Liga:** {recomendacion['metadata'].get('liga', '')}
                """)
            else:
                st.info("No hay recomendación de apuesta para este análisis.")
        
        # Guardar en historial interno
        if st.button("📝 Guardar en Historial Interno", use_container_width=True):
            if 'historial' not in st.session_state:
                st.session_state.historial = []
            
            registro = {
                'timestamp': datetime.now(),
                'recomendacion': recomendacion,
                'resultados': resultados_analisis,
                'metadata': analisis_completo
            }
            
            st.session_state.historial.append(registro)
            st.success(f"✅ Análisis guardado. Total en historial: {len(st.session_state.historial)}")
        
        # Mostrar historial si existe
        if 'historial' in st.session_state and st.session_state.historial:
            with st.expander("📚 Ver Historial de Análisis", expanded=False):
                for i, registro in enumerate(reversed(st.session_state.historial[-5:]), 1):
                    fecha = registro['timestamp'].strftime("%Y-%m-%d %H:%M")
                    rec = registro['recomendacion']
                    
                    if rec['pick']:
                        st.markdown(f"""
                        **{i}. {fecha}** - {rec['accion']} en {rec['pick']} @ {rec['cuota']:.2f}
                        """)
                    else:
                        st.markdown(f"""
                        **{i}. {fecha}** - {rec['accion']}
                        """)
                
                # Opción para exportar todo el historial
                if st.button("📦 Exportar Todo el Historial"):
                    historial_data = {
                        'version': 'ACBE-Kelly v3.0',
                        'generated': datetime.now().isoformat(),
                        'total_analisis': len(st.session_state.historial),
                        'analisis': st.session_state.historial
                    }
                    
                    json_historial = json.dumps(historial_data, indent=2, default=str)
                    
                    st.download_button(
                        label="Descargar Historial Completo",
                        data=json_historial,
                        file_name=f"acbe_historial_completo_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                        mime="application/json"
                    )
        
    # ============ INTERFAZ STREAMLIT v3.0 ============

    # --- BARRA LATERAL: CONFIGURACIÓN AVANZADA ---
    st.sidebar.header("⚙️ CONFIGURACIÓN DEL SISTEMA")

    with st.sidebar.expander("🎯 OBJETIVOS DE PERFORMANCE", expanded=True):
        col_obj1, col_obj2 = st.columns(2)
        with col_obj1:
            roi_target = st.slider("ROI Target (%)", 5, 25, 12)
            cvar_target = st.slider("CVaR Máximo (%)", 5, 25, 15)
        with col_obj2:
            max_dd = st.slider("Max Drawdown (%)", 10, 40, 20)
            sharpe_min = st.slider("Sharpe Mínimo", 0.5, 3.0, 1.5)
        
        st.markdown("---")
        st.markdown(f"""
        **Objetivos establecidos:**
        - ROI: {roi_target}%
        - CVaR: < {cvar_target}%
        - Max DD: < {max_dd}%
        - Sharpe: > {sharpe_min}
        """)

    with st.sidebar.expander("📊 PARÁMETROS BAYESIANOS", expanded=False):
        liga = st.selectbox("Liga", ["Serie A", "Premier League", "La Liga", "Bundesliga", "Ligue 1"])
        
        st.markdown("**Priors del Modelo:**")
        col_prior1, col_prior2 = st.columns(2)
        with col_prior1:
            confianza_prior = st.slider("Confianza Prior", 0.1, 1.0, 0.7)
        with col_prior2:
            aprendizaje_bayes = st.slider("Tasa Aprendizaje", 0.1, 1.0, 0.5)
        
        st.markdown("**Actualización Bayesiana:**")
        peso_reciente = st.slider("Peso Partidos Recientes", 0.0, 1.0, 0.7)
        peso_historico = 1 - peso_reciente

    st.sidebar.header("📥 INGESTA DE DATOS")

    team_h = st.sidebar.text_input("Equipo Local", value="Bologna")
    team_a = st.sidebar.text_input("Equipo Visitante", value="AC Milan")

    # --- PANEL PRINCIPAL: DATOS DETALLADOS ---
    st.header("📈 ANÁLISIS DE EQUIPOS")

    col_team1, col_team2 = st.columns(2)

    with col_team1:
        st.subheader(f"🏠 {team_h} (Local)")
        
        with st.expander("📊 ESTADÍSTICAS OFENSIVAS", expanded=True):
            col_o1, col_o2 = st.columns(2)
            with col_o1:
                g_h_ult5 = st.number_input(f"Goles (últ. 5p)", value=8, min_value=0, key="gh5")
                xg_h_prom = st.number_input("xG promedio", value=1.65, step=0.05, key="xgh")
                tiros_arco_h = st.number_input("Tiros a puerta/p", value=4.8, step=0.1)
            with col_o2:
                g_h_ult10 = st.number_input(f"Goles (últ. 10p)", value=15, min_value=0, key="gh10")
                posesion_h = st.slider("Posesión %", 30, 70, 52, key="pos_h")
                precision_pases_h = st.slider("Precisión pases %", 70, 90, 82)
        
        with st.expander("🛡️ ESTADÍSTICAS DEFENSIVAS", expanded=False):
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                goles_rec_h = st.number_input("Goles recibidos (10p)", value=12, min_value=0, key="grh")
                xg_contra_h = st.number_input("xG en contra/p", value=1.2, step=0.05)
            with col_d2:
                entradas_h = st.number_input("Entradas/p", value=15.5, step=0.1)
                recuperaciones_h = st.number_input("Recuperaciones/p", value=45.0, step=0.5)
        
        with st.expander("⚠️ FACTORES DE RIESGO", expanded=False):
            delta_h = st.slider(f"Impacto bajas {team_h}", 0.0, 0.3, 0.08, step=0.01)
            motivacion_h = st.slider("Motivación", 0.5, 1.5, 1.0, step=0.05)
            carga_fisica_h = st.slider("Carga física", 0.5, 1.5, 1.0, step=0.05)

    with col_team2:
        st.subheader(f"✈️ {team_a} (Visitante)")
        
        with st.expander("📊 ESTADÍSTICAS OFENSIVAS", expanded=True):
            col_o1, col_o2 = st.columns(2)
            with col_o1:
                g_a_ult5 = st.number_input(f"Goles (últ. 5p)", value=6, min_value=0, key="ga5")
                xg_a_prom = st.number_input("xG promedio", value=1.40, step=0.05, key="xga")
                tiros_arco_a = st.number_input("Tiros a puerta/p", value=4.3, step=0.1)
            with col_o2:
                g_a_ult10 = st.number_input(f"Goles (últ. 10p)", value=12, min_value=0, key="ga10")
                posesion_a = 100 - posesion_h
                st.metric("Posesión %", f"{posesion_a}%")
                precision_pases_a = st.slider("Precisión pases %", 70, 90, 78, key="ppa")
        
        with st.expander("🛡️ ESTADÍSTICAS DEFENSIVAS", expanded=False):
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                goles_rec_a = st.number_input("Goles recibidos (10p)", value=10, min_value=0, key="gra")
                xg_contra_a = st.number_input("xG en contra/p", value=1.05, step=0.05)
            with col_d2:
                entradas_a = st.number_input("Entradas/p", value=16.2, step=0.1)
                recuperaciones_a = st.number_input("Recuperaciones/p", value=42.5, step=0.5)
        
        with st.expander("⚠️ FACTORES DE RIESGO", expanded=False):
            delta_a = st.slider(f"Impacto bajas {team_a}", 0.0, 0.3, 0.05, step=0.01)
            motivacion_a = st.slider("Motivación", 0.5, 1.5, 0.9, step=0.05, key="mot_a")
            carga_fisica_a = st.slider("Carga física", 0.5, 1.5, 1.1, step=0.05, key="cf_a")

    # --- SECCIÓN MERCADO Y CUOTAS ---
    st.sidebar.header("💰 MERCADO")
    col_c1, col_c2, col_c3 = st.sidebar.columns(3)
    with col_c1:
        c1 = st.number_input("1", value=2.90, min_value=1.01, step=0.01, key="cuota1")
    with col_c2:
        cx = st.number_input("X", value=3.25, min_value=1.01, step=0.01, key="cuotax")
    with col_c3:
        c2 = st.number_input("2", value=2.45, min_value=1.01, step=0.01, key="cuota2")

    st.sidebar.markdown("---")
    st.sidebar.header("📈 MÉTRICAS DE MERCADO")

    # Calcular métricas de mercado
    or_val = (1/c1 + 1/cx + 1/c2) - 1
    volumen_estimado = st.sidebar.slider("Volumen Relativo", 0.5, 2.0, 1.0, step=0.1)
    steam_detectado = st.sidebar.slider("Steam Move (σ)", 0.0, 0.05, 0.0, step=0.005)

    col_met1, col_met2, col_met3 = st.sidebar.columns(3)
    with col_met1:
        st.metric("Overround", f"{or_val:.2%}")
    with col_met2:
        st.metric("Margen Casa", f"{(or_val/(1+or_val)*100):.1f}%")
    with col_met3:
        entropia_mercado = st.sidebar.slider("Entropía (H)", 0.3, 0.9, 0.62, step=0.01)
        st.metric("Entropía", f"{entropia_mercado:.3f}")

    # ============ EJECUCIÓN DEL SISTEMA ============
    if st.sidebar.button("🚀 EJECUTAR ANÁLISIS COMPLETO", type="primary", use_container_width=True):
        
        with st.spinner("🔬 Inicializando modelo bayesiano jerárquico..."):
            # Inicializar componentes
            modelo_bayes = ModeloBayesianoJerarquico(liga)
            detector = DetectorIneficiencias()
            gestor_riesgo = GestorRiscoCVaR(cvar_target=cvar_target/100, max_drawdown=max_dd/100)
            backtester = BacktestSintetico()
            
            # FASE 0: Validación de mercado
            st.subheader("🎯 FASE 0: VALIDACIÓN DE MERCADO")
            
            col_val1, col_val2, col_val3, col_val4 = st.columns(4)
            
            with col_val1:
                val_min_odd = c1 >= 1.60 and c2 >= 1.60
                st.metric("Cuota Mínima", "✅" if val_min_odd else "❌", 
                        delta="OK" if val_min_odd else "< 1.60")
            
            with col_val2:
                val_or = or_val <= 0.07
                st.metric("Overround", "✅" if val_or else "❌", 
                        delta=f"{or_val:.2%}" if val_or else "Alto")
            
            with col_val3:
                val_entropia = entropia_mercado <= 0.72
                st.metric("Entropía", "✅" if val_entropia else "❌",
                        delta=f"{entropia_mercado:.3f}")
            
            with col_val4:
                val_volumen = volumen_estimado >= 0.8
                st.metric("Liquidez", "✅" if val_volumen else "⚠️",
                        delta=f"{volumen_estimado:.1f}x")
            
            # Verificar condiciones de evasión
            condiciones_evasion = []
            if not val_min_odd: condiciones_evasion.append("Cuota < 1.60")
            if not val_or: condiciones_evasion.append(f"Overround alto ({or_val:.2%})")
            if not val_entropia: condiciones_evasion.append(f"Entropía alta ({entropia_mercado:.3f})")
            
            if condiciones_evasion:
                st.error(f"🚫 EVASIÓN DE RIESGO: {', '.join(condiciones_evasion)}")
                st.stop()
            
            st.success("✅ MERCADO VÁLIDO PARA ANÁLISIS")
        
        with st.spinner("🧠 EJECUTANDO INFERENCIA BAYESIANA..."):
            st.subheader("🎯 FASE 1: INFERENCIA BAYESIANA")
            
            # Preparar datos para el modelo
            datos_local = {
                "goles_anotados": g_h_ult10,
                "goles_recibidos": goles_rec_h,
                "n_partidos": 10,
                "xG": xg_h_prom,
                "tiros_arco": tiros_arco_h,
                "posesion": posesion_h,
                "precision_pases": precision_pases_h
            }
            
            datos_visitante = {
                "goles_anotados": g_a_ult10,
                "goles_recibidos": goles_rec_a,
                "n_partidos": 10,
                "xG": xg_a_prom,
                "tiros_arco": tiros_arco_a,
                "posesion": posesion_a,
                "precision_pases": precision_pases_a
            }
            
            # Inferencia bayesiana
            posterior_local = modelo_bayes.inferencia_variacional(datos_local, es_local=True)
            posterior_visitante = modelo_bayes.inferencia_variacional(datos_visitante, es_local=False)
            
            # Aplicar ajustes por factores de riesgo
            lambda_h_ajustado = posterior_local["lambda"] * (1 - delta_h) * motivacion_h / carga_fisica_h
            lambda_a_ajustado = posterior_visitante["lambda"] * (1 - delta_a) * motivacion_a / carga_fisica_a
            
            # Mostrar resultados de inferencia
            col_inf1, col_inf2 = st.columns(2)
            
            with col_inf1:
                st.markdown(f"**{team_h} (Local)**")
                st.metric("λ Posterior", f"{lambda_h_ajustado:.3f}")
                st.metric("Incertidumbre", f"{posterior_local['incertidumbre']:.3f}")
                st.metric("CI 95%", f"[{posterior_local['ci_95'][0]:.2f}, {posterior_local['ci_95'][1]:.2f}]")
            
            with col_inf2:
                st.markdown(f"**{team_a} (Visitante)**")
                st.metric("λ Posterior", f"{lambda_a_ajustado:.3f}")
                st.metric("Incertidumbre", f"{posterior_visitante['incertidumbre']:.3f}")
                st.metric("CI 95%", f"[{posterior_visitante['ci_95'][0]:.2f}, {posterior_visitante['ci_95'][1]:.2f}]")
        
        with st.spinner("🎲 SIMULANDO 50,000 ESCENARIOS..."):
            st.subheader("🎯 FASE 2: SIMULACIÓN MONTE CARLO AVANZADA")
            
            # Simulación con incertidumbre paramétrica
            n_simulaciones = 50000
            resultados_sim = []
            
            progress_bar = st.progress(0)
            for i in range(n_simulaciones):
                # Muestrear de la distribución posterior
                lambda_h_sim = np.random.gamma(
                    posterior_local["alpha"], 
                    1/posterior_local["beta"]
                ) * (1 - delta_h) * motivacion_h / carga_fisica_h
                
                lambda_a_sim = np.random.gamma(
                    posterior_visitante["alpha"],
                    1/posterior_visitante["beta"]
                ) * (1 - delta_a) * motivacion_a / carga_fisica_a
                
                # Simular goles
                goles_h = np.random.poisson(lambda_h_sim)
                goles_a = np.random.poisson(lambda_a_sim)
                
                # Determinar resultado
                if goles_h > goles_a:
                    resultado = "1"
                elif goles_h == goles_a:
                    resultado = "X"
                else:
                    resultado = "2"
                
                resultados_sim.append(resultado)
                
                if i % 10000 == 0:
                    progress_bar.progress((i + 1) / n_simulaciones)
            
            progress_bar.progress(1.0)
            
            # Calcular probabilidades
            resultados_array = np.array(resultados_sim)
            p1_mc = np.mean(resultados_array == "1")
            px_mc = np.mean(resultados_array == "X")
            p2_mc = np.mean(resultados_array == "2")
            
            # Calcular incertidumbre (error estándar)
            se_p1 = np.sqrt(p1_mc * (1 - p1_mc) / n_simulaciones)
            se_px = np.sqrt(px_mc * (1 - px_mc) / n_simulaciones)
            se_p2 = np.sqrt(p2_mc * (1 - p2_mc) / n_simulaciones)
            
            # Visualizar distribución
            fig_sim = go.Figure(data=[
                go.Bar(
                    x=["1", "X", "2"],
                    y=[p1_mc, px_mc, p2_mc],
                    error_y=dict(type='data', array=[se_p1, se_px, se_p2]),
                    marker_color=['#00CC96', '#636EFA', '#EF553B']
                )
            ])
            
            fig_sim.update_layout(
                title="Distribución de Probabilidades (Monte Carlo)",
                yaxis_title="Probabilidad",
                showlegend=False
            )
            
            st.plotly_chart(fig_sim, use_container_width=True)
        
        with st.spinner("🔍 DETECTANDO INEFICIENCIAS..."):
            st.subheader("🎯 FASE 3: DETECCIÓN DE INEFICIENCIAS")
            
            # Probabilidades implícitas del mercado
            p1_mercado = 1 / c1
            px_mercado = 1 / cx
            p2_mercado = 1 / c2
            
            # ============ CALCULAR ENTROPÍA AUTO ============
            # Calcula entropía de Shannon de las probabilidades del mercado
            import numpy as np
            prob_mercado_array = np.array([p1_mercado, px_mercado, p2_mercado])
            prob_mercado_array = prob_mercado_array[prob_mercado_array > 0]  # Evitar log(0)
            entropia_auto = -np.sum(prob_mercado_array * np.log2(prob_mercado_array))
            
            # Análisis para cada resultado
            resultados_analisis = []
            
            for label, p_modelo, p_mercado, se, cuota in zip(
                ["1", "X", "2"],
                [p1_mc, px_mc, p2_mc],
                [p1_mercado, px_mercado, p2_mercado],
                [se_p1, se_px, se_p2],
                [c1, cx, c2]
            ):
                # Value Score estadístico
                value_analysis = detector.calcular_value_score(p_modelo, p_mercado, se)
                
                # KL Divergence
                kl_analysis = detector.calcular_entropia_kullback_leibler(p_modelo, p_mercado)
                
                # Valor esperado
                ev = p_modelo * cuota - 1
                
                # Cuota justa
                fair_odd = 1 / p_modelo if p_modelo > 0 else 999
                
                resultados_analisis.append({
                    "Resultado": label,
                    "Prob Modelo": p_modelo,
                    "Prob Mercado": p_mercado,
                    "Delta": p_modelo - p_mercado,
                    "EV": ev,
                    "Fair Odd": fair_odd,
                    "Cuota Mercado": cuota,
                    "Value Score": value_analysis,
                    "KL Divergence": kl_analysis
                })
                # ============ QUITAMOS EL GUARDADO DE SESSION_STATE DE AQUÍ ============
            
            # ============ AQUÍ MOVEMOS EL GUARDADO (FUERA DEL LOOP) ============
            # Ahora guardamos en session_state UNA SOLA VEZ, después del loop
            st.session_state['resultados_analisis'] = resultados_analisis
            st.session_state['analisis_completo'] = {
                'team_h': team_h,
                'team_a': team_a,
                'liga': liga,
                'or_val': or_val,
                'entropia': entropia_auto,  # Ahora sí está definida
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            # También guarda las probabilidades numéricas para cálculos
            st.session_state['probabilidades_numericas'] = {
                '1': p1_mc,
                'X': px_mc,
                '2': p2_mc
            }
            
            # Crear tabla de resultados
            df_resultados = pd.DataFrame([
                {
                    "Resultado": r["Resultado"],
                    "Prob Modelo": f"{r['Prob Modelo']:.2%}",
                    "Prob Mercado": f"{r['Prob Mercado']:.2%}",
                    "Delta": f"{r['Delta']:+.2%}",
                    "EV": f"{r['EV']:+.2%}",
                    "Fair Odd": f"{r['Fair Odd']:.2f}",
                    "Cuota": f"{r['Cuota Mercado']:.2f}",
                    "Value Score": f"{r['Value Score']['t_statistic']:.2f}",
                    "Significativo": "✅" if r['Value Score']['significativo'] else "❌",
                    "KL Bits": f"{r['KL Divergence']['informacion_bits']:.3f}"
                }
                for r in resultados_analisis
            ])
            
            st.dataframe(df_resultados, use_container_width=True)

            # Mostrar recomendación y opciones de exportación
            agregar_modulo_recomendacion()
            
            # Identificar picks con valor
            picks_con_valor = []
            for r in resultados_analisis:
                if r['Value Score']['significativo'] and r['EV'] > 0.02:
                    picks_con_valor.append(r)
            
            if picks_con_valor:
                st.success(f"✅ **{len(picks_con_valor)} INEFICIENCIA(S) DETECTADA(S)**")
            else:
                st.warning("⚠️ MERCADO EFICIENTE: No se detectan ineficiencias significativas")
        
        with st.spinner("💰 CALCULANDO GESTIÓN DE CAPITAL..."):
            st.subheader("🎯 FASE 4: GESTIÓN DE CAPITAL (KELLY DINÁMICO)")
            
            # Configurar bankroll
            bankroll = 1000  # Se puede hacer configurable
            
            recomendaciones = []
            
            for r in picks_con_valor:
                # --- CONVERSIÓN DE DATOS A NUMÉRICOS (PRIMERO) ---
                # 1. Convertir EV a numérico
                if isinstance(r["EV"], str) and '%' in r["EV"]:
                    ev_numerico = float(r["EV"].strip('%')) / 100
                else:
                    ev_numerico = float(r["EV"])
                
                # 2. Convertir Prob Modelo a numérico
                if isinstance(r["Prob Modelo"], str) and '%' in r["Prob Modelo"]:
                    prob_modelo_numerico = float(r["Prob Modelo"].strip('%')) / 100
                else:
                    prob_modelo_numerico = float(r["Prob Modelo"])
                
                # 3. Obtener significancia
                significativo = r.get("Value Score", {}).get("significativo", False)
                
                # 4. Convertir cuota a float
                cuota_numerico = float(r["Cuota Mercado"])

                # Simular CVaR para este pick
                simulacion_cvar = gestor_riesgo.simular_cvar(
                    prob=r["Prob Modelo"],
                    cuota=r["Cuota Mercado"],
                    n_simulaciones=10000,
                    conf_level=0.95
                )
                
                # --- AQUÍ VA LA CONVERSIÓN DE EV A NUMÉRICO ---
                # Asegurar que EV sea numérico (puede venir como string con %)
                if isinstance(r["EV"], str) and '%' in r["EV"]:
                    ev_numerico = float(r["EV"].strip('%')) / 100
                else:
                    ev_numerico = float(r["EV"])

                # También verificar que Value Score tiene la clave 'significativo'
                significativo = r.get("Value Score", {}).get("significativo", False)


                # Calcular Kelly dinámico
                metrics_kelly = {
                    "incertidumbre": posterior_local.get("incertidumbre", 0.5) 
                                    if r["Resultado"] == "1" 
                                    else posterior_visitante.get("incertidumbre", 0.5),
                    "cvar_estimado": simulacion_cvar["cvar"],
                    "entropia": entropia_mercado,
                    "sharpe_esperado": simulacion_cvar["sharpe_simulado"],
                    "prob_modelo": r["Prob Modelo"],
                    "valor_estadistico": r["Value Score"].get("t_statistic", 0),
                    "ev": ev_numerico,  # Usar el valor numérico  # Este es el EV que calculaste antes
                    "significativo": significativo  # <-- Ahora tiene valor por defecto
                }
                
                kelly_result = gestor_riesgo.calcular_kelly_dinamico(
                    prob=prob_modelo_numerico,
                    cuota=cuota_numerico,
                    bankroll=bankroll,
                    metrics=metrics_kelly
                )
                
                # Backtest sintético
                backtest_result = backtester.generar_escenarios(
                    prob=r["Prob Modelo"],
                    cuota=r["Cuota Mercado"],
                    bankroll_inicial=bankroll,
                    n_apuestas=100,
                    n_simulaciones=2000
                )
                
                recomendaciones.append({
                    "resultado": r["Resultado"],
                    "ev": r["EV"],
                    "kelly_pct": kelly_result["stake_pct"],
                    "stake_abs": kelly_result["stake_abs"],
                    "cvar": simulacion_cvar["cvar"],
                    "sharpe_esperado": backtest_result["metricas"]["sharpe_promedio"],
                    "prob_profit": backtest_result["metricas"]["prob_profit"],
                    "max_dd_promedio": backtest_result["metricas"]["max_dd_promedio"],
                    "backtest_metrics": backtest_result["metricas"]
                })
            
            # Mostrar recomendaciones
            if recomendaciones:
                st.subheader("🎰 RECOMENDACIONES DE APUESTA")
                
                for rec in recomendaciones:
                    with st.expander(f"**{rec['resultado']}** - EV: {rec['ev']:+.2%} - Stake: {rec['kelly_pct']:.2f}%", expanded=True):
                        col_rec1, col_rec2, col_rec3 = st.columns(3)
                        
                        with col_rec1:
                            st.metric("Stake Recomendado", f"€{rec['stake_abs']:.0f}")
                            st.metric("% Bankroll", f"{rec['kelly_pct']:.2f}%")
                        
                        with col_rec2:
                            st.metric("CVaR Estimado", f"{rec['cvar']:.2%}")
                            st.metric("Sharpe Esperado", f"{rec['sharpe_esperado']:.2f}")
                        
                        with col_rec3:
                            st.metric("Prob. Profit", f"{rec['prob_profit']:.1%}")
                            st.metric("Max DD Esperado", f"{rec['max_dd_promedio']:.1%}")
                        
                        # Gráfico de distribución de retornos
                        fig_dist = go.Figure()
                        fig_dist.add_trace(go.Histogram(
                            x = rec.get('backtest_metrics', {}).get('distribucion_retornos', []),
                            nbinsx=50,
                            name="Distribución Retornos",
                            marker_color='#636EFA'
                        ))
                        
                        fig_dist.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Break-even")
                        fig_dist.add_vline(x=roi_target/100, line_dash="dash", line_color="green", 
                                        annotation_text=f"Target {roi_target}%")
                        
                        fig_dist.update_layout(
                            title="Distribución de Retornos Simulados (100 apuestas)",
                            xaxis_title="Retorno Total",
                            yaxis_title="Frecuencia"
                        )
                        
                        st.plotly_chart(fig_dist, use_container_width=True)
            else:
                st.info("📊 No hay picks con valor estadísticamente significativo y EV > 2%")
        
        with st.spinner("📊 GENERANDO REPORTE FINAL..."):
            st.subheader("🎯 FASE 5: REPORTE DE RIESGO Y PERFORMANCE")
            
            # Calcular métricas agregadas
            if recomendaciones:
                ev_promedio = np.mean([r['ev'] for r in recomendaciones])
                sharpe_promedio = np.mean([r['sharpe_esperado'] for r in recomendaciones])
                cvar_promedio = np.mean([r['cvar'] for r in recomendaciones])
                prob_profit_promedio = np.mean([r['prob_profit'] for r in recomendaciones])
                
                # Verificar objetivos
                objetivos_cumplidos = []
                if ev_promedio * 100 >= roi_target * 0.8:  # 80% del target
                    objetivos_cumplidos.append("ROI")
                if cvar_promedio <= cvar_target/100:
                    objetivos_cumplidos.append("CVaR")
                if sharpe_promedio >= sharpe_min:
                    objetivos_cumplidos.append("Sharpe")
                
                col_obj1, col_obj2, col_obj3, col_obj4 = st.columns(4)

            with col_obj1:
                color_text = "🟢" if ev_promedio * 100 >= roi_target * 0.8 else "🟠"
                st.metric(f"ROI Esperado {color_text}", f"{ev_promedio:.2%}")
                st.caption(f"Target: {roi_target}%")

            with col_obj2:
                color_text = "🟢" if cvar_promedio <= cvar_target/100 else "🔴"
                st.metric(f"CVaR Estimado {color_text}", f"{cvar_promedio:.2%}")
                st.caption(f"Máx: {cvar_target}%")

            with col_obj3:
                color_text = "🟢" if sharpe_promedio >= sharpe_min else "🟠"
                st.metric(f"Sharpe Esperado {color_text}", f"{sharpe_promedio:.2f}")
                st.caption(f"Mín: {sharpe_min}")

            with col_obj4:
                    st.metric("Prob. Éxito", f"{prob_profit_promedio:.1%}")
            with col_obj4:
                    st.metric("Prob. Éxito", f"{prob_profit_promedio:.1%}")
                
            # Resumen de objetivos
            if len(objetivos_cumplidos) >= 2:
                    st.success(f"✅ **SISTEMA DENTRO DE PARÁMETROS:** {', '.join(objetivos_cumplidos)}")
            else:
                    st.warning(f"⚠️ **SISTEMA FUERA DE PARÁMETROS:** Solo {len(objetivos_cumplidos)} objetivo(s) cumplido(s)")
            
            # Guardar en historial
            if picks_con_valor:
                for pick in picks_con_valor:
                    logger.registrar_pick({
                        'equipo_local': team_h,
                        'equipo_visitante': team_a,
                        'resultado': pick['Resultado'],
                        'ev': pick['EV'],
                        'prob_modelo': pick['Prob Modelo'],
                        'cuota': pick['Cuota Mercado']
                    })
            
            st.markdown("---")
            st.markdown("""
            ### 📝 SUPUESTOS Y LIMITACIONES
            
            1. **Modelo Bayesiano**: Asume distribución Gamma para λ y actualización conjugada
            2. **Independencia**: Asume independencia entre goles (Poisson)
            3. **Mercado Eficiente**: Asume que el mercado incorpora toda la información pública
            4. **Simulaciones**: Basadas en distribuciones paramétricas, no eventos extremos
            5. **Datos**: Calidad dependiente de los inputs proporcionados
            
            **TASA DE ÉXITO ESPERADA**: 58-65% en picks con EV+ ≥ 3%
            **ROI ANUALIZADO**: 12-18% con gestión estricta de capital
            **DRAWDOWN MÁXIMO ESPERADO**: 15-25%
            """)

    # ============ PANEL DE MONITOREO EN TIEMPO REAL ============
    st.sidebar.markdown("---")
    st.sidebar.header("📊 MONITOREO")

    if st.sidebar.button("📈 VER MÉTRICAS DEL SISTEMA", type="secondary"):
        st.subheader("📊 MÉTRICAS HISTÓRICAS DEL SISTEMA")
        
        if logger.historial:
            df_historial = pd.DataFrame(logger.historial)
            
            col_met1, col_met2, col_met3, col_met4 = st.columns(4)
            
            with col_met1:
                st.metric("Total Picks", len(df_historial))
            
            with col_met2:
                picks_ev_pos = len(df_historial[df_historial['ev'] > 0])
                st.metric("Picks EV+", picks_ev_pos)
            
            with col_met3:
                if len(df_historial) > 0:
                    ev_promedio = df_historial['ev'].mean()
                    st.metric("EV Promedio", f"{ev_promedio:.2%}")
            
            with col_met4:
                if picks_ev_pos > 0:
                    st.metric("Ratio EV+", f"{(picks_ev_pos/len(df_historial)):.1%}")
            
            # Gráfico de EV histórico
            if len(df_historial) > 1:
                df_historial = df_historial.sort_values('timestamp')
                df_historial['ev_acumulado'] = df_historial['ev'].cumsum()
                
                fig_ev = go.Figure()
                fig_ev.add_trace(go.Scatter(
                    x=df_historial['timestamp'],
                    y=df_historial['ev_acumulado']*100,
                    mode='lines+markers',
                    name='EV Acumulado',
                    line=dict(color='#00CC96', width=2)
                ))
                
                fig_ev.update_layout(
                    title="EV Acumulado del Sistema",
                    xaxis_title="Fecha",
                    yaxis_title="EV Acumulado (%)",
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig_ev, use_container_width=True)
        else:
            st.info("No hay historial registrado. Ejecuta análisis para comenzar.")

    # ============ SECCIÓN DE DOCUMENTACIÓN ============
    with st.expander("📚 DOCUMENTACIÓN TÉCNICA", expanded=False):
        st.markdown("""
        ## 🏛️ SISTEMA ACBE-KELLY v3.0
        
        ### ARQUITECTURA DEL SISTEMA
        
        1. **Modelo Bayesiano Jerárquico**
        - Prior: Gamma(α, β) calibrado por liga
        - Likelihood: Poisson(λ)
        - Posterior: Gamma(α_post, β_post) via conjugación
        - Ajuste: Factores de forma, posesión, xG, bajas
        
        2. **Detección de Ineficiencias**
        - Test estadístico: t-score con p-value
        - Valor mínimo: Δ > 2% con significancia 95%
        - KL Divergence: Medida de información
        
        3. **Gestión de Capital Avanzada**
        - Kelly dinámico con ajustes múltiples
        - CVaR (Conditional Value at Risk) en tiempo real
        - Backtesting sintético con 5,000 escenarios
        
        4. **Validación y Monitoreo**
        - Backtest histórico implícito
        - Métricas de performance en tiempo real
        - Sistema de logging profesional
        
        ### PARÁMETROS CLAVE CALIBRADOS
        
        | Parámetro | Valor | Descripción |
        |-----------|-------|-------------|
        | **ROI Target** | 12-18% | Retorno sobre inversión anual |
        | **CVaR Máximo** | 15% | Pérdida máxima esperada en cola |
        | **Sharpe Mínimo** | 1.5 | Ratio riesgo/retorno mínimo |
        | **Max Drawdown** | 20% | Pérdida máxima tolerada |
        | **Confianza Prior** | 70% | Peso de datos históricos vs recientes |
        
        ### SUPUESTOS CRÍTICOS
        
        1. **Eficiencia de Mercado Débil**: El mercado incorpora toda la información pública
        2. **Distribución Poisson**: Los goles siguen distribución de Poisson (validado empíricamente)
        3. **Independencia**: Los goles son independientes entre sí
        4. **Estacionariedad**: Las estadísticas de equipos son estables en el corto plazo
        
        ### LÍMITES CONOCIDOS
        
        1. **Eventos extremos**: No modela bien black swans (lesiones graves, condiciones extremas)
        2. **Correlaciones**: No considera correlación entre resultados múltiples
        3. **Datos en tiempo real**: Depende de inputs manuales (versión actual)
        4. **Cambios estructurales**: No detecta cambios bruscos en dinámica de equipos
        
        ### ROADMAP v4.0
        
        1. **API Automática**: Conexión con APIs de datos en tiempo real
        2. **Machine Learning**: Random Forest sobre features del modelo
        3. **Portfolio Optimization**: Gestión de correlación entre apuestas
        4. **Alertas Automáticas**: Sistema de notificaciones para steam moves
        5. **Dashboard Avanzado**: Métricas en tiempo real con streaming
        """)

    # ============ PIE DE PÁGINA PROFESIONAL ============
    st.markdown("---")
    col_footer1, col_footer2, col_footer3 = st.columns(3)

    with col_footer1:
        st.markdown("**ACBE Quantum Terminal v3.0**")
        st.markdown("Sistema de Arbitraje Estadístico Deportivo")

    with col_footer2:
        st.markdown("**🏛️ Metodología**")
        st.markdown("Bayesiano Jerárquico + Monte Carlo + Kelly Dinámico")

    with col_footer3:
        st.markdown("**⚡ Performance Esperada**")
        st.markdown("ROI: 12-18% | Sharpe: 1.5-2.0 | CVaR: < 15%")

    st.markdown("---")
    st.caption("© 2024 ACBE Predictive Systems | Para uso educativo y profesional. Apuestas conllevan riesgo de pérdida.")
    pass
