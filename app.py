import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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
import uuid

# ============================================
# CONFIGURACIÓN DE PÁGINA
# ============================================
st.set_page_config(page_title="Sistema ACBE-Kelly", layout="wide")

# ============================================
# INICIALIZACIÓN DEL ESTADO DE LA SESIÓN
# ============================================

# Diccionario Maestro (PATRÓN COCINA/SALÓN)
if 'dm' not in st.session_state:
    st.session_state['dm'] = {}

# Bandera de análisis ejecutado
if 'analisis_ejecutado' not in st.session_state:
    st.session_state['analisis_ejecutado'] = False

# Mantener compatibilidad con sistema anterior
if 'entropia_mercado' not in st.session_state:
    st.session_state['entropia_mercado'] = 0.620

# INICIALIZACIÓN CORREGIDA: Sistema de doble entrada
if 'bankroll_actual' not in st.session_state:
    st.session_state.bankroll_actual = 1000.0

if 'bankroll_inicial_sesion' not in st.session_state:
    st.session_state.bankroll_inicial_sesion = st.session_state.bankroll_actual

# NUEVA VARIABLE: Beneficio neto exclusivo de trading
if 'beneficio_neto' not in st.session_state:
    st.session_state.beneficio_neto = 0.0

if 'historial_bankroll' not in st.session_state:
    st.session_state.historial_bankroll = []

if 'historial_apuestas' not in st.session_state:
    st.session_state.historial_apuestas = []

# ============================================
# CLASES DEL NÚCLEO MATEMÁTICO (GLOBALES)
# ============================================

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

class ModeloBayesianoJerarquico:
    def __init__(self, liga="Serie A"):
        self.priors = self._inicializar_priors(liga)
        
    def _inicializar_priors(self, liga):
        datos_ligas = {
            "Serie A": {"mu_goles": 1.32, "sigma_goles": 0.85, "home_adv": 1.18},
            "Premier League": {"mu_goles": 1.48, "sigma_goles": 0.92, "home_adv": 1.15},
            "La Liga": {"mu_goles": 1.35, "sigma_goles": 0.88, "home_adv": 1.16},
            "Bundesliga": {"mu_goles": 1.56, "sigma_goles": 0.95, "home_adv": 1.12},
            "Ligue 1": {"mu_goles": 1.28, "sigma_goles": 0.82, "home_adv": 1.20}
        }
        
        data = datos_ligas.get(liga, datos_ligas["Serie A"])
        alpha_prior = (data["mu_goles"] ** 2) / (data["sigma_goles"] ** 2)
        beta_prior = data["mu_goles"] / (data["sigma_goles"] ** 2)
        
        return {
            "alpha": alpha_prior,
            "beta": beta_prior,
            "home_advantage": data["home_adv"],
            "sigma_liga": data["sigma_goles"]
        }
    
    def inferencia_variacional(self, datos_equipo, es_local=True):
        goles_anotados = datos_equipo.get("goles_anotados", 0)
        goles_recibidos = datos_equipo.get("goles_recibidos", 0)
        n_partidos = datos_equipo.get("n_partidos", 10)
        xG_promedio = datos_equipo.get("xG", 1.5)
        
        alpha_posterior = self.priors["alpha"] + goles_anotados
        beta_posterior = self.priors["beta"] + n_partidos
        lambda_posterior = alpha_posterior / beta_posterior
        
        if xG_promedio > 0:
            ratio_xg = min(max(xG_promedio / max(lambda_posterior, 0.1), 0.7), 1.3)
            lambda_posterior *= ratio_xg
        
        if es_local:
            lambda_posterior *= self.priors["home_advantage"]
        else:
            lambda_posterior *= (2 - self.priors["home_advantage"])
        
        varianza_posterior = alpha_posterior / (beta_posterior ** 2)
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
    @staticmethod
    def calcular_value_score(p_modelo, p_mercado, sigma_modelo):
        if sigma_modelo < 1e-10:
            return {"score": 0, "p_value": 1, "significativo": False}
        
        t_stat = (p_modelo - p_mercado) / sigma_modelo
        df = 10000
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        
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
        from scipy.stats import norm
        z_alpha = norm.ppf(1 - alpha/2)
        z_beta = (efecto * np.sqrt(n)) / sigma - z_alpha
        poder = norm.cdf(z_beta)
        return max(0, min(poder, 1))
    
    @staticmethod
    def calcular_entropia_kullback_leibler(p_modelo, p_mercado):
        epsilon = 1e-10
        p_modelo = max(p_modelo, epsilon)
        p_mercado = max(p_mercado, epsilon)
        
        kl_div = p_modelo * np.log(p_modelo / p_mercado)
        kl_norm = 1 - np.exp(-kl_div)
        
        return {
            "kl_divergence": kl_div,
            "incertidumbre_modelo": kl_norm,
            "informacion_bits": kl_div / np.log(2)
        }

class GestorRiscoCVaR:
    def __init__(self, cvar_target=0.15, max_drawdown=0.20):
        self.cvar_target = cvar_target
        self.max_drawdown = max_drawdown
        self.historial_riesgo = []
    
    def calcular_kelly_dinamico(self, prob, cuota, bankroll, metrics):
        try:
            if prob is None or cuota is None or bankroll is None:
                return {"stake_pct": 0, "stake_abs": 0, "razon": "Datos incompletos"}
            
            try:
                prob_num = float(prob)
                cuota_num = float(cuota)
                bankroll_num = float(bankroll)
            except (ValueError, TypeError):
                return {"stake_pct": 0, "stake_abs": 0, "razon": "Datos no numéricos"}
            
            if cuota_num <= 1.0:
                return {"stake_pct": 0, "stake_abs": 0, "razon": "Cuota <= 1.0"}
            
            ev = float(metrics.get("ev", 0)) if metrics else 0
            condiciones_minimas = (
                prob_num > 0.30,
                cuota_num > 1.40,
                ev > 0.01,
            )
            
            if not all(condiciones_minimas):
                return {
                    "stake_pct": 0, 
                    "stake_abs": 0, 
                    "razon": f"Condiciones: prob={prob_num:.2f} cuota={cuota_num:.2f} ev={ev:.2%}"
                }
            
            b = cuota_num - 1
            kelly_base = (prob_num * b - (1 - prob_num)) / b
            kelly_base = max(0, min(kelly_base, 0.5))
            
            cvar_actual = float(metrics.get("cvar_estimado", 0.15))
            cvar_actual = min(1.0, max(0.0, cvar_actual))
            
            if cvar_actual >= 1.0:
                return {
                    "stake_pct": 0.0, 
                    "stake_abs": 0.0, 
                    "razon": "🚫 EVASIÓN: Riesgo de cola inaceptable (100%+)"
                }
            
            incertidumbre = float(metrics.get("incertidumbre", 0.5))
            cvar_actual = float(metrics.get("cvar_estimado", 0.15))
            sharpe_actual = float(metrics.get("sharpe_esperado", 1.0))
            max_dd_actual = float(metrics.get("max_dd_promedio", 0.10))
            
            adj_incertidumbre = 1.0 / (1.0 + incertidumbre * 2.0)
            
            if cvar_actual <= 0.15:
                adj_cvar = 1.0
            elif cvar_actual <= 0.25:
                adj_cvar = 0.15 / cvar_actual
            else:
                adj_cvar = 0.15 / cvar_actual * 0.5
            
            adj_cvar = max(0.1, adj_cvar)
            
            if sharpe_actual >= 1.5:
                adj_sharpe = min(1.2, 1.0 + (sharpe_actual - 1.5) * 0.2)
            else:
                adj_sharpe = max(0.5, sharpe_actual / 1.5)
            
            if max_dd_actual <= 0.20:
                adj_dd = 1.0
            elif max_dd_actual <= 0.30:
                adj_dd = 0.20 / max_dd_actual
            else:
                adj_dd = 0.20 / max_dd_actual * 0.5
            
            adj_dd = max(0.1, adj_dd)
            
            if ev > 0.12:
                adj_ev = min(1.3, 1.0 + (ev - 0.12) * 2.5)
            else:
                adj_ev = max(0.3, ev / 0.12)
            
            kelly_ajustado = kelly_base * adj_incertidumbre * adj_cvar * adj_sharpe * adj_dd * adj_ev
            kelly_final = kelly_ajustado * 0.5
            
            if (ev > 0.20 and cvar_actual < 0.15 and sharpe_actual > 2.0):
                limite_max = 0.05
            else:
                limite_max = 0.03
            
            kelly_final = max(0.005, min(kelly_final, limite_max))
            stake_abs = kelly_final * bankroll_num
            stake_abs = max(5.0, stake_abs)
            
            return {
                "stake_pct": kelly_final * 100,
                "stake_abs": stake_abs,
                "kelly_base": kelly_base * 100,
                "ajustes": {
                    "incertidumbre": adj_incertidumbre,
                    "cvar": adj_cvar,
                    "sharpe": adj_sharpe,
                    "drawdown": adj_dd,
                    "ev": adj_ev
                },
                "razon": f"CVaR: {cvar_actual:.1%} | Sharpe: {sharpe_actual:.2f} | DD: {max_dd_actual:.1%} | EV: {ev:.1%}"
            }
            
        except Exception as e:
            return {
                "stake_pct": 0.5,
                "stake_abs": max(5.0, bankroll * 0.005),
                "razon": f"Error: {str(e)[:50]}"
            }
    
    def simular_cvar(self, prob, cuota, n_simulaciones=10000, conf_level=0.95):
        try:
            if prob <= 0 or prob >= 1 or cuota <= 1:
                return {
                    "cvar": 0.25,
                    "var": 0.20,
                    "esperanza": 0,
                    "desviacion": 0.1,
                    "sharpe_simulado": 0,
                    "max_perdida_simulada": -1,
                    "prob_perdida": 0.5
                }
            
            ganancias = []
            for _ in range(n_simulaciones):
                if np.random.random() < prob:
                    ganancias.append(cuota - 1)
                else:
                    ganancias.append(-1)
            
            ganancias = np.array(ganancias)
            percentil = 5
            var = np.percentile(ganancias, percentil)
            perdidas_extremas = ganancias[ganancias <= var]
            
            if len(perdidas_extremas) > 0:
                cvar = abs(perdidas_extremas.mean())
            else:
                cvar = abs(var) if var < 0 else 0.0
            
            if var < 0:
                cvar = max(cvar, abs(var))
            cvar = min(cvar, 1.0)
            
            esperanza = ganancias.mean()
            desviacion = ganancias.std()
            sharpe = esperanza / max(desviacion, 0.001)
            prob_perdida = np.mean(ganancias < 0)
            max_perdida = ganancias.min()
            
            return {
                "cvar": float(cvar),
                "var": float(abs(var)),
                "esperanza": float(esperanza),
                "desviacion": float(desviacion),
                "sharpe_simulado": float(sharpe),
                "max_perdida_simulada": float(max_perdida),
                "prob_perdida": float(prob_perdida)
            }
            
        except Exception as e:
            return {
                "cvar": 0.20,
                "var": 0.15,
                "esperanza": 0,
                "desviacion": 0.1,
                "sharpe_simulado": 0,
                "max_perdida_simulada": -1,
                "prob_perdida": 0.5,
                "error": str(e)[:100]
            }

class BacktestSintetico:
    @staticmethod
    def generar_escenarios(prob, cuota, bankroll_inicial=1000, n_apuestas=100, n_simulaciones=5000):
        resultados = []
        metricas_por_simulacion = []
        
        for sim in range(n_simulaciones):
            bankroll = bankroll_inicial
            historial_br = [bankroll]
            drawdown_actual = 0
            drawdown_maximo = 0
            peak = bankroll
            
            for apuesta in range(n_apuestas):
                stake_pct = 0.02
                stake = bankroll * stake_pct
                
                gana = np.random.random() < prob
                
                if gana:
                    bankroll += stake * (cuota - 1)
                else:
                    bankroll -= stake
                
                if bankroll > peak:
                    peak = bankroll
                
                drawdown_actual = (peak - bankroll) / peak
                drawdown_maximo = max(drawdown_maximo, drawdown_actual)
                
                historial_br.append(bankroll)
            
            retorno_total = (bankroll - bankroll_inicial) / bankroll_inicial
            volatilidad = np.std(np.diff(historial_br) / historial_br[:-1]) if len(historial_br) > 1 else 0
            sharpe = retorno_total / max(volatilidad, 0.01) * np.sqrt(252/365)
            
            metricas_por_simulacion.append({
                "final_balance": bankroll,
                "return": retorno_total,
                "max_drawdown": drawdown_maximo,
                "sharpe": sharpe,
                "ruin": bankroll < bankroll_inicial * 0.5
            })
            
            resultados.append(historial_br)
        
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

# ============================================
# FUNCIONES AUXILIARES (GLOBALES)
# ============================================

def convertir_datos_python(obj):
    """Convierte todos los datos numpy a tipos nativos de Python"""
    if isinstance(obj, np.integer): 
        return int(obj)
    if isinstance(obj, np.floating): 
        return float(obj)
    if isinstance(obj, np.ndarray): 
        return obj.tolist()
    if isinstance(obj, dict): 
        return {k: convertir_datos_python(v) for k, v in obj.items()}
    if isinstance(obj, list): 
        return [convertir_datos_python(x) for x in obj]
    return obj

def calcular_or_val_seguro(c1_val, cx_val, c2_val):
    """Calcula el overround de manera segura"""
    try:
        c1_float = float(c1_val) if c1_val else 1.01
        cx_float = float(cx_val) if cx_val else 1.01
        c2_float = float(c2_val) if c2_val else 1.01
        
        c1_float = max(1.01, c1_float)
        cx_float = max(1.01, cx_float)
        c2_float = max(1.01, c2_float)
        
        resultado = (1/c1_float + 1/cx_float + 1/c2_float) - 1
        
        if np.isfinite(resultado):
            return float(resultado)
        else:
            return 0.0
    except Exception as e:
        return 0.0

def actualizar_bankroll(resultado_apuesta, monto_apostado, cuota=None, pick=None, descripcion=""):
    """
    Actualiza el bankroll según el resultado de una apuesta
    SISTEMA DE DOBLE ENTRADA: Modifica bankroll_actual Y beneficio_neto
    """
    if 'bankroll_actual' not in st.session_state:
        st.session_state.bankroll_actual = 1000.0
    
    if 'beneficio_neto' not in st.session_state:
        st.session_state.beneficio_neto = 0.0
    
    if 'historial_bankroll' not in st.session_state:
        st.session_state.historial_bankroll = []
    
    if 'historial_apuestas' not in st.session_state:
        st.session_state.historial_apuestas = []
    
    registro_apuesta = {
        'timestamp': datetime.now(),
        'resultado': resultado_apuesta,
        'stake': monto_apostado,
        'cuota': cuota if cuota else 0,
        'pick': pick,
        'descripcion': descripcion
    }
    
    if resultado_apuesta == "ganada" and cuota:
        ganancia_neta = monto_apostado * (cuota - 1)
        # ACTUALIZACIÓN DOBLE ENTRADA
        st.session_state.bankroll_actual += ganancia_neta
        st.session_state.beneficio_neto += ganancia_neta
        
        registro_apuesta['ganancia'] = ganancia_neta
        registro_apuesta['resultado_final'] = f"+€{ganancia_neta:.2f}"
        
        registro_bankroll = {
            'timestamp': datetime.now(),
            'operacion': 'apuesta_ganada',
            'monto': ganancia_neta,
            'detalle': descripcion,
            'bankroll_final': st.session_state.bankroll_actual,
            'beneficio_neto': st.session_state.beneficio_neto
        }
        
        st.session_state.historial_bankroll.append(registro_bankroll)
        st.session_state.historial_apuestas.append(registro_apuesta)
        
        return ganancia_neta
        
    elif resultado_apuesta == "perdida":
        # ACTUALIZACIÓN DOBLE ENTRADA
        st.session_state.bankroll_actual -= monto_apostado
        st.session_state.beneficio_neto -= monto_apostado
        
        registro_apuesta['perdida'] = monto_apostado
        registro_apuesta['resultado_final'] = f"-€{monto_apostado:.2f}"
        
        registro_bankroll = {
            'timestamp': datetime.now(),
            'operacion': 'apuesta_perdida',
            'monto': -monto_apostado,
            'detalle': descripcion,
            'bankroll_final': st.session_state.bankroll_actual,
            'beneficio_neto': st.session_state.beneficio_neto
        }
        
        st.session_state.historial_bankroll.append(registro_bankroll)
        st.session_state.historial_apuestas.append(registro_apuesta)
        
        return -monto_apostado
    
    else:  # empatada (stake devuelto)
        registro_apuesta['resultado_final'] = f"€0.00 (stake devuelto)"
        st.session_state.historial_apuestas.append(registro_apuesta)
        return 0

def exportar_estado_json():
    """Exporta el estado actual a JSON"""
    estado = {
        'bankroll_actual': st.session_state.bankroll_actual,
        'bankroll_inicial_sesion': st.session_state.bankroll_inicial_sesion,
        'beneficio_neto': st.session_state.beneficio_neto,
        'historial_apuestas': st.session_state.historial_apuestas,
        'historial_bankroll': st.session_state.historial_bankroll,
        'dm': st.session_state.dm,
        'timestamp': datetime.now().isoformat()
    }
    
    # Convertir a tipos nativos de Python
    estado = convertir_datos_python(estado)
    
    return json.dumps(estado, indent=2, ensure_ascii=False, default=str)

def importar_estado_json(json_data):
    """Importa el estado desde JSON"""
    try:
        estado = json.loads(json_data)
        
        st.session_state.bankroll_actual = float(estado.get('bankroll_actual', 1000.0))
        st.session_state.bankroll_inicial_sesion = float(estado.get('bankroll_inicial_sesion', 1000.0))
        st.session_state.beneficio_neto = float(estado.get('beneficio_neto', 0.0))
        st.session_state.historial_apuestas = estado.get('historial_apuestas', [])
        st.session_state.historial_bankroll = estado.get('historial_bankroll', [])
        st.session_state.dm = estado.get('dm', {})
        
        return True
    except Exception as e:
        st.error(f"Error al importar JSON: {str(e)}")
        return False

# ============================================
# NAVEGACIÓN PRINCIPAL
# ============================================

# Sidebar navigation
menu = st.sidebar.selectbox(
    "Navegación",
    ["🏠 App Principal", "📊 Historial"],
    key="nav_menu"
)

# ============================================
# APP PRINCIPAL
# ============================================

if menu == "🏠 App Principal":
    st.title("🏛️ Sistema ACBE-Kelly v3.0 (Bayesiano Completo)")
    st.markdown("---")
    
    # ============================================
    # SIDEBAR: GESTIÓN DE CAPITAL Y DATOS
    # ============================================
    
    st.sidebar.header("⚙️ CONFIGURACIÓN DEL SISTEMA")
    
    # --- GESTIÓN DE CAPITAL (NUEVA ESTRUCTURA HÍBRIDA) ---
    st.sidebar.header("💰 GESTIÓN DE CAPITAL")
    
    # 1. ZONA DE CONFIGURACIÓN (Oculta por seguridad)
    with st.sidebar.expander("⚙️ Configurar Capital Inicial"):
        nuevo_inicio = st.number_input("Monto Inicial (€)", value=1000.0, step=100.0, key="input_reset_bankroll")
        if st.button("💾 Reiniciar Bankroll", use_container_width=True):
            # REINICIO COMPLETO DEL SISTEMA DE DOBLE ENTRADA
            st.session_state.bankroll_actual = nuevo_inicio
            st.session_state.bankroll_inicial_sesion = nuevo_inicio
            st.session_state.beneficio_neto = 0.0
            st.rerun()

    # 2. VISUALIZADOR PRINCIPAL (Solo lectura, se actualiza solo)
    st.sidebar.metric(
        label="🏦 BANKROLL ACTUAL",
        value=f"€{st.session_state.bankroll_actual:,.2f}",
        delta_color="normal"
    )
    
    # 3. MÉTRICAS DE PERFORMANCE (BASADAS EN BENEFICIO NETO)
    beneficio_neto = st.session_state.get('beneficio_neto', 0.0)
    roi = (beneficio_neto / st.session_state.bankroll_inicial_sesion * 100) if st.session_state.bankroll_inicial_sesion > 0 else 0
    
    st.sidebar.metric(
        label="📈 BENEFICIO NETO",
        value=f"€{beneficio_neto:,.2f}",
        delta=f"{roi:.1f}% ROI"
    )
    
    st.sidebar.metric(
        label="🎯 ROI ACUMULADO",
        value=f"{roi:.1f}%"
    )
    
    # 4. TRANSACCIONES (Depósitos y Retiros - NO AFECTAN BENEFICIO_NETO)
    st.sidebar.subheader("Transacciones")
    c_dep, c_ret = st.sidebar.columns(2)
    
    with c_dep:
        dep_val = st.number_input("Depósito", 0.0, step=50.0, key="in_dep")
        if st.button("📥 Ingresar", use_container_width=True):
            if dep_val > 0:
                # SOLO MODIFICA BANKROLL, NO BENEFICIO_NETO
                st.session_state.bankroll_actual += dep_val
                
                # Registrar en historial
                registro = {
                    'timestamp': datetime.now(),
                    'operacion': 'deposito',
                    'monto': dep_val,
                    'detalle': "Depósito manual",
                    'bankroll_final': st.session_state.bankroll_actual,
                    'beneficio_neto': st.session_state.beneficio_neto  # No cambia
                }
                st.session_state.historial_bankroll.append(registro)
                st.rerun()
                
    with c_ret:
        ret_val = st.number_input("Retiro", 0.0, step=50.0, key="in_ret")
        if st.button("📤 Retirar", use_container_width=True):
            if ret_val > 0 and ret_val <= st.session_state.bankroll_actual:
                # SOLO MODIFICA BANKROLL, NO BENEFICIO_NETO
                st.session_state.bankroll_actual -= ret_val
                
                # Registrar en historial
                registro = {
                    'timestamp': datetime.now(),
                    'operacion': 'retiro',
                    'monto': -ret_val,
                    'detalle': "Retiro manual",
                    'bankroll_final': st.session_state.bankroll_actual,
                    'beneficio_neto': st.session_state.beneficio_neto  # No cambia
                }
                st.session_state.historial_bankroll.append(registro)
                st.rerun()
            elif ret_val > st.session_state.bankroll_actual:
                st.sidebar.error("Fondos insuficientes")
    
    st.sidebar.markdown("---")
    
    # BACKUP/IMPORT JSON
    st.sidebar.subheader("💾 PERSISTENCIA DE DATOS")
    
    col_backup1, col_backup2 = st.sidebar.columns(2)
    
    with col_backup1:
        # Exportar
        json_export = exportar_estado_json()
        st.download_button(
            label="📤 Exportar",
            data=json_export,
            file_name=f"acbe_backup_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col_backup2:
        # Importar
        uploaded_file = st.sidebar.file_uploader("📥 Importar", type=["json"], key="json_uploader")
        if uploaded_file is not None:
            try:
                json_data = uploaded_file.getvalue().decode("utf-8")
                if importar_estado_json(json_data):
                    st.sidebar.success("✅ Estado importado correctamente")
                    st.rerun()
            except Exception as e:
                st.sidebar.error(f"❌ Error al importar: {str(e)}")
    
    with st.sidebar.expander("🎯 OBJETIVOS DE PERFORMANCE", expanded=True):
        col_obj1, col_obj2 = st.columns(2)
        with col_obj1:
            roi_target = st.slider("ROI Target (%)", 5, 25, 12, key="roi_target_main") / 100
            cvar_target = st.slider("CVaR Máximo (%)", 5, 25, 15, key="cvar_target_main") / 100
        with col_obj2:
            max_dd = st.slider("Max Drawdown (%)", 10, 40, 20, key="max_dd_main") / 100
            sharpe_min = st.slider("Sharpe Mínimo", 0.5, 3.0, 1.50, key="sharpe_min_main")
        
        st.markdown("---")
        st.markdown(f"""
        **Objetivos establecidos:**
        - ROI: {roi_target:.0%}
        - CVaR: < {cvar_target:.0%}
        - Max DD: < {max_dd:.0%}
        - Sharpe: > {sharpe_min}
        """)
    
    # ============ INGESTA DE DATOS ============
    st.sidebar.header("📥 INGESTA DE DATOS")
    
    team_h = st.sidebar.text_input("Equipo Local", value="Bologna", key="team_h_input")
    team_a = st.sidebar.text_input("Equipo Visitante", value="AC Milan", key="team_a_input")
    
    st.sidebar.header("💰 MERCADO")
    c_col1, c_col2, c_col3 = st.sidebar.columns(3)
    c1 = c_col1.number_input("1", value=2.90, min_value=1.01, step=0.01, key="c1_input_pro")
    cx = c_col2.number_input("X", value=3.25, min_value=1.01, step=0.01, key="cx_input_pro")
    c2 = c_col3.number_input("2", value=2.45, min_value=1.01, step=0.01, key="c2_input_pro")
    
    st.sidebar.markdown("---")
    st.sidebar.header("📈 MÉTRICAS DE MERCADO")
    
    # Calcular overround
    c1_f, cx_f, c2_f = float(c1), float(cx), float(c2)
    or_val = (1/c1_f + 1/cx_f + 1/c2_f) - 1
    
    entropia_mercado = st.sidebar.slider(
        "Entropía (H)", 0.3, 0.9, 
        value=st.session_state['entropia_mercado'], 
        key="ent_slider_v3"
    )
    st.session_state['entropia_mercado'] = entropia_mercado
    
    volumen_estimado = st.sidebar.slider("Volumen Relativo", 0.5, 2.0, 1.0, step=0.1, key="vol_slider_v3")
    steam_detectado = st.sidebar.slider("Steam Move (σ)", 0.0, 0.05, 0.0, step=0.005, key="steam_slider_v3")
    
    col_met1, col_met2, col_met3 = st.sidebar.columns(3)
    with col_met1:
        st.metric("Overround", f"{or_val*100:.2f}%")
    
    with col_met2:
        margen_num = (or_val / (1 + or_val)) * 100 if or_val > -1 else 0.0
        st.metric("Margen Casa", f"{margen_num:.1f}%")
    
    with col_met3:
        st.metric("Entropía", f"{entropia_mercado:.3f}")
        
    liga = st.sidebar.selectbox("Liga", ["Serie A", "Premier League", "La Liga", "Bundesliga", "Ligue 1"], 
                               key="liga_selector_sidebar")
    
    # ============================================
    # SECCIÓN DE INPUTS DE EQUIPOS (EN APP PRINCIPAL)
    # ============================================
    
    st.header("📊 DATOS DE LOS EQUIPOS")
    
    # Crear tabs para separar equipo local y visitante
    tab_h, tab_a = st.tabs([f"🏠 {team_h} (Local)", f"✈️ {team_a} (Visitante)"])
    
    with tab_h:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("⚽ Ataque")
            g_h_ult5 = st.number_input("Goles últimos 5 partidos", 0, 30, 8, key="g_h_ult5")
            g_h_ult10 = st.number_input("Goles últimos 10 partidos", 0, 60, 15, key="g_h_ult10")
            xg_h_prom = st.number_input("xG promedio", 0.0, 5.0, 1.5, step=0.1, key="xg_h_prom")
            tiros_arco_h = st.number_input("Tiros a puerta p/p", 0.0, 20.0, 4.5, step=0.1, key="tiros_arco_h")
        
        with col2:
            st.subheader("🛡️ Defensa")
            goles_rec_h = st.number_input("Goles recibidos últ. 10", 0, 30, 12, key="goles_rec_h")
            xg_contra_h = st.number_input("xG contra p/p", 0.0, 5.0, 1.2, step=0.1, key="xg_contra_h")
            entradas_h = st.number_input("Entradas p/p", 0.0, 30.0, 15.5, step=0.1, key="entradas_h")
            recuperaciones_h = st.number_input("Recuperaciones p/p", 0.0, 100.0, 45.0, step=0.1, key="recuperaciones_h")
        
        with col3:
            st.subheader("📈 Control & Estado")
            posesion_h = st.slider("Posesión (%)", 0, 100, 52, key="posesion_h")
            precision_pases_h = st.slider("Precisión pases (%)", 0, 100, 78, key="precision_pases_h")
            delta_h = st.slider("Delta forma (1=mejor)", 0.5, 1.5, 1.0, step=0.05, key="delta_h")
            motivacion_h = st.slider("Motivación", 0.5, 1.5, 1.0, step=0.05, key="motivacion_h")
            carga_fisica_h = st.slider("Carga física (1=normal)", 0.5, 2.0, 1.0, step=0.05, key="carga_fisica_h")
    
    with tab_a:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("⚽ Ataque")
            g_a_ult5 = st.number_input("Goles últimos 5 partidos", 0, 30, 7, key="g_a_ult5")
            g_a_ult10 = st.number_input("Goles últimos 10 partidos", 0, 60, 14, key="g_a_ult10")
            xg_a_prom = st.number_input("xG promedio", 0.0, 5.0, 1.4, step=0.1, key="xg_a_prom")
            tiros_arco_a = st.number_input("Tiros a puerta p/p", 0.0, 20.0, 4.2, step=0.1, key="tiros_arco_a")
        
        with col2:
            st.subheader("🛡️ Defensa")
            goles_rec_a = st.number_input("Goles recibidos últ. 10", 0, 30, 10, key="goles_rec_a")
            xg_contra_a = st.number_input("xG contra p/p", 0.0, 5.0, 1.1, step=0.1, key="xg_contra_a")
            entradas_a = st.number_input("Entradas p/p", 0.0, 30.0, 14.5, step=0.1, key="entradas_a")
            recuperaciones_a = st.number_input("Recuperaciones p/p", 0.0, 100.0, 42.0, step=0.1, key="recuperaciones_a")
        
        with col3:
            st.subheader("📈 Control & Estado")
            posesion_a = st.slider("Posesión (%)", 0, 100, 48, key="posesion_a")
            precision_pases_a = st.slider("Precisión pases (%)", 0, 100, 76, key="precision_pases_a")
            delta_a = st.slider("Delta forma (1=mejor)", 0.5, 1.5, 1.0, step=0.05, key="delta_a")
            motivacion_a = st.slider("Motivación", 0.5, 1.5, 1.0, step=0.05, key="motivacion_a")
            carga_fisica_a = st.slider("Carga física (1=normal)", 0.5, 2.0, 1.0, step=0.05, key="carga_fisica_a")
    
    st.markdown("---")
    
    # ============================================
    # BOTÓN ÚNICO DE EJECUCIÓN (LA COCINA)
    # ============================================
    
    # Botón en la app principal (no en sidebar) para asegurar scope
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        ejecutar_analisis = st.button("🚀 EJECUTAR ANÁLISIS COMPLETO", 
                                      type="primary", 
                                      key="btn_final_maestro",
                                      use_container_width=True)
    
    if ejecutar_analisis:
        try:
            with st.spinner("🧠 EJECUTANDO ANÁLISIS COMPLETO..."):
                
                # ============================================
                # FASE 1: INFERENCIA VARIACIONAL
                # ============================================
                with st.spinner("🔮 Fase 1: Inferencia Bayesiana..."):
                    # Obtener inputs de los widgets DIRECTAMENTE DE SESSION_STATE
                    datos = {
                        'team_h': st.session_state['team_h_input'],
                        'team_a': st.session_state['team_a_input'],
                        'g_h_ult5': st.session_state['g_h_ult5'],
                        'g_h_ult10': st.session_state['g_h_ult10'],
                        'xg_h_prom': st.session_state['xg_h_prom'],
                        'tiros_arco_h': st.session_state['tiros_arco_h'],
                        'posesion_h': st.session_state['posesion_h'],
                        'precision_pases_h': st.session_state['precision_pases_h'],
                        'goles_rec_h': st.session_state['goles_rec_h'],
                        'xg_contra_h': st.session_state['xg_contra_h'],
                        'entradas_h': st.session_state['entradas_h'],
                        'recuperaciones_h': st.session_state['recuperaciones_h'],
                        'delta_h': st.session_state['delta_h'],
                        'motivacion_h': st.session_state['motivacion_h'],
                        'carga_fisica_h': st.session_state['carga_fisica_h'],
                        'g_a_ult5': st.session_state['g_a_ult5'],
                        'g_a_ult10': st.session_state['g_a_ult10'],
                        'xg_a_prom': st.session_state['xg_a_prom'],
                        'tiros_arco_a': st.session_state['tiros_arco_a'],
                        'posesion_a': st.session_state['posesion_a'],
                        'precision_pases_a': st.session_state['precision_pases_a'],
                        'goles_rec_a': st.session_state['goles_rec_a'],
                        'xg_contra_a': st.session_state['xg_contra_a'],
                        'entradas_a': st.session_state['entradas_a'],
                        'recuperaciones_a': st.session_state['recuperaciones_a'],
                        'delta_a': st.session_state['delta_a'],
                        'motivacion_a': st.session_state['motivacion_a'],
                        'carga_fisica_a': st.session_state['carga_fisica_a'],
                        'cuotas': {'1': st.session_state['c1_input_pro'], 
                                  'X': st.session_state['cx_input_pro'], 
                                  '2': st.session_state['c2_input_pro']},
                        'overround': or_val, 
                        'liga': liga,
                        'volumen_estimado': st.session_state['vol_slider_v3'],
                        'steam_detectado': st.session_state['steam_slider_v3'],
                        'entropia_mercado': st.session_state['ent_slider_v3']
                    }
                    
                    # Guardar inputs
                    st.session_state['dm']['inputs'] = datos
                    
                    # Preparar datos para inferencia
                    datos_local = {
                        "goles_anotados": st.session_state['g_h_ult10'],
                        "goles_recibidos": st.session_state['goles_rec_h'],
                        "n_partidos": 10,
                        "xG": st.session_state['xg_h_prom'],
                        "tiros_arco": st.session_state['tiros_arco_h'],
                        "posesion": st.session_state['posesion_h'],
                        "precision_pases": st.session_state['precision_pases_h']
                    }
                    
                    datos_visitante = {
                        "goles_anotados": st.session_state['g_a_ult10'],
                        "goles_recibidos": st.session_state['goles_rec_a'],
                        "n_partidos": 10,
                        "xG": st.session_state['xg_a_prom'],
                        "tiros_arco": st.session_state['tiros_arco_a'],
                        "posesion": st.session_state['posesion_a'],
                        "precision_pases": st.session_state['precision_pases_a']
                    }
                    
                    # Ejecutar inferencia
                    modelo_bayes = ModeloBayesianoJerarquico(liga)
                    post_h = modelo_bayes.inferencia_variacional(datos_local, es_local=True)
                    post_a = modelo_bayes.inferencia_variacional(datos_visitante, es_local=False)
                    
                    # Aplicar ajustes de factores de riesgo
                    l_h_adj = post_h["lambda"] * (1 - st.session_state['delta_h']) * st.session_state['motivacion_h'] / st.session_state['carga_fisica_h']
                    l_a_adj = post_a["lambda"] * (1 - st.session_state['delta_a']) * st.session_state['motivacion_a'] / st.session_state['carga_fisica_a']
                    
                    # Guardar resultados Fase 1
                    st.session_state['dm']['fase1'] = {
                        'modelo': modelo_bayes,
                        'post_h': post_h,
                        'post_a': post_a,
                        'l_h_adj': l_h_adj,
                        'l_a_adj': l_a_adj,
                        'inc_h': post_h['incertidumbre'],
                        'inc_a': post_a['incertidumbre'],
                        'ci_h': post_h['ci_95'],
                        'ci_a': post_a['ci_95']
                    }
                
                # ============================================
                # FASE 2: SIMULACIÓN MONTE CARLO
                # ============================================
                with st.spinner("🎲 Fase 2: Simulación Monte Carlo (50k escenarios)..."):
                    n_sim = 50000
                    post_h = st.session_state['dm']['fase1']['post_h']
                    post_a = st.session_state['dm']['fase1']['post_a']
                    l_h_adj = st.session_state['dm']['fase1']['l_h_adj']
                    l_a_adj = st.session_state['dm']['fase1']['l_a_adj']
                    
                    # Simulación vectorizada de goles
                    goles_h = np.random.poisson(l_h_adj, n_sim)
                    goles_a = np.random.poisson(l_a_adj, n_sim)
                    
                    # Probabilidades
                    p1_mc = float(np.mean(goles_h > goles_a))
                    px_mc = float(np.mean(goles_h == goles_a))
                    p2_mc = float(np.mean(goles_h < goles_a))
                    
                    # Errores estándar
                    se_p1 = float(np.sqrt(p1_mc*(1-p1_mc)/n_sim))
                    se_px = float(np.sqrt(px_mc*(1-px_mc)/n_sim))
                    se_p2 = float(np.sqrt(p2_mc*(1-p2_mc)/n_sim))
                    
                    # Guardar resultados Fase 2
                    st.session_state['dm']['fase2'] = {
                        'p1': p1_mc,
                        'px': px_mc,
                        'p2': p2_mc,
                        'se': [se_p1, se_px, se_p2],
                        'goles_h_sims': goles_h,
                        'goles_a_sims': goles_a,
                        'n_sim': n_sim
                    }
                
                # ============================================
                # FASE 3: DETECCIÓN DE INEFICIENCIAS
                # ============================================
                with st.spinner("🔍 Fase 3: Detectando ineficiencias..."):
                    # Inicializar detector
                    detector = DetectorIneficiencias()
                    
                    # Probabilidades del mercado
                    c1_val = st.session_state['c1_input_pro']
                    cx_val = st.session_state['cx_input_pro']
                    c2_val = st.session_state['c2_input_pro']
                    
                    p1_mercado = 1 / c1_val if c1_val > 0 else 0.33
                    px_mercado = 1 / cx_val if cx_val > 0 else 0.33
                    p2_mercado = 1 / c2_val if c2_val > 0 else 0.33
                    
                    # Entropía de Shannon
                    prob_mercado_array = np.array([p1_mercado, px_mercado, p2_mercado])
                    prob_mercado_array = prob_mercado_array[prob_mercado_array > 0]
                    entropia_auto = -np.sum(prob_mercado_array * np.log2(prob_mercado_array))
                    
                    # Análisis para cada resultado
                    resultados_analisis = []
                    picks_con_valor = []
                    
                    for label, p_modelo, p_mercado, se, cuota in zip(
                        ["1", "X", "2"],
                        [p1_mc, px_mc, p2_mc],
                        [p1_mercado, px_mercado, p2_mercado],
                        [se_p1, se_px, se_p2],
                        [c1_val, cx_val, c2_val]
                    ):
                        # Value Score
                        value_analysis = detector.calcular_value_score(p_modelo, p_mercado, se)
                        
                        # KL Divergence
                        kl_analysis = detector.calcular_entropia_kullback_leibler(p_modelo, p_mercado)
                        
                        # Valor esperado
                        ev = p_modelo * cuota - 1
                        
                        # Cuota justa
                        fair_odd = 1 / p_modelo if p_modelo > 0 else 999
                        
                        resultado = {
                            "Resultado": label,
                            "Prob Modelo": float(p_modelo),
                            "Prob Mercado": float(p_mercado),
                            "Delta": float(p_modelo - p_mercado),
                            "EV": float(ev),
                            "Fair Odd": float(fair_odd),
                            "Cuota Mercado": float(cuota),
                            "Value Score": {
                                "t_statistic": float(value_analysis.get("t_statistic", 0)),
                                "significativo": bool(value_analysis.get("significativo", False))
                            },
                            "KL Divergence": {
                                "informacion_bits": float(kl_analysis.get("informacion_bits", 0))
                            }
                        }
                        
                        resultados_analisis.append(resultado)
                        
                        # Identificar picks con valor
                        if value_analysis.get("significativo", False) and ev > 0.02:
                            picks_con_valor.append(resultado)
                    
                    # Guardar resultados Fase 3
                    st.session_state['dm']['fase3'] = {
                        'detector': detector,
                        'resultados_analisis': resultados_analisis,
                        'picks_con_valor': picks_con_valor,
                        'entropia_auto': entropia_auto
                    }
                
                # ============================================
                # FASE 4: GESTIÓN DE CAPITAL
                # ============================================
                with st.spinner("💰 Fase 4: Gestión de capital (Kelly Dinámico)..."):
                    # Inicializar componentes
                    gestor_riesgo = GestorRiscoCVaR(cvar_target=cvar_target, max_drawdown=max_dd)
                    backtester = BacktestSintetico()
                    
                    bankroll = st.session_state.get('bankroll_actual', 1000.0)
                    picks_con_valor = st.session_state['dm']['fase3']['picks_con_valor']
                    post_h = st.session_state['dm']['fase1']['post_h']
                    post_a = st.session_state['dm']['fase1']['post_a']
                    entropia_auto = st.session_state['dm']['fase3']['entropia_auto']
                    
                    # Ejecutar gestión de capital para cada pick con valor
                    recomendaciones = []
                    
                    for r in picks_con_valor:
                        try:
                            # Datos numéricos
                            prob_modelo_numerico = r["Prob Modelo"]
                            cuota_numerico = r["Cuota Mercado"]
                            ev_numerico = r["EV"]
                            significativo = r["Value Score"]["significativo"]
                            
                            # Simulación CVaR
                            simulacion_cvar = gestor_riesgo.simular_cvar(
                                prob=prob_modelo_numerico,
                                cuota=cuota_numerico,
                                n_simulaciones=10000,
                                conf_level=0.95
                            )
                            
                            # Incertidumbre según resultado
                            if r["Resultado"] == "1":
                                incertidumbre_valor = post_h.get("incertidumbre", 0.5)
                            elif r["Resultado"] in ["2", "X"]:
                                incertidumbre_valor = post_a.get("incertidumbre", 0.5)
                            else:
                                incertidumbre_valor = 0.5
                            
                            # Métricas para Kelly
                            metrics_kelly = {
                                "incertidumbre": incertidumbre_valor,
                                "cvar_estimado": simulacion_cvar.get("cvar", 0.15),
                                "entropia": entropia_auto,
                                "sharpe_esperado": simulacion_cvar.get("sharpe_simulado", 1.0),
                                "prob_modelo": prob_modelo_numerico,
                                "valor_estadistico": r["Value Score"]["t_statistic"],
                                "ev": ev_numerico,
                                "significativo": significativo
                            }
                            
                            # Kelly Dinámico
                            kelly_result = gestor_riesgo.calcular_kelly_dinamico(
                                prob=prob_modelo_numerico,
                                cuota=cuota_numerico,
                                bankroll=bankroll,
                                metrics=metrics_kelly
                            )
                            
                            # Backtest Sintético
                            backtest_result = backtester.generar_escenarios(
                                prob=prob_modelo_numerico,
                                cuota=cuota_numerico,
                                bankroll_inicial=bankroll,
                                n_apuestas=100,
                                n_simulaciones=2000
                            )
                            
                            recomendacion = {
                                "resultado": r["Resultado"],
                                "ev": f"{ev_numerico:.2%}",
                                "kelly_pct": kelly_result.get("stake_pct", 0),
                                "stake_abs": kelly_result.get("stake_abs", 0),
                                "cvar": simulacion_cvar.get("cvar", 0.15),
                                "sharpe_esperado": backtest_result["metricas"].get("sharpe_promedio", 0),
                                "prob_profit": backtest_result["metricas"].get("prob_profit", 0),
                                "max_dd_promedio": backtest_result["metricas"].get("max_dd_promedio", 0),
                                "backtest_metrics": backtest_result["metricas"],
                                "razon_kelly": kelly_result.get("razon", "Sin información"),
                                "prob_modelo_numerico": prob_modelo_numerico,
                                "cuota_numerico": cuota_numerico,
                                "ev_numerico": ev_numerico
                            }
                            
                            recomendaciones.append(recomendacion)
                            
                        except Exception as e:
                            st.warning(f"⚠️ Error procesando pick {r.get('Resultado', 'N/A')}: {str(e)}")
                            continue
                    
                    # Guardar resultados Fase 4
                    st.session_state['dm']['fase4'] = {
                        'gestor_riesgo': gestor_riesgo,
                        'backtester': backtester,
                        'recomendaciones': recomendaciones,
                        'bankroll': bankroll
                    }
                
                # ============================================
                # FASE 5: MÉTRICAS DE PERFORMANCE
                # ============================================
                with st.spinner("📊 Fase 5: Generando métricas de performance..."):
                    recomendaciones = st.session_state['dm']['fase4']['recomendaciones']
                    
                    # Calcular métricas agregadas
                    ev_promedio = 0
                    sharpe_promedio = 0
                    cvar_promedio = 0
                    prob_profit_promedio = 0
                    objetivos_cumplidos = []
                    
                    if recomendaciones:
                        ev_valores = [r['ev_numerico'] for r in recomendaciones]
                        ev_promedio = np.mean(ev_valores)
                        sharpe_promedio = np.mean([r.get('sharpe_esperado', 0) for r in recomendaciones])
                        cvar_promedio = np.mean([r.get('cvar', 0.15) for r in recomendaciones])
                        prob_profit_promedio = np.mean([r.get('prob_profit', 0) for r in recomendaciones])
                        
                        # Verificar objetivos
                        if ev_promedio * 100 >= roi_target * 0.8:
                            objetivos_cumplidos.append("ROI")
                        if cvar_promedio <= cvar_target/100:
                            objetivos_cumplidos.append("CVaR")
                        if sharpe_promedio >= sharpe_min:
                            objetivos_cumplidos.append("Sharpe")
                    
                    # Guardar resultados Fase 5
                    st.session_state['dm']['fase5'] = {
                        'ev_promedio': ev_promedio,
                        'sharpe_promedio': sharpe_promedio,
                        'cvar_promedio': cvar_promedio,
                        'prob_profit_promedio': prob_profit_promedio,
                        'objetivos_cumplidos': objetivos_cumplidos,
                        'roi_target': roi_target,
                        'cvar_target': cvar_target,
                        'sharpe_min': sharpe_min
                    }
                
                # ============================================
                # FINALIZACIÓN
                # ============================================
                st.session_state['analisis_ejecutado'] = True
                st.success("✅ ANÁLISIS COMPLETO EJECUTADO")
                
        except Exception as e:
            st.error(f"❌ Error crítico en el análisis: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            st.info("Por favor, verifica que todos los datos de entrada sean correctos.")
        
        # ÚNICO st.rerun() AL FINAL DE LA EJECUCIÓN
        st.rerun()
    
    # ============================================
    # RENDERIZADO PERSISTENTE (EL SALÓN)
    # ============================================
    
    if st.session_state.get('analisis_ejecutado', False) and 'dm' in st.session_state:
        dm = st.session_state['dm']
        
        # ============ VALIDACIÓN DE MERCADO ============
        if 'inputs' in dm:
            inputs = dm['inputs']
            st.subheader("🎯 VALIDACIÓN DE MERCADO")
            
            col_val1, col_val2, col_val3, col_val4 = st.columns(4)
            
            with col_val1:
                val_min_odd = inputs['cuotas']['1'] >= 1.60 and inputs['cuotas']['2'] >= 1.60
                st.metric("Cuota Mínima", "✅" if val_min_odd else "❌", 
                        delta="OK" if val_min_odd else "< 1.60")
            
            with col_val2:
                val_or = inputs['overround'] <= 0.07
                st.metric("Overround", "✅" if val_or else "❌", 
                        delta=f"{inputs['overround']:.2%}" if val_or else "Alto")
            
            with col_val3:
                val_entropia = inputs['entropia_mercado'] <= 0.72
                st.metric("Entropía", "✅" if val_entropia else "❌",
                        delta=f"{inputs['entropia_mercado']:.3f}")
            
            with col_val4:
                val_volumen = inputs.get('volumen_estimado', 1) >= 0.8
                st.metric("Liquidez", "✅" if val_volumen else "⚠️",
                        delta=f"{inputs.get('volumen_estimado', 1):.1f}x")
            
            # Verificar condiciones
            condiciones_evasion = []
            if not val_min_odd: condiciones_evasion.append("Cuota < 1.60")
            if not val_or: condiciones_evasion.append(f"Overround alto ({inputs['overround']:.2%})")
            if not val_entropia: condiciones_evasion.append(f"Entropía alta ({inputs['entropia_mercado']:.3f})")
            
            if condiciones_evasion:
                st.error(f"🚫 EVASIÓN DE RIESGO: {', '.join(condiciones_evasion)}")
                st.stop()
            
            st.success("✅ MERCADO VÁLIDO PARA ANÁLISIS")
        
        # ============ FASE 1: INFERENCIA BAYESIANA ============
        if 'fase1' in dm:
            f1 = dm['fase1']
            inputs = dm['inputs']
            
            st.subheader("🧠 FASE 1: INFERENCIA BAYESIANA")
            col_inf1, col_inf2 = st.columns(2)
            
            with col_inf1:
                st.markdown(f"**🏠 {inputs['team_h']}**")
                st.metric("λ Posterior", f"{f1['l_h_adj']:.3f}", help="Goles esperados ajustados")
                st.metric("Incertidumbre", f"{f1['inc_h']:.3%}")
                st.caption(f"Intervalo Credibilidad: {f1['ci_h'][0]:.2f} - {f1['ci_h'][1]:.2f}")
            
            with col_inf2:
                st.markdown(f"**✈️ {inputs['team_a']}**")
                st.metric("λ Posterior", f"{f1['l_a_adj']:.3f}", help="Goles esperados ajustados")
                st.metric("Incertidumbre", f"{f1['inc_a']:.3%}")
                st.caption(f"Intervalo Credibilidad: {f1['ci_a'][0]:.2f} - {f1['ci_a'][1]:.2f}")
        
        # ============ FASE 2: SIMULACIÓN MONTE CARLO ============
        if 'fase2' in dm:
            f2 = dm['fase2']
            
            st.subheader("🎲 FASE 2: SIMULACIÓN MONTE CARLO (50,000 escenarios)")
            
            # Gráfico de barras
            fig_sim = go.Figure(data=[
                go.Bar(
                    x=["1 (Local)", "X (Empate)", "2 (Visitante)"],
                    y=[f2['p1'], f2['px'], f2['p2']],
                    error_y=dict(type='data', array=f2['se']),
                    marker_color=['#00CC96', '#636EFA', '#EF553B'],
                    text=[f"{f2['p1']:.1%}", f"{f2['px']:.1%}", f"{f2['p2']:.1%}"],
                    textposition='auto',
                )
            ])
            fig_sim.update_layout(
                template="plotly_dark", 
                height=450, 
                showlegend=False,
                title="Distribución de Probabilidades - Simulación Monte Carlo"
            )
            st.plotly_chart(fig_sim, use_container_width=True)
        
        # ============ FASE 3: DETECCIÓN DE INEFICIENCIAS ============
        if 'fase3' in dm:
            f3 = dm['fase3']
            
            st.subheader("🔍 FASE 3: DETECCIÓN DE INEFICIENCIAS")
            
            # Tabla de resultados
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
                for r in f3['resultados_analisis']
            ])
            st.dataframe(df_resultados, use_container_width=True)
            
            # Picks con valor
            picks_con_valor = f3['picks_con_valor']
            if picks_con_valor:
                st.success(f"✅ **{len(picks_con_valor)} INEFICIENCIA(S) DETECTADA(S)**")
            else:
                st.warning("⚠️ MERCADO EFICIENTE: No se detectan ineficiencias significativas")
        
        # ============ FASE 4: GESTIÓN DE CAPITAL ============
        if 'fase4' in dm:
            f4 = dm['fase4']
            
            st.subheader("💰 FASE 4: GESTIÓN DE CAPITAL (KELLY DINÁMICO)")
            
            recomendaciones = f4['recomendaciones']
            bankroll = f4['bankroll']
            
            if not recomendaciones:
                st.info("📭 No hay picks con valor para gestionar capital")
            else:
                # Mostrar stake total
                stake_total = sum([r.get('stake_abs', 0) for r in recomendaciones])
                st.info(f"📊 **Stake Total Recomendado:** €{stake_total:,.2f} ({stake_total/bankroll*100:.1f}% del bankroll)")
                
                if stake_total > bankroll * 0.25:
                    st.warning("⚠️ **ALERTA:** Estás apostando más del 25% de tu bankroll. Considera reducir stakes.")
                
                # Mostrar cada recomendación - SIEMPRE VISIBLE (sin condiciones)
                for i, rec in enumerate(recomendaciones):
                    with st.expander(
                        f"🎯 **RECOMENDACIÓN {i+1}: {rec['resultado']}** - EV: {rec['ev']} - Stake: {rec['kelly_pct']:.2f}%",
                        expanded=True
                    ):
                        # Fila 1: Métricas
                        col_met1, col_met2, col_met3, col_met4 = st.columns(4)
                        
                        with col_met1:
                            st.metric("💰 Stake", f"€{rec['stake_abs']:.0f}")
                            st.caption(f"{rec['kelly_pct']:.2f}% bankroll")
                        
                        with col_met2:
                            st.metric("🎯 EV", f"{rec['ev']}")
                            st.caption("Valor Esperado")
                        
                        with col_met3:
                            st.metric("⚠️ CVaR", f"{rec['cvar']:.2%}")
                            st.caption("Riesgo de cola")
                        
                        with col_met4:
                            st.metric("📈 Sharpe", f"{rec['sharpe_esperado']:.2f}")
                            st.caption("Ratio riesgo/retorno")
                        
                        # Fila 2: BOTONES DE ACCIÓN - EN 3 COLUMNAS SEPARADAS
                        st.markdown("---")
                        st.subheader("📝 REGISTRAR RESULTADO")
                        
                        col_btn1, col_btn2, col_btn3 = st.columns(3)
                        
                        with col_btn1:
                            if st.button(f"✅ GANADA", key=f"win_{i}_{uuid.uuid4()}", 
                                      type="primary", use_container_width=True):
                                ganancia = rec.get('stake_abs', 0) * (rec.get('cuota_numerico', 2.0) - 1)
                                resultado = actualizar_bankroll(
                                    resultado_apuesta="ganada",
                                    monto_apostado=rec.get('stake_abs', 0),
                                    cuota=rec.get('cuota_numerico', 2.0),
                                    pick=rec['resultado'],
                                    descripcion=f"Apuesta {rec['resultado']} ganada"
                                )
                                st.success(f"✅ Ganancia registrada: €{ganancia:.2f}")
                                st.rerun()
                        
                        with col_btn2:
                            if st.button(f"❌ PERDIDA", key=f"loss_{i}_{uuid.uuid4()}", 
                                      type="secondary", use_container_width=True):
                                resultado = actualizar_bankroll(
                                    resultado_apuesta="perdida",
                                    monto_apostado=rec.get('stake_abs', 0),
                                    pick=rec['resultado'],
                                    descripcion=f"Apuesta {rec['resultado']} perdida"
                                )
                                st.error(f"❌ Pérdida registrada: €{rec.get('stake_abs', 0):.2f}")
                                st.rerun()
                        
                        with col_btn3:
                            if st.button(f"🔄 VOID", key=f"void_{i}_{uuid.uuid4()}", 
                                      type="secondary", use_container_width=True):
                                resultado = actualizar_bankroll(
                                    resultado_apuesta="void",
                                    monto_apostado=rec.get('stake_abs', 0),
                                    pick=rec['resultado'],
                                    descripcion=f"Apuesta {rec['resultado']} anulada"
                                )
                                st.info("💰 Apuesta anulada - Stake devuelto")
                                st.rerun()
                        
                        # Información adicional
                        with st.expander("📊 Métricas detalladas", expanded=False):
                            col_det1, col_det2 = st.columns(2)
                            with col_det1:
                                st.metric("🎯 Prob. Profit", f"{rec['prob_profit']:.1%}")
                                st.metric("📉 Max DD Esperado", f"{rec['max_dd_promedio']:.1%}")
                            with col_det2:
                                st.metric("📊 Kelly Base", f"{rec.get('kelly_base', 0):.2f}%" if 'kelly_base' in rec else "N/A")
                                st.caption(f"**Razón:** {rec.get('razon_kelly', 'Sin información')}")
        
        # ============ FASE 5: MÉTRICAS DE PERFORMANCE ============
        if 'fase5' in dm:
            f5 = dm['fase5']
            
            st.subheader("📊 FASE 5: REPORTE DE RIESGO Y PERFORMANCE")
            
            col_obj1, col_obj2, col_obj3, col_obj4 = st.columns(4)
            
            with col_obj1:
                color_text = "🟢" if f5['ev_promedio'] * 100 >= f5['roi_target'] * 0.8 else "🟠"
                st.metric(f"ROI Esperado {color_text}", f"{f5['ev_promedio']:.2%}")
                st.caption(f"Target: {f5['roi_target']:.0%}")
            
            with col_obj2:
                color_text = "🟢" if f5['cvar_promedio'] <= f5['cvar_target']/100 else "🔴"
                st.metric(f"CVaR Estimado {color_text}", f"{f5['cvar_promedio']:.2%}")
                st.caption(f"Máx: {f5['cvar_target']:.0%}")
            
            with col_obj3:
                color_text = "🟢" if f5['sharpe_promedio'] >= f5['sharpe_min'] else "🟠"
                st.metric(f"Sharpe Esperado {color_text}", f"{f5['sharpe_promedio']:.2f}")
                st.caption(f"Mín: {f5['sharpe_min']}")
            
            with col_obj4:
                st.metric("Prob. Éxito", f"{f5['prob_profit_promedio']:.1%}")
                st.caption("Probabilidad de ganar")
            
            # Resumen de objetivos
            objetivos_cumplidos = f5['objetivos_cumplidos']
            if len(objetivos_cumplidos) >= 2:
                st.success(f"✅ **SISTEMA DENTRO DE PARÁMETROS:** {', '.join(objetivos_cumplidos)}")
            else:
                st.warning(f"⚠️ **SISTEMA FUERA DE PARÁMETROS:** Solo {len(objetivos_cumplidos)} objetivo(s) cumplido(s)")
    
    # ============================================
    # SECCIÓN SIEMPRE VISIBLE: REGISTRO DE APUESTAS
    # ============================================
    
    st.markdown("---")
    st.subheader("🎰 REGISTRO MANUAL DE APUESTAS")
    
    # Mostrar métricas del bankroll (SIEMPRE VISIBLE)
    col_br1, col_br2, col_br3 = st.columns(3)
    
    with col_br1:
        st.metric(
            "💰 Bankroll Actual", 
            f"€{st.session_state.get('bankroll_actual', 1000):,.2f}"
        )
    
    with col_br2:
        beneficio_neto = st.session_state.get('beneficio_neto', 0.0)
        st.metric(
            "📈 Beneficio Neto", 
            f"€{beneficio_neto:,.2f}"
        )
    
    with col_br3:
        roi = (beneficio_neto / st.session_state.get('bankroll_inicial_sesion', 1000) * 100) if st.session_state.get('bankroll_inicial_sesion', 1000) > 0 else 0
        st.metric(
            "🎯 ROI Acumulado",
            f"{roi:.1f}%"
        )
    
    # ============================================
    # PIE DE PÁGINA PROFESIONAL
    # ============================================
    
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

# ============================================
# HISTORIAL
# ============================================

elif menu == "📊 Historial":
    st.title("📊 Historial de Apuestas")
    st.markdown("---")
    
    if st.session_state.get('historial_apuestas'):
        df_historial = pd.DataFrame(st.session_state.historial_apuestas)
        
        # Mostrar estadísticas
        col1, col2, col_beneficio = st.columns(3)
        
        with col1:
            total_apuestas = len(df_historial)
            st.metric("Total Apuestas", total_apuestas)
        
        with col2:
            apuestas_ganadas = len(df_historial[df_historial['resultado'] == 'ganada'])
            st.metric("Apuestas Ganadas", apuestas_ganadas)
        
        with col_beneficio:
            tasa_acierto = (apuestas_ganadas / total_apuestas * 100) if total_apuestas > 0 else 0
            st.metric("Tasa de Acierto", f"{tasa_acierto:.1f}%")
        
        # Mostrar beneficio neto en el historial también
        st.metric("💰 Beneficio Neto Acumulado", f"€{st.session_state.get('beneficio_neto', 0):,.2f}")
        
        # Mostrar tabla de historial
        st.subheader("📋 Historial Detallado")
        st.dataframe(df_historial, use_container_width=True)
        
        # Opción para exportar historial
        if st.button("📥 Exportar Historial a CSV"):
            csv = df_historial.to_csv(index=False)
            st.download_button(
                label="Descargar CSV",
                data=csv,
                file_name=f"historial_apuestas_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
    else:
        st.info("📭 No hay historial de apuestas registrado todavía.")