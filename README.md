# 🏛️ ACBE-Sports-Betting v2.0
**Algoritmo de Convergencia Bayesiana Estructural + Kelly Fraccional.**

## 🔬 Metodología
Este sistema utiliza un enfoque de **Arbitraje Estadístico** basado en:
- **Fase 0**: Filtros de mercado (Overround < 7%, Entropía < 0.72).
- **Fase 2**: Modelado de Poisson y Simulación Monte Carlo (10k iteraciones).
- **Fase 4**: Gestión de riesgo mediante Kelly Bayesiano ajustado por Entropía de Shannon.

## 🛠️ Configuración
1. Clonar repo: `git clone ...`
2. Instalar dependencias: `pip install -r requirements.txt`
3. Configurar `.env` con tu llave de API-Football.
