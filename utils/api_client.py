import requests
import streamlit as st
from dotenv import load_dotenv
import os

class FootballDataClient:
    def __init__(self):
        """Inicializa el cliente usando Secrets de Streamlit o archivo .env local."""
        load_dotenv()
        # Prioriza Streamlit Secrets (Nube) sobre os.getenv (Local)
        self.api_key = st.secrets.get("RAPIDAPI_KEY") or os.getenv("RAPIDAPI_KEY")
        self.host = st.secrets.get("RAPIDAPI_HOST") or os.getenv("RAPIDAPI_HOST")
        
        self.headers = {
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": self.host
        }

    def get_fixtures_by_date(self, league_id, date_obj):
        """
        Busca partidos por fecha para automatizar la selección en app.py.
        """
        fecha_str = date_obj.strftime('%Y-%m-%d')
        
        # Lógica de Temporadas ACBE+: 2025 para Europa, 2026 para Latam/Asia
        ligas_euro = [2, 3, 39, 140, 135, 78, 61]
        season = 2025 if league_id in ligas_euro else 2026
            
        url = f"https://{self.host}/v3/fixtures"
        params = {"league": league_id, "season": season, "date": fecha_str}
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json().get('response', [])
        except Exception as e:
            st.error(f"Error al consultar partidos: {e}")
            return []

    def get_match_stats(self, fixture_id):
        """
        Data Mining (Fase 1): Recupera estadísticas detalladas para los Lambdas.
        """
        url = f"https://{self.host}/v3/fixtures/statistics"
        params = {"fixture": fixture_id}
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json().get('response', [])
            
            if not data:
                raise ValueError("No hay estadísticas disponibles para este ID.")
            return data
        except Exception as e:
            raise Exception(f"Falla en Data Mining: {e}")