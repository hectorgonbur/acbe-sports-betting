import requests
import os
from dotenv import load_dotenv

# Carga las variables de entorno (.env en local, Secrets en Streamlit Cloud)
load_dotenv()

class FootballDataClient:
    def __init__(self):
        self.api_key = os.getenv("RAPIDAPI_KEY")
        self.host = os.getenv("RAPIDAPI_HOST")
        self.headers = {
            "X-RapidAPI-Key": self.api_key,
            "X-RapidAPI-Host": self.host
        }

    def get_match_stats(self, fixture_id):
        """
        Extrae estadísticas avanzadas (xG, tiros, posesión) del partido.
        """
        url = f"https://{self.host}/v3/fixtures/statistics"
        params = {"fixture": fixture_id}
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()['response']
        except Exception as e:
            return f"Error en conexión: {e}"