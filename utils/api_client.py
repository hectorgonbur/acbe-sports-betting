def get_todays_fixtures(self, league_id):
        """
        Detecta la temporada y recupera partidos sin saturar la memoria.
        """
        import datetime
        now = datetime.datetime.now()
        today = now.strftime('%Y-%m-%d')
        
        # Lógica de Temporada ACBE+
        # Ligas Europeas (Finalizan en 2026 pero su ID de registro es 2025)
        ligas_euro = [2, 3, 39, 140, 135, 78, 61]
        season = 2025 if league_id in ligas_euro else 2026
            
        url = f"https://{self.host}/v3/fixtures"
        params = {"league": league_id, "season": season, "date": today}
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json().get('response', [])
        except Exception as e:
            print(f"Error consultando fixtures: {e}")
            return []