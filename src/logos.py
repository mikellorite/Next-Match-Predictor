"""
logos.py — Descarga y gestión de escudos/logos de equipos.
Usa base64 data URIs para renderizado 100% fiable en Streamlit.
"""
import os
import base64
import requests
from pathlib import Path
from src.config import LOGOS_DIR
from src.data_loader import clean_team_name

# Mapeo exacto de nombres a los archivos de GitHub
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/luukhopman/football-logos/master/logos/England%20-%20Premier%20League"

EXACT_GITHUB_FILES = {
    "Arsenal": "Arsenal%20FC.png",
    "Aston Villa": "Aston%20Villa.png",
    "Bournemouth": "AFC%20Bournemouth.png",
    "Brentford": "Brentford%20FC.png",
    "Brighton & Hove Albion": "Brighton%20%26%20Hove%20Albion.png",
    "Chelsea": "Chelsea%20FC.png",
    "Coventry City": "Coventry%20City.png",
    "Crystal Palace": "Crystal%20Palace.png",
    "Everton": "Everton%20FC.png",
    "Fulham": "Fulham%20FC.png",
    "Hull City": "Hull%20City.png",
    "Ipswich Town": "Ipswich%20Town.png",
    "Leeds United": "Leeds%20United.png",
    "Liverpool": "Liverpool%20FC.png",
    "Manchester City": "Manchester%20City.png",
    "Manchester United": "Manchester%20United.png",
    "Newcastle United": "Newcastle%20United.png",
    "Nottingham Forest": "Nottingham%20Forest.png",
    "Sunderland": "Sunderland%20AFC.png",
    "Tottenham Hotspur": "Tottenham%20Hotspur.png",
    "West Ham United": "https://crests.football-data.org/563.png",
    "Wolverhampton Wanderers": "https://crests.football-data.org/76.png",
    "Burnley": "https://crests.football-data.org/328.png",
    "Leicester City": "https://crests.football-data.org/338.png",
    "Southampton": "https://crests.football-data.org/340.png",
}

def get_logo_url(clean_name: str) -> str:
    val = EXACT_GITHUB_FILES.get(clean_name)
    if val:
        if val.startswith("http"):
            return val
        return f"{GITHUB_RAW_BASE}/{val}"
    # Fallback general
    from urllib.parse import quote
    return f"{GITHUB_RAW_BASE}/{quote(clean_name)}.png"

def download_and_cache_logo(clean_name: str) -> Path | None:
    LOGOS_DIR.mkdir(parents=True, exist_ok=True)
    local_path = LOGOS_DIR / f"{clean_name}.png"
    if local_path.exists() and local_path.stat().st_size > 300:
        return local_path
    
    url = get_logo_url(clean_name)
    try:
        resp = requests.get(url, timeout=6)
        if resp.status_code == 200 and len(resp.content) > 300:
            with open(local_path, "wb") as f:
                f.write(resp.content)
            return local_path
    except Exception:
        pass
    return None

def get_logo_base64_data_uri(team_name: str) -> str:
    """Devuelve un data URI base64 listo para usar en <img src='...'>"""
    clean = clean_team_name(team_name)
    path = download_and_cache_logo(clean)
    if path and path.exists():
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
            return f"data:image/png;base64,{encoded}"
    
    # Fallback a URL directa
    return get_logo_url(clean)
