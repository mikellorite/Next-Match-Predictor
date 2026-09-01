"""
config.py — Constantes y configuración global del proyecto.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── API ──────────────────────────────────────────────────────────────────────
API_KEY = os.getenv("FOOTBAL_DATA_ORG_API_KEY", "")
BASE_URL = "https://api.football-data.org/v4"
COMP_CODE = "PL"
RATE_LIMIT_SLEEP = 6.2  # segundos entre peticiones (10 req/min max)

# ── Rutas ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGOS_DIR = PROJECT_ROOT / "logos"
CACHE_FILE = PROJECT_ROOT / "api_cache.json"

MODELS_DIR.mkdir(exist_ok=True)
LOGOS_DIR.mkdir(exist_ok=True)

# ── Feature columns ─────────────────────────────────────────────────────────
TEAM_FEATURE_COLS = [
    'GF_5', 'GC_5', 'PPG_5', 'FormaHT_5', 'DifGol_5', 'PPG_15pts',
    'FatigaDias', 'GF_Casa5', 'GC_Casa5', 'PPG_Casa5',
    'GF_Fuera5', 'GC_Fuera5', 'PPG_Fuera5',
]

L_COLS = [f'L_{c}' for c in TEAM_FEATURE_COLS]
V_COLS = [f'V_{c}' for c in TEAM_FEATURE_COLS]
D_COLS = [f'Dif_{c}' for c in TEAM_FEATURE_COLS if c not in ['FatigaDias']]
ELO_COLS = ['EloLocal', 'EloVisitante', 'EloDiff']
H2H_COLS = ['H2H_DifGoles', 'H2H_WinRateLocal']

FEATURE_COLS = L_COLS + V_COLS + D_COLS + ELO_COLS + H2H_COLS

# ── Train/Val/Test splits ───────────────────────────────────────────────────
TRAIN_UNTIL = '2024-08-01'
VAL_UNTIL = '2025-08-01'
