"""
data_loader.py — Carga de datos históricos desde los Excels en data/.
"""
import re
import numpy as np
import pandas as pd
from pathlib import Path


def clean_team_name(name: str) -> str:
    """Normaliza nombres de equipo quitando FC y AFC."""
    n = re.sub(r'\bFC\b', '', str(name))
    n = re.sub(r'\bAFC\b', '', n)
    return ' '.join(n.split())


def load_historical_data(data_dir: Path) -> pd.DataFrame:
    """
    Lee todos los Excel de las subcarpetas de data/ y los une en un único DF.
    Normaliza nombres de equipos y calcula el resultado (H/D/A).
    """
    dfs = []
    for xlsx_path in sorted(data_dir.glob("**/*.xlsx")):
        season = xlsx_path.parent.name
        df = pd.read_excel(xlsx_path)
        df['Temporada'] = season
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No se encontraron archivos .xlsx en {data_dir}")

    df_all = pd.concat(dfs, ignore_index=True)

    # Renombrar columnas
    df_all.rename(columns={
        'GolesMarcadosLocal': 'GL',
        'GolesMarcadosVisitante': 'GV',
        'GolesMarcadosDescansoLocal': 'HGL',
        'GolesMarcadosDescansoVisitante': 'HGV',
    }, inplace=True)

    # Normalizar nombres
    df_all['EquipoLocal'] = df_all['EquipoLocal'].apply(clean_team_name)
    df_all['EquipoVisitante'] = df_all['EquipoVisitante'].apply(clean_team_name)

    # Parsear fechas y ordenar
    df_all['Fecha'] = pd.to_datetime(df_all['Fecha'])
    df_all.sort_values('Fecha', inplace=True)
    df_all.reset_index(drop=True, inplace=True)

    # Resultado del partido (perspectiva del equipo local)
    df_all['Resultado'] = np.where(
        df_all['GL'] > df_all['GV'], 'H',
        np.where(df_all['GL'] < df_all['GV'], 'A', 'D')
    )

    return df_all
