"""
Módulo de cálculo de Elo, ingeniería de características y construcción del dataset.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.config import FEATURE_COLS, TEAM_FEATURE_COLS


def calcular_elo(df, k_inicial=40.0, k_estable=20.0, umbral_partidos=30, base=1500.0, home_advantage=50.0):
    """Calcula Elo para todos los partidos. Devuelve (df_con_elo, ratings_dict, n_partidos_dict)."""
    ratings = {}
    n_partidos = {}
    elo_local_list = []
    elo_visit_list = []
    for _, row in df.iterrows():
        home, away = row['EquipoLocal'], row['EquipoVisitante']
        r_h = ratings.get(home, base)
        r_a = ratings.get(away, base)
        elo_local_list.append(r_h)
        elo_visit_list.append(r_a)
        r_h_adj = r_h + home_advantage
        e_h = 1 / (1 + 10 ** ((r_a - r_h_adj) / 400))
        e_a = 1 - e_h
        resultado = row['Resultado']
        s_h = 1.0 if resultado == 'H' else (0.5 if resultado == 'D' else 0.0)
        s_a = 1.0 - s_h
        k_h = k_inicial if n_partidos.get(home, 0) < umbral_partidos else k_estable
        k_a = k_inicial if n_partidos.get(away, 0) < umbral_partidos else k_estable
        ratings[home] = r_h + k_h * (s_h - e_h)
        ratings[away] = r_a + k_a * (s_a - e_a)
        n_partidos[home] = n_partidos.get(home, 0) + 1
        n_partidos[away] = n_partidos.get(away, 0) + 1
    df = df.copy()
    df['EloLocal'] = elo_local_list
    df['EloVisitante'] = elo_visit_list
    df['EloDiff'] = df['EloLocal'] - df['EloVisitante']
    return df, ratings, n_partidos


def _rolling_safe(series, window, agg='mean'):
    """Aplica agregaciones móviles con un desplazamiento de 1 para evitar fuga de datos (data leakage)."""
    shifted = series.shift(1)
    if agg == 'mean':
        return shifted.rolling(window, min_periods=1).mean()
    elif agg == 'sum':
        return shifted.rolling(window, min_periods=1).sum()
    raise ValueError(f"agg '{agg}' not supported")


def build_team_features(df_hist):
    """Construye características rolling por equipo a partir del histórico."""
    rows = []
    for idx, row in df_hist.iterrows():
        for role in ['Local', 'Visitante']:
            es_local = (role == 'Local')
            equipo = row['EquipoLocal'] if es_local else row['EquipoVisitante']
            rival = row['EquipoVisitante'] if es_local else row['EquipoLocal']
            gf = row['GL'] if es_local else row['GV']
            gc = row['GV'] if es_local else row['GL']
            hgf = row['HGL'] if es_local else row['HGV']
            hgc = row['HGV'] if es_local else row['HGL']
            if es_local:
                resultado_prop = row['Resultado']
            else:
                resultado_prop = 'A' if row['Resultado'] == 'H' else ('H' if row['Resultado'] == 'A' else 'D')
            puntos = 3 if resultado_prop == 'H' else (1 if resultado_prop == 'D' else 0)
            rows.append({
                'match_idx': idx, 'Fecha': row['Fecha'], 'Temporada': row['Temporada'],
                'Equipo': equipo, 'Rival': rival, 'EsLocal': int(es_local),
                'GF': gf, 'GC': gc, 'HGF': hgf, 'HGC': hgc,
                'Puntos': puntos, 'Resultado': resultado_prop,
            })
    df_long = pd.DataFrame(rows)
    df_long.sort_values(['Equipo', 'Fecha'], inplace=True)
    df_long.reset_index(drop=True, inplace=True)
    grp = df_long.groupby('Equipo', group_keys=False)
    df_long['GF_5'] = grp['GF'].transform(lambda s: _rolling_safe(s, 5))
    df_long['GC_5'] = grp['GC'].transform(lambda s: _rolling_safe(s, 5))
    df_long['PPG_5'] = grp['Puntos'].transform(lambda s: _rolling_safe(s, 5))
    df_long['DifHT_bruto'] = df_long['HGF'].fillna(0) - df_long['HGC'].fillna(0)
    df_long['FormaHT_5'] = grp['DifHT_bruto'].transform(lambda s: _rolling_safe(s, 5))
    df_long['DifGol_bruto'] = df_long['GF'] - df_long['GC']
    df_long['DifGol_5'] = grp['DifGol_bruto'].transform(lambda s: _rolling_safe(s, 5))
    df_long['PPG_15pts'] = grp['Puntos'].transform(lambda s: _rolling_safe(s, 5, 'sum'))
    df_long['FatigaDias'] = grp['Fecha'].transform(lambda s: s.diff().dt.days.fillna(7))
    df_long['FatigaDias'] = grp['FatigaDias'].transform(lambda s: s.shift(1).fillna(7))
    for ctx_eslocal, sufijo in [(1, 'Casa'), (0, 'Fuera')]:
        df_ctx = df_long[df_long['EsLocal'] == ctx_eslocal].copy()
        df_ctx.sort_values(['Equipo', 'Fecha'], inplace=True)
        grp_ctx = df_ctx.groupby('Equipo', group_keys=False)
        df_ctx[f'GF_{sufijo}5'] = grp_ctx['GF'].transform(lambda s: _rolling_safe(s, 5))
        df_ctx[f'GC_{sufijo}5'] = grp_ctx['GC'].transform(lambda s: _rolling_safe(s, 5))
        df_ctx[f'PPG_{sufijo}5'] = grp_ctx['Puntos'].transform(lambda s: _rolling_safe(s, 5))
        df_long = df_long.merge(
            df_ctx[['match_idx', 'Equipo', f'GF_{sufijo}5', f'GC_{sufijo}5', f'PPG_{sufijo}5']],
            on=['match_idx', 'Equipo'], how='left'
        )
    return df_long


def build_h2h_features(df_hist, max_years=3, n_matches=3):
    """Calcula características Head-to-Head (H2H) para cada partido."""
    h2h_records = []
    for idx, row in df_hist.iterrows():
        home, away, fecha = row['EquipoLocal'], row['EquipoVisitante'], row['Fecha']
        cutoff = fecha - pd.DateOffset(years=max_years)
        h2h_df = df_hist[
            (df_hist['Fecha'] < fecha) & (df_hist['Fecha'] >= cutoff) &
            (((df_hist['EquipoLocal'] == home) & (df_hist['EquipoVisitante'] == away)) |
             ((df_hist['EquipoLocal'] == away) & (df_hist['EquipoVisitante'] == home)))
        ].tail(n_matches)
        if len(h2h_df) == 0:
            h2h_records.append({'match_idx': idx, 'H2H_DifGoles': 0.0, 'H2H_WinRateLocal': 0.5})
            continue
        dif_list, win_local_list = [], []
        for _, h_row in h2h_df.iterrows():
            if h_row['EquipoLocal'] == home:
                dif_list.append(h_row['GL'] - h_row['GV'])
                win_local_list.append(1 if h_row['Resultado'] == 'H' else 0)
            else:
                dif_list.append(h_row['GV'] - h_row['GL'])
                win_local_list.append(1 if h_row['Resultado'] == 'A' else 0)
        h2h_records.append({
            'match_idx': idx, 'H2H_DifGoles': float(np.mean(dif_list)),
            'H2H_WinRateLocal': float(np.mean(win_local_list)), 'H2H_N': len(h2h_df),
        })
    return pd.DataFrame(h2h_records)


def build_model_dataset(df_hist, df_long, df_h2h):
    """Combina métricas de equipo, Elo y H2H para construir el dataset de entrenamiento/inferencia."""
    df_local = df_long[df_long['EsLocal'] == 1][['match_idx', 'Equipo'] + TEAM_FEATURE_COLS].copy()
    df_local.columns = ['match_idx', 'Equipo'] + [f'L_{c}' for c in TEAM_FEATURE_COLS]
    df_visit = df_long[df_long['EsLocal'] == 0][['match_idx', 'Equipo'] + TEAM_FEATURE_COLS].copy()
    df_visit.columns = ['match_idx', 'Equipo'] + [f'V_{c}' for c in TEAM_FEATURE_COLS]
    df_base = df_hist[['Fecha', 'Temporada', 'EquipoLocal', 'EquipoVisitante', 'GL', 'GV', 'Resultado', 'EloLocal', 'EloVisitante', 'EloDiff']].copy()
    df_base['match_idx'] = df_base.index
    df_features = (df_base
        .merge(df_local.drop(columns='Equipo'), on='match_idx', how='left')
        .merge(df_visit.drop(columns='Equipo'), on='match_idx', how='left')
        .merge(df_h2h, on='match_idx', how='left'))
    for c in TEAM_FEATURE_COLS:
        if c not in ['FatigaDias']:
            df_features[f'Dif_{c}'] = df_features[f'L_{c}'] - df_features[f'V_{c}']
    le = LabelEncoder()
    df_features['Target'] = le.fit_transform(df_features['Resultado'])
    class_names = le.classes_
    df_model = df_features.dropna(subset=['L_PPG_5', 'V_PPG_5', 'EloLocal']).copy()
    for col in FEATURE_COLS:
        if col in df_model.columns:
            df_model[col] = df_model[col].fillna(df_model[col].median())
    return df_model, le, class_names
