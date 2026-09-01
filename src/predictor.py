"""
predictor.py — Lógica de cálculo de features para el partido a predecir,
cálculo de probabilidades y extracción de métricas para la UI.
"""
import numpy as np
import pandas as pd
from src.config import TEAM_FEATURE_COLS, FEATURE_COLS
from src.data_loader import clean_team_name
from src.logos import get_logo_base64_data_uri


def get_team_snapshot(
    team_name_hist: str,
    prediction_date: pd.Timestamp,
    df_hist: pd.DataFrame,
    df_long: pd.DataFrame,
    elo_ratings: dict,
    context: str = 'any'
) -> dict:
    """
    Obtiene el snapshot de features de un equipo antes del próximo partido.
    FatigaDias siempre se calcula desde el último partido (context='any').
    """
    pred_date_naive = (
        prediction_date.tz_localize(None)
        if prediction_date.tzinfo is None
        else prediction_date.tz_convert('UTC').tz_localize(None)
    )

    mask_any = (df_long['Equipo'] == team_name_hist) & (df_long['Fecha'] < pred_date_naive)
    team_any = df_long[mask_any].sort_values('Fecha')

    if len(team_any) == 0:
        snap = {c: 0.0 for c in TEAM_FEATURE_COLS}
        snap['EloActual'] = elo_ratings.get(team_name_hist, 1500.0)
        snap['FatigaDias'] = 7
        return snap

    ultimo_partido_any = team_any['Fecha'].iloc[-1]
    fatiga_real = max(0, (pred_date_naive - ultimo_partido_any).days)

    if context == 'any':
        team_df = team_any
    else:
        mask_ctx = mask_any.copy()
        if context == 'home':
            mask_ctx &= (df_long['EsLocal'] == 1)
        elif context == 'away':
            mask_ctx &= (df_long['EsLocal'] == 0)
        team_df = df_long[mask_ctx].sort_values('Fecha')
        if len(team_df) == 0:
            team_df = team_any

    last = team_df.iloc[-1]

    snap = {}
    for col in TEAM_FEATURE_COLS:
        snap[col] = last.get(col, np.nan)

    snap['FatigaDias'] = fatiga_real
    snap['EloActual'] = elo_ratings.get(team_name_hist, 1500.0)

    return snap


def get_team_form_detailed(team_name: str, df_long: pd.DataFrame, df_hist: pd.DataFrame, pred_date: pd.Timestamp, n: int = 5):
    """
    Obtiene la racha de los últimos N partidos antes de pred_date con metadatos completos
    para tooltips en hover (fecha, resultado, goles, rival, escudos).
    Devuelve (form_str, form_score, list_of_match_details).
    """
    pred_date_naive = (
        pred_date.tz_localize(None)
        if pred_date.tzinfo is None
        else pred_date.tz_convert('UTC').tz_localize(None)
    )
    mask = (df_long['Equipo'] == team_name) & (df_long['Fecha'] < pred_date_naive)
    team_df = df_long[mask].sort_values('Fecha').tail(n)

    if team_df.empty:
        return "-----", 0.5, []

    form_chars = []
    matches_info = []
    total_pts = 0

    for _, row in team_df.iterrows():
        res = row['Resultado'] # 'H'=victoria, 'D'=empate, 'A'=derrota
        m_idx = row['match_idx']
        
        # Obtener datos del partido original
        if m_idx in df_hist.index:
            h_row = df_hist.loc[m_idx]
            home_team = h_row['EquipoLocal']
            away_team = h_row['EquipoVisitante']
            gl = int(h_row['GL'])
            gv = int(h_row['GV'])
            date_dt = pd.to_datetime(h_row['Fecha'])
        else:
            home_team = team_name if row['EsLocal'] == 1 else row['Rival']
            away_team = row['Rival'] if row['EsLocal'] == 1 else team_name
            gl = int(row['GF']) if row['EsLocal'] == 1 else int(row['GC'])
            gv = int(row['GC']) if row['EsLocal'] == 1 else int(row['GF'])
            date_dt = pd.to_datetime(row['Fecha'])

        if res == 'H':
            char = 'W'
            desc = "Victoria"
            total_pts += 3
        elif res == 'D':
            char = 'D'
            desc = "Empate"
            total_pts += 1
        else:
            char = 'L'
            desc = "Derrota"

        form_chars.append(char)

        matches_info.append({
            'char': char,
            'desc': desc,
            'date_str': date_dt.strftime("%d/%m/%Y"),
            'home_team': home_team,
            'away_team': away_team,
            'home_logo': get_logo_base64_data_uri(home_team),
            'away_logo': get_logo_base64_data_uri(away_team),
            'score_str': f"{gl} - {gv}",
            'is_home': (row['EsLocal'] == 1)
        })

    form_str = "".join(form_chars)
    max_pts = len(team_df) * 3
    form_score = total_pts / max_pts if max_pts > 0 else 0.5

    return form_str, form_score, matches_info


def build_prediction_vector(
    home_name: str,
    away_name: str,
    pred_date: pd.Timestamp,
    df_hist: pd.DataFrame,
    df_long: pd.DataFrame,
    elo_ratings: dict,
    scaler
):
    """
    Construye el vector X_pred_sc listo para pasar al modelo predict_proba.
    """
    pred_date_naive = (
        pred_date.tz_localize(None)
        if pred_date.tzinfo is None
        else pred_date.tz_convert('UTC').tz_localize(None)
    )

    home_clean = clean_team_name(home_name)
    away_clean = clean_team_name(away_name)

    snap_home_gen = get_team_snapshot(home_clean, pred_date_naive, df_hist, df_long, elo_ratings, 'any')
    snap_home_ctx = get_team_snapshot(home_clean, pred_date_naive, df_hist, df_long, elo_ratings, 'home')
    snap_away_gen = get_team_snapshot(away_clean, pred_date_naive, df_hist, df_long, elo_ratings, 'any')
    snap_away_ctx = get_team_snapshot(away_clean, pred_date_naive, df_hist, df_long, elo_ratings, 'away')

    # H2H últimos 3 años (máximo 3 partidos)
    cutoff_h2h = pred_date_naive - pd.DateOffset(years=3)
    h2h_df_pred = df_hist[
        (df_hist['Fecha'] < pred_date_naive) &
        (df_hist['Fecha'] >= cutoff_h2h) &
        (
            ((df_hist['EquipoLocal'] == home_clean) & (df_hist['EquipoVisitante'] == away_clean)) |
            ((df_hist['EquipoLocal'] == away_clean) & (df_hist['EquipoVisitante'] == home_clean))
        )
    ].tail(3)

    if len(h2h_df_pred) > 0:
        h2h_difs = []
        h2h_wins = []
        for _, r in h2h_df_pred.iterrows():
            if r['EquipoLocal'] == home_clean:
                h2h_difs.append(r['GL'] - r['GV'])
                h2h_wins.append(1 if r['Resultado'] == 'H' else 0)
            else:
                h2h_difs.append(r['GV'] - r['GL'])
                h2h_wins.append(1 if r['Resultado'] == 'A' else 0)
        h2h_dif_goles = float(np.mean(h2h_difs))
        h2h_win_rate_loc = float(np.mean(h2h_wins))
    else:
        h2h_dif_goles = 0.0
        h2h_win_rate_loc = 0.5

    elo_home = elo_ratings.get(home_clean, 1500.0)
    elo_away = elo_ratings.get(away_clean, 1500.0)
    elo_diff = elo_home - elo_away

    vec = {}
    for c in TEAM_FEATURE_COLS:
        if 'Casa' in c:
            val = snap_home_ctx.get(c, np.nan)
            if pd.isna(val):
                val = snap_home_gen.get(c, np.nan)
            vec[f'L_{c}'] = val
        elif 'Fuera' in c:
            vec[f'L_{c}'] = snap_home_gen.get(c, np.nan)
        else:
            vec[f'L_{c}'] = snap_home_gen.get(c, np.nan)

        if 'Fuera' in c:
            val = snap_away_ctx.get(c, np.nan)
            if pd.isna(val):
                val = snap_away_gen.get(c, np.nan)
            vec[f'V_{c}'] = val
        elif 'Casa' in c:
            vec[f'V_{c}'] = snap_away_gen.get(c, np.nan)
        else:
            vec[f'V_{c}'] = snap_away_gen.get(c, np.nan)

    for c in TEAM_FEATURE_COLS:
        if c not in ['FatigaDias']:
            l_val = vec.get(f'L_{c}', 0)
            v_val = vec.get(f'V_{c}', 0)
            vec[f'Dif_{c}'] = l_val - v_val if not (pd.isna(l_val) or pd.isna(v_val)) else 0.0

    vec['EloLocal'] = elo_home
    vec['EloVisitante'] = elo_away
    vec['EloDiff'] = elo_diff
    vec['H2H_DifGoles'] = h2h_dif_goles
    vec['H2H_WinRateLocal'] = h2h_win_rate_loc

    X_pred_raw = pd.DataFrame([vec])[FEATURE_COLS]
    X_pred_raw = X_pred_raw.fillna(0.0)
    X_pred_sc = scaler.transform(X_pred_raw.values)

    snapshots = (snap_home_gen, snap_home_ctx, snap_away_gen, snap_away_ctx)
    h2h_info = (h2h_dif_goles, h2h_win_rate_loc, h2h_df_pred)

    return X_pred_sc, vec, snapshots, h2h_info


def predict_match(model, X_pred_sc, class_names) -> dict:
    """Calcula las probabilidades de victoria, empate y derrota."""
    probs = model.predict_proba(X_pred_sc)[0]
    class_list = list(class_names)

    prob_away = float(probs[class_list.index('A')])
    prob_draw = float(probs[class_list.index('D')])
    prob_home = float(probs[class_list.index('H')])

    return {
        'prob_home': prob_home,
        'prob_draw': prob_draw,
        'prob_away': prob_away
    }
