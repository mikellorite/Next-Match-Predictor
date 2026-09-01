"""
api.py — Interfaz Streamlit para Next Match Predictor (Matchday IQ).
Diseño optimizado sin cortes en la parte superior y con etiquetas sencillas e intuitivas.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from src.config import DATA_DIR, API_KEY
from src.data_loader import load_historical_data, clean_team_name
from src.features import calcular_elo, build_team_features, build_h2h_features, build_model_dataset
from src.model import load_model, model_exists, train_and_save, get_test_predictions
from src.api_client import ApiClient
from src.predictor import build_prediction_vector, predict_match, get_team_form_detailed
from src.logos import get_logo_base64_data_uri

# Configuración de página
st.set_page_config(
    page_title="MATCHDAY IQ | Premier League Match Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Estilos CSS globales (Corregidos para evitar cortes superiores) ──────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Espaciado superior ajustado para evitar que se corte con el header nativo de Streamlit */
.block-container {
    padding-top: 2.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 1250px;
}

/* Alineación vertical para la fila del selector superior */
[data-testid="column"] {
    display: flex;
    align-items: center;
}

/* Ajustes de Selectbox e Inputs superiores */
[data-testid="stSelectbox"] {
    width: 100%;
    margin-top: 0px;
}

/* Header Dark Bar */
.header-bar {
    background-color: #0f172a;
    color: #ffffff;
    padding: 14px 24px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 10px;
    margin-bottom: 18px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

.brand-title {
    font-size: 20px;
    font-weight: 900;
    letter-spacing: -0.5px;
    color: #ffffff;
}

.brand-badge {
    background-color: #2563eb;
    color: white;
    font-size: 10px;
    font-weight: 800;
    padding: 2px 8px;
    border-radius: 4px;
    margin-left: 8px;
    vertical-align: middle;
}

.fixture-pill {
    background-color: #1e293b;
    padding: 6px 18px;
    border-radius: 30px;
    border: 1px solid #334155;
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 14px;
    font-weight: 600;
}

.fixture-crest {
    width: 26px;
    height: 26px;
    object-fit: contain;
    vertical-align: middle;
}

.fixture-time {
    color: #94a3b8;
    font-size: 12px;
    font-weight: 500;
    margin-left: 6px;
}

.team-highlight {
    background-color: #3b82f6;
    color: #ffffff;
    font-size: 10px;
    font-weight: 800;
    padding: 2px 6px;
    border-radius: 4px;
    margin-left: 4px;
}

/* Match Context Info Banner */
.match-context-banner {
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #2563eb;
    border-radius: 8px;
    padding: 10px 16px;
    margin-bottom: 20px;
    font-size: 13px;
    color: #334155;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

/* Outcome Cards */
.outcome-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 22px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    margin-bottom: 20px;
}

.outcome-tag {
    font-size: 11px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    padding: 4px 9px;
    border-radius: 4px;
    display: inline-block;
    margin-bottom: 10px;
}

.tag-green { background-color: #dcfce7; color: #15803d; }
.tag-blue { background-color: #dbeafe; color: #1d4ed8; }
.tag-red { background-color: #fee2e2; color: #b91c1c; }

.outcome-title {
    font-size: 22px;
    font-weight: 800;
    margin-bottom: 2px;
}
.title-green { color: #15803d; }
.title-blue { color: #1d4ed8; }
.title-red { color: #b91c1c; }

.prob-number {
    font-size: 54px;
    font-weight: 900;
    line-height: 1;
    margin-bottom: 4px;
}

.prob-subtext {
    font-size: 10px;
    color: #94a3b8;
    text-transform: uppercase;
    font-weight: 700;
    letter-spacing: 0.5px;
    margin-bottom: 16px;
}

/* Stat Box */
.stat-box {
    margin-top: 14px;
    padding-top: 10px;
    border-top: 1px solid #f1f5f9;
}

.stat-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 4px;
}

.stat-label {
    font-size: 11px;
    font-weight: 700;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.3px;
}

.stat-value {
    font-size: 14px;
    font-weight: 800;
    color: #0f172a;
}

.custom-progress-bg {
    background-color: #f1f5f9;
    height: 7px;
    border-radius: 6px;
    overflow: hidden;
    margin-top: 4px;
}

.custom-progress-fill {
    height: 100%;
    border-radius: 6px;
}

.fill-green { background-color: #16a34a; }
.fill-blue { background-color: #2563eb; }
.fill-red { background-color: #dc2626; }

/* ── TOOLTIPS EN HOVER ── */
.form-badge-container {
    display: flex;
    gap: 6px;
    margin-top: 5px;
    margin-bottom: 6px;
}

.tooltip-wrap {
    position: relative;
    display: inline-block;
}

.form-pill {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    border-radius: 5px;
    font-size: 11px;
    font-weight: 800;
    color: white;
    cursor: pointer;
    transition: transform 0.15s ease;
}

.form-pill:hover {
    transform: scale(1.15);
}

.pill-W { background-color: #16a34a; }
.pill-D { background-color: #64748b; }
.pill-L { background-color: #dc2626; }

.tooltip-card {
    visibility: hidden;
    opacity: 0;
    width: 220px;
    background-color: #0f172a;
    color: #ffffff;
    border-radius: 10px;
    padding: 10px 12px;
    position: absolute;
    z-index: 99999;
    bottom: 135%;
    left: 50%;
    transform: translateX(-50%);
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
    border: 1px solid #334155;
    transition: opacity 0.2s cubic-bezier(0.16, 1, 0.3, 1), transform 0.2s cubic-bezier(0.16, 1, 0.3, 1), visibility 0.2s;
    pointer-events: none;
    text-align: center;
}

.tooltip-card::after {
    content: "";
    position: absolute;
    top: 100%;
    left: 50%;
    margin-left: -6px;
    border-width: 6px;
    border-style: solid;
    border-color: #0f172a transparent transparent transparent;
}

.tooltip-wrap:hover .tooltip-card {
    visibility: visible;
    opacity: 1;
    transform: translateX(-50%) translateY(-6px);
}

.tooltip-date {
    font-size: 10px;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 5px;
}

.tooltip-fixture {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    font-size: 13px;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 5px;
}

.tooltip-crest {
    width: 18px;
    height: 18px;
    object-fit: contain;
    vertical-align: middle;
}

.tooltip-badge {
    display: inline-block;
    font-size: 10px;
    font-weight: 800;
    text-transform: uppercase;
    padding: 2px 7px;
    border-radius: 4px;
    letter-spacing: 0.5px;
}
.tt-W { background-color: #16a34a; color: #ffffff; }
.tt-D { background-color: #475569; color: #ffffff; }
.tt-L { background-color: #dc2626; color: #ffffff; }

/* Bottom Panel Cards */
.panel-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}

.panel-title {
    font-size: 12px;
    font-weight: 800;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 14px;
}

.prediction-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 14px;
    border-radius: 8px;
    background-color: #f8fafc;
    margin-bottom: 8px;
    border: 1px solid #f1f5f9;
}

.prediction-match {
    font-size: 13px;
    font-weight: 700;
    color: #0f172a;
}

.prediction-details {
    font-size: 11px;
    color: #64748b;
}

.badge-correct {
    color: #15803d;
    font-size: 11px;
    font-weight: 700;
    background-color: #dcfce7;
    padding: 3px 8px;
    border-radius: 4px;
}

.badge-wrong {
    color: #b91c1c;
    font-size: 11px;
    font-weight: 700;
    background-color: #fee2e2;
    padding: 3px 8px;
    border-radius: 4px;
}

.metric-mini {
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 14px;
    text-align: center;
}
.metric-mini-val {
    font-size: 26px;
    font-weight: 900;
}
.metric-mini-lbl {
    font-size: 11px;
    font-weight: 700;
    color: #64748b;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)


# ── CARGA Y CACHÉ DE DATOS ──────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_all_pipeline_data():
    df_hist = load_historical_data(DATA_DIR)
    df_hist, elo_ratings, _ = calcular_elo(df_hist)
    df_long = build_team_features(df_hist)
    df_h2h = build_h2h_features(df_hist)
    df_model, le, class_names = build_model_dataset(df_hist, df_long, df_h2h)
    return df_hist, elo_ratings, df_long, df_h2h, df_model, class_names

@st.cache_resource(show_spinner=False)
def get_or_train_model(_df_model, _class_names):
    if model_exists():
        model, scaler, class_names, best_name = load_model()
    else:
        model, scaler, class_names, best_name = train_and_save(_df_model, _class_names)
    return model, scaler, class_names, best_name

with st.spinner("Cargando pipeline..."):
    df_hist, elo_ratings, df_long, df_h2h, df_model, class_names = load_all_pipeline_data()
    model, scaler, class_names, model_name = get_or_train_model(df_model, class_names)
    api = ApiClient(api_key=API_KEY)

# Obtener los 20 equipos oficiales de la Premier League directamente de la API
@st.cache_data(ttl=3600)
def get_pl_teams_list():
    df_teams = api.get_teams()
    if not df_teams.empty:
        return sorted([clean_team_name(n) for n in df_teams['name'].dropna().tolist()])
    return [
        'Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton & Hove Albion',
        'Chelsea', 'Coventry City', 'Crystal Palace', 'Everton', 'Fulham',
        'Hull City', 'Ipswich Town', 'Leeds United', 'Liverpool', 'Manchester City',
        'Manchester United', 'Newcastle United', 'Nottingham Forest', 'Sunderland', 'Tottenham Hotspur'
    ]

pl_teams = get_pl_teams_list()

# ── TOP SELECTOR & BOTÓN DE REENTRENAMIENTO (Alineación perfecta) ─────────────
col_brand, col_sel, col_retrain = st.columns([1.2, 2.0, 0.8])

with col_brand:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:8px; padding-top:4px;">
        <span style="font-size:24px; font-weight:900; color:#0f172a;">⚽ MATCHDAY <span style="color:#2563eb;">IQ</span></span>
        <span class="brand-badge">PREMIER LEAGUE</span>
    </div>
    """, unsafe_allow_html=True)

with col_sel:
    selected_team = st.selectbox(
        "🔍 Selecciona un equipo:",
        options=pl_teams,
        index=pl_teams.index("Arsenal") if "Arsenal" in pl_teams else 0,
        label_visibility="collapsed"
    )

with col_retrain:
    retrain_clicked = st.button("🔄 Reentrenar Modelo", use_container_width=True, help="Vuelve a leer los Excels en data/ y actualiza el modelo con datos recientes.")

# Contenedor dedicado de estado para no desplazar ni cortar la barra superior
if retrain_clicked:
    with st.status("🚀 Reentrenando modelo con datos actualizados...", expanded=True) as status:
        st.write("1. Leyendo archivos Excel en `data/`...")
        df_hist_new = load_historical_data(DATA_DIR)
        st.write(f"   Partidos encontrados: **{len(df_hist_new):,}**")
        
        st.write("2. Recalculando sistema Elo y features rolling...")
        df_hist_new, elo_ratings_new, _ = calcular_elo(df_hist_new)
        df_long_new = build_team_features(df_hist_new)
        df_h2h_new = build_h2h_features(df_hist_new)
        df_model_new, le_new, class_names_new = build_model_dataset(df_hist_new, df_long_new, df_h2h_new)
        
        st.write("3. Ejecutando GridSearchCV y calibración...")
        model_new, scaler_new, class_names_new, best_name_new = train_and_save(df_model_new, class_names_new)
        
        st.cache_resource.clear()
        st.cache_data.clear()
        status.update(label=f"✅ ¡Modelo ({best_name_new}) reentrenado y guardado con éxito!", state="complete", expanded=False)
        st.success(f"Modelo actualizado con {len(df_hist_new):,} partidos históricos.")
        st.rerun()

# ── API: PRÓXIMO PARTIDO DEL EQUIPO SELECCIONADO ─────────────────────────────
team_id, team_name_api = api.find_team_id(selected_team)
next_match = api.get_next_match(team_id)

if next_match is not None:
    home_name_api = next_match['home_name']
    away_name_api = next_match['away_name']
    match_date = pd.to_datetime(next_match['utcDate'])
    match_time_str = match_date.strftime("%a %d/%m %H:%M UTC")
    is_selected_home = (next_match['home_id'] == team_id)
else:
    home_name_api = selected_team
    away_name_api = "Aston Villa" if selected_team != "Aston Villa" else "Arsenal"
    match_date = pd.Timestamp.now() + pd.Timedelta(days=5)
    match_time_str = match_date.strftime("%a %d/%m %H:%M UTC")
    is_selected_home = True

home_clean = clean_team_name(home_name_api)
away_clean = clean_team_name(away_name_api)

# Logos en base64
logo_home = get_logo_base64_data_uri(home_clean)
logo_away = get_logo_base64_data_uri(away_clean)

# Rol del equipo seleccionado
if is_selected_home:
    badge_home_html = '<span class="team-highlight">TÚ · LOCAL</span>'
    badge_away_html = '<span style="font-size:10px; color:#94a3b8; font-weight:700;">RIVAL</span>'
    role_text = f"<b>{selected_team}</b> juega como <b>LOCAL 🏠</b> en su estadio contra <b>{away_clean}</b>."
else:
    badge_home_html = '<span style="font-size:10px; color:#94a3b8; font-weight:700;">RIVAL</span>'
    badge_away_html = '<span class="team-highlight">TÚ · VISITANTE</span>'
    role_text = f"<b>{selected_team}</b> juega como <b>VISITANTE ✈️</b> en el estadio de <b>{home_clean}</b>."

# ── CABECERA NEGRA ESTILO MATCHDAY IQ ─────────────────────────────────────────
header_html = f"""<div class="header-bar">
<div class="brand-title">⚽ MATCHDAY IQ</div>
<div class="fixture-pill">
<img class="fixture-crest" src="{logo_home}">
<span style="color:#ffffff;">{home_clean}</span>
{badge_home_html}
<span style="color:#64748b; font-weight:400; margin:0 4px;">vs</span>
<img class="fixture-crest" src="{logo_away}">
<span style="color:#ffffff;">{away_clean}</span>
{badge_away_html}
<span class="fixture-time">| {match_time_str}</span>
</div>
<div style="font-size:11px; color:#94a3b8; font-weight:600;">
MODELO: <span style="color:#38bdf8;">{model_name.upper()}</span>
</div>
</div>"""
st.markdown(header_html, unsafe_allow_html=True)

# Banner explicativo de contexto
st.markdown(f"""
<div class="match-context-banner">
    <div>📌 <b>Próximo partido oficial:</b> {role_text}</div>
    <div style="font-size:11px; color:#64748b;">Fecha: {match_time_str}</div>
</div>
""", unsafe_allow_html=True)


# ── CÁLCULO DE PREDICCIÓN & FEATURES ─────────────────────────────────────────
X_pred_sc, vec, (snap_h_gen, snap_h_ctx, snap_a_gen, snap_a_ctx), (h2h_dif, h2h_winrate, h2h_df) = build_prediction_vector(
    home_name=home_clean,
    away_name=away_clean,
    pred_date=match_date,
    df_hist=df_hist,
    df_long=df_long,
    elo_ratings=elo_ratings,
    scaler=scaler
)

preds = predict_match(model, X_pred_sc, class_names)
prob_home = preds['prob_home']
prob_draw = preds['prob_draw']
prob_away = preds['prob_away']

# Formas detalladas con metadatos de partidos (últimos 5 partidos)
form_h_str, form_h_score, form_h_details = get_team_form_detailed(home_clean, df_long, df_hist, match_date, n=5)
form_a_str, form_a_score, form_a_details = get_team_form_detailed(away_clean, df_long, df_hist, match_date, n=5)

# Métricas para Local
fatiga_h = int(snap_h_gen.get('FatigaDias', 7))
gf_h = float(snap_h_gen.get('GF_5', 1.5))
gc_h = float(snap_h_gen.get('GC_5', 1.2))
dif_gol_h = float(snap_h_gen.get('DifGol_5', 0.0))
ppg_casa_h = float(snap_h_ctx.get('PPG_Casa5', snap_h_gen.get('PPG_5', 1.5)))

# Métricas para Visitante
fatiga_a = int(snap_a_gen.get('FatigaDias', 7))
gf_a = float(snap_a_gen.get('GF_5', 1.5))
gc_a = float(snap_a_gen.get('GC_5', 1.2))
dif_gol_a = float(snap_a_gen.get('DifGol_5', 0.0))
ppg_fuera_a = float(snap_a_ctx.get('PPG_Fuera5', snap_a_gen.get('PPG_5', 1.2)))

# Helper para renderizar badges de forma CON TOOLTIP INTERACTIVO
def make_form_interactive_html(match_list):
    badges_html = '<div class="form-badge-container">'
    for m in match_list:
        c = m['char']
        badges_html += f"""<div class="tooltip-wrap">
<span class="form-pill pill-{c}">{c}</span>
<div class="tooltip-card">
<div class="tooltip-date">🗓️ {m['date_str']}</div>
<div class="tooltip-fixture">
<img class="tooltip-crest" src="{m['home_logo']}">
<span>{m['home_team']} <b>{m['score_str']}</b> {m['away_team']}</span>
<img class="tooltip-crest" src="{m['away_logo']}">
</div>
<span class="tooltip-badge tt-{c}">{m['desc']}</span>
</div>
</div>"""
    badges_html += '</div>'
    return badges_html

def make_bar_html(val_pct, color_class):
    pct = max(0, min(100, int(val_pct * 100)))
    return f'<div class="custom-progress-bg"><div class="custom-progress-fill {color_class}" style="width:{pct}%;"></div></div>'

def make_stat_box(label, val_str, bar_pct, color_class, extra_html=""):
    return f"""<div class="stat-box">
<div class="stat-header"><span class="stat-label">{label}</span><span class="stat-value">{val_str}</span></div>
{extra_html}
{make_bar_html(bar_pct, color_class)}
</div>"""

# ── 3 COLUMNAS PRINCIPALES DE PROBABILIDAD & ESTADÍSTICAS ───────────────────
col_home, col_draw, col_away = st.columns(3)

home_title_suffix = "(Tu Equipo)" if is_selected_home else "(Rival)"
away_title_suffix = "(Tu Equipo)" if not is_selected_home else "(Rival)"

# 1. COLUMNA LOCAL (VERDE)
with col_home:
    tag_h = 'MOST LIKELY' if prob_home >= max(prob_draw, prob_away) else 'HOME'
    h_html = f"""<div class="outcome-card">
<div class="outcome-tag tag-green">OUTCOME A &bull; {tag_h}</div>
<div class="outcome-title title-green">{home_clean} win <span style="font-size:13px; font-weight:600; color:#15803d;">{home_title_suffix}</span></div>
<div class="prob-number title-green">{prob_home*100:.0f}%</div>
<div class="prob-subtext">MODEL PROBABILITY &bull; LOCAL</div>
{make_stat_box("FORMA (ÚLTIMOS 5)", form_h_str, form_h_score, "fill-green", make_form_interactive_html(form_h_details))}
{make_stat_box("DÍAS DE DESCANSO", f"{fatiga_h} días", min(fatiga_h / 8.0, 1.0), "fill-green")}
{make_stat_box("Goles por partido en los últimos 5 partidos", f"{gf_h:.2f} GF", min(gf_h / 3.5, 1.0), "fill-green")}
{make_stat_box("Goles encajados por partido en los últimos 5 partidos", f"{gc_h:.2f} GC", max(0.0, 1.0 - (gc_h / 3.0)), "fill-green")}
{make_stat_box("diferencial de tiros en sus últimos 5 partidos", f"{'+' if dif_gol_h > 0 else ''}{dif_gol_h:.2f}", max(0.0, min(1.0, (dif_gol_h + 3.0) / 6.0)), "fill-green")}
{make_stat_box("Promedio de puntos como local en sus últimos 5 partidos", f"{ppg_casa_h:.2f} PPG", min(ppg_casa_h / 3.0, 1.0), "fill-green")}
</div>"""
    st.markdown(h_html, unsafe_allow_html=True)


# 2. COLUMNA EMPATE (AZUL)
with col_draw:
    empates_pct = (df_hist['Resultado'] == 'D').mean()
    tag_d = 'MOST LIKELY' if prob_draw >= max(prob_home, prob_away) else 'DRAW'
    
    # Calcular empates en los últimos 3 partidos H2H
    n_empates_h2h = int(sum(1 for _, r in h2h_df.iterrows() if r['Resultado'] == 'D')) if len(h2h_df) > 0 else 0
    
    d_html = f"""<div class="outcome-card">
<div class="outcome-tag tag-blue">OUTCOME B &bull; {tag_d}</div>
<div class="outcome-title title-blue">Draw</div>
<div class="prob-number title-blue">{prob_draw*100:.0f}%</div>
<div class="prob-subtext">MODEL PROBABILITY</div>
{make_stat_box("TASA EMPATES HISTÓRICA", f"{empates_pct*100:.1f}%", empates_pct, "fill-blue")}
{make_stat_box("Certeza de las predicciones de ambos equipos", f"Δ {abs(vec.get('EloDiff', 0)):.1f} pts", max(0.0, 1.0 - abs(vec.get('EloDiff', 0))/400.0), "fill-blue")}
{make_stat_box("número de empates en los últimos 3 partidos", f"{n_empates_h2h} de {len(h2h_df)}", (n_empates_h2h / len(h2h_df)) if len(h2h_df)>0 else 0.25, "fill-blue")}
<div style="background-color:#eff6ff; border:1px solid #bfdbfe; border-radius:8px; padding:14px; margin-top:24px; text-align:center;">
<div style="font-size:13px; font-weight:800; color:#1e40af; margin-bottom:4px;">🎯 Marcador más probable: 1 - 1</div>
<div style="font-size:11px; color:#60a5fa; font-weight:500;">Los empates presentan la mayor dispersión en modelos de Machine Learning.</div>
</div>
</div>"""
    st.markdown(d_html, unsafe_allow_html=True)


# 3. COLUMNA VISITANTE (ROJO)
with col_away:
    tag_a = 'MOST LIKELY' if prob_away >= max(prob_home, prob_draw) else 'AWAY'
    a_html = f"""<div class="outcome-card">
<div class="outcome-tag tag-red">OUTCOME C &bull; {tag_a}</div>
<div class="outcome-title title-red">{away_clean} win <span style="font-size:13px; font-weight:600; color:#b91c1c;">{away_title_suffix}</span></div>
<div class="prob-number title-red">{prob_away*100:.0f}%</div>
<div class="prob-subtext">MODEL PROBABILITY &bull; VISITANTE</div>
{make_stat_box("FORMA (ÚLTIMOS 5)", form_a_str, form_a_score, "fill-red", make_form_interactive_html(form_a_details))}
{make_stat_box("DÍAS DE DESCANSO", f"{fatiga_a} días", min(fatiga_a / 8.0, 1.0), "fill-red")}
{make_stat_box("Goles por partido en los últimos 5 partidos", f"{gf_a:.2f} GF", min(gf_a / 3.5, 1.0), "fill-red")}
{make_stat_box("Goles encajados por partido en los últimos 5 partidos", f"{gc_a:.2f} GC", max(0.0, 1.0 - (gc_a / 3.0)), "fill-red")}
{make_stat_box("diferencial de tiros en sus últimos 5 partidos", f"{'+' if dif_gol_a > 0 else ''}{dif_gol_a:.2f}", max(0.0, min(1.0, (dif_gol_a + 3.0) / 6.0)), "fill-red")}
{make_stat_box("Promedio de puntos como visitante en sus últimos 5 partidos", f"{ppg_fuera_a:.2f} PPG", min(ppg_fuera_a / 3.0, 1.0), "fill-red")}
</div>"""
    st.markdown(a_html, unsafe_allow_html=True)


# ── SECCIÓN INFERIOR: ÚLTIMAS 6 PREDICCIONES & ESTADÍSTICAS COMPARTIDAS ──────
col_bottom_left, col_bottom_right = st.columns([1.2, 1.0])

# PREDICCIONES RECIENTES EN EL TEST SET
with col_bottom_left:
    test_preds = get_test_predictions(df_model, model, scaler, class_names, n=6)
    
    pred_rows_html = ""
    if test_preds:
        for p in test_preds:
            badge_class = "badge-correct" if p['correct'] else "badge-wrong"
            badge_icon = "✓ Correct" if p['correct'] else "✗ Wrong"
            pred_rows_html += f"""<div class="prediction-row">
<div>
<div class="prediction-match">{p['home']} {p['score']} {p['away']}</div>
<div class="prediction-details">Predicción: <b>{p['prediction']}</b> ({p['confidence']:.0f}%)</div>
</div>
<div><span class="{badge_class}">{badge_icon}</span></div>
</div>"""
    else:
        pred_rows_html = "<div style='color:#94a3b8; font-size:12px;'>Sin predicciones de test registradas.</div>"

    bottom_left_html = f"""<div class="panel-card">
<div class="panel-title">📋 Últimas 6 Predicciones (Test Set Liga Completa)</div>
<div style="display:grid; grid-template-columns:1fr 1fr; gap:10px;">
{pred_rows_html}
</div>
</div>"""
    st.markdown(bottom_left_html, unsafe_allow_html=True)


# ESTADÍSTICOS COMPARTIDOS (HEAD-TO-HEAD)
with col_bottom_right:
    h2h_rows_html = ""
    if len(h2h_df) > 0:
        for _, r in h2h_df.iterrows():
            h2h_rows_html += f"""<div style="display:flex; justify-content:space-between; font-size:12px; padding:4px 8px; background:#f8fafc; border-radius:4px; margin-bottom:4px;">
<span>{r['Fecha'].strftime('%d/%m/%Y')}</span>
<span style="font-weight:700;">{r['EquipoLocal']} {int(r['GL'])} - {int(r['GV'])} {r['EquipoVisitante']}</span>
</div>"""
    else:
        h2h_rows_html = "<div style='font-size:12px; color:#94a3b8;'>Sin enfrentamientos en los últimos 3 años.</div>"

    dif_color = "#15803d" if h2h_dif >= 0 else "#b91c1c"
    
    bottom_right_html = f"""<div class="panel-card">
<div class="panel-title">⚔️ Estadísticos Compartidos ({home_clean} vs {away_clean})</div>
<div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:12px;">
<div class="metric-mini">
<div class="metric-mini-val" style="color:#15803d;">{h2h_winrate*100:.0f}%</div>
<div class="metric-mini-lbl">Tasa de victorias del local</div>
</div>
<div class="metric-mini">
<div class="metric-mini-val" style="color:{dif_color};">{'+' if h2h_dif > 0 else ''}{h2h_dif:.2f}</div>
<div class="metric-mini-lbl">Promedio goles marcados menos encajados del local</div>
</div>
</div>
<div style="font-size:11px; color:#64748b; line-height:1.4; margin-bottom:10px; background:#f8fafc; padding:8px 10px; border-radius:6px; border:1px solid #f1f5f9;">
💡 <b>Interpretación H2H:</b> Un valor positivo en el promedio indica que {home_clean} marca más goles en sus duelos directos; uno negativo favorece a {away_clean}.
</div>
<div style="font-size:11px; font-weight:700; color:#64748b; margin-bottom:4px;">HISTORIAL DIRECTO:</div>
{h2h_rows_html}
</div>"""
    st.markdown(bottom_right_html, unsafe_allow_html=True)


# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Panel de Control")
    st.write(f"**Modelo en uso:** {model_name}")
    st.write(f"**Partidos históricos:** {len(df_hist):,}")
    
    st.markdown("---")
    st.subheader("🔄 Actualizar Entrenamiento")
    st.write("Si añades nuevos partidos o archivos Excel a `data/`, pulsa este botón para reentrenar el modelo con toda la información nueva:")
    
    if st.button("🚀 Reentrenar desde Sidebar", use_container_width=True):
        with st.status("Reentrenando...", expanded=True) as status:
            df_hist_new = load_historical_data(DATA_DIR)
            df_hist_new, elo_ratings_new, _ = calcular_elo(df_hist_new)
            df_long_new = build_team_features(df_hist_new)
            df_h2h_new = build_h2h_features(df_hist_new)
            df_model_new, le_new, class_names_new = build_model_dataset(df_hist_new, df_long_new, df_h2h_new)
            model_new, scaler_new, class_names_new, best_name_new = train_and_save(df_model_new, class_names_new)
            st.cache_resource.clear()
            st.cache_data.clear()
            status.update(label="¡Modelo reentrenado y guardado!", state="complete")
            st.rerun()
            
    st.markdown("---")
    if st.button("🗑️ Limpiar Caché de la API", use_container_width=True):
        api.clear_cache()
        st.success("Caché de API vaciada. Las próximas consultas irán directas a football-data.org.")
        st.rerun()
