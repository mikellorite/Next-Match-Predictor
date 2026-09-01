# ⚽ MATCHDAY IQ — Premier League Next Match Predictor

Sistema avanzado de Machine Learning y visualización interactiva para predecir probabilidades exactas de resultados (**Victoria Local**, **Empate**, **Victoria Visitante**) en los próximos partidos de la **Premier League**.

El pipeline combina datos históricos locales desde la temporada 2016-17 con la API oficial en vivo de `football-data.org`, integrando ratings **Elo adaptativos**, features de forma sin data leakage (rolling con shift), estadísticas Head-to-Head (H2H) y modelos calibrados (`Random Forest` y `XGBoost`).

---

## 🚀 Inicio Rápido (Lanzar la Aplicación)

### 1. Requisitos previos
- **Python 3.10+**
- Clave de API gratuita de [football-data.org](https://www.football-data.org/) (añadida en el archivo `.env`).

### 2. Instalación de dependencias

Si utilizas `uv`:
```bash
uv sync
```

O utilizando `pip` en tu entorno virtual:
```bash
pip install -r requirements.txt
# o directamente:
pip install streamlit pandas numpy scikit-learn xgboost joblib requests python-dotenv openpyxl
```

### 3. Configuración del entorno (`.env`)
Asegúrate de que el archivo `.env` en la raíz del proyecto contenga tu API Key:
```env
FOOTBAL_DATA_ORG_API_KEY=tu_api_key_aqui
```

---

## 🌐 Comando para Ejecutar la Interfaz Web

Para iniciar el servidor de Streamlit y desplegar la interfaz en `http://localhost:8501`:

```bash
streamlit run api.py --server.port 8501
```

*(También puedes usar indistintamente `streamlit run app.py --server.port 8501`)*

Una vez ejecutado, abre tu navegador web en:
👉 **[http://localhost:8501](http://localhost:8501)**

---

## 📁 Estructura del Proyecto

```
Next Match Predictor/
├── data/                          # Excels de resultados históricos organizados por temporada
│   ├── 2016-17/ ... 2026-27/
├── models/                        # Artefactos del modelo entrenado y serializado
│   ├── model.joblib               # Mejor clasificador entrenado y calibrado
│   ├── scaler.joblib              # Escalador StandardScaler
│   ├── class_names.joblib         # Clases ('A', 'D', 'H')
│   └── model_name.joblib          # Nombre del modelo ganador
├── logos/                         # Escudos oficiales descargados en local
├── src/                           # Código fuente modular
│   ├── __init__.py
│   ├── config.py                  # Constantes, rutas y definiciones de features
│   ├── data_loader.py             # Carga y normalización de nombres de equipos
│   ├── features.py                # Sistema Elo, rolling stats y construcción de dataset
│   ├── model.py                   # GridSearchCV, calibración isotónica y métricas
│   ├── predictor.py               # Extracción de snapshot de features y predicción
│   └── logos.py                   # Descarga y resolución de escudos en Base64
├── api.py                         # Aplicación web interactiva en Streamlit
├── app.py                         # Wrapper directo de arranque
├── api_cache.json                 # Caché local para respetar rate limits de la API
└── .env                           # Variables de entorno con clave de API
```

---

## 🧠 Características del Pipeline de Machine Learning

1. **Ratings Elo Dinámicos**:
   - Cada club cuenta con un rating inicial de 1500 puntos.
   - Ventaja de campo (*Home Advantage*) de +50 puntos Elo para el equipo local.
   - Factor $K$ adaptativo según la experiencia y consolidación del club en la liga.

2. **Feature Engineering Anti-Data Leakage**:
   - Estructura *long-format* por equipo y partido con orden cronológico estricto.
   - Aplicación de `shift(1)` antes de cualquier cálculo rodante (*rolling window* de 5 partidos).
   - Métricas de fatiga real basadas en días de descanso entre encuentros.
   - Features contextuales separadas para rendimiento exclusivamente en casa (`PPG_Casa5`, `GF_Casa5`) y fuera (`PPG_Fuera5`, `GF_Fuera5`).
   - Métricas históricas directas Head-to-Head (*H2H*) en los últimos 3 años.

3. **Entrenamiento y Calibración**:
   - Validación temporal *Walk-Forward* (entrenamiento en histórico previo, validación en la última temporada completa y test en la temporada actual).
   - Optimización de hiperparámetros mediante `GridSearchCV` con métrica `neg_log_loss`.
   - Comparativa automática entre `Random Forest` y `XGBoost`.
   - Calibración de probabilidades con `CalibratedClassifierCV (isotonic)` para obtener porcentajes reales y fiables de probabilidad.

---

## 🔄 Actualización y Reentrenamiento Futuro

Cuando se jueguen nuevas jornadas (por ejemplo, dentro de 3 semanas):
1. Añade los nuevos resultados al Excel correspondiente dentro de `data/2026-27/`.
2. Abre la aplicación en `http://localhost:8501`.
3. Haz clic en el botón **`🔄 Reentrenar Modelo`** ubicado en la barra superior (o en la barra lateral).
4. El sistema reentrenará automáticamente todo el pipeline y actualizará los pesos del modelo en `models/`.
