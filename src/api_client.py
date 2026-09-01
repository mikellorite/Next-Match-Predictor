"""
football-data.org API v4 client with caching, rate limiting, and match retrieval.
"""

import json
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import pandas as pd
import requests

from src.config import API_KEY, BASE_URL, COMP_CODE, RATE_LIMIT_SLEEP, CACHE_FILE


class ApiClient:
    """Wrapper for the football-data.org v4 REST API."""

    def __init__(self, api_key: str = API_KEY, cache_file: Path = CACHE_FILE):
        self.headers = {"X-Auth-Token": api_key}
        self.cache_file = cache_file
        self._load_cache()
        self._last_request_time = 0.0

    def _load_cache(self):
        if self.cache_file.exists():
            with open(self.cache_file, "r", encoding="utf-8") as f:
                self._cache = json.load(f)
        else:
            self._cache = {}

    def _save_cache(self):
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, ensure_ascii=False, indent=2)

    def clear_cache(self):
        self._cache = {}
        if self.cache_file.exists():
            self.cache_file.unlink()

    def _get(self, endpoint: str, params: Optional[dict] = None, cache_ttl_hours: float = 1.0) -> dict:
        """GET with cache, rate limiting, and retries on 429."""
        cache_key = f"{endpoint}|{json.dumps(params, sort_keys=True)}"
        # Check cache
        if cache_key in self._cache and cache_ttl_hours != 0:
            entry = self._cache[cache_key]
            age_h = (time.time() - entry["ts"]) / 3600
            if cache_ttl_hours < 0 or age_h < cache_ttl_hours:
                return entry["data"]
        # Rate limit
        elapsed = time.time() - self._last_request_time
        if elapsed < RATE_LIMIT_SLEEP:
            time.sleep(RATE_LIMIT_SLEEP - elapsed)
        url = f"{BASE_URL}{endpoint}"
        for _ in range(3):
            try:
                resp = requests.get(url, headers=self.headers, params=params, timeout=20)
            except requests.RequestException:
                return {}
            self._last_request_time = time.time()
            if resp.status_code == 200:
                data = resp.json()
                if cache_ttl_hours != 0:
                    self._cache[cache_key] = {"ts": time.time(), "data": data}
                    self._save_cache()
                return data
            elif resp.status_code == 429:
                time.sleep(65)
            else:
                return {}
        return {}

    def get_teams(self) -> pd.DataFrame:
        data = self._get(f"/competitions/{COMP_CODE}/teams", cache_ttl_hours=24)
        if not isinstance(data, dict) or not data:
            return pd.DataFrame(columns=["id", "name", "shortName", "tla"])
        teams = data.get("teams", [])
        if not teams:
            return pd.DataFrame(columns=["id", "name", "shortName", "tla"])
        rows = [
            {
                "id": t.get("id"),
                "name": t.get("name", ""),
                "shortName": t.get("shortName", ""),
                "tla": t.get("tla", ""),
            }
            for t in teams
            if isinstance(t, dict)
        ]
        return pd.DataFrame(rows, columns=["id", "name", "shortName", "tla"])

    def find_team_id(self, team_name: str) -> Tuple[int, str]:
        df = self.get_teams()
        if df.empty:
            raise ValueError("No se pudieron cargar los equipos desde la API.")
        name_lower = team_name.lower().replace(" fc", "").strip()
        for col in ["name", "shortName"]:
            normalized = df[col].fillna("").astype(str).str.lower().str.replace(" fc", "", regex=False).str.strip()
            mask = normalized == name_lower
            if mask.any():
                row = df.loc[mask].iloc[0]
                return int(row["id"]), str(row["name"])
        first_token = name_lower.split()[0]
        for col in ["name", "shortName"]:
            mask = df[col].fillna("").astype(str).str.lower().str.contains(first_token, regex=False, na=False)
            if mask.any():
                row = df.loc[mask].iloc[0]
                return int(row["id"]), str(row["name"])
        available = df["name"].dropna().astype(str).tolist()
        raise ValueError(f"Equipo '{team_name}' no encontrado. Disponibles: {available}")

    def get_matches_season(self, season: Optional[int] = None, cache_ttl_hours: float = 2.0) -> pd.DataFrame:
        params = {}
        if season is not None:
            params["season"] = season
        data = self._get(
            f"/competitions/{COMP_CODE}/matches",
            params=params if params else None,
            cache_ttl_hours=cache_ttl_hours,
        )
        if not data:
            return pd.DataFrame()
        matches = data.get("matches", [])
        if not matches:
            return pd.DataFrame()
        rows = []
        for m in matches:
            score = m.get("score", {})
            ht = score.get("halfTime", {}) or {}
            ft = score.get("fullTime", {}) or {}
            rows.append({
                "match_id": m.get("id"),
                "utcDate": m.get("utcDate"),
                "status": m.get("status"),
                "matchday": m.get("matchday"),
                "home_id": m["homeTeam"]["id"],
                "home_name": m["homeTeam"]["name"],
                "away_id": m["awayTeam"]["id"],
                "away_name": m["awayTeam"]["name"],
                "goals_home_ft": ft.get("home"),
                "goals_away_ft": ft.get("away"),
                "goals_home_ht": ht.get("home"),
                "goals_away_ht": ht.get("away"),
                "winner": score.get("winner"),
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df["utcDate"] = pd.to_datetime(df["utcDate"], utc=True)
        return df

    def get_next_match(self, team_id: int) -> Optional[Dict[str, Any]]:
        df = self.get_matches_season(season=None, cache_ttl_hours=1)
        if not df.empty:
            now_utc = pd.Timestamp.now(tz="UTC")
            future = df[
                (df["status"].isin(["SCHEDULED", "TIMED"]))
                & (df["utcDate"] > now_utc)
                & ((df["home_id"] == team_id) | (df["away_id"] == team_id))
            ].sort_values("utcDate")
            if not future.empty:
                return future.iloc[0].to_dict()
        # Fallback
        data = self._get(f"/teams/{team_id}/matches", params={"status": "SCHEDULED", "limit": 5}, cache_ttl_hours=1)
        matches = data.get("matches", [])
        pl_matches = [m for m in matches if m.get("competition", {}).get("code") == COMP_CODE]
        if not pl_matches:
            pl_matches = matches
        if not pl_matches:
            return None
        m = pl_matches[0]
        score = m.get("score", {})
        ht = score.get("halfTime", {}) or {}
        ft = score.get("fullTime", {}) or {}
        return {
            "match_id": m.get("id"),
            "utcDate": pd.to_datetime(m.get("utcDate"), utc=True),
            "status": m.get("status"),
            "matchday": m.get("matchday"),
            "home_id": m["homeTeam"]["id"],
            "home_name": m["homeTeam"]["name"],
            "away_id": m["awayTeam"]["id"],
            "away_name": m["awayTeam"]["name"],
            "goals_home_ft": ft.get("home"),
            "goals_away_ft": ft.get("away"),
            "goals_home_ht": ht.get("home"),
            "goals_away_ht": ht.get("away"),
            "winner": score.get("winner"),
        }
