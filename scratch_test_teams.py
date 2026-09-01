import sys
sys.stdout.reconfigure(encoding='utf-8')

from src.api_client import ApiClient
from src.config import API_KEY
from src.data_loader import clean_team_name

api = ApiClient(api_key=API_KEY)
df_teams = api.get_teams()
print(f"Total equipos oficiales en la API: {len(df_teams)}")

for _, r in df_teams.iterrows():
    raw_name = r['name']
    clean = clean_team_name(raw_name)
    tid, tname = api.find_team_id(clean)
    nxt = api.get_next_match(tid)
    if nxt:
        h = clean_team_name(nxt['home_name'])
        a = clean_team_name(nxt['away_name'])
        dt = nxt['utcDate'].strftime('%d/%m %H:%M')
        role = "LOCAL [H]" if nxt['home_id'] == tid else "VISITANTE [A]"
        print(f"Equipo: {clean:25s} | Rol: {role:15s} | Proximo Partido: {h} vs {a} ({dt})")
    else:
        print(f"Equipo: {clean:25s} | Sin proximo partido")
