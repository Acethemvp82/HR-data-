
import math
from datetime import date, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
from pybaseball import statcast

MLB_API = "https://statsapi.mlb.com/api/v1"

st.set_page_config(
    page_title="MLB HR Model",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------
# Mobile-first CSS
# -----------------------------
st.markdown("""
<style>
:root {
  --card-radius: 18px;
}
.block-container {
  max-width: 980px;
  padding-top: 0.8rem;
  padding-left: 0.8rem;
  padding-right: 0.8rem;
  padding-bottom: 4rem;
}
h1 { font-size: 1.75rem !important; margin-bottom: .15rem !important; }
h2 { font-size: 1.35rem !important; }
h3 { font-size: 1.1rem !important; }
[data-testid="stMetric"] {
  border: 1px solid rgba(128,128,128,.25);
  border-radius: 14px;
  padding: .65rem;
}
div[data-testid="stExpander"] {
  border-radius: var(--card-radius);
  border: 1px solid rgba(128,128,128,.25);
  overflow: hidden;
}
.stButton > button {
  width: 100%;
  min-height: 48px;
  border-radius: 14px;
  font-weight: 700;
}
.stDownloadButton > button {
  width: 100%;
  min-height: 46px;
  border-radius: 14px;
}
.mobile-card {
  border: 1px solid rgba(128,128,128,.24);
  border-radius: var(--card-radius);
  padding: 14px 15px;
  margin: 10px 0;
  box-shadow: 0 2px 9px rgba(0,0,0,.06);
}
.rankline {
  display:flex;
  justify-content:space-between;
  gap:12px;
  align-items:center;
}
.player-name {
  font-size:1.12rem;
  font-weight:800;
  line-height:1.2;
}
.score-pill {
  font-size:1.05rem;
  font-weight:900;
  padding:7px 11px;
  border-radius:999px;
  border:1px solid rgba(128,128,128,.28);
  white-space:nowrap;
}
.muted {
  opacity:.72;
  font-size:.88rem;
}
.grade {
  margin-top:8px;
  font-weight:800;
}
.stat-grid {
  display:grid;
  grid-template-columns: repeat(3, 1fr);
  gap:7px;
  margin-top:10px;
}
.stat-box {
  text-align:center;
  border-radius:11px;
  padding:7px 4px;
  background:rgba(128,128,128,.08);
}
.stat-value { font-weight:850; font-size:.97rem; }
.stat-label { opacity:.70; font-size:.70rem; }
.section-note {
  border-radius:14px;
  padding:10px 12px;
  background:rgba(128,128,128,.08);
  font-size:.88rem;
}
@media (max-width: 600px) {
  .block-container { padding-left:.55rem; padding-right:.55rem; }
  h1 { font-size:1.55rem !important; }
  .stat-grid { grid-template-columns: repeat(3, 1fr); }
  [data-testid="column"] { min-width: 0 !important; }
}
</style>
""", unsafe_allow_html=True)

BATTER_WEIGHTS = {
    "barrel_pct": 0.25,
    "hard_hit_pct": 0.15,
    "avg_ev": 0.10,
    "max_ev": 0.10,
    "pull_air_pct": 0.15,
    "pull_barrel_pct": 0.15,
    "sweet_spot_pct": 0.10,
}
BATTER_RANGES = {
    "barrel_pct": (5, 25),
    "hard_hit_pct": (35, 65),
    "avg_ev": (86, 96),
    "max_ev": (100, 115),
    "pull_air_pct": (10, 45),
    "pull_barrel_pct": (0, 15),
    "sweet_spot_pct": (20, 50),
}
PITCHER_RANGES = {
    "barrel_allowed_pct": (5, 18),
    "hard_hit_allowed_pct": (35, 55),
    "avg_ev_allowed": (86, 93),
    "air_pct_allowed": (25, 50),
    "hr_per_bbe": (1, 10),
}
PITCHER_WEIGHTS = {
    "barrel_allowed_pct": 0.30,
    "hard_hit_allowed_pct": 0.20,
    "avg_ev_allowed": 0.15,
    "air_pct_allowed": 0.15,
    "hr_per_bbe": 0.20,
}

def clip_score(value, low, high):
    if value is None or pd.isna(value):
        return 50.0
    return float(np.clip((value - low) / (high - low) * 100, 0, 100))

def request_json(url, params=None):
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=1800, show_spinner=False)
def get_schedule(game_date: str):
    data = request_json(
        f"{MLB_API}/schedule",
        params={"sportId": 1, "date": game_date, "hydrate": "probablePitcher,team,venue"},
    )
    games = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            away = g["teams"]["away"]["team"]
            home = g["teams"]["home"]["team"]
            games.append({
                "game_pk": g["gamePk"],
                "away_id": away["id"], "away": away["name"],
                "home_id": home["id"], "home": home["name"],
                "away_sp_id": g["teams"]["away"].get("probablePitcher", {}).get("id"),
                "away_sp": g["teams"]["away"].get("probablePitcher", {}).get("fullName"),
                "home_sp_id": g["teams"]["home"].get("probablePitcher", {}).get("id"),
                "home_sp": g["teams"]["home"].get("probablePitcher", {}).get("fullName"),
                "venue": g.get("venue", {}).get("name", ""),
                "status": g.get("status", {}).get("detailedState", ""),
            })
    return pd.DataFrame(games)

@st.cache_data(ttl=1800, show_spinner=False)
def get_active_roster(team_id: int):
    data = request_json(f"{MLB_API}/teams/{team_id}/roster", params={"rosterType": "active"})
    rows = []
    for r in data.get("roster", []):
        pos = r.get("position", {}).get("abbreviation", "")
        rows.append({
            "player_id": r["person"]["id"],
            "name": r["person"]["fullName"],
            "position": pos,
        })
    return pd.DataFrame(rows)

@st.cache_data(ttl=1800, show_spinner=True)
def get_statcast_window(start_dt: str, end_dt: str):
    df = statcast(start_dt=start_dt, end_dt=end_dt)
    return pd.DataFrame() if df is None else df.copy()

def is_pull(row):
    x, stand = row.get("hc_x"), row.get("stand")
    if pd.isna(x) or stand not in ("L", "R"):
        return False
    return (stand == "R" and x < 125.42) or (stand == "L" and x > 125.42)

def bbe_metrics(df: pd.DataFrame, n: int):
    if df.empty:
        return {}
    x = df.dropna(subset=["launch_speed", "launch_angle"]).copy()
    if x.empty:
        return {}
    sort_cols = [c for c in ["game_date", "game_pk", "at_bat_number", "pitch_number"] if c in x.columns]
    if sort_cols:
        x = x.sort_values(sort_cols)
    x = x.tail(n)

    lsa = pd.to_numeric(x.get("launch_speed_angle"), errors="coerce")
    ev = pd.to_numeric(x["launch_speed"], errors="coerce")
    la = pd.to_numeric(x["launch_angle"], errors="coerce")

    x["is_barrel"] = lsa.eq(6)
    x["is_hard_hit"] = ev.ge(95)
    x["is_sweet_spot"] = la.between(8, 32)
    x["is_pull"] = x.apply(is_pull, axis=1)
    x["is_air"] = la.ge(10)
    x["is_pull_air"] = x["is_pull"] & x["is_air"]
    x["is_pull_barrel"] = x["is_pull"] & x["is_barrel"]

    return {
        "bbe": len(x),
        "barrel_pct": x["is_barrel"].mean() * 100,
        "hard_hit_pct": x["is_hard_hit"].mean() * 100,
        "avg_ev": ev.mean(),
        "max_ev": ev.max(),
        "pull_air_pct": x["is_pull_air"].mean() * 100,
        "pull_barrel_pct": x["is_pull_barrel"].mean() * 100,
        "sweet_spot_pct": x["is_sweet_spot"].mean() * 100,
    }

def score_batter_metrics(m):
    if not m:
        return np.nan
    total = used = 0.0
    for k, w in BATTER_WEIGHTS.items():
        v = m.get(k)
        if v is not None and not pd.isna(v):
            lo, hi = BATTER_RANGES[k]
            total += clip_score(v, lo, hi) * w
            used += w
    return total / used if used else np.nan

def pitcher_metrics(df):
    x = df.dropna(subset=["launch_speed", "launch_angle"]).copy()
    if x.empty:
        return {}
    ev = pd.to_numeric(x["launch_speed"], errors="coerce")
    la = pd.to_numeric(x["launch_angle"], errors="coerce")
    lsa = pd.to_numeric(x.get("launch_speed_angle"), errors="coerce")
    x["is_barrel"] = lsa.eq(6)
    x["is_hard_hit"] = ev.ge(95)
    x["is_air"] = la.ge(10)
    events = x["events"] if "events" in x.columns else pd.Series(index=x.index, dtype=object)
    x["is_hr"] = events.eq("home_run")
    return {
        "bbe_allowed": len(x),
        "barrel_allowed_pct": x["is_barrel"].mean() * 100,
        "hard_hit_allowed_pct": x["is_hard_hit"].mean() * 100,
        "avg_ev_allowed": ev.mean(),
        "air_pct_allowed": x["is_air"].mean() * 100,
        "hr_per_bbe": x["is_hr"].mean() * 100,
    }

def score_pitcher(m):
    if not m:
        return 50.0
    total = used = 0.0
    for k, w in PITCHER_WEIGHTS.items():
        v = m.get(k)
        if v is not None and not pd.isna(v):
            lo, hi = PITCHER_RANGES[k]
            total += clip_score(v, lo, hi) * w
            used += w
    return total / used if used else 50.0

def pitcher_pitch_mix(df, min_usage=.20):
    if df.empty or "pitch_type" not in df:
        return []
    mix = df.dropna(subset=["pitch_type"])["pitch_type"].value_counts(normalize=True)
    major = mix[mix >= min_usage]
    if major.empty:
        major = mix.head(2)
    return [(str(pt), float(pct)) for pt, pct in major.items()]

def pitch_match_score(batter_df, pitcher_mix):
    if batter_df.empty or not pitcher_mix:
        return 50.0
    scores, weights = [], []
    for pitch_type, usage in pitcher_mix:
        d = batter_df[batter_df["pitch_type"] == pitch_type]
        n = len(d.dropna(subset=["launch_speed", "launch_angle"]))
        if n < 2:
            continue
        m = bbe_metrics(d, min(15, n))
        scores.append(score_batter_metrics(m))
        weights.append(usage)
    return float(np.average(scores, weights=weights)) if scores else 50.0

def classify(score):
    if score >= 90: return "🚀 ELITE"
    if score >= 84: return "🔥 STRONG"
    if score >= 78: return "✅ PLAYABLE"
    if score >= 70: return "👀 WATCH"
    return "— PASS"

@st.cache_data(ttl=1800, show_spinner=False)
def build_rankings(game_date: str, lookback_days: int):
    schedule = get_schedule(game_date)
    if schedule.empty:
        return pd.DataFrame(), schedule

    end_dt = (pd.Timestamp(game_date) - pd.Timedelta(days=1)).date()
    start_dt = end_dt - timedelta(days=lookback_days - 1)
    sc = get_statcast_window(str(start_dt), str(end_dt))
    if sc.empty:
        return pd.DataFrame(), schedule

    for c in ["batter", "pitcher"]:
        sc[c] = pd.to_numeric(sc[c], errors="coerce")

    candidates = []
    for _, g in schedule.iterrows():
        sides = [
            (int(g["away_id"]), g["away"], g["home"], g["home_sp_id"], g["home_sp"]),
            (int(g["home_id"]), g["home"], g["away"], g["away_sp_id"], g["away_sp"]),
        ]
        for team_id, team_name, opp_name, opp_sp_id, opp_sp in sides:
            if opp_sp_id is None or pd.isna(opp_sp_id):
                continue
            roster = get_active_roster(team_id)
            roster = roster[~roster["position"].isin(["P", "TWP"])]

            sp_rows = sc[sc["pitcher"] == int(opp_sp_id)]
            pm = pitcher_metrics(sp_rows)
            pscore = score_pitcher(pm)
            pmix = pitcher_pitch_mix(sp_rows)

            for _, p in roster.iterrows():
                pid = int(p["player_id"])
                br = sc[sc["batter"] == pid]
                bbe_count = len(br.dropna(subset=["launch_speed", "launch_angle"]))
                if bbe_count < 10:
                    continue

                m10 = bbe_metrics(br, 10)
                m15 = bbe_metrics(br, 15)
                s10 = score_batter_metrics(m10)
                s15 = score_batter_metrics(m15)
                recent = .60 * s10 + .40 * s15
                match = pitch_match_score(br, pmix)
                total = .50 * recent + .25 * match + .25 * pscore

                candidates.append({
                    "Batter": p["name"],
                    "Team": team_name,
                    "Opponent": opp_name,
                    "Opp SP": opp_sp or "TBD",
                    "HR Score": round(total, 1),
                    "Grade": classify(total),
                    "Recent Score": round(recent, 1),
                    "Pitch Match": round(match, 1),
                    "Pitcher Vulnerability": round(pscore, 1),
                    "SP Primary Pitches": ", ".join(f"{pt} {u:.0%}" for pt, u in pmix) or "N/A",
                    "L10 Barrel%": round(m10.get("barrel_pct", np.nan), 1),
                    "L10 HardHit%": round(m10.get("hard_hit_pct", np.nan), 1),
                    "L10 AvgEV": round(m10.get("avg_ev", np.nan), 1),
                    "L10 MaxEV": round(m10.get("max_ev", np.nan), 1),
                    "L10 PullAir%": round(m10.get("pull_air_pct", np.nan), 1),
                    "L10 PullBarrel%": round(m10.get("pull_barrel_pct", np.nan), 1),
                    "L10 SweetSpot%": round(m10.get("sweet_spot_pct", np.nan), 1),
                    "L15 Barrel%": round(m15.get("barrel_pct", np.nan), 1),
                    "L15 HardHit%": round(m15.get("hard_hit_pct", np.nan), 1),
                    "L15 AvgEV": round(m15.get("avg_ev", np.nan), 1),
                    "L15 MaxEV": round(m15.get("max_ev", np.nan), 1),
                    "Pitcher BarrelAllowed%": round(pm.get("barrel_allowed_pct", np.nan), 1),
                    "Pitcher HardHitAllowed%": round(pm.get("hard_hit_allowed_pct", np.nan), 1),
                    "Pitcher AvgEVAllowed": round(pm.get("avg_ev_allowed", np.nan), 1),
                    "Pitcher HR/BBE%": round(pm.get("hr_per_bbe", np.nan), 1),
                    "Recent BBE Available": bbe_count,
                })

    out = pd.DataFrame(candidates)
    if not out.empty:
        out = out.sort_values("HR Score", ascending=False).reset_index(drop=True)
        out.insert(0, "Rank", np.arange(1, len(out)+1))
    return out, schedule

def fmt(v, suffix=""):
    return "—" if pd.isna(v) else f"{v}{suffix}"

def render_player_card(row):
    st.markdown(f"""
    <div class="mobile-card">
      <div class="rankline">
        <div>
          <div class="muted">#{int(row['Rank'])} · {row['Team']} vs {row['Opponent']}</div>
          <div class="player-name">{row['Batter']}</div>
        </div>
        <div class="score-pill">{row['HR Score']}</div>
      </div>
      <div class="grade">{row['Grade']}</div>
      <div class="muted">vs {row['Opp SP']} · {row['SP Primary Pitches']}</div>
      <div class="stat-grid">
        <div class="stat-box"><div class="stat-value">{fmt(row['L10 Barrel%'],'%')}</div><div class="stat-label">L10 BARREL</div></div>
        <div class="stat-box"><div class="stat-value">{fmt(row['L10 HardHit%'],'%')}</div><div class="stat-label">L10 HARD HIT</div></div>
        <div class="stat-box"><div class="stat-value">{fmt(row['L10 AvgEV'])}</div><div class="stat-label">L10 AVG EV</div></div>
        <div class="stat-box"><div class="stat-value">{fmt(row['L10 PullAir%'],'%')}</div><div class="stat-label">PULL AIR</div></div>
        <div class="stat-box"><div class="stat-value">{fmt(row['Pitch Match'])}</div><div class="stat-label">PITCH MATCH</div></div>
        <div class="stat-box"><div class="stat-value">{fmt(row['Pitcher Vulnerability'])}</div><div class="stat-label">SP VULN</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("View full breakdown"):
        c1, c2, c3 = st.columns(3)
        c1.metric("Recent", row["Recent Score"])
        c2.metric("Pitch Match", row["Pitch Match"])
        c3.metric("SP Vulnerability", row["Pitcher Vulnerability"])

        st.markdown("**Last 10 BBE**")
        l10 = pd.DataFrame({
            "Metric": ["Barrel%", "HardHit%", "Avg EV", "Max EV", "Pull-Air%", "Pull-Barrel%", "SweetSpot%"],
            "Value": [
                row["L10 Barrel%"], row["L10 HardHit%"], row["L10 AvgEV"], row["L10 MaxEV"],
                row["L10 PullAir%"], row["L10 PullBarrel%"], row["L10 SweetSpot%"]
            ]
        })
        st.dataframe(l10, hide_index=True, use_container_width=True)

        st.markdown("**Last 15 BBE**")
        l15 = pd.DataFrame({
            "Metric": ["Barrel%", "HardHit%", "Avg EV", "Max EV"],
            "Value": [row["L15 Barrel%"], row["L15 HardHit%"], row["L15 AvgEV"], row["L15 MaxEV"]]
        })
        st.dataframe(l15, hide_index=True, use_container_width=True)

        st.markdown("**Opposing pitcher**")
        p = pd.DataFrame({
            "Metric": ["Barrel Allowed%", "HardHit Allowed%", "Avg EV Allowed", "HR / BBE%"],
            "Value": [
                row["Pitcher BarrelAllowed%"], row["Pitcher HardHitAllowed%"],
                row["Pitcher AvgEVAllowed"], row["Pitcher HR/BBE%"]
            ]
        })
        st.dataframe(p, hide_index=True, use_container_width=True)

# -----------------------------
# UI
# -----------------------------
st.title("⚾ MLB HR Model")
st.caption("Daily Statcast-powered HR opportunity rankings")

with st.expander("⚙️ Slate settings", expanded=True):
    selected_date = st.date_input("Slate date", value=date.today())
    c1, c2 = st.columns(2)
    lookback = c1.selectbox("Statcast history", [30, 45, 60], index=1, format_func=lambda x: f"{x} days")
    top_n = c2.selectbox("Show", [5, 10, 15, 20, 30], index=1, format_func=lambda x: f"Top {x}")
    min_score = st.slider("Minimum HR Score", 0, 100, 70)
    run = st.button("🔥 BUILD TODAY'S HR BOARD", type="primary")

if run:
    with st.spinner("Building the slate from Baseball Savant / Statcast..."):
        rankings, schedule = build_rankings(str(selected_date), int(lookback))

    if schedule.empty:
        st.warning("No MLB games were found for this date.")
        st.stop()

    st.markdown("### 📅 Slate")
    for _, g in schedule.iterrows():
        away_sp = g["away_sp"] or "TBD"
        home_sp = g["home_sp"] or "TBD"
        st.markdown(
            f"<div class='section-note'><b>{g['away']} @ {g['home']}</b><br>"
            f"<span class='muted'>{away_sp} vs {home_sp} · {g['venue']}</span></div>",
            unsafe_allow_html=True
        )

    if rankings.empty:
        st.warning("No hitters were scored. Probable pitchers may not be posted yet, or there may be insufficient Statcast data.")
        st.stop()

    qualified = rankings[rankings["HR Score"] >= min_score].head(int(top_n))

    m1, m2, m3 = st.columns(3)
    m1.metric("Scored", len(rankings))
    m2.metric("Qualified", len(rankings[rankings["HR Score"] >= min_score]))
    m3.metric("Top", rankings.iloc[0]["HR Score"])

    tabs = st.tabs(["🔥 Best HR Spots", "🚀 Elite", "📊 Full Board"])

    with tabs[0]:
        if qualified.empty:
            st.info("No hitters meet the current minimum score.")
        else:
            for _, row in qualified.iterrows():
                render_player_card(row)

    with tabs[1]:
        elite = rankings[rankings["HR Score"] >= 84].head(20)
        if elite.empty:
            st.info("No STRONG/ELITE hitters on this slate.")
        else:
            for _, row in elite.iterrows():
                render_player_card(row)

    with tabs[2]:
        display_cols = [
            "Rank", "Batter", "Team", "Opponent", "Opp SP", "HR Score", "Grade",
            "Recent Score", "Pitch Match", "Pitcher Vulnerability",
            "L10 Barrel%", "L10 HardHit%", "L10 AvgEV", "L10 PullAir%", "L10 PullBarrel%"
        ]
        st.dataframe(rankings[display_cols], use_container_width=True, hide_index=True)

        st.download_button(
            "Download full CSV",
            data=rankings.to_csv(index=False).encode("utf-8"),
            file_name=f"mlb_hr_board_{selected_date}.csv",
            mime="text/csv",
        )

    st.markdown("""
    <div class="section-note">
      <b>V1 scope:</b> This board ranks HR opportunities from baseball data. It does not yet include sportsbook price,
      weather, park-factor adjustments, or calibrated fair odds. Those are the next modules.
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="section-note">
      Pick the slate date, then tap <b>BUILD TODAY'S HR BOARD</b>. Once this app is deployed, you can bookmark it
      on your phone and use it like a normal website.
    </div>
    """, unsafe_allow_html=True)

with st.expander("ℹ️ How the HR Score works"):
    st.markdown("""
**50% Recent contact quality**
- Last 10 BBE = 60% of recent score
- Last 15 BBE = 40%

**25% Pitch-type matchup**
- Starter pitches used at least 20% are evaluated.
- If none reach 20%, the top two pitches are used.

**25% Pitcher vulnerability**
- Barrel rate allowed
- Hard-hit rate allowed
- Average EV allowed
- Air-ball rate
- HR per BBE

**Contact metrics**
- Barrel%
- Hard-hit%
- Average EV
- Max EV
- Pull-Air%
- Pull-Barrel%
- Sweet-Spot%
""")
