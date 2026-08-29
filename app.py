
from datetime import date, timedelta
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import requests
import streamlit as st
from pybaseball import statcast

MLB_API = "https://statsapi.mlb.com/api/v1"

st.set_page_config(
    page_title="MLB HR Model V3.1",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #0b1117 0%, #111827 100%);
    color: #e6edf3;
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #0b1117 0%, #111827 100%);
}

[data-testid="stHeader"] {
    background: rgba(11, 17, 23, 0.85);
}

/* HR board / dataframe styling */
[data-testid="stDataFrame"] {
    background: #0f1720;
    border: 1px solid #1f2937;
    border-radius: 12px;
    overflow: visible;
}

[data-testid="stDataFrame"] [role="columnheader"] {
    background: #16202c !important;
    color: #f8fafc !important;
    font-weight: 700 !important;
}

[data-testid="stDataFrame"] [role="gridcell"] {
    background: #0f1720;
    color: #e6edf3;
    border-color: #263241 !important;
}
.block-container{max-width:980px;padding:.75rem .65rem 4rem}
h1{font-size:1.6rem!important;margin-bottom:.15rem!important}
[data-testid="stMetric"]{border:1px solid rgba(128,128,128,.22);border-radius:14px;padding:.6rem}
div[data-testid="stExpander"]{border-radius:16px;overflow:hidden}
.stButton>button{width:100%;min-height:48px;border-radius:14px;font-weight:800}
.mobile-card{border:1px solid rgba(128,128,128,.22);border-radius:18px;padding:14px;margin:10px 0;box-shadow:0 2px 8px rgba(0,0,0,.05)}
.rankline{display:flex;justify-content:space-between;gap:10px;align-items:center}
.player{font-size:1.08rem;font-weight:850;line-height:1.2}
.muted{opacity:.72;font-size:.84rem}
.score{font-size:1.12rem;font-weight:900;padding:7px 11px;border-radius:999px;border:1px solid rgba(128,128,128,.3)}
.grade{font-weight:850;margin-top:7px}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-top:10px}
.box{text-align:center;border-radius:10px;padding:7px 3px;background:rgba(128,128,128,.08)}
.val{font-weight:850;font-size:.95rem}.lab{opacity:.68;font-size:.67rem}
.note{border-radius:14px;padding:10px 12px;background:rgba(128,128,128,.08);font-size:.86rem}
</style>
""", unsafe_allow_html=True)

# Transparent metric recipe. These create RAW component scores.
BATTER_WEIGHTS = {
    "barrel_pct": .25,
    "hard_hit_pct": .15,
    "avg_ev": .10,
    "max_ev": .10,
    "pull_air_pct": .15,
    "pull_barrel_pct": .15,
    "sweet_spot_pct": .10,
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
PITCHER_WEIGHTS = {
    "barrel_allowed_pct": .30,
    "hard_hit_allowed_pct": .20,
    "avg_ev_allowed": .15,
    "air_pct_allowed": .15,
    "hr_per_bbe": .20,
}
PITCHER_RANGES = {
    "barrel_allowed_pct": (5, 18),
    "hard_hit_allowed_pct": (35, 55),
    "avg_ev_allowed": (86, 93),
    "air_pct_allowed": (25, 50),
    "hr_per_bbe": (1, 10),
}

def clip_score(v, lo, hi):
    if v is None or pd.isna(v):
        return 50.0
    return float(np.clip((v-lo)/(hi-lo)*100, 0, 100))

def req_json(url, params=None):
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=300, show_spinner=False)
def get_schedule(game_date):
    data = req_json(f"{MLB_API}/schedule", {
        "sportId":1, "date":game_date, "hydrate":"probablePitcher,team,venue"
    })
    rows=[]
    for d in data.get("dates",[]):
        for g in d.get("games",[]):
            a=g["teams"]["away"]["team"]; h=g["teams"]["home"]["team"]
            rows.append({
                "game_pk": g.get("gamePk"),
                "away_id":a["id"],"away":a["name"],"home_id":h["id"],"home":h["name"],
                "away_sp_id":g["teams"]["away"].get("probablePitcher",{}).get("id"),
                "away_sp":g["teams"]["away"].get("probablePitcher",{}).get("fullName"),
                "home_sp_id":g["teams"]["home"].get("probablePitcher",{}).get("id"),
                "home_sp":g["teams"]["home"].get("probablePitcher",{}).get("fullName"),
                "venue":g.get("venue",{}).get("name",""),
                "game_time": g.get("gameDate", ""),
                "status":g.get("status",{}).get("detailedState",""),
            })
    return pd.DataFrame(rows)
    @st.cache_data(ttl=60, show_spinner=False)
    def get_game_homers(game_pk):
        data = req_json(f"{MLB_API}/game/{int(game_pk)}/feed/live")

        homers = []

        plays = data.get("liveData", {}).get("plays", {}).get("allPlays", [])

        for play in plays:
            result = play.get("result", {})

            if result.get("eventType") == "home_run":
                batter = play.get("matchup", {}).get("batter", {})

                homers.append({
                    "Batter": batter.get("fullName", ""),
                    "batter_id": batter.get("id"),
                    "HR": "💣"
                })

    return homers
@st.cache_data(ttl=300, show_spinner=False)
def get_lineup_spots(game_date):
    schedule = req_json(f"{MLB_API}/schedule", {
        "sportId": 1,
        "date": game_date
    })

    rows = []

    for d in schedule.get("dates", []):
        for g in d.get("games", []):
            game_pk = g["gamePk"]

            feed = req_json(
                f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
            )

            box = feed.get("liveData", {}).get("boxscore", {}).get("teams", {})

            for side in ["away", "home"]:
                team_data = box.get(side, {})
                team_name = team_data.get("team", {}).get("name")

                for p in team_data.get("players", {}).values():
                    batting_order = p.get("battingOrder")

                    if batting_order:
                        rows.append({
                            "Batter": p["person"]["fullName"],
                            "Team": team_name,
                            "Lineup Spot": int(str(batting_order)) // 100
                        })

    return pd.DataFrame(rows)
@st.cache_data(ttl=1800, show_spinner=False)
def get_roster(team_id):
    data=req_json(f"{MLB_API}/teams/{team_id}/roster",{"rosterType":"active"})
    rows=[]
    for r in data.get("roster",[]):
        rows.append({
            "player_id":r["person"]["id"],
            "name":r["person"]["fullName"],
            "position":r.get("position",{}).get("abbreviation","")
        })
    return pd.DataFrame(rows)

@st.cache_data(ttl=1800, show_spinner=True)
def get_statcast(start_dt,end_dt):
    x=statcast(start_dt=start_dt,end_dt=end_dt)
    return pd.DataFrame() if x is None else x.copy()

def is_pull(row):
    x=row.get("hc_x"); stand=row.get("stand")
    if pd.isna(x) or stand not in ("L","R"): return False
    return (stand=="R" and x<125.42) or (stand=="L" and x>125.42)

def bbe_metrics(df,n):
    x=df.dropna(subset=["launch_speed","launch_angle"]).copy()
    if x.empty:return {}
    cols=[c for c in ["game_date","game_pk","at_bat_number","pitch_number"] if c in x.columns]
    if cols:x=x.sort_values(cols)
    x=x.tail(n)
    ev=pd.to_numeric(x["launch_speed"],errors="coerce")
    la=pd.to_numeric(x["launch_angle"],errors="coerce")
    lsa=pd.to_numeric(x.get("launch_speed_angle"),errors="coerce")
    x["barrel"]=lsa.eq(6)
    x["hard"]=ev.ge(95)
    x["sweet"]=la.between(8,32)
    x["pull"]=x.apply(is_pull,axis=1)
    x["air"]=la.ge(10)
    x["pull_air"]=x["pull"]&x["air"]
    x["pull_barrel"]=x["pull"]&x["barrel"]
    return {
        "bbe":len(x),
        "barrel_pct":x["barrel"].mean()*100,
        "hard_hit_pct":x["hard"].mean()*100,
        "avg_ev":ev.mean(),
        "max_ev":ev.max(),
        "pull_air_pct":x["pull_air"].mean()*100,
        "pull_barrel_pct":x["pull_barrel"].mean()*100,
        "sweet_spot_pct":x["sweet"].mean()*100,
    }

def score_batter(m):
    if not m:return np.nan
    s=w=0
    for k,wt in BATTER_WEIGHTS.items():
        v=m.get(k)
        if v is not None and not pd.isna(v):
            lo,hi=BATTER_RANGES[k]
            s+=clip_score(v,lo,hi)*wt; w+=wt
    return s/w if w else np.nan

def pitcher_metrics(df):
    x=df.dropna(subset=["launch_speed","launch_angle"]).copy()
    if x.empty:return {}
    ev=pd.to_numeric(x["launch_speed"],errors="coerce")
    la=pd.to_numeric(x["launch_angle"],errors="coerce")
    lsa=pd.to_numeric(x.get("launch_speed_angle"),errors="coerce")
    events=x["events"] if "events" in x else pd.Series(index=x.index,dtype=object)
    return {
        "barrel_allowed_pct":lsa.eq(6).mean()*100,
        "hard_hit_allowed_pct":ev.ge(95).mean()*100,
        "avg_ev_allowed":ev.mean(),
        "air_pct_allowed":la.ge(10).mean()*100,
        "hr_per_bbe":events.eq("home_run").mean()*100,
        "bbe_allowed":len(x),
    }

def score_pitcher(m):
    if not m:return 50.0
    s=w=0
    for k,wt in PITCHER_WEIGHTS.items():
        v=m.get(k)
        if v is not None and not pd.isna(v):
            lo,hi=PITCHER_RANGES[k]
            s+=clip_score(v,lo,hi)*wt; w+=wt
    return s/w if w else 50.0

def pitch_mix(df,min_usage=.20):
    if df.empty or "pitch_type" not in df:return []
    mix=df.dropna(subset=["pitch_type"])["pitch_type"].value_counts(normalize=True)
    major=mix[mix>=min_usage]
    if major.empty:major=mix.head(2)
    return [(str(pt),float(u)) for pt,u in major.items()]

def pitch_match_raw(bdf,pmix):
    # V3: missing pitch-match evidence stays missing instead of receiving a
    # synthetic 50 that can become an inflated slate percentile.
    if not pmix:
        return np.nan
    scores=[]; weights=[]
    for pt,u in pmix:
        d=bdf[bdf["pitch_type"]==pt]
        n=len(d.dropna(subset=["launch_speed","launch_angle"]))
        if n<3:
            continue
        m=bbe_metrics(d,min(15,n))
        scores.append(score_batter(m)); weights.append(u)
    return float(np.average(scores,weights=weights)) if scores else np.nan

def recent_pa_count(df,cutoff):
    d=df[pd.to_datetime(df["game_date"])>=pd.Timestamp(cutoff)]
    if d.empty:return 0
    if "at_bat_number" in d.columns and "game_pk" in d.columns:
        return int(d[["game_pk","at_bat_number"]].drop_duplicates().shape[0])
    return int(len(d))

def percentile_score(series):
    # 50-99 range keeps the board intuitive while preserving relative separation.
    pct=series.rank(method="average",pct=True)
    return 50 + 49*pct

def grade_from_row(row):
    """
    V3.1: HR Score determines rank, but the displayed tier also requires
    actual Last-10-BBE contact quality. Matchup context cannot manufacture
    an ELITE label.
    """
    score = float(row["HR Score"])
    barrel = float(row["L10 Barrel%"])
    hard = float(row["L10 HardHit%"])
    avg_ev = float(row["L10 AvgEV"])
    max_ev = float(row["L10 MaxEV"])
    pull_air = float(row["L10 PullAir%"])
    pull_barrel = float(row["L10 PullBarrel%"])
    sweet = float(row["L10 SweetSpot%"])

    # Core power/contact confirmations.
    power_checks = sum([
        barrel >= 20,
        hard >= 50,
        avg_ev >= 90,
        max_ev >= 105,
        pull_air >= 30,
        pull_barrel >= 10,
        sweet >= 30,
    ])

    # ELITE: strong score plus multiple real power indicators.
    elite_core = (
        (barrel >= 20 and hard >= 50 and avg_ev >= 89)
        or (barrel >= 30 and avg_ev >= 88)
        or (hard >= 55 and avg_ev >= 91)
    )
    if score >= 90 and elite_core and power_checks >= 4:
        return "🚀 ELITE"

    # STRONG: score remains high, but contact can be a little less complete.
    strong_core = (
        (barrel >= 15 and hard >= 45 and avg_ev >= 88)
        or (barrel >= 20 and avg_ev >= 88)
        or (hard >= 55 and avg_ev >= 90)
    )
    if score >= 85 and strong_core and power_checks >= 3:
        return "🔥 STRONG"

    # GOOD: legitimate HR contact evidence, even if not fully confirmed.
    good_core = (
        barrel >= 15
        or (hard >= 50 and avg_ev >= 89)
        or (pull_barrel >= 10 and avg_ev >= 88)
    )
    if score >= 80 and good_core and power_checks >= 2:
        return "✅ GOOD"

    if score >= 75:
        return "👀 WATCH"
    return "— PASS"

def contact_flags(row):
    """V3 contact-first confirmations using the hitter's last 10 BBE."""
    checks = {
        "Barrel": row["L10 Barrel%"] >= 20,
        "HardHit": row["L10 HardHit%"] >= 50,
        "AvgEV": row["L10 AvgEV"] >= 90,
        "MaxEV": row["L10 MaxEV"] >= 105,
        "PullAir": row["L10 PullAir%"] >= 30,
        "PullBarrel": row["L10 PullBarrel%"] >= 10,
        "SweetSpot": row["L10 SweetSpot%"] >= 30,
    }
    return checks

def contact_gate(row):
    c=contact_flags(row)
    return int(sum(c.values()))

def contact_label(n):
    if n>=5:return "🔥 CRUSHING"
    if n>=4:return "✅ STRONG CONTACT"
    if n>=3:return "👀 LIVE"
    return "⚠️ WEAK CONTACT"

@st.cache_data(ttl=1800,show_spinner=False)
def build(game_date,lookback_days,projected_pool):
    schedule=get_schedule(game_date)
    if schedule.empty:return pd.DataFrame(),schedule
    end=(pd.Timestamp(game_date)-pd.Timedelta(days=1)).date()
    start=end-timedelta(days=lookback_days-1)
    sc=get_statcast(str(start),str(end))
    if sc.empty:return pd.DataFrame(),schedule
    for c in ["batter","pitcher"]:
        sc[c]=pd.to_numeric(sc[c],errors="coerce")
    sc["game_date"]=pd.to_datetime(sc["game_date"])
    recent_cutoff=end-timedelta(days=13)

    rows=[]
    for _,g in schedule.iterrows():
        sides=[
            (int(g["away_id"]),g["away"],g["home"],g["home_sp_id"],g["home_sp"]),
            (int(g["home_id"]),g["home"],g["away"],g["away_sp_id"],g["away_sp"]),
        ]
        for team_id,team,opp,spid,spname in sides:
            if spid is None or pd.isna(spid):continue
            roster=get_roster(team_id)
            roster=roster[~roster["position"].isin(["P","TWP"])].copy()

            # Build likely hitter pool from recent MLB playing time.
            play=[]
            for _,p in roster.iterrows():
                pid=int(p["player_id"])
                bdf=sc[sc["batter"]==pid]
                bbe=len(bdf.dropna(subset=["launch_speed","launch_angle"]))
                pa=recent_pa_count(bdf,recent_cutoff)
                if bbe>=10:
                    play.append((pid,p["name"],p["position"],pa,bbe))
            play=sorted(play,key=lambda z:(z[3],z[4]),reverse=True)[:projected_pool]

            spdf=sc[sc["pitcher"]==int(spid)]
            pm=pitcher_metrics(spdf)
            praw=score_pitcher(pm)
            pmix=pitch_mix(spdf)

            for pid,name,pos,pa,bbe in play:
                bdf=sc[sc["batter"]==pid]
                m10=bbe_metrics(bdf,10);m15=bbe_metrics(bdf,15)
                s10=score_batter(m10);s15=score_batter(m15)
                recent_raw=.60*s10+.40*s15
                match_raw=pitch_match_raw(bdf,pmix)
                rows.append({
                    "Batter":name,"Team":team,"Opponent":opp,"Opp SP":spname or "TBD",
                    "Recent PA (14d)":pa,"Recent BBE Available":bbe,
                    "Recent Raw":recent_raw,"Pitch Match Raw":match_raw,"Pitcher Raw":praw,
                    "SP Primary Pitches":", ".join(f"{pt} {u:.0%}" for pt,u in pmix) or "N/A",
                    "L10 Barrel%":round(pd.to_numeric(m10.get("barrel_pct",np.nan),errors="coerce"),1),
                    "L10 HardHit%":round(pd.to_numeric(m10.get("hard_hit_pct",np.nan),errors="coerce"),1),
                    "L10 AvgEV":round(pd.to_numeric(m10.get("avg_ev",np.nan),errors="coerce"),1),
                    "L10 MaxEV":round(pd.to_numeric(m10.get("max_ev",np.nan),errors="coerce"),1),
                    "L10 PullAir%":round(pd.to_numeric(m10.get("pull_air_pct",np.nan),errors="coerce"),1),
                    "L10 PullBarrel%":round(pd.to_numeric(m10.get("pull_barrel_pct",np.nan),errors="coerce"),1),
                    "L10 SweetSpot%":round(pd.to_numeric(m10.get("sweet_spot_pct",np.nan),errors="coerce"),1),
                    "L15 Barrel%":round(pd.to_numeric(m15.get("barrel_pct",np.nan),errors="coerce"),1),
                    "L15 HardHit%":round(pd.to_numeric(m15.get("hard_hit_pct",np.nan),errors="coerce"),1),
                    "L15 AvgEV":round(pd.to_numeric(m15.get("avg_ev",np.nan),errors="coerce"),1),
                    "Pitcher BarrelAllowed%":round(pd.to_numeric(pm.get("barrel_allowed_pct",np.nan),errors="coerce"),1),
                    "Pitcher HardHitAllowed%":round(pd.to_numeric(pm.get("hard_hit_allowed_pct",np.nan),errors="coerce"),1),
                    "Pitcher AvgEVAllowed":round(pd.to_numeric(pm.get("avg_ev_allowed",np.nan),errors="coerce"),1),
                    "Pitcher HR/BBE%":round(pd.to_numeric(pm.get("hr_per_bbe",np.nan),errors="coerce"),1),
                })
    out=pd.DataFrame(rows)
    if out.empty:return out,schedule

    # V3 CALIBRATION: recent hitter contact is the foundation.
    out["Recent Score"]=percentile_score(out["Recent Raw"])
    out["Pitcher Vulnerability"]=percentile_score(out["Pitcher Raw"])

    # Only percentile-rank actual pitch-match observations.
    valid_match=out["Pitch Match Raw"].notna()
    out["Pitch Match"]=50.0
    if valid_match.any():
        out.loc[valid_match,"Pitch Match"]=percentile_score(
            out.loc[valid_match,"Pitch Match Raw"]
        )
    out["Pitch Match Available"]=valid_match

    # Explicit L10 contact confirmations prevent matchup context from
    # manufacturing an ELITE hitter with weak recent contact.
    out["Contact Checks"]=out.apply(contact_gate,axis=1)
    out["Contact Form"]=out["Contact Checks"].apply(contact_label)

    # Contact-first blend: 65% hitter, 15% pitch-type, 20% opposing SP.
    base=(
        .65*out["Recent Score"]+
        .15*out["Pitch Match"]+
        .20*out["Pitcher Vulnerability"]
    )

    # Missing pitch-match data is neutral and carries a small confidence penalty.
    missing_penalty=np.where(out["Pitch Match Available"],0.0,3.0)

    # Hard contact gate. Fewer than three L10 confirmations cannot be ELITE;
    # four+ confirmations are required for a 90+ grade.
    score=base-missing_penalty
    score=np.where(out["Contact Checks"]<3,np.minimum(score,79.9),score)
    score=np.where(out["Contact Checks"]==3,np.minimum(score,87.9),score)
    score=np.where(out["Contact Checks"]==4,np.minimum(score,92.9),score)
    out["HR Score"]=np.round(score,1)
    out["Grade"]=out.apply(grade_from_row,axis=1)
    out=out.sort_values(["HR Score","Recent Raw"],ascending=False).reset_index(drop=True)
    out.insert(0,"Rank",np.arange(1,len(out)+1))
    return out,schedule

def f(v,s=""):
    return "—" if pd.isna(v) else f"{v}{s}"

def card(r):
    st.markdown(f"""
    <div class="mobile-card">
      <div class="rankline">
        <div><div class="muted">#{int(r['Rank'])} · {r['Team']} vs {r['Opponent']}</div>
        <div class="player">{r['Batter']}</div></div>
        <div class="score">{r['HR Score']}</div>
      </div>
      <div class="grade">{r['Grade']} · {r['Contact Form']} ({int(r['Contact Checks'])}/7)</div>
      <div class="muted">vs {r['Opp SP']} · {r['SP Primary Pitches']}</div>
      <div class="grid">
        <div class="box"><div class="val">{f(r['L10 Barrel%'],'%')}</div><div class="lab">L10 BARREL</div></div>
        <div class="box"><div class="val">{f(r['L10 HardHit%'],'%')}</div><div class="lab">L10 HARD HIT</div></div>
        <div class="box"><div class="val">{f(r['L10 AvgEV'])}</div><div class="lab">L10 AVG EV</div></div>
        <div class="box"><div class="val">{f(round(r['Recent Score'],1))}</div><div class="lab">RECENT</div></div>
        <div class="box"><div class="val">{f(round(r['Pitch Match'],1))}</div><div class="lab">PITCH MATCH</div></div>
        <div class="box"><div class="val">{f(round(r['Pitcher Vulnerability'],1))}</div><div class="lab">SP VULN</div></div>
      </div>
    </div>
    """,unsafe_allow_html=True)
    with st.expander("View full breakdown"):
        c1,c2,c3=st.columns(3)
        c1.metric("Recent",round(r["Recent Score"],1))
        c2.metric("Pitch",round(r["Pitch Match"],1))
        c3.metric("SP Vuln",round(r["Pitcher Vulnerability"],1))
        st.caption(f"Recent playing time: {int(r['Recent PA (14d)'])} PA-equivalents in the last 14 days")
        st.dataframe(pd.DataFrame({
            "Metric":["L10 Barrel%","L10 HardHit%","L10 Avg EV","L10 Max EV","L10 Pull-Air%","L10 Pull-Barrel%","L10 SweetSpot%",
                      "L15 Barrel%","L15 HardHit%","L15 Avg EV"],
            "Value":[r["L10 Barrel%"],r["L10 HardHit%"],r["L10 AvgEV"],r["L10 MaxEV"],r["L10 PullAir%"],
                     r["L10 PullBarrel%"],r["L10 SweetSpot%"],r["L15 Barrel%"],r["L15 HardHit%"],r["L15 AvgEV"]]
        }),hide_index=True,use_container_width=True)
        st.markdown("**Opposing pitcher**")
        st.dataframe(pd.DataFrame({
            "Metric":["Barrel Allowed%","HardHit Allowed%","Avg EV Allowed","HR/BBE%"],
            "Value":[r["Pitcher BarrelAllowed%"],r["Pitcher HardHitAllowed%"],r["Pitcher AvgEVAllowed"],r["Pitcher HR/BBE%"]]
        }),hide_index=True,use_container_width=True)

st.title("⚾ MLB HR Model V3.1")
st.caption("Contact-first daily Statcast HR opportunity rankings")

with st.expander("⚙️ Slate settings",expanded=True):
    ny_today=datetime.now(ZoneInfo("America/New_York")).date()
    selected=st.date_input("Slate date",value=ny_today)
    c1,c2=st.columns(2)
    lookback=c1.selectbox("Statcast history",[30,45,60],index=1,format_func=lambda x:f"{x} days")
    shown=c2.selectbox("Show",[5,10,15,20,30],index=1,format_func=lambda x:f"Top {x}")
    pool=st.selectbox("Likely hitter pool per team",[9,10,11,12],index=1,
                      help="Uses recent playing time to reduce bench noise. This is not a confirmed lineup.")
    minimum=st.slider("Minimum HR Score",50,99,80)
    run=st.button("🔥 BUILD TODAY'S HR BOARD",type="primary")

if run:
    with st.spinner("Building and calibrating today's Statcast board..."):
        rankings,schedule=build(str(selected),int(lookback),int(pool)) 
        lineups = get_lineup_spots(str(selected))
        if not lineups.empty:
           rankings = rankings.merge(lineups, on=["Batter", "Team"], how="left")
    if schedule.empty:
        st.warning("No MLB games found.");st.stop()
    # Collect home runs from today's slate
    homer_rows = []

    for _, game in schedule.iterrows():
        game_pk = game.get("game_pk")

        if pd.notna(game_pk):
            homer_rows.extend(get_game_homers(game_pk))

    homers_today = pd.DataFrame(homer_rows)
    st.markdown(f"### Slate — {selected.strftime('%B %d, %Y')}")
    for _, g in schedule.iterrows():
        game_time = pd.to_datetime(g["game_time"], utc=True).tz_convert("America/New_York")
        time_text = game_time.strftime("%-I:%M %p ET")

        st.markdown(
            f"<div class='note'><b>{g['away']} @ {g['home']}</b><br>"
            f"<span class='muted'>{g['away_sp'] or 'TBD'} vs {g['home_sp'] or 'TBD'} · {time_text} · {g['venue']}</span></div>",
            unsafe_allow_html=True
    )
    if rankings.empty:
        st.warning("No hitters were scored.");st.stop()

    q=rankings[rankings["HR Score"]>=minimum].head(int(shown))
    m1,m2,m3=st.columns(3)
    m1.metric("Scored",len(rankings))
    m2.metric("Qualified",len(rankings[rankings["HR Score"]>=minimum]))
    m3.metric("Top",rankings.iloc[0]["HR Score"])

    tabs=st.tabs(["🔥 Best HR Spots","🎯 Pitch Mix","🔗 Pairings","🚀 Elite","📊 Full Board","🏠 HR Tracker"])
    with tabs[0]:
        if q.empty:st.info("No hitters meet the current minimum score.")
        else:
            for _,r in q.iterrows():card(r)
    def color_pitch_match(v):
        if pd.isna(v):
            return ""
        if v >= 85:
            return "background-color:#2e7d32;color:white;font-weight:700;"
        elif v >= 75:
            return "background-color:#81c784;color:black;font-weight:700;"
        elif v >= 60:
            return "background-color:#ffd54f;color:black;font-weight:700;"
        else:
            return "background-color:#ef5350;color:white;font-weight:700;"

    def color_hr_score(v):
        if pd.isna(v):
            return ""
        if v >= 90:
            return "background-color:#66bb6a;color:black;font-weight:700;"
        elif v >= 80:
            return "background-color:#dce775;color:black;font-weight:700;"
        elif v >= 70:
            return "background-color:#ffcc80;color:black;font-weight:700;"
        else:
            return "background-color:#ff8a65;color:black;font-weight:700;"

    def color_contact(v):
        if pd.isna(v):
            return ""
        if v >= 50:
            return "background-color:#66bb6a;color:black;font-weight:700;"
        elif v >= 30:
            return "background-color:#c5e1a5;color:black;font-weight:700;"
        elif v >= 20:
            return "background-color:#fff176;color:black;font-weight:700;"
        elif v > 0:
            return "background-color:#ffe0b2;color:black;font-weight:700;"
        return ""
    def color_pitcher_vulnerability(v):
        if pd.isna(v):
            return ""
        if v >= 85:
          return "background-color:#66bb6a;color:black;font-weight:bold"
        elif v >= 75:
            return "background-color:#dce775;color:black;font-weight:bold"
        elif v >= 65:
            return "background-color:#ffcc80;color:black;font-weight:bold"
        else:
            return "background-color:#ff7043;color:black;font-weight:bold"
    with tabs[1]:
        pm=rankings.sort_values("Pitch Match",ascending=False).reset_index(drop=True)
        pm["Pitch Rank"]=range(1,len(pm)+1)
        pm["Pitch Grade"]=pm["Pitch Match"].apply(lambda x:"🔥 ELITE" if x>=85 else "🟢 STRONG" if x>=75 else "🟡 GOOD" if x>=65 else "⚪ BELOW")
        team_abbr = {
    "Arizona Diamondbacks": "ARI",
    "Athletics": "ATH",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",
    "New York Yankees": "NYY",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD",
    "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH"
}
        pm["Team"] = pm["Team"].replace(team_abbr)

        cols = ["Pitch Rank", "Batter", "Lineup Spot", "Team", "Opp SP", "Pitch Match", "HR Score", "Pitcher Vulnerability", "Pitch Grade", "L10 AvgEV", "L10 Barrel%", "L10 HardHit%"]
        cols = [c for c in cols if c in pm.columns]
        styled_pm = pm[cols].style

        if "Pitch Match" in cols:
            styled_pm = styled_pm.map(
                color_pitch_match,
                subset=["Pitch Match"]
            )

        if "Pitcher Vulnerability" in cols:
            styled_pm = styled_pm.map(
                color_pitcher_vulnerability,
                subset=["Pitcher Vulnerability"]
            )

        if "HR Score" in cols:
            styled_pm = styled_pm.map(
                color_hr_score,
                subset=["HR Score"]
            )

        for c in ["L10 Barrel%", "L10 HardHit%"]:
            if c in cols:
                styled_pm = styled_pm.map(
                    color_contact,
                    subset=[c]
                )

        st.dataframe(
            styled_pm,
            hide_index=True,
            use_container_width=True
        )
        
        with tabs[2]:
            st.subheader("🔗 Pitch Mix Pairings")
            st.caption("Pairings are generated directly from the Pitch Mix board — Pitch Match is the primary signal.")

            from itertools import combinations

            pair_pool = pm.copy()
            # Remove hitters from games that have already started or finished
            active_games = schedule[
                schedule["status"].isin([
                    "Scheduled",
                    "Pre-Game",
                    "Warmup",
                    "Delayed Start"
                 ])
             ].copy()

            active_teams = set(
                active_games["away"].replace(team_abbr)
            ).union(
                set(active_games["home"].replace(team_abbr))
            )

            pair_pool = pair_pool[
                pair_pool["Team"].isin(active_teams)
            ].copy()

            # Only use legitimate Pitch Mix HR candidates
            pair_pool = pair_pool[
                (pair_pool["Pitch Match"] >= 70) 
    
            ].copy()

            # Lineup bonus when confirmed lineup data is available
            if "Lineup Spot" in pair_pool.columns:
                def lineup_score(x):
                    try:
                        x = int(x)
                        if x in [2, 3]:
                            return 5
                        elif x in [1, 4]:
                            return 3
                        elif x == 5:
                            return 1
                        return 0
                    except:
                        return 0

                pair_pool["Lineup Bonus"] = pair_pool["Lineup Spot"].apply(lineup_score)
            else:
                pair_pool["Lineup Bonus"] = 0

            # Build individual pairing score
            pair_pool["Pair Score"] = (
                pair_pool["Pitch Match"] * 0.60 +
                pair_pool["HR Score"] * 0.20 +
                pair_pool["Pitcher Vulnerability"] * 0.10 +
                pair_pool["Lineup Bonus"] * 2
            )

            # Reward recent barrel production
            if "L10 Barrel%" in pair_pool.columns:
                pair_pool["Pair Score"] += pair_pool["L10 Barrel%"].fillna(0) * 0.10

            # Reward recent hard contact
            if "L10 HardHit%" in pair_pool.columns:
                pair_pool["Pair Score"] += pair_pool["L10 HardHit%"].fillna(0) * 0.05

            # Keep strongest Pitch Mix candidates
            pair_pool = pair_pool.sort_values(
                ["Pair Score", "Pitch Match"],
                ascending=False
            ).head(15)

            # ---------- 2-LEG PAIRINGS ----------
            pair_rows = []

            for a, b in combinations(pair_pool.to_dict("records"), 2):
                score = a["Pair Score"] + b["Pair Score"]

                # Small diversification bonus for different teams
                if a.get("Team") != b.get("Team"):
                    score += 2

                pair_rows.append({
                    "2-Leg Pairing": f'{a["Batter"]} + {b["Batter"]}',
                    "Pairing Score": round(score, 2),
                    "Teams": f'{a.get("Team", "")} / {b.get("Team", "")}'
                })

            pairs = pd.DataFrame(pair_rows)

            if not pairs.empty:
                pairs = pairs.sort_values(
                    "Pairing Score",
                    ascending=False
                ).head(35)

                st.markdown("### 🔗 Best 2-Leg Pairings")
                st.dataframe(
                    pairs,
                    hide_index=True,
                    use_container_width=True
                )
            else:
                st.info("Not enough qualifying Pitch Mix hitters for 2-leg pairings.")

            # ---------- 3-LEG PAIRINGS ----------
            triple_rows = []

            for a, b, c in combinations(pair_pool.to_dict("records"), 3):
                score = a["Pair Score"] + b["Pair Score"] + c["Pair Score"]

                teams = {a.get("Team"), b.get("Team"), c.get("Team")}
                score += (len(teams) - 1) * 2

                triple_rows.append({
                    "3-Leg Pairing": f'{a["Batter"]} + {b["Batter"]} + {c["Batter"]}',
                    "Pairing Score": round(score, 2),
                    "Teams": f'{a.get("Team", "")} / {b.get("Team", "")} / {c.get("Team", "")}'
                })

            triples = pd.DataFrame(triple_rows)

            if not triples.empty:
                triples = triples.sort_values(
                    "Pairing Score",
                    ascending=False
                ).head(35)

                st.markdown("### 🚀 Best 3-Leg Pairings")
                st.dataframe(
                    triples,
                    hide_index=True,
                    use_container_width=True
                )
            else:
                st.info("Not enough qualifying Pitch Mix hitters for 3-leg pairings.")
    with tabs[3]:
        elite=rankings[rankings["HR Score"]>=90].head(20)
        if elite.empty:st.info("No ELITE hitters on this slate.")
        else:
            for _,r in elite.iterrows():card(r)
    with tabs[4]:
        cols=["Rank","Batter","Team","Opponent","Opp SP","HR Score","Grade","Recent Score",
              "Pitch Match","Pitcher Vulnerability","Contact Checks","Contact Form","Pitch Match Available","Recent PA (14d)","L10 Barrel%","L10 HardHit%",
              "L10 AvgEV","L10 PullAir%","L10 PullBarrel%"]
        st.dataframe(rankings[cols],hide_index=True,use_container_width=True)
        st.download_button("Download full CSV",rankings.to_csv(index=False).encode(),
                           file_name=f"mlb_hr_v3_1_{selected}.csv",mime="text/csv")
    st.markdown("<div class='note'><b>V3.1:</b> HR Score ranks the board, while tier labels require raw L10 contact qualification. Pitch-type matchup and "
                "pitcher vulnerability confirm the spot rather than overpowering weak contact. The HR Score is "
                "a <b>ranking score, not HR probability</b>.</div>",unsafe_allow_html=True)
else:
    st.markdown("<div class='note'>V3 is contact-first. Hitters need real L10 contact confirmations before "
                "the model can label them GOOD, STRONG, or ELITE. Tap <b>BUILD TODAY'S HR BOARD</b> to rank the slate."
    with tabs[5]:
        st.subheader("🏠 HR Tracker")
        st.caption("Live home runs from today's slate")

        if homers_today.empty:
            st.info("No home runs recorded yet.")
        else:
            hr_counts = (
                homers_today.groupby("Batter")
                .size()
                .reset_index(name="HRs")
            )

            tracker = hr_counts.merge(
                rankings,
                on="Batter",
                how="left"
            )  

            tracker["Model Match"] = tracker["Rank"].apply(
                lambda x: "✅ In Model" if pd.notna(x) else "❌ Not Ranked"
            )

            tracker_cols = [
                "Batter",
                "HRs",
                "Rank",
                "Team",
                "Lineup Spot",
                "Pitch Match",
                "HR Score",
                "Pitcher Vulnerability",
                "Grade",
                "Model Match"
            ]

            tracker_cols = [c for c in tracker_cols if c in tracker.columns]

            st.dataframe(
                tracker[tracker_cols],
                hide_index=True,
                use_container_width=True
            )

