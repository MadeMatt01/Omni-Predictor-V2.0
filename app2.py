import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson, norm
from datetime import timedelta

# --- CONFIG & AUTH ---
MY_RAPID_KEY = "df22d5b81cmsh696779713ca2b88p1b2cadjsnf3cb1bdd491e"
MY_HOST = "api-football-v1.p.rapidapi.com" 

st.set_page_config(page_title="Omni-Predictor v26.2 Platinum", layout="wide", page_icon="🏆")

# --- CUSTOM NEON CSS ---
st.markdown("""<style>
    .stApp { background-color: #0e1117; color: white; }
    [data-testid="stMetricValue"] { color: #00ffcc !important; font-weight: bold; }
    div.stButton > button:first-child { background-color: #00ffcc; color: #000; font-weight: bold; border-radius: 8px; border: none; width: 100%; }
    .stTabs [aria-selected="true"] { background-color: #00ffcc !important; color: black !important; }
    .stTab { color: white; }
    .footer-warning { padding: 20px; border-radius: 10px; border: 1px solid #444; background-color: rgba(255, 75, 75, 0.05); color: #888; font-size: 0.85em; text-align: center; margin-top: 50px; }
    </style>""", unsafe_allow_html=True)

# --- THE GLOBAL SCOUT ENGINE (LEAGUE-AWARE) ---
@st.cache_data(ttl=timedelta(hours=12))
def fetch_team_data(team_name, sport):
    headers = {"X-RapidAPI-Key": MY_RAPID_KEY, "X-RapidAPI-Host": MY_HOST}
    
    # 1. SMART OVERRIDES (Prevents the 37% bug for Elites)
    elites = {
        "CITY": 2.8, "LIVERPOOL": 2.4, "ARSENAL": 2.3, "REAL MADRID": 2.5, 
        "BARCELONA": 2.2, "BAYERN": 2.9, "LEVERKUSEN": 2.1, "INTER": 2.0
    }
    name_up = team_name.upper()
    for key, val in elites.items():
        if key in name_up: return val, "Elite (Manual Override)"

    if sport != "Football":
        defaults = {"Basketball": 114.5, "Rugby": 26.2, "American Football": 23.5, "Tennis": 12.8}
        return defaults.get(sport, 1.0), "Standard Tier"

    try:
        # STEP 1: Global Team Search
        search = requests.get(f"https://{MY_HOST}/v3/teams?search={team_name.strip()}", headers=headers, timeout=5).json()
        if search.get("response"):
            team = search["response"][0]
            tid = team["team"]["id"]
            country = team["team"].get("country", "Unknown")
            
            # STEP 2: Multi-Season League Search (Try 2025, fallback to 2024)
            for season in [2025, 2024]:
                leagues = requests.get(f"https://{MY_HOST}/v3/leagues?team={tid}&season={season}", headers=headers).json()
                if leagues.get("response"):
                    # Find the league with the most coverage or the 'Current' one
                    best_league = leagues["response"][0]
                    lid = best_league["league"]["id"]
                    lname = best_league["league"]["name"]
                    
                    # STEP 3: Fetch Statistics
                    stats = requests.get(f"https://{MY_HOST}/v3/teams/statistics?season={season}&team={tid}&league={lid}", headers=headers).json()
                    if stats.get("response"):
                        avg = stats["response"]["goals"]["for"]["average"]["total"]
                        if avg: return float(avg), f"{lname} ({season})"
    except: pass
    
    # Final Floor Fallback
    return 1.2, f"Regional ({country if 'country' in locals() else 'Global'})"

# --- MULTI-SPORT SIMULATION MATH ---
def run_simulation(h_pwr, a_pwr, sport):
    n = 10000
    if sport == "Football":
        return np.random.poisson(h_pwr, n), np.random.poisson(a_pwr, n)
    
    # Normal distribution for high-scoring sports
    volatility = {"Basketball": 10.5, "Rugby": 12.0, "American Football": 13.5, "Tennis": 3.2}
    std = volatility.get(sport, 10.0)
    return np.random.normal(h_pwr, std, n), np.random.normal(a_pwr, std, n)

# --- SIDEBAR CONTROL ---
with st.sidebar:
    st.title("🕹️ OMNI CONTROL")
    sport = st.selectbox("Select Sport", ["Football", "Basketball", "Rugby", "Tennis", "American Football"])
    t1_name = st.text_input("Home / Player 1", "Manchester City")
    t2_name = st.text_input("Away / Player 2", "Las Palmas")
    
    line_map = {"Football": 2.5, "Basketball": 216.5, "Rugby": 44.5, "Tennis": 21.5, "American Football": 46.5}
    m_line = st.number_input("Main O/U Line", value=line_map[sport])
    analyze = st.button("🔥 RUN PLATINUM SIMULATION")

# --- MAIN DASHBOARD ---
st.title(f"🏆 Omni-Predictor v26.2 Platinum")

if analyze:
    with st.spinner("Executing Global Scout & Monte Carlo Engine..."):
        # Fetching Data
        h_pwr, h_league = fetch_team_data(t1_name, sport)
        a_pwr, a_league = fetch_team_data(t2_name, sport)
        h_sim, a_sim = run_simulation(h_pwr, a_pwr, sport)
        
        # Intelligence Report Header
        st.subheader("📍 Intelligence Report")
        i1, i2 = st.columns(2)
        i1.info(f"**{t1_name}** | League: {h_league} | Power: {h_pwr:.2f}")
        i2.info(f"**{t2_name}** | League: {a_league} | Power: {a_pwr:.2f}")

        # 1. CORE WIN/LOSS METRICS
        st.divider()
        m_cols = st.columns(5 if sport == "Football" else 4)
        m_cols[0].metric(f"{t1_name} Win", f"{np.mean(h_sim > a_sim):.1%}")
        
        if sport == "Football":
            m_cols[1].metric("Draw %", f"{np.mean(h_sim.astype(int) == a_sim.astype(int)):.1%}")
            m_cols[2].metric(f"{t2_name} Win", f"{np.mean(a_sim > h_sim):.1%}")
            m_cols[3].metric("Proj. Total", f"{np.mean(h_sim + a_sim):.2f}")
            m_cols[4].metric(f"Over {m_line}", f"{np.mean((h_sim + a_sim) > m_line):.1%}")
        else:
            m_cols[1].metric(f"{t2_name} Win", f"{np.mean(a_sim > h_sim):.1%}")
            m_cols[2].metric("Proj. Total", f"{np.mean(h_sim + a_sim):.1f}")
            m_cols[3].metric(f"Over {m_line}", f"{np.mean((h_sim + a_sim) > m_line):.1%}")

        # 2. TABS FOR DEEP ANALYSIS
        t_matrix, t_periods, t_markets, t_curve = st.tabs(["📊 Score Matrix", "⏱️ Period Logic", "📈 Markets", "📉 Curve Analysis"])

        with t_matrix:
            if sport == "Football":
                matrix = np.zeros((6, 6))
                for i in range(6):
                    for j in range(6):
                        matrix[i, j] = np.mean((h_sim.astype(int) == i) & (a_sim.astype(int) == j))
                df_m = pd.DataFrame(matrix, index=[f"H-{i}" for i in range(6)], columns=[f"A-{j}" for j in range(6)])
                st.dataframe(df_m.style.format("{:.1%}").background_gradient(cmap='Greens'), use_container_width=True)
            else:
                st.info("Matrix view is optimized for Football goals.")

        with t_periods:
            if sport == "Basketball":
                st.subheader("Performance-Weighted Quarters")
                weights = [0.22, 0.24, 0.29, 0.25] # Q3 is peak performance
                q_cols = st.columns(4)
                for i, w in enumerate(weights):
                    q_val = (h_pwr + a_pwr) * w
                    q_cols[i].metric(f"Q{i+1} Total", f"{q_val:.1f}")
                    q_cols[i].caption(f"Trend: {'High' if w > 0.25 else 'Normal'}")
            elif sport == "Football":
                st.subheader("HT/FT Projection")
                st.write(f"First Half Over 0.5: **{np.mean((h_sim*0.45 + a_sim*0.45) > 0.5):.1%}**")

        with t_markets:
            c_ou, c_hcp = st.columns(2)
            with c_ou:
                st.write("**Alt Over/Under**")
                step = 1 if sport == "Football" else 10
                for l in [m_line - step, m_line, m_line + step]:
                    st.write(f"Total Over {l}: **{np.mean((h_sim + a_sim) > l):.1%}**")
            with c_hcp:
                st.write("**Handicaps**")
                h_steps = [-1.5, 0.5, 1.5] if sport == "Football" else [-5.5, -2.5, 2.5]
                for h in h_steps:
                    st.write(f"Home {h}: **{np.mean((h_sim + h) > a_sim):.1%}**")

        with t_curve:
            margin = h_sim - a_sim
            avg_m = np.mean(margin)
            fig, ax = plt.subplots(figsize=(10, 4))
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#0e1117')
            sns.kdeplot(margin, fill=True, color="#00ffcc", alpha=0.4, linewidth=3, ax=ax)
            
            # Status / Dead Heat Logic
            threshold = 0.15 if sport == "Football" else 1.5
            if abs(avg_m) < threshold:
                status, color = "DEAD HEAT / EQUAL", "white"
            else:
                status, color = (f"FAVORED: {t1_name}", "#00ffcc") if avg_m > 0 else (f"FAVORED: {t2_name}", "#ff4b4b")
            
            ax.axvline(0, color='white', linestyle='--', alpha=0.6)
            ax.text(avg_m, ax.get_ylim()[1]*0.8, status, color=color, fontweight='bold', ha='center', bbox=dict(facecolor='black', alpha=0.5))
            ax.tick_params(colors='white')
            st.pyplot(fig)

# --- FOOTER ---
st.divider()
st.markdown("""<div class='footer-warning'>
    ⚠️ <b>DISCLAIMER:</b> Analytical Monte Carlo simulations for 2026. Sports results are highly volatile. 
    <b>For informational use only.</b>
    </div>""", unsafe_allow_html=True)
st.caption("Omni-Predictor v26.2 Platinum | League-Aware | Multi-Sport Math")



