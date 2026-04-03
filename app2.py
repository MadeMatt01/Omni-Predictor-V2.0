import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson
from datetime import timedelta

# This must be the first Streamlit command in every page file
st.set_page_config(page_title="Omni-SubPage", layout="wide")
# --- CONFIG ---
st.set_page_config(page_title="Omni-Predictor v2.0", layout="wide", page_icon="🏆")

# --- CSS ---
st.markdown("""<style>
    .stApp { background-color: #0e1117; color: white; }
    [data-testid="stMetricValue"] { color: #00ffcc !important; font-weight: bold; }
    div.stButton > button:first-child { background-color: #00ffcc; color: #000; font-weight: bold; border-radius: 8px; }
    .footer-warning { padding: 20px; border-radius: 10px; border: 1px solid #444; background-color: rgba(255, 75, 75, 0.05); color: #888; font-size: 0.85em; text-align: center; margin-top: 50px; }
    </style>""", unsafe_allow_html=True)

# --- YOUR ORIGINAL ELITE DATABASE (RESTORED FORMAT - UNTOUCHED) ---
ELITE_DB = {
    # --- YOUR ORIGINAL ELITES ---
    "MAN CITY": [2.85, 0.9], "LIVERPOOL": [2.45, 1.1], "ARSENAL": [2.30, 0.85], 
    "ASTON VILLA": [1.95, 1.3], "TOTTENHAM": [2.10, 1.4], "MAN UNITED": [1.55, 1.5],
    "CHELSEA": [1.75, 1.4], "NEWCASTLE": [1.85, 1.3],
    "REAL MADRID": [2.55, 0.8], "BARCELONA": [2.25, 1.0], "GIRONA": [2.15, 1.2],
    "ATLETICO MADRID": [1.85, 0.9], "BILBAO": [1.65, 1.1], "SOCIEDAD": [1.45, 1.0],
    "BAYERN": [2.95, 1.2], "LEVERKUSEN": [2.35, 0.9], "STUTTGART": [2.20, 1.2],
    "DORTMUND": [2.05, 1.3], "RB LEIPZIG": [2.10, 1.1],
    "INTER": [2.40, 0.75], "AC MILAN": [2.05, 1.2], "JUVENTUS": [1.65, 0.8],
    "ROMA": [1.70, 1.2], "NAPOLI": [1.60, 1.3], "ATALANTA": [1.95, 1.1],
    "PSG": [2.50, 0.9], "MONACO": [1.90, 1.2], "LILLE": [1.65, 1.1],
    "SPORTING CP": [2.70, 0.85], "BENFICA": [2.35, 0.95], "PORTO": [2.15, 0.8],
    "BRAGA": [2.05, 1.35], "VITORIA": [1.45, 1.1],

    # --- OTHER PREMIER LEAGUE ---
    "WEST HAM": [1.45, 1.6], "BRIGHTON": [1.70, 1.5], "WOLVES": [1.25, 1.5], 
    "FULHAM": [1.35, 1.4], "CRYSTAL PALACE": [1.30, 1.4], "BRENTFORD": [1.40, 1.6],
    "EVERTON": [1.10, 1.3], "NOTTINGHAM": [1.20, 1.5], "IPSWICH": [1.15, 1.8], 
    "LEICESTER": [1.20, 1.7], "SOUTHAMPTON": [1.10, 1.9],

   # --- CHAMPIONSHIP (EXPANDED TO 30+) ---
    "LEEDS": [1.85, 1.1], "BURNLEY": [1.65, 1.0], "SHEFFIELD UTD": [1.55, 1.1],
    "LUTON": [1.50, 1.4], "MIDDLESBROUGH": [1.60, 1.3], "COVENTRY": [1.45, 1.4],
    "WEST BROM": [1.35, 1.1], "NORWICH": [1.55, 1.5], "WATFORD": [1.40, 1.5],
    "HULL CITY": [1.35, 1.5], "SUNDERLAND": [1.50, 1.3], "BRISTOL CITY": [1.25, 1.3],
    "CARDIFF": [1.15, 1.6], "SWANSEA": [1.20, 1.4], "QPR": [1.10, 1.5],
    "PRESTON": [1.20, 1.4], "BLACKBURN": [1.40, 1.7], "STOKE": [1.25, 1.5],
    "MILLWALL": [1.15, 1.2], "DERBY": [1.10, 1.3], "OXFORD UTD": [1.20, 1.7],
    "PORTSMOUTH": [1.15, 1.8], "SHEFFIELD WED": [1.25, 1.6], "PLYMOUTH": [1.20, 1.9],
    "BLACKPOOL": [1.30, 1.4], "WIGAN": [1.15, 1.3], "BIRMINGHAM": [1.55, 1.1], # League 1 giants included
    "HUDDERSFIELD": [1.35, 1.2], "ROTHERHAM": [1.10, 1.4], "CHARLTON": [1.30, 1.5],
    "CHAMPIONSHIP_GENERIC": [1.30, 1.40],

   # --- LA LIGA (OTHER) ---
    "VILLARREAL": [1.75, 1.6], "BETIS": [1.40, 1.1], "VALENCIA": [1.15, 1.3],
    "SEVILLA": [1.25, 1.4], "OSASUNA": [1.20, 1.3], "CELTA VIGO": [1.40, 1.6],
    "GETAFE": [1.05, 1.1], "ALAVES": [1.10, 1.3], "RAYO VALLECANO": [1.15, 1.4],
    "LAS PALMAS": [1.10, 1.5], "MALLORCA": [1.05, 1.1], "LEGANES": [1.05, 1.1],
    "VALLADOLID": [0.95, 1.6], "ESPANYOL": [1.10, 1.5],

    # --- LA LIGA 2 & PROMOTION CONTENDERS ---
    "EIBAR": [1.45, 1.2], "ALMERIA": [1.60, 1.5], "GRANADA": [1.50, 1.4],
    "LEVANTE": [1.35, 1.3], "ELCHE": [1.25, 1.2], "CADIZ": [1.15, 1.3],
    "OVIEDO": [1.10, 1.0], "SPORTING GIJON": [1.20, 1.2], "RACING SANTANDER": [1.40, 1.5],
    "TENERIFE": [1.00, 1.1], "ZARAGOZA": [1.15, 1.2], "MALAGA": [1.05, 1.1],
    "CASTELLON": [1.45, 1.7], "ALBACETE": [1.25, 1.5], "BURGOS": [1.10, 1.2],
    "LALIGA2_GENERIC": [1.15, 1.10],
    # --- EREDIVISIE (1st Tier) ---
    "PSV": [2.80, 1.1], "FEYENOORD": [2.20, 1.2], "AJAX": [1.90, 1.0],
    "AZ ALKMAAR": [1.70, 1.4], "TWENTE": [1.70, 1.1], "UTRECHT": [1.50, 1.1],
    "NEC NIJMEGEN": [2.10, 1.5], "HEERENVEEN": [1.75, 1.6], "GO AHEAD EAGLES": [1.60, 1.6],
    "SPARTA ROTTERDAM": [1.25, 1.7], "GRONINGEN": [1.45, 1.3], "FORTUNA SITTARD": [1.55, 1.8],
    "ZWOLLE": [1.35, 1.9], "ALMERE CITY": [1.25, 1.6], "HERACLES": [1.20, 2.3],
    "NAC BREDA": [1.05, 1.8], "WAALWIJK": [1.10, 1.7], "WILLEM II": [1.40, 1.2],

    # --- EERSTE DIVISIE (2nd Tier) ---
    "DEN HAAG": [1.85, 1.2], "CAMBUUR": [1.80, 1.3], "DE GRAAFSCHAP": [1.65, 1.4],
    "EXCELSIOR": [1.40, 1.5], "VITESSE": [1.50, 1.4], "RODA": [1.35, 1.2],
    "FC EMMEN": [1.45, 1.7], "VOLENDAM": [1.30, 1.6], "TELSTAR": [1.25, 1.5],
    "EINDHOVEN FC": [1.20, 1.5], "DORDRECHT": [1.15, 1.8], "MAASTRICHT": [1.05, 1.9],
    "DEN BOSCH": [1.40, 1.8], "HELMOND SPORT": [1.10, 1.6], "TOP OSS": [1.10, 1.7],
    
    # --- JONG TEAMS (RESERVES - HIGH VOLATILITY) ---
    "JONG PSV": [1.90, 1.8], "JONG AJAX": [1.40, 1.8], "JONG AZ": [1.55, 1.8],
    "JONG UTRECHT": [1.35, 1.7], "JONG TWENTE": [1.50, 1.9], "JONG HEERENVEEN": [1.45, 1.9],

    # --- SERIE A (EXPANDED) ---
    "LAZIO": [1.65, 1.2], "FIORENTINA": [1.55, 1.3], "BOLOGNA": [1.35, 1.1],
    "TORINO": [1.10, 1.1], "UDINESE": [1.15, 1.3], "EMPOLI": [0.95, 1.2],
    "VERONA": [1.10, 1.6], "CAGLIARI": [1.05, 1.5], "LECCE": [0.90, 1.4],
    "GENOA": [1.10, 1.3], "MONZA": [1.05, 1.3], "PARMA": [1.25, 1.5],
    "COMO": [1.20, 1.6], "VENEZIA": [1.05, 1.7],

    # --- SERIE B & PROMOTION CONTENDERS ---
    "SASSUOLO": [1.55, 1.4], "PALERMO": [1.30, 1.2], "CREMONESE": [1.25, 1.1],
    "SAMPDORIA": [1.35, 1.4], "PISA": [1.40, 1.3], "SPEZIA": [1.15, 1.1],
    "FROSINONE": [1.20, 1.5], "SALERNITANA": [1.10, 1.6], "MODENA": [1.15, 1.3],
    "BARI": [1.10, 1.2], "BRESCIA": [1.20, 1.4], "CESENA": [1.25, 1.3],
    "CATANZARO": [1.30, 1.5], "SERIEB_GENERIC": [1.10, 1.15],

   # --- BUNDESLIGA 1 (EXPANDED) ---
    "RB LEIPZIG": [2.10, 1.1], "FRANKFURT": [1.80, 1.5],
    "HOFFENHEIM": [1.75, 1.8], "FREIBURG": [1.45, 1.4], "WOLFSBURG": [1.35, 1.5],
    "GLADBACH": [1.50, 1.7], "WERDER BREMEN": [1.40, 1.6], "AUGSBURG": [1.30, 1.7],
    "MAINZ": [1.25, 1.4], "HEIDENHEIM": [1.35, 1.6], "UNION BERLIN": [1.10, 1.2],
    "BOCHUM": [1.15, 2.1], "ST PAULI": [1.15, 1.6], "KIEL": [1.10, 1.9],

    # --- BUNDESLIGA 2 & PROMOTION CONTENDERS ---
    "HAMBURGER SV": [1.85, 1.4], "SCHALKE": [1.55, 1.8], "HERTHA BERLIN": [1.70, 1.6],
    "FORTUNA DUSSELDORF": [1.60, 1.2], "COLONIA": [1.65, 1.3], "HANNOVER": [1.45, 1.2],
    "PADERBORN": [1.50, 1.5], "KARLSRUHER": [1.55, 1.5], "NURNBERG": [1.30, 1.6],
    "KAISERSLAUTERN": [1.40, 1.7], "MAGDEBURG": [1.35, 1.6], "GREUTHER FURTH": [1.30, 1.5],
    "DARMSTADT": [1.25, 1.8], "ELVERSBERG": [1.40, 1.6], "BUNDESLIGA2_GENERIC": [1.30, 1.45],

    # --- PORTUGAL (OTHER) ---
    "FAMALICAO": [1.20, 1.2], "MOREIRENSE": [1.10, 1.2], "RIO AVE": [1.05, 1.3],
    "GIL VICENTE": [1.15, 1.5], "ESTORIL": [1.10, 1.6], "BOAVISTA": [1.00, 1.5],
    
    # --- LIGA MX (1st Tier) ---
    "AMERICA": [2.10, 0.95], "TIGRES": [1.90, 0.85], "MONTERREY": [1.95, 0.90],
    "CRUZ AZUL": [1.85, 1.00], "GUADALAJARA": [1.55, 1.10], "TOLUCA": [2.20, 1.40], # High Alt High Score
    "PACHUCA": [1.70, 1.35], "PUMAS": [1.60, 1.20], "LEON": [1.45, 1.40],
    "TIGRES UANL": [1.90, 0.85], "CHIVAS": [1.55, 1.10], "SANTOS LAGUNA": [1.40, 1.55],
    "ATLAS": [1.15, 1.25], "NECAXA": [1.25, 1.45], "MAZATLAN": [1.20, 1.70],
    "JUAREZ": [1.10, 1.65], "QUERETARO": [1.05, 1.50], "PUEBLA": [1.30, 1.80],
    "TIJUANA": [1.35, 1.55], "SAN LUIS": [1.40, 1.40],

    # --- LIGA DE EXPANSIÓN (2nd Tier) ---
    "ATLANTE": [1.55, 0.95], "LEONES NEGROS": [1.40, 1.10], "CELAYA": [1.50, 1.25],
    "VENADOS": [1.35, 1.30], "MINEROS": [1.65, 1.55], "CANCUN FC": [1.25, 1.15],
    "CORRECAMINOS": [1.20, 1.45], "TLAXCALA": [1.10, 1.40], "TEPATITLAN": [1.05, 1.35],
    "MORELIA": [1.30, 1.30], "DORADOS": [1.15, 1.70], "ALEBRIJES": [1.20, 1.65],
    "MEXICO_GENERIC": [1.30, 1.35],
    
    # --- MLS EASTERN CONFERENCE ---
    "INTER MIAMI": [2.65, 1.45], "COLUMBUS CREW": [2.25, 1.10], "FC CINCINNATI": [1.85, 1.15],
    "ORLANDO CITY": [1.70, 1.25], "NEW YORK CITY": [1.65, 1.30], "NY RED BULLS": [1.55, 1.10],
    "CHARLOTTE FC": [1.35, 1.05], "PHILADELPHIA": [1.75, 1.50], "MONTREAL": [1.50, 1.75],
    "ATLANTA UNITED": [1.60, 1.65], "DC UNITED": [1.55, 1.80], "TORONTO FC": [1.30, 1.70],
    "NEW ENGLAND": [1.25, 1.85], "CHICAGO FIRE": [1.35, 1.75], "NASHVILLE SC": [1.20, 1.35],

    # --- MLS WESTERN CONFERENCE ---
    "LAFC": [2.15, 1.20], "LA GALAXY": [2.35, 1.55], "REAL SALT LAKE": [2.05, 1.40],
    "SOUNDERS": [1.60, 1.05], "HOUSTON DYNAMO": [1.45, 1.10], "COLORADO RAPIDS": [1.80, 1.65],
    "PORTLAND TIMBERS": [2.10, 1.85], "MINNESOTA UTD": [1.75, 1.60], "VANCOUVER": [1.65, 1.45],
    "AUSTIN FC": [1.30, 1.55], "FC DALLAS": [1.35, 1.50], "ST LOUIS CITY": [1.50, 1.70],
    "SPORTING KC": [1.60, 1.85], "SAN JOSE": [1.40, 2.10],

    # --- USL CHAMPIONSHIP (2nd Tier) ---
    "LOUISVILLE CITY": [1.90, 1.10], "CHARLESTON BATTERY": [1.80, 1.15], "NEW MEXICO UTD": [1.60, 1.30],
    "SACRAMENTO REPUBLIC": [1.50, 1.05], "TAMPA BAY ROWDIES": [1.70, 1.40], "DETROIT CITY": [1.20, 1.00],
    "USA_GENERIC": [1.45, 1.45],
    
    # --- TIER 1: THE ELITES (1750+ PTS) ---
    "FRANCE": [2.65, 0.75], "SPAIN": [2.50, 0.80], "ARGENTINA": [2.35, 0.70],
    "ENGLAND": [2.20, 0.85], "PORTUGAL": [2.40, 0.90], "BRAZIL": [2.15, 0.85],
    "NETHERLANDS": [2.10, 0.95], "MOROCCO": [1.90, 0.80], "BELGIUM": [2.05, 1.10],
    "GERMANY": [2.45, 1.20],

    # --- TIER 2: CONTINENTAL CONTENDERS (1600-1749 PTS) ---
    "CROATIA": [1.70, 0.90], "ITALY": [1.65, 0.85], "COLOMBIA": [1.80, 1.05],
    "SENEGAL": [1.75, 1.00], "MEXICO": [1.60, 1.15], "USA": [1.70, 1.20],
    "URUGUAY": [1.85, 0.95], "JAPAN": [1.95, 1.10], "SWITZERLAND": [1.55, 1.05],
    "DENMARK": [1.60, 1.10], "IRAN": [1.65, 1.00], "TURKIYE": [1.85, 1.35],

    # --- TIER 3: RISING FORCES (1450-1599 PTS) ---
    "NIGERIA": [1.80, 1.40], "SOUTH KOREA": [1.75, 1.30], "AUSTRALIA": [1.50, 1.25],
    "ALGERIA": [1.60, 1.15], "EGYPT": [1.55, 1.05], "CANADA": [1.65, 1.45],
    "NORWAY": [1.90, 1.55], "UKRAINE": [1.50, 1.25], "IVORY COAST": [1.70, 1.35],
    "POLAND": [1.45, 1.40], "SWEDEN": [1.65, 1.35], "SERBIA": [1.60, 1.50],
    "UZBEKISTAN": [1.40, 1.10], "MALI": [1.35, 1.15], "QATAR": [1.50, 1.40],

    # --- TIER 4: MID-RANKED (1200-1449 PTS) ---
    "IRAQ": [1.35, 1.25], "SOUTH AFRICA": [1.30, 1.20], "GHANA": [1.40, 1.50],
    "GEORGIA": [1.45, 1.40], "JAMAICA": [1.30, 1.35], "VENEZUELA": [1.25, 1.10],
    "BOLIVIA": [1.10, 1.80], "SYRIA": [1.15, 1.30], "NEW ZEALAND": [1.20, 1.40],
    "THAILAND": [1.25, 1.60], "VIETNAM": [1.10, 1.55],

    "DEFAULT": [1.35, 1.35]
}

# --- THE SCOUT ENGINE ---
@st.cache_data(ttl=timedelta(hours=12))
def fetch_scout_data(team_name, sport):
    name_up = team_name.upper()
    for key in ELITE_DB:
        if key in name_up: return ELITE_DB[key], "Elite (Manual)"
    
    if sport != "Football":
        defaults = {"Rugby": [26.2, 22.1], "American Football": [23.5, 21.0], "Tennis": [12.8, 10.5]}
        return defaults.get(sport, [1.5, 1.5]), "Standard Tier"
    
    return ELITE_DB["DEFAULT"], "Regional Fallback"

# --- SIMULATION ENGINE ---
def run_simulation(h_stats, a_stats, sport, h_adv):
    n = 10000
    h_exp = ((h_stats[0] * h_adv) + a_stats[1]) / 2
    a_exp = (a_stats[0] + (h_stats[1] / h_adv)) / 2
    
    if sport == "Football":
        return np.random.poisson(h_exp, n), np.random.poisson(a_exp, n), h_exp, a_exp
    
    vol = {"Rugby": 12.0, "American Football": 13.5, "Tennis": 3.2}
    return np.random.normal(h_exp, vol.get(sport, 10.0), n), np.random.normal(a_exp, vol.get(sport, 10.0), n), h_exp, a_exp

# --- SIDEBAR ---
with st.sidebar:
    st.title("🕹️ GLOBAL CONTROL")
    sport = st.selectbox("Select Sport", ["Football", "Rugby", "Tennis", "American Football"])
    t1_in = st.text_input("Home / P1", "Sporting CP")
    t2_in = st.text_input("Away / P2", "Real Madrid")
    h_adv_input = st.slider("Home Advantage Boost", 1.0, 1.3, 1.1)
    
    line_map = {"Football": 2.5, "Rugby": 44.5, "Tennis": 21.5, "American Football": 46.5}
    m_line = st.number_input("Main O/U Line", value=line_map[sport])
    run = st.button("🔥 RUN PLATINUM SIM")

# --- MAIN DASHBOARD ---
st.title(f"🏆 Omni-Predictor v2.0 Platinum | {sport}")

if run:
    h_stats, h_src = fetch_scout_data(t1_in, sport)
    a_stats, a_src = fetch_scout_data(t2_in, sport)
    h_sim, a_sim, h_exp, a_exp = run_simulation(h_stats, a_stats, sport, h_adv_input)
    
    st.subheader(f"📍 Intelligence: {t1_in} vs {t2_in}")
    
    # Probabilities
    m = st.columns(4)
    m[0].metric(f"{t1_in} Win", f"{np.mean(h_sim > a_sim):.1%}")
    m[1].metric(f"{t2_in} Win", f"{np.mean(a_sim > h_sim):.1%}")
    m[2].metric("Proj. Total", f"{np.mean(h_sim + a_sim):.2f}")
    m[3].metric(f"Over {m_line}", f"{np.mean((h_sim + a_sim) > m_line):.1%}")

    # TABS
    t_mat, t_half, t_markets, col2= st.tabs(["📊 Score Matrix", "⏱️ Half/Period Analytics", "📈 All Markets & Spreads", "📊 Probability Distribution"])
    
    with t_mat:

        if sport == "Football":
            max_g = 6
            matrix = np.outer(poisson.pmf(range(max_g), h_exp), poisson.pmf(range(max_g), a_exp))
            fig_mat, ax_mat = plt.subplots(figsize=(5, 2.5))
            sns.heatmap(matrix, annot=True, fmt=".1%", cmap="Greens", cbar=False, ax=ax_mat)
            ax_mat.set_xlabel(f"{t2_in} Goals")
            ax_mat.set_ylabel(f"{t1_in} Goals")
            st.pyplot(fig_mat)
        else:
            st.info("Matrix optimized for Football.")
        
    with col2:
        st.write("**Probability Distribution**")
        
        # Calculate Average Margin and Status
        avg_m = np.mean(h_sim - a_sim)
        t1_name = t1_in
        t2_name = t2_in
        
        # Determine Favoritism / Tightness
        if abs(avg_m) < 0.15:
            status = "DEAD HEAT (TIGHT)"
        elif avg_m > 0:
            status = f"FAVORED: {t1_name}"
        else:
            status = f"FAVORED: {t2_name}"
        
        # Plotting
        fig_dist, ax_dist = plt.subplots(figsize=(5, 2.5))
        fig_dist.patch.set_facecolor('#0e1117')
        ax_dist.set_facecolor('#1e2130')
        
        diff = h_sim - a_sim
        sns.kdeplot(diff, fill=True, color="#00ffcc", ax=ax_dist, label="Density")
        
        # Draw line and add text
        ax_dist.axvline(0, color='red', linestyle='--', alpha=0.7)
        ax_dist.set_title(status, color="#00ffcc", fontsize=10)
        ax_dist.set_xlabel("Margin (Home - Away)", color="white")
        ax_dist.set_ylabel("Probability Density", color="white")
        ax_dist.tick_params(colors='white')
        
        st.pyplot(fig_dist)

    with t_half:
        h1, h2 = st.columns(2)
        total_exp = h_exp + a_exp
        with h1:
            st.write("**1st Half Analytics**")
            h1_exp = total_exp * 0.45
            st.write(f"1H Expected Goals: **{h1_exp:.2f}**")
            st.write(f"1H Over 0.5: **{1 - poisson.pmf(0, h1_exp):.1%}**")
            st.write(f"1H Over 1.5: **{1 - (poisson.pmf(0, h1_exp) + poisson.pmf(1, h1_exp)):.1%}**")
        
        with h2:
            st.write("**2nd Half Analytics**")
            h2_exp = total_exp * 0.55
            st.write(f"2H Expected Goals: **{h2_exp:.2f}**")
            st.write(f"2H Over 0.5: **{1 - poisson.pmf(0, h2_exp):.1%}**")
            st.write(f"2H Over 1.5: **{1 - (poisson.pmf(0, h2_exp) + poisson.pmf(1, h2_exp)):.1%}**")

    with t_markets:
        m1, m2 = st.columns(2)
        with m1:
            st.write("**All Over/Under Markets**")
            lines = [1.5, 2.5, 3.5, 4.5] if sport == "Football" else [m_line-10, m_line, m_line+10]
            for l in lines:
                prob = np.mean((h_sim + a_sim) > l)
                st.write(f"Total Over {l}: **{prob:.1%}**")
        
        with m2:
            st.write("**Handicap / Spreads**")
            if sport == "Football":
                spreads = [-1.5, -0.5, 0.5, 1.5]
                for s in spreads:
                    label = f"Home {s}" if s < 0 else f"Home +{s}"
                    prob = np.mean((h_sim + s) > a_sim)
                    st.write(f"{label}: **{prob:.1%}**")
            else:
                s_val = -5.5
                st.write(f"Home {s_val}: **{np.mean((h_sim + s_val) > a_sim):.1%}**")

st.divider()
st.warning("⚠️ **RESPONSIBLE GAMING WARNING:** This tool provides mathematical probabilities based on historical averages and Poisson distribution. It does not guarantee results.")
st.markdown("<div class='footer-warning'>v2.0 Platinum | All Elite Data Protected | Multi-Sport Active</div>", unsafe_allow_html=True)