import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Football Match Predictor", page_icon="⚽", layout="wide")

# ---------- تحويل الأرقام والاتجاه للإنجليزي بالقوة ----------
st.markdown("""
    <style>
    /* إجبار الصفحة والمدخلات على الاتجاه من اليسار لليمين */
    html, body, [class*="css"], input, button, select {
        direction: ltr !important;
        text-align: left !important;
        font-family: 'Consolas', 'Courier New', monospace !important;
    }
    
    /* ضبط خانة العداد والأرقام */
    .stNumberInput input {
        direction: ltr !important;
        text-align: left !important;
        font-variant-numeric: lining-nums !important;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Load Assets ----------
@st.cache_resource
def load_all():
    model    = joblib.load("football_model.pkl")
    columns  = joblib.load("columns.pkl")
    defaults = joblib.load("defaults.pkl")
    ranges   = joblib.load("ranges.pkl")
    return model, columns, defaults, ranges

model, COLUMNS, DEFAULTS, RANGES = load_all()

# ---------- Find home_ / away_ Pairs ----------
PAIRS = [(c, c.replace("home_", "away_")) for c in COLUMNS
         if c.startswith("home_") and c.replace("home_", "away_") in COLUMNS]

imp = dict(zip(COLUMNS, model.feature_importances_))
PAIRS.sort(key=lambda p: imp.get(p[0], 0) + imp.get(p[1], 0), reverse=True)
TOP_PAIRS = PAIRS[:8]

def label(name):
    return name.replace("home_", "").replace("away_", "").replace("_", " ").title()

def get_range_bounds(col):
    if col in RANGES:
        r = RANGES[col]
        if isinstance(r, dict):
            min_val = float(r.get(0.0, r.get("0.0", r.get("min", min(r.values())))))
            max_val = float(r.get(1.0, r.get("1.0", r.get("max", max(r.values())))))
            return min_val, max_val
        elif isinstance(r, (tuple, list)):
            return float(r[0]), float(r[-1])
    return 0.0, 3000.0

def rand_value(col):
    if col in RANGES:
        r = RANGES[col]
        if isinstance(r, dict):
            lo = float(r.get(0.1, r.get("0.1", min(r.values()))))
            hi = float(r.get(0.9, r.get("0.9", max(r.values()))))
        elif isinstance(r, (tuple, list)):
            lo, hi = float(r[0]), float(r[-1])
        else:
            lo, hi = 0.0, 100.0
    else:
        lo, hi = 0.0, 100.0
    
    v = np.random.uniform(lo, hi)
    return round(float(v), 2)

def make_frame(values):
    row = {c: float(DEFAULTS[c]) for c in COLUMNS}
    row.update(values)
    return pd.DataFrame([row])[COLUMNS]

# ---------- State Management ----------
if "vals" not in st.session_state:
    st.session_state.vals = {c: float(DEFAULTS[c]) for c in COLUMNS}

# ---------- UI Layout ----------
st.title("⚽ Football Match Result Predictor")
st.caption("Random Forest Classifier — Home Win / Draw / Away Win")

b1, b2, _ = st.columns([1, 1, 3])
if b1.button("🎲 Random Fill", use_container_width=True):
    for h, a in TOP_PAIRS:
        st.session_state.vals[h] = rand_value(h)
        st.session_state.vals[a] = rand_value(a)
    st.rerun()

if b2.button("↺ Reset", use_container_width=True):
    st.session_state.vals = {c: float(DEFAULTS[c]) for c in COLUMNS}
    st.rerun()

st.divider()
c1, c2 = st.columns(2)

with c1:
    team1 = st.text_input("🔵 Team 1 Name", "Team A")
    st.subheader(team1)
    for h, a in TOP_PAIRS:
        min_v, max_v = get_range_bounds(h)
        # العداد اليدوي بزيادة ونقصان بمقدار 1.0 مع دعم الكتابة اليدوية
        st.session_state.vals[h] = st.number_input(
            label=f"{label(h)}", 
            min_value=min_v,
            max_value=max_v,
            value=float(st.session_state.vals[h]),
            step=1.0, 
            format="%.2f", 
            key="t1_" + h
        )

with c2:
    team2 = st.text_input("🔴 Team 2 Name", "Team B")
    st.subheader(team2)
    for h, a in TOP_PAIRS:
        min_v, max_v = get_range_bounds(a)
        st.session_state.vals[a] = st.number_input(
            label=f"{label(a)}", 
            min_value=min_v,
            max_value=max_v,
            value=float(st.session_state.vals[a]),
            step=1.0, 
            format="%.2f", 
            key="t2_" + a
        )

with st.expander("⚙️ Match Settings"):
    neutral = st.checkbox("Neutral Venue", value=True)
    if "neutral" in COLUMNS:
        st.session_state.vals["neutral"] = 1.0 if neutral else 0.0

st.divider()

# ---------- Prediction Logic ----------
if st.button("🔮 Predict Result", type="primary", use_container_width=True):
    vals = dict(st.session_state.vals)

    p1 = model.predict_proba(make_frame(vals))[0]

    swapped = dict(vals)
    for h, a in PAIRS:
        swapped[h], swapped[a] = vals[a], vals[h]
    p2 = model.predict_proba(make_frame(swapped))[0]

    cls = list(model.classes_)
    iH, iD, iA = cls.index("Home Win"), cls.index("Draw"), cls.index("Away Win")

    win1 = (p1[iH] + p2[iA]) / 2
    win2 = (p1[iA] + p2[iH]) / 2
    draw = (p1[iD] + p2[iD]) / 2
    total = win1 + win2 + draw
    win1, win2, draw = win1 / total, win2 / total, draw / total

    results = {f"{team1} Win": win1, "Draw": draw, f"{team2} Win": win2}
    best = max(results, key=results.get)

    st.success(f"🏆 **{best}** — Probability: {results[best] * 100:.1f}%")

    m1, m2, m3 = st.columns(3)
    m1.metric(f"🔵 {team1}", f"{win1 * 100:.1f}%")
    m2.metric("🤝 Draw",       f"{draw * 100:.1f}%")
    m3.metric(f"🔴 {team2}", f"{win2 * 100:.1f}%")

    st.progress(float(win1), text=f"{team1}")
    st.progress(float(draw), text="Draw")
    st.progress(float(win2), text=f"{team2}")
    st.bar_chart(pd.Series(results))

    st.info("Note: Model accuracy on test data is approximately 55–60%, which is standard for 3-way match classification.")