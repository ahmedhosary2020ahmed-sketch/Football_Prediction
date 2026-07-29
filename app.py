import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Football Match Predictor",
    page_icon="⚽",
    layout="wide"
)

# ---------- Page Style ----------
st.markdown("""
    <style>
    html, body, [class*="css"], input, button, select {
        direction: ltr !important;
        text-align: left !important;
        font-family: 'Consolas', 'Courier New', monospace !important;
    }

    .stSlider {
        direction: ltr !important;
    }
    </style>
""", unsafe_allow_html=True)


# ---------- Load Assets ----------
@st.cache_resource
def load_all():
    model = joblib.load("football_model.pkl")
    columns = joblib.load("columns.pkl")
    defaults = joblib.load("defaults.pkl")
    ranges = joblib.load("ranges.pkl")

    return model, columns, defaults, ranges


model, COLUMNS, DEFAULTS, RANGES = load_all()


# ---------- Find Home / Away Feature Pairs ----------
PAIRS = [
    (c, c.replace("home_", "away_"))
    for c in COLUMNS
    if c.startswith("home_")
    and c.replace("home_", "away_") in COLUMNS
]


# ---------- Feature Importance ----------
imp = dict(zip(COLUMNS, model.feature_importances_))

PAIRS.sort(
    key=lambda p: imp.get(p[0], 0) + imp.get(p[1], 0),
    reverse=True
)

TOP_PAIRS = PAIRS[:8]


# ---------- Labels ----------
def label(name):
    return (
        name
        .replace("home_", "")
        .replace("away_", "")
        .replace("_", " ")
        .title()
    )


# ---------- Get Min / Max ----------
def get_range_bounds(col):

    if col in RANGES:

        r = RANGES[col]

        if isinstance(r, dict):

            values = [float(v) for v in r.values()]

            min_val = min(values)
            max_val = max(values)

            return min_val, max_val

        elif isinstance(r, (tuple, list)):

            return float(r[0]), float(r[-1])

    return 0.0, 100.0


# ---------- Random Value ----------
def rand_value(col):

    min_val, max_val = get_range_bounds(col)

    value = np.random.uniform(
        min_val,
        max_val
    )

    return round(float(value), 2)


# ---------- Create Model Input ----------
def make_frame(values):

    row = {
        c: float(DEFAULTS[c])
        for c in COLUMNS
    }

    row.update(values)

    return pd.DataFrame([row])[COLUMNS]


# ---------- Session State ----------
if "vals" not in st.session_state:

    st.session_state.vals = {
        c: float(DEFAULTS[c])
        for c in COLUMNS
    }


# Used to refresh sliders after Random / Reset
if "widget_version" not in st.session_state:

    st.session_state.widget_version = 0


# ---------- Title ----------
st.title("⚽ Football Match Result Predictor")

st.caption(
    "Random Forest Classifier — Home Win / Draw / Away Win"
)


# ---------- Buttons ----------
b1, b2, _ = st.columns([1, 1, 3])


# ---------- Random Fill ----------
if b1.button(
    "🎲 Random Fill",
    use_container_width=True
):

    for h, a in TOP_PAIRS:

        st.session_state.vals[h] = rand_value(h)

        st.session_state.vals[a] = rand_value(a)

    st.session_state.widget_version += 1

    st.rerun()


# ---------- Reset ----------
if b2.button(
    "↺ Reset",
    use_container_width=True
):

    st.session_state.vals = {
        c: float(DEFAULTS[c])
        for c in COLUMNS
    }

    st.session_state.widget_version += 1

    st.rerun()


st.divider()


# ---------- Team Columns ----------
c1, c2 = st.columns(2)


# =========================================================
# TEAM 1
# =========================================================

with c1:

    team1 = st.text_input(
        "🔵 Team 1 Name",
        "Team A"
    )

    st.subheader(team1)

    for h, a in TOP_PAIRS:

        min_v, max_v = get_range_bounds(h)

        st.session_state.vals[h] = st.slider(

            label=label(h),

            min_value=float(min_v),

            max_value=float(max_v),

            value=float(
                st.session_state.vals[h]
            ),

            step=1.0,

            key=f"t1_{h}_{st.session_state.widget_version}"
        )


# =========================================================
# TEAM 2
# =========================================================

with c2:

    team2 = st.text_input(
        "🔴 Team 2 Name",
        "Team B"
    )

    st.subheader(team2)

    for h, a in TOP_PAIRS:

        min_v, max_v = get_range_bounds(a)

        st.session_state.vals[a] = st.slider(

            label=label(a),

            min_value=float(min_v),

            max_value=float(max_v),

            value=float(
                st.session_state.vals[a]
            ),

            step=1.0,

            key=f"t2_{a}_{st.session_state.widget_version}"
        )


# ---------- Match Settings ----------
with st.expander("⚙️ Match Settings"):

    neutral = st.checkbox(
        "Neutral Venue",
        value=True
    )

    if "neutral" in COLUMNS:

        st.session_state.vals["neutral"] = (
            1.0 if neutral else 0.0
        )


st.divider()


# =========================================================
# PREDICTION
# =========================================================

if st.button(
    "🔮 Predict Result",
    type="primary",
    use_container_width=True
):

    vals = dict(
        st.session_state.vals
    )


    # ---------- Normal Prediction ----------
    p1 = model.predict_proba(
        make_frame(vals)
    )[0]


    # ---------- Swap Home / Away ----------
    swapped = dict(vals)

    for h, a in PAIRS:

        swapped[h], swapped[a] = (
            vals[a],
            vals[h]
        )


    # ---------- Second Prediction ----------
    p2 = model.predict_proba(
        make_frame(swapped)
    )[0]


    # ---------- Classes ----------
    cls = list(model.classes_)

    iH = cls.index("Home Win")
    iD = cls.index("Draw")
    iA = cls.index("Away Win")


    # ---------- Final Probabilities ----------
    win1 = (
        p1[iH] +
        p2[iA]
    ) / 2

    win2 = (
        p1[iA] +
        p2[iH]
    ) / 2

    draw = (
        p1[iD] +
        p2[iD]
    ) / 2


    # ---------- Normalize ----------
    total = win1 + win2 + draw

    win1 = win1 / total
    win2 = win2 / total
    draw = draw / total


    # ---------- Results ----------
    results = {

        f"{team1} Win": win1,

        "Draw": draw,

        f"{team2} Win": win2
    }


    best = max(
        results,
        key=results.get
    )


    # ---------- Winner ----------
    st.success(
        f"🏆 **{best}** — "
        f"Probability: "
        f"{results[best] * 100:.1f}%"
    )


    # ---------- Metrics ----------
    m1, m2, m3 = st.columns(3)


    m1.metric(
        f"🔵 {team1}",
        f"{win1 * 100:.1f}%"
    )


    m2.metric(
        "🤝 Draw",
        f"{draw * 100:.1f}%"
    )


    m3.metric(
        f"🔴 {team2}",
        f"{win2 * 100:.1f}%"
    )


    # ---------- Progress ----------
    st.progress(
        float(win1),
        text=team1
    )

    st.progress(
        float(draw),
        text="Draw"
    )

    st.progress(
        float(win2),
        text=team2
    )


    # ---------- Chart ----------
    st.bar_chart(
        pd.Series(results)
    )


    # ---------- Note ----------
    st.info(
        "Note: Model accuracy on test data is "
        "approximately 55–60%, which is standard "
        "for 3-way match classification."
    )