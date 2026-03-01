import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import base64
from datetime import datetime

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AutoVal — Car Price Predictor",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# THEME
# ─────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

is_dark = st.session_state.theme == "dark"

if is_dark:
    BG          = "#0A0C10"
    SURFACE     = "#12161E"
    SURFACE2    = "#1C2230"
    BORDER      = "#2A3347"
    TEXT        = "#E8EDF5"
    SUBTEXT     = "#8892A4"
    ACCENT      = "#4EFFA0"
    ACCENT2     = "#4EC9FF"
    BADGE_BG    = "#1E2A1E"
    BADGE_TEXT  = "#4EFFA0"
    SHADOW      = "rgba(0,0,0,0.6)"
    CARD_BORDER = "#2A3347"
    INPUT_BG    = "#1C2230"
else:
    BG          = "#F0F4FA"
    SURFACE     = "#FFFFFF"
    SURFACE2    = "#EBF0F8"
    BORDER      = "#D0D8E8"
    TEXT        = "#0E1420"
    SUBTEXT     = "#5A6478"
    ACCENT      = "#00C46A"
    ACCENT2     = "#0077CC"
    BADGE_BG    = "#E0F7EC"
    BADGE_TEXT  = "#007845"
    SHADOW      = "rgba(0,0,0,0.12)"
    CARD_BORDER = "#D0D8E8"
    INPUT_BG    = "#F5F8FF"

# Pre-computed glow values — avoids f-string tokenizer errors
GLOW       = "rgba(78,255,160,0.08)"  if is_dark else "rgba(0,196,106,0.07)"
GLOW_RING  = "rgba(78,255,160,0.12)"  if is_dark else "rgba(0,196,106,0.12)"
GLOW_SLIDE = "rgba(78,255,160,0.2)"   if is_dark else "rgba(0,196,106,0.2)"
GLOW_BTN   = "rgba(78,255,160,0.06)"  if is_dark else "rgba(0,196,106,0.06)"
GLOW_SHAD  = "rgba(78,255,160,0.25)"  if is_dark else "rgba(0,196,106,0.25)"
BADGE_BORD = "rgba(78,255,160,0.2)"   if is_dark else "rgba(0,196,106,0.2)"

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

*, *::before, *::after {{ box-sizing: border-box; }}

html, body, [class*="css"] {{
    font-family: sans-serif;
    color: {TEXT} !important;
}}

.stApp {{
    background-color: {BG} !important;
    background-image: radial-gradient(ellipse 80% 60% at 50% -10%, {GLOW}, transparent) !important;
    min-height: 100vh;
}}

/* ─── FULLSCREEN VIDEO — same technique as working reference ─── */
video {{
    position: fixed;
    top: 0;
    left: 0;
    min-width: 100%;
    min-height: 100%;
    object-fit: cover;
    z-index: -2;
}}

.video-overlay {{
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background: rgba(0, 0, 0, 0.6);
    z-index: -1;
}}

/* Hide Streamlit's native video chrome (controls wrapper) */
div[data-testid="stVideo"] {{
    position: fixed !important;
    top: 0 !important; left: 0 !important;
    width: 0 !important; height: 0 !important;
    overflow: visible !important;
    padding: 0 !important; margin: 0 !important;
    pointer-events: none !important;
    z-index: -3 !important;
}}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background: {SURFACE} !important;
    border-right: 1px solid {BORDER} !important;
    z-index: 100 !important;
}}
section[data-testid="stSidebar"] * {{ color: {TEXT} !important; }}
section[data-testid="stSidebar"] p  {{ color: {SUBTEXT} !important; font-size: 0.82rem !important; }}

.block-container {{
    max-width: 760px !important;
    padding: 2.5rem 2rem 4rem !important;
    position: relative;
    z-index: 10;
}}

h1, h2, h3 {{
    font-family: sans-serif !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em;
    color: {TEXT} !important;
}}

div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div {{
    background: {INPUT_BG} !important;
    border: 1.5px solid {BORDER} !important;
    border-radius: 10px !important;
    transition: border-color 0.2s;
}}
div[data-baseweb="input"] > div:focus-within,
div[data-baseweb="textarea"] > div:focus-within {{
    border-color: {ACCENT} !important;
    box-shadow: 0 0 0 3px {GLOW_RING} !important;
}}
input, textarea {{
    color: {TEXT} !important;
    font-family: sans-serif !important;
    font-size: 0.95rem !important;
}}

div[data-baseweb="select"] > div {{
    background: {INPUT_BG} !important;
    border: 1.5px solid {BORDER} !important;
    border-radius: 10px !important;
    color: {TEXT} !important;
    transition: border-color 0.2s;
}}
div[data-baseweb="select"] > div:focus-within {{ border-color: {ACCENT} !important; }}
div[data-baseweb="popover"] {{
    background: {SURFACE2} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
}}
li[role="option"] {{ color: {TEXT} !important; font-size: 0.9rem !important; }}
li[role="option"]:hover {{ background: {SURFACE} !important; }}

div[data-testid="stSlider"] .rc-slider-track {{ background: {ACCENT} !important; }}
div[data-testid="stSlider"] .rc-slider-handle {{
    border-color: {ACCENT} !important;
    background: {ACCENT} !important;
    box-shadow: 0 0 0 4px {GLOW_SLIDE} !important;
}}

label, .stSelectbox label, .stNumberInput label, .stSlider label {{
    font-family: sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    color: {SUBTEXT} !important;
    letter-spacing: 0.01em;
    text-transform: uppercase;
    margin-bottom: 4px !important;
}}

div[data-testid="stRadio"] label {{
    text-transform: none !important;
    font-size: 0.85rem !important;
    letter-spacing: 0 !important;
}}

div[data-testid="stFormSubmitButton"] > button {{
    background: linear-gradient(135deg, {ACCENT}, {ACCENT2}) !important;
    color: #0A0C10 !important;
    font-family: sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.03em;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: opacity 0.2s, transform 0.15s !important;
    box-shadow: 0 4px 24px {GLOW_SHAD} !important;
}}
div[data-testid="stFormSubmitButton"] > button:hover {{
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}}

.stButton > button {{
    background: {SURFACE2} !important;
    color: {TEXT} !important;
    border: 1.5px solid {BORDER} !important;
    border-radius: 10px !important;
    font-family: sans-serif !important;
    font-weight: 500 !important;
    padding: 0.4rem 1.1rem !important;
    transition: background 0.2s, border-color 0.2s !important;
    width: 100% !important;
}}
.stButton > button:hover {{
    border-color: {ACCENT} !important;
    background: {GLOW_BTN} !important;
}}

div[data-testid="stAlert"] {{
    border-radius: 12px !important;
    border-left-width: 4px !important;
    font-size: 0.9rem !important;
}}

details {{
    background: {SURFACE2} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 12px !important;
    padding: 0.5rem !important;
}}
details summary {{
    font-family: sans-serif !important;
    font-weight: 500 !important;
    color: {SUBTEXT} !important;
    cursor: pointer;
}}

hr {{ border-color: {BORDER} !important; margin: 2rem 0 !important; }}

button[data-testid="stNumberInputStepUp"],
button[data-testid="stNumberInputStepDown"] {{
    background: {SURFACE2} !important;
    border-color: {BORDER} !important;
    color: {TEXT} !important;
}}

::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 3px; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_models():
    models = joblib.load(os.path.join(BASE_DIR, "all_models.pkl"))
    scores = joblib.load(os.path.join(BASE_DIR, "model_scores.pkl"))
    return models, scores

try:
    all_models, model_scores = load_models()
    best_model_name = max(model_scores, key=model_scores.get)
    best_r2         = model_scores[best_model_name]
    model_loaded    = True
except FileNotFoundError:
    st.error("Model files (all_models.pkl / model_scores.pkl) not found.")
    model_loaded = False
except Exception as e:
    st.error(f"Error loading models: {e}")
    model_loaded = False

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div>"
        "</div>",
        unsafe_allow_html=True
    )

    theme_label = "Switch to Light Mode" if is_dark else "Switch to Dark Mode"
    if st.button(theme_label, use_container_width=True):
        toggle_theme()
        st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Background mode ──
    st.markdown(
        "<p style='font-size:0.72rem;font-weight:600;text-transform:uppercase;"
        "letter-spacing:0.08em;color:" + SUBTEXT + ";margin:0 0 0.4rem;'>Background</p>",
        unsafe_allow_html=True
    )
    bg_mode = st.radio(
        "Background",
        ["Default", "Static", "Dynamic"],
        index=0,
        label_visibility="collapsed"
    )

    # Audio shown only for Video + Sound
    if bg_mode == "Dynamic":
        video_path = os.path.join(BASE_DIR, "background.mp4")
        if os.path.exists(video_path):
            st.audio(video_path, loop=True, autoplay=True)
        else:
            st.warning("background.mp4 not found.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Model selector ──
    if model_loaded:
        selected_model_name = st.selectbox(
            "Select Model",
            list(all_models.keys()),
            index=list(all_models.keys()).index(best_model_name),
            help="Auto-selects highest R2 model by default"
        )
        chosen_model = all_models[selected_model_name]
        chosen_r2    = model_scores[selected_model_name]

        st.markdown("<hr>", unsafe_allow_html=True)

        st.markdown(
            "<div>"
            "<p style='font-size:0.72rem;font-weight:600;text-transform:uppercase;"
            "letter-spacing:0.08em;color:" + SUBTEXT + ";margin:0 0 0rem'>Model Stats</p>",
            unsafe_allow_html=True
        )
        for name, score in sorted(model_scores.items(), key=lambda x: -x[1]):
            bar_pct      = int(score * 100)
            is_best      = name == best_model_name
            bar_color    = ACCENT if is_best else ACCENT2
            label_color  = TEXT if is_best else SUBTEXT
            label_weight = "700" if is_best else "400"
            st.markdown(
                "<div style='margin-top:1rem'>"
                "<div style='margin-bottom:0.2rem'>"
                "<div style='display:flex;justify-content:space-between;"
                "align-items:center;margin-bottom:3px'>"
                "<span style='font-size:0.78rem;font-weight:" + label_weight +
                ";color:" + label_color + ";'>" + name + "</span>"
                "<span style='font-size:0.78rem;color:" + bar_color +
                ";font-weight:600;'>" + f"{score:.3f}" + "</span></div>"
                "<div style='background:" + BORDER + ";border-radius:4px;"
                "height:4px;overflow:hidden;'>"
                "<div style='width:" + str(bar_pct) + "%;height:100%;background:" +
                bar_color + ";border-radius:4px;'></div></div></div>",
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        chosen_model        = None
        selected_model_name = "N/A"
        chosen_r2           = 0.0

    st.markdown(
        "<div style='margin-top:1.5rem;'><p style='font-size:0.72rem;color:" +
        SUBTEXT + ";line-height:1.6;'>Supports KNN, Decision Tree, Random Forest"
        " &amp; Gradient Boosting. Inputs auto-handle missing optional fields."
        " Predictions logged to CSV for analysis.</p></div>",
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────
# BACKGROUND — image or video depending on mode
# ─────────────────────────────────────────────
if bg_mode == "Static":
    # Show background.jpg as a static fullscreen image
    img_path = os.path.join(BASE_DIR, "background.jpg")
    if os.path.exists(img_path):
        with open(img_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        st.markdown(
            "<style>"
            ".stApp {"
            "  background: url('data:image/jpg;base64," + img_b64 + "') !important;"
            "  background-size: cover !important;"
            "  background-position: center !important;"
            "  background-repeat: no-repeat !important;"
            "  background-attachment: fixed !important;"
            "  background-image: none;"  # cancel the radial gradient
            "}"
            "</style>",
            unsafe_allow_html=True
        )
    else:
        st.warning("background.jpg not found in the app directory.")

elif bg_mode == "Dynamic":
    video_path = os.path.join(BASE_DIR, "background.mp4")
    if os.path.exists(video_path):
        # Transparent app background so video shows through
        st.markdown(
            "<style>"
            ".stApp { background-color: transparent !important;"
            "         background-image: none !important; }"
            "</style>",
            unsafe_allow_html=True
        )
        # Dark overlay (same as reference code)
        st.markdown('<div class="video-overlay"></div>', unsafe_allow_html=True)
        # st.video injects <video>; bare CSS `video {}` makes it fullscreen
        st.video("background.mp4", autoplay=True, loop=True, muted=True)
    else:
        st.warning("background.mp4 not found in the app directory.")

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown(
    "<div style='text-align:center;padding:1rem 0 2.5rem;'>"
    "<h1 style='font-size:clamp(2rem,5vw,3rem);margin:0.2rem 0 0.6rem;"
    "background:linear-gradient(135deg," + TEXT + "," + SUBTEXT + ");"
    "-webkit-background-clip:text;-webkit-text-fill-color:transparent;"
    "background-clip:text;'>Car Selling Price Predictor</h1>"
    "<p style='color:" + SUBTEXT + ";font-size:1rem;font-weight:300;margin:0;"
    "max-width:420px;margin-left:auto;margin-right:auto;line-height:1.6;'>"
    "Predict your car's resale value using Machine Learning"
    "</p></div>",
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────
# FORM
# ─────────────────────────────────────────────
def section_header(title, subtitle=""):
    sub_html = (
        "<p style='font-size:0.8rem;color:" + SUBTEXT + ";margin:0;'>" + subtitle + "</p>"
        if subtitle else ""
    )
    st.markdown(
        "<div style='margin:2rem 0 1rem;'>"
        "<p style='font-family:sans-serif;font-size:1.2rem;font-weight:700;"
        "color:" + TEXT + ";margin:0 0 2px;letter-spacing:-0.01em;'>" + title + "</p>"
        + sub_html + "</div>",
        unsafe_allow_html=True
    )

with st.form("car_form", clear_on_submit=False):

    section_header("Core Details", "The essentials that affect value most")

    col1, col2 = st.columns(2)
    with col1:
        year = st.slider("Year of Manufacture", 1995, 2024, 2015,
                            help="Newer cars typically hold more value")
    with col2:
        km_driven = st.number_input(
            "Kilometers Driven",
            min_value=0, max_value=500_000, value=50_000, step=1_000,
            help="Total odometer reading in km"
        )

    col3, col4 = st.columns(2)
    with col3:
        fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG"])
    with col4:
        transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

    col5, col6 = st.columns(2)
    with col5:
        seller_type = st.selectbox("Seller Type",
                                    ["Individual", "Dealer", "Trustmark Dealer"])
    with col6:
        owner_map = {
            0: "Test / New Car",
            1: "First Owner",
            2: "Second Owner",
            3: "Third Owner",
            4: "Fourth Owner+"
        }
        owner = st.selectbox(
            "Ownership History",
            options=list(owner_map.keys()),
            format_func=lambda x: owner_map[x]
        )

    section_header("Performance & Specs", "Leave at 0 to auto-handle as missing")

    col7, col8 = st.columns(2)
    with col7:
        engine = st.number_input("Engine Capacity (CC)", min_value=0, value=0,
                                    help="e.g. 1197 for 1.2L engine")
    with col8:
        max_power = st.number_input("Max Power (bhp)", min_value=0.0, value=0.0,
                                    step=0.5, help="e.g. 82.0 bhp")

    col9, col10 = st.columns(2)
    with col9:
        seats = st.slider("Number of Seats", 2, 9, 5)
    with col10:
        mileage_unit = st.selectbox("Mileage Unit", ["kmpl", "km/kg"])

    mileage_value = st.number_input(
        "Mileage", min_value=0.0, value=0.0, step=0.5,
        help="Fuel efficiency — leave 0 if unknown"
    )

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    submit = st.form_submit_button("Calculate Resale Value", use_container_width=True)

# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
if submit and model_loaded:

    def zero_to_nan(x):
        return np.nan if x == 0 else x

    input_df = pd.DataFrame([{
        "year":          year,
        "km_driven":     km_driven,
        "fuel":          fuel,
        "seller_type":   seller_type,
        "transmission":  transmission,
        "owner":         owner,
        "engine":        zero_to_nan(engine),
        "max_power":     zero_to_nan(max_power),
        "seats":         seats,
        "mileage_value": zero_to_nan(mileage_value),
        "mileage_unit":  mileage_unit
    }])

    try:
        # Impute NaN — GradientBoosting / KNN / DecisionTree can't handle NaN natively
        NUMERIC_FILL = {
            "engine":        1248.0,
            "max_power":     82.0,
            "mileage_value": 18.0,
        }
        for col, val in NUMERIC_FILL.items():
            if col in input_df.columns:
                input_df[col] = input_df[col].fillna(val)

        prediction = chosen_model.predict(input_df)[0]
        prediction = max(0, prediction)

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Result card ──
        st.markdown(
            "<div style='background:linear-gradient(135deg," + SURFACE + "," + SURFACE2 + ");"
            "border:1px solid " + CARD_BORDER + ";border-radius:20px;"
            "padding:2rem 2rem 1.75rem;margin-bottom:1.5rem;"
            "box-shadow:0 8px 40px " + SHADOW + ";position:relative;overflow:hidden;'>"
            "<div style='position:absolute;top:-40px;right:-40px;width:160px;height:160px;"
            "background:radial-gradient(circle," + GLOW + ",transparent);"
            "border-radius:50%;'></div>"
            "<p style='font-size:0.75rem;font-weight:600;text-transform:uppercase;"
            "letter-spacing:0.1em;color:" + SUBTEXT + ";margin:0 0 0.5rem;'>"
            "Estimated Resale Value</p>"
            "<h1 style='font-family:Syne,sans-serif;font-size:clamp(2rem,6vw,3.25rem);"
            "font-weight:800;margin:0 0 0.25rem;color:" + ACCENT + ";letter-spacing:-0.03em;'>"
            "&#8377; " + f"{int(prediction):,}" + "</h1>"
            "<p style='font-size:0.82rem;color:" + SUBTEXT + ";margin:0;'>"
            "Based on " + selected_model_name + " &mdash; R&sup2; score: "
            "<span style='color:" + ACCENT2 + ";font-weight:600;'>" + f"{chosen_r2:.3f}" +
            "</span></p></div>",
            unsafe_allow_html=True
        )

        # ── Metric row ──
        age          = 2024 - year
        price_per_km = int(prediction / max(km_driven, 1))
        c1, c2, c3  = st.columns(3)

        def metric_card(label, value, note=""):
            note_html = (
                "<p style='font-size:0.7rem;color:" + SUBTEXT + ";margin:4px 0 0;'>" +
                note + "</p>" if note else ""
            )
            return (
                "<div style='background:" + SURFACE2 + ";border:1px solid " + BORDER +
                ";border-radius:14px;padding:2rem;text-align:center;'>"
                "<p style='font-size:0.7rem;text-transform:uppercase;letter-spacing:0.08em;"
                "color:" + SUBTEXT + ";margin:0 0 4px;'>" + label + "</p>"
                "<p style='font-family:Syne,sans-serif;font-size:1.3rem;font-weight:700;"
                "color:" + TEXT + ";margin:2;'>" + value + "</p>" +
                note_html + "</div>"
            )

        with c1:
            st.markdown(metric_card("Car Age", str(age) + " yrs", "from manufacture"),
                        unsafe_allow_html=True)
        with c2:
            st.markdown(metric_card("Value / km", "&#8377;" + str(price_per_km), "residual ratio"),
                        unsafe_allow_html=True)
        with c3:
            st.markdown(metric_card("Ownership", owner_map.get(owner, str(owner))),
                        unsafe_allow_html=True)

        # ── Model comparison ──
        with st.expander("Full Model Comparison"):
            comp_data = {
                "Model":    list(model_scores.keys()),
                "R2 Score": [f"{v:.4f}" for v in model_scores.values()],
                "Status":   [
                    "Best" if k == best_model_name
                    else ("Selected" if k == selected_model_name else "")
                    for k in model_scores.keys()
                ]
            }
            st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

        # ── Save to CSV ──
        record                    = input_df.copy()
        record["predicted_price"] = int(prediction)
        record["model_used"]      = selected_model_name
        record["r2_score"]        = chosen_r2
        record["timestamp"]       = datetime.now()

        csv_path = os.path.join(BASE_DIR, "user_inputs.csv")
        record.to_csv(csv_path, mode="a",
                        header=not os.path.exists(csv_path), index=False)

        st.success("Data saved.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

elif submit and not model_loaded:
    st.error("Cannot predict — model files are missing.")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown(
    "<div style='text-align:center;padding:3rem 0 1rem;color:" + SUBTEXT +
    ";font-size:0.75rem;'>"
    "Prakash"
    "&nbsp;|&nbsp;"
    "Zaid"
    "&nbsp;|&nbsp;"
    "Rimjhim"
    "&nbsp;|&nbsp;"
    "Ansh"
    "&nbsp;|&nbsp;"
    "Harsh"
    "&nbsp;|&nbsp;"
    "Anirudh"
    "</div>",
    unsafe_allow_html=True
)
