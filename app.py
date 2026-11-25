import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import streamlit as st
import json
import os
import streamlit.components.v1 as components

st.set_page_config(page_title="AgroPredict", page_icon="🌽", layout="wide")

COLS = ["PRECTOTCORR", "RH2M", "T2M_MAX", "T2M_MIN", "T2M_RANGE", "sin_doy", "cos_doy"]
INPUTS = ["PRECTOTCORR", "RH2M", "T2M_MAX", "T2M_MIN"]
LAT, LON = 8.5, 4.55

EXAMPLES = {
    "dry": {
        "date": pd.Timestamp("2026-03-19").date(),
        "data": {
            "PRECTOTCORR": [0, 0, 0, 0, 0, 0.3, 0, 0, 0, 0, 0, 0, 0.1, 0],
            "RH2M": [48, 45, 50, 47, 46, 52, 44, 43, 45, 44, 42, 43, 46, 41],
            "T2M_MAX": [35, 35.5, 36, 35.8, 36.2, 35, 36.5, 36.8, 37, 36.6, 37.2, 37, 36.8, 37.3],
            "T2M_MIN": [23, 23.5, 24, 23.8, 24.2, 23.6, 24.5, 24.4, 24.8, 24.6, 25, 24.7, 24.5, 25],
        },
    },
    "wet": {
        "date": pd.Timestamp("2026-06-12").date(),
        "data": {
            "PRECTOTCORR": [12, 25, 8, 0, 30, 15, 5, 22, 18, 10, 0, 28, 14, 9],
            "RH2M": [88, 90, 86, 80, 91, 89, 85, 90, 88, 87, 82, 91, 89, 86],
            "T2M_MAX": [29, 28.5, 29.5, 31, 28, 29, 30, 28.5, 29, 29.5, 30.5, 28, 29, 29.5],
            "T2M_MIN": [22, 21.8, 22.2, 22.5, 21.5, 22, 22.1, 21.7, 22, 22.3, 22.4, 21.6, 21.9, 22],
        },
    },
}

# ---------- translations ----------
# NOTE: Pidgin and Yoruba text is a first draft. Native speakers must review it.
LANGS = ["English", "Pidgin", "Yorùbá"]

TEXT = {
    "tagline": {
        "English": "Rainfall-based planting and dry-spell guidance for maize farmers in Ilorin.",
        "Pidgin": "Rain information to help maize farmers for Ilorin know when to plant and when dry spell fit come.",
        "Yorùbá": "Ìmọ̀ràn nípa òjò àti ọ̀dá fún àwọn àgbẹ̀ àgbàdo ní Ìlọrin.",
    },
    "tab_warning": {
        "English": "Dry-spell warning",
        "Pidgin": "Dry spell warning",
        "Yorùbá": "Ìkìlọ̀ ọ̀dá",
    },
    "tab_calendar": {
        "English": "Planting calendar",
        "Pidgin": "Planting calendar",
        "Yorùbá": "Kàlẹ́ńdà ogbìn",
    },
    "tab_charts": {
        "English": "Weather charts",
        "Pidgin": "Weather charts",
        "Yorùbá": "Àwòrán ojú ọjọ́",
    },
    "tab_how": {
        "English": "How it works",
        "Pidgin": "How e dey work",
        "Yorùbá": "Bí ó ṣe ń ṣiṣẹ́",
    },
    "listen": {
        "English": "Listen",
        "Pidgin": "Hear am",
        "Yorùbá": "Gbọ́",
    },
    "predict": {
        "English": "Predict",
        "Pidgin": "Check am",
        "Yorùbá": "Ṣàyẹ̀wò",
    },
    "risk_label": {
        "English": "Risk score (0 to 1)",
        "Pidgin": "Risk score (0 to 1)",
        "Yorùbá": "Odíwọ̀n ewu (0 sí 1)",
    },
    "warn_title": {
        "English": "Warning: a dry spell is likely",
        "Pidgin": "Warning: dry spell fit come",
        "Yorùbá": "Ìkìlọ̀: ó ṣeé ṣe kí ọ̀dá dé",
    },
    "warn_body": {
        "English": "A dry spell may start within the next 7 days. Consider delaying planting by a few days, or plan to protect young plants (mulch, water if available).",
        "Pidgin": "Dry spell fit start inside the next 7 days. Better wait small before you plant, or protect the young plants (cover the ground with dry grass, and water dem if you fit).",
        "Yorùbá": "Ọ̀dá lè bẹ̀rẹ̀ láàrín ọjọ́ méje tó ń bọ̀. Ẹ lè dúró díẹ̀ kí ẹ tó gbìn, tàbí kí ẹ dáàbò bo àwọn ọ̀gbìn tuntun (ẹ fi koríko gbígbẹ bo ilẹ̀, kí ẹ sì bomi rin wọ́n tí ẹ bá lè).",
    },
    "strong": {
        "English": "Strong signal.",
        "Pidgin": "The signal strong well well.",
        "Yorùbá": "Àmì náà lágbára gan-an.",
    },
    "ok_title": {
        "English": "No warning",
        "Pidgin": "No warning",
        "Yorùbá": "Kò sí ìkìlọ̀",
    },
    "ok_body": {
        "English": "A dry spell is not expected in the next 7 days. Keep checking, because conditions can change quickly.",
        "Pidgin": "We no expect dry spell for the next 7 days. Keep checking, because weather fit change quick quick.",
        "Yorùbá": "A ò retí ọ̀dá láàrín ọjọ́ méje tó ń bọ̀. Ẹ máa ṣàyẹ̀wò lẹ́ẹ̀kan sí i, nítorí ojú ọjọ́ lè yí padà kíákíá.",
    },
    "cal_title": {
        "English": "When do the rains usually start in Ilorin?",
        "Pidgin": "When rain dey usually start for Ilorin?",
        "Yorùbá": "Ìgbà wo ni òjò máa ń bẹ̀rẹ̀ ní Ìlọrin?",
    },
    "plant_q": {
        "English": "Pick a date you are thinking of planting",
        "Pidgin": "Pick the date wey you wan plant",
        "Yorùbá": "Yan ọjọ́ tí ẹ fẹ́ gbìn",
    },
    "verdict_early": {
        "English": "Before this date, the rains had NOT yet started in most years. Planting now would be early in most years.",
        "Pidgin": "Before this date, rain never start for most years. To plant now go be early for most years.",
        "Yorùbá": "Ṣáájú ọjọ́ yìí, òjò kò tíì bẹ̀rẹ̀ ní ọ̀pọ̀ ọdún. Ìgbìn ní ọjọ́ yìí máa tètè jù ní ọ̀pọ̀ ọdún.",
    },
    "verdict_mid": {
        "English": "In more than half of the years the rains had started by this date, but a good number of years were still late.",
        "Pidgin": "For more than half of the years, rain don start by this date, but plenty years rain still late.",
        "Yorùbá": "Ní ìdajì ọdún tó pọ̀ jù, òjò ti bẹ̀rẹ̀ ní ọjọ́ yìí, ṣùgbọ́n ọ̀pọ̀ ọdún ni ó pẹ́.",
    },
    "verdict_late": {
        "English": "In most years the rains had already started by this date.",
        "Pidgin": "For most years, rain don already start by this date.",
        "Yorùbá": "Ní ọ̀pọ̀ ọdún, òjò ti bẹ̀rẹ̀ ní ọjọ́ yìí.",
    },
}

# ---------- sidebar (language picked first) ----------
with st.sidebar:
    lang = st.selectbox("Language / Èdè", LANGS)

    def t(key):
        return TEXT[key][lang]

    st.header("Settings")
    mode = st.radio(
        "Alert sensitivity",
        ["Catch more dry spells (0.70)", "Fewer false alarms (0.80)"],
    )
    THRESHOLD = 0.70 if "0.70" in mode else 0.80
    st.caption(
        "A lower cut-off catches more dry spells but gives more false alarms. "
        "We chose 0.70 because a missed dry spell usually costs a farmer more "
        "than an unnecessary precaution."
    )
    st.divider()
    st.caption("Location: Ilorin (8.5°N, 4.55°E)")
    

# ---------- model ----------
class RainLSTM(nn.Module):
    def __init__(self, n_features, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.out(h[-1]).squeeze(-1)


@st.cache_resource
def load_model():
    model = RainLSTM(len(COLS))
    model.load_state_dict(torch.load("Ilorin_rain_lstm.pt", map_location="cpu"))
    model.eval()
    return model


@st.cache_data
def load_stats():
    return pd.read_csv("scaling_stats.csv", index_col=0)


@st.cache_data
def load_onset():
    return pd.read_csv("onset_history.csv", index_col=0)["onset_doy"].dropna()


def doy_to_text(d):
    return (pd.Timestamp("2001-01-01") + pd.Timedelta(days=int(round(d)) - 1)).strftime("%d %b")


def get_score(days, last_day):
    model = load_model()
    stats = load_stats()

    x = days[INPUTS].copy()
    doy = pd.date_range(end=last_day, periods=14).day_of_year.values

    x["T2M_RANGE"] = x["T2M_MAX"] - x["T2M_MIN"]
    x["PRECTOTCORR"] = np.log1p(x["PRECTOTCORR"])
    x["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    x["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    x = x[COLS]

    x = (x - stats["mean"]) / stats["std"]
    xt = torch.tensor(x.values.astype("float32")).unsqueeze(0)

    with torch.no_grad():
        return torch.sigmoid(model(xt)).item()


def fetch_weather(last_day):
    end = pd.Timestamp(last_day)
    start = end - pd.Timedelta(days=13)
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": "PRECTOTCORR,RH2M,T2M_MAX,T2M_MIN",
        "community": "AG",
        "longitude": LON,
        "latitude": LAT,
        "start": start.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
        "format": "JSON",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()["properties"]["parameter"]
    live = pd.DataFrame(data).replace(-999, np.nan)
    return live[INPUTS].reset_index(drop=True)


LANG_CODES = {"English": "en-GB", "Pidgin": "en-NG", "Yorùbá": "yo-NG"}


def listen_button(text, lang_label, button_label):
    code = LANG_CODES[lang_label]
    html = f"""
    <button id="b" style="background:#2E7D32;color:white;border:none;padding:10px 18px;
    border-radius:8px;font-size:16px;cursor:pointer;">🔊 {button_label}</button>
    <span id="m" style="margin-left:10px;font-family:sans-serif;color:#555;"></span>
    <script>
    const text = {json.dumps(text)};
    const code = {json.dumps(code)};
    const base = code.split('-')[0];
    const msg = document.getElementById('m');

    function pickVoice() {{
      const vs = window.speechSynthesis.getVoices();
      return vs.find(v => v.lang === code) || vs.find(v => v.lang.startsWith(base));
    }}

    document.getElementById('b').onclick = () => {{
      if (!('speechSynthesis' in window)) {{
        msg.textContent = 'This browser cannot read aloud.';
        return;
      }}
      window.speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(text);
      const v = pickVoice();
      if (v) {{
        u.voice = v;
        u.lang = v.lang;
        msg.textContent = '';
      }} else if (base === 'yo') {{
        msg.textContent = 'No Yoruba voice found on this device.';
        return;
      }} else {{
        u.lang = code;
      }}
      window.speechSynthesis.speak(u);
    }};
    </script>
    """
    components.html(html, height=60)

AUDIO_SLUG = {"Pidgin": "pidgin", "Yorùbá": "yoruba"}


def play_message(kind, speech, lang_label, button_label):
    slug = AUDIO_SLUG.get(lang_label)
    if slug:
        for ext in ("mp3", "wav", "m4a"):
            path = f"audio/{slug}_{kind}.{ext}"
            if os.path.exists(path):
                st.audio(path)
                return
    listen_button(speech, lang_label, button_label)

# ---------- state ----------
def set_example(name):
    st.session_state["days"] = pd.DataFrame(EXAMPLES[name]["data"])
    st.session_state["last_day"] = EXAMPLES[name]["date"]
    st.session_state["version"] += 1


if "version" not in st.session_state:
    st.session_state["version"] = 0
    st.session_state["days"] = pd.DataFrame(EXAMPLES["dry"]["data"])
    st.session_state["last_day"] = EXAMPLES["dry"]["date"]

# ---------- style ----------
st.markdown(
    """
    <style>
    .hero {
        background: linear-gradient(135deg, #1B5E20 0%, #43A047 100%);
        padding: 28px 32px; border-radius: 16px; color: white; margin-bottom: 18px;
    }
    .hero h1 { margin: 0; font-size: 2rem; color: white; }
    .hero p { margin: 6px 0 0 0; font-size: 1.05rem; opacity: 0.95; }
    .card { padding: 20px 24px; border-radius: 14px; margin-top: 10px; }
    .card h2 { margin: 0 0 6px 0; }
    .card-warn { background: #FFEBEE; border-left: 8px solid #C62828; color: #5D1010; }
    .card-ok { background: #E8F5E9; border-left: 8px solid #2E7D32; color: #123D16; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="hero">
      <h1>🌽 AgroPredict</h1>
      <p>{t("tagline")}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- weather input ----------
st.subheader("1. Choose the weather")

c1, c2, c3, c4 = st.columns([2, 1, 1, 1.4])
with c1:
    last_day = st.date_input("Date of the LAST day in the table", key="last_day")
with c2:
    st.write("")
    st.button("Dry March example", on_click=set_example, args=("dry",))
with c3:
    st.write("")
    st.button("Wet June example", on_click=set_example, args=("wet",))
with c4:
    st.write("")
    load_clicked = st.button("Load weather from NASA")

if load_clicked:
    try:
        with st.spinner("Getting weather from NASA POWER..."):
            live = fetch_weather(last_day)
        if len(live) != 14 or live.isna().any().any():
            st.error("That date is not fully available yet (data arrives a few days late). Pick an earlier date.")
        else:
            st.session_state["days"] = live
            st.session_state["version"] += 1
            st.success("Weather loaded. You can still edit the numbers.")
    except Exception as e:
        st.error(f"Could not load weather: {e}. You can type the numbers in manually.")

st.caption("Oldest day first, last day at the bottom. Click a cell to edit it.")
days = st.data_editor(
    st.session_state["days"],
    key=f"editor_{st.session_state['version']}",
    column_config={
        "PRECTOTCORR": st.column_config.NumberColumn("Rain (mm)", min_value=0.0, format="%.1f"),
        "RH2M": st.column_config.NumberColumn("Humidity (%)", min_value=0.0, max_value=100.0, format="%.0f"),
        "T2M_MAX": st.column_config.NumberColumn("Max temp (°C)", format="%.1f"),
        "T2M_MIN": st.column_config.NumberColumn("Min temp (°C)", format="%.1f"),
    },
)

# ---------- weather summary ----------
rain = days["PRECTOTCORR"].values
dates = pd.date_range(end=last_day, periods=14)

dry_run = 0
for r in reversed(rain):
    if r < 1.0:
        dry_run += 1
    else:
        break

m1, m2, m3, m4 = st.columns(4)
m1.metric("Rain in 14 days", f"{np.nansum(rain):.0f} mm")
m2.metric("Rainy days", f"{int((rain >= 1.0).sum())} of 14")
m3.metric("Dry days in a row now", dry_run)
m4.metric("Average max temp", f"{days['T2M_MAX'].mean():.1f} °C")

# ---------- tabs ----------
tab1, tab2, tab3, tab4 = st.tabs(
    [t("tab_warning"), t("tab_calendar"), t("tab_charts"), t("tab_how")]
)

with tab1:
    if st.button(t("predict"), type="primary"):
        if days.isna().any().any():
            st.error("Some cells are empty. Fill in every cell and try again.")
        else:
            if last_day.month < 3 or last_day.month > 10:
                st.warning(
                    "The model was trained on March to October only. This date is outside "
                    "that period, so treat the result with caution."
                )

            score = get_score(days, last_day)
            warn = score >= THRESHOLD

            st.progress(float(min(max(score, 0.0), 1.0)))
            st.metric(t("risk_label"), f"{score:.2f}")

            if warn:
                strong = (" " + t("strong")) if score >= 0.85 else ""
                st.markdown(
                    f"""<div class="card card-warn"><h2>⚠️ {t("warn_title")}</h2>
                    <p>{t("warn_body")}{strong}</p></div>""",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""<div class="card card-ok"><h2>✅ {t("ok_title")}</h2>
                    <p>{t("ok_body")}</p></div>""",
                    unsafe_allow_html=True,
                )
            speech = (
                t("warn_title") + ". " + t("warn_body")
                + ((" " + t("strong")) if score >= 0.85 else "")
                if warn
                else t("ok_title") + ". " + t("ok_body")
            )
            kind = ("warn_strong" if score >= 0.85 else "warn") if warn else "ok"
            play_message(kind, speech, lang, t("listen"))
            signal_date = None
            for i in range(len(rain) - 2):
                if np.nansum(rain[i:i + 3]) >= 20:
                    signal_date = dates[i + 2]
            if signal_date is not None:
                st.info(
                    f"🌱 Rain-start signal: 20 mm or more fell over 3 days ending "
                    f"{signal_date.strftime('%d %b %Y')}. This is a simple rule, not the model. "
                    "The full onset rule also needs no long dry spell afterwards."
                )

            out = days.copy()
            out.insert(0, "date", dates.date)
            out["risk_score"] = round(score, 3)
            out["warning"] = "YES" if warn else "NO"
            st.download_button(
                "Download this result (CSV)",
                out.to_csv(index=False),
                file_name="dry_spell_result.csv",
                mime="text/csv",
            )

            st.caption(
                "The score ranks risk and is not an exact probability. In tests on "
                "2013-2025 the model caught about 6 in 10 dry spells at the 0.70 cut-off, "
                "with many false alarms."
            )

with tab2:
    st.subheader(t("cal_title"))
    try:
        onset = load_onset()
        n_years = len(onset)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Median start", doy_to_text(onset.median()))
        k2.metric("Earliest on record", doy_to_text(onset.min()))
        k3.metric("Latest on record", doy_to_text(onset.max()))
        k4.metric(
            "Middle half of years",
            f"{doy_to_text(onset.quantile(0.25))} to {doy_to_text(onset.quantile(0.75))}",
        )

        st.write("**Start of the rains each year (day of the year)**")
        st.line_chart(onset)
        st.caption("Day 107 is 17 April. A higher line means a later start. The start varies a lot from year to year.")

        pick = st.date_input(
            t("plant_q"),
            value=pd.Timestamp("2026-04-20").date(),
            key="plant_date",
        )
        pick_doy = pd.Timestamp(2001, pick.month, pick.day).day_of_year
        started = int((onset <= pick_doy).sum())
        share = started / n_years * 100

        st.metric(
            f"Years when the rains had already started by {pick.strftime('%d %b')}",
            f"{started} of {n_years}",
            f"{share:.0f}% of years",
            delta_color="off",
        )
        if share < 50:
            st.warning(t("verdict_early"))
        elif share < 80:
            st.info(t("verdict_mid"))
        else:
            st.success(t("verdict_late"))

        st.caption(
            "This is a history, not a forecast for this year. The rains' start is defined as 20 mm of "
            "rain over 3 days, followed by no dry spell longer than 7 days in the next 30 days. "
            "Use it together with the dry-spell warning before you plant."
        )
    except FileNotFoundError:
        st.error("onset_history.csv was not found. Put it in the same folder as app.py.")

with tab3:
    chart_df = days.copy()
    chart_df.index = dates
    st.write("**Daily rainfall (mm)**")
    st.bar_chart(chart_df["PRECTOTCORR"])
    st.write("**Temperature (°C)**")
    st.line_chart(chart_df[["T2M_MAX", "T2M_MIN"]])

with tab4:
    st.markdown(
        """
**AgroPredict has two parts:**

1. **Planting calendar:** shows when the rains started in each year from 1981 to 2025, so a farmer can see how early or late a planting date is compared with history.
2. **Dry-spell warning:** an LSTM model reads 14 days of rainfall, humidity and temperature plus the time of year, and gives a risk score for a dry spell (8 or more days in a row with less than 1 mm of rain) starting in the next 7 days.

**How the warning was tested:** trained on 1981-2006, checked on 2007-2012, and tested on 2013-2025 (never seen in training).

**Honest results:**
- Test AUC about 0.83 (0.5 is guessing, 1.0 is perfect).
- At the 0.70 cut-off it caught about 62% of dry spells, with many false alarms.
- Most of the skill comes from the time of year. Recent weather adds a small gain, and simple models scored similarly.
- It works well in some years and poorly in others.

**Next steps:** add forecast and regional or satellite data, predict the start of the rains, and test with real farmers.
        """
    )