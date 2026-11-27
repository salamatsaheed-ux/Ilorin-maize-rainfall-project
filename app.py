import json
import time

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import streamlit as st
import streamlit.components.v1 as components
from google import genai
from google.genai import types

st.set_page_config(page_title="MaizeGuard", page_icon="🌽", layout="wide")

GEMINI_MODELS = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
THRESHOLD = 0.70

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
# Pidgin and Yoruba lines marked NEW were written by the assistant and must be checked by a native speaker.
LANGS = ["English", "Pidgin", "Yorùbá"]

TEXT = {
    "tagline": {
        "English": "Rainfall-based planting and dry-spell guidance for maize farmers in Ilorin.",
        "Pidgin": "Rain information wey fit help maize farmers for Ilorin to sabi when to plant and when dry spell fit come.",
        "Yorùbá": "Ìmọ̀ràn nípa òjò àti ọ̀dá fún àwọn àgbẹ̀ àgbàdo ní Ìlọrin.",
    },
    "tab_warning": {
        "English": "Dry-spell warning",
        "Pidgin": "Dry spell warning",
        "Yorùbá": "Ìkìlọ̀ ọ̀dá",
    },
    "tab_ask": {  # NEW
        "English": "Ask MaizeGuard",
        "Pidgin": "Ask MaizeGuard",
        "Yorùbá": "Béèrè lọ́wọ́ MaizeGuard",
    },
    "date_label": {  # NEW
        "English": "Date to check",
        "Pidgin": "Date wey you wan check",
        "Yorùbá": "Ọjọ́ tí ẹ fẹ́ ṣàyẹ̀wò",
    },
    "load_btn": {  # NEW
        "English": "Load weather",
        "Pidgin": "Load weather",
        "Yorùbá": "Gbé ojú ọjọ́ wọlé",
    },
    "loaded": {  # NEW
        "English": "Weather loaded.",
        "Pidgin": "Weather don load.",
        "Yorùbá": "A ti gbé ojú ọjọ́ wọlé.",
    },
    "not_ready": {  # NEW
        "English": "Weather for this date is not ready yet. Please pick an earlier date.",
        "Pidgin": "Weather for this date never ready. Abeg pick earlier date.",
        "Yorùbá": "Ojú ọjọ́ ọjọ́ yìí kò tíì wà. Ẹ jọ̀wọ́, ẹ yan ọjọ́ tó ṣáájú.",
    },
    "load_failed": {  # NEW
        "English": "Could not load the weather. Please check your internet and try again.",
        "Pidgin": "We no fit load the weather. Abeg check your network make you try again.",
        "Yorùbá": "A kò lè gbé ojú ọjọ́ wọlé. Ẹ jọ̀wọ́, ẹ ṣàyẹ̀wò íńtánẹ́ẹ̀tì yín, kí ẹ tún gbìyànjú.",
    },
    "predict": {
        "English": "Predict",
        "Pidgin": "Check am",
        "Yorùbá": "Ṣàyẹ̀wò",
    },
    "risk_label": {  # NEW wording
        "English": "Dry-spell risk (0 to 1)",
        "Pidgin": "Dry spell risk (0 to 1)",
        "Yorùbá": "Ewu ọ̀dá (0 sí 1)",
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
    "off_season": {  # NEW wording (calendar tab removed)
        "English": "This date is outside the rainy season (March to October), when the warning works. Dry weather is normal at this time of year, so no warning is given. You can ask MaizeGuard when the rains usually start.",
        "Pidgin": "This date no dey inside the rainy season (March to October) wey the warning dey work. Dry weather normal for this time, so we no dey give warning. You fit ask MaizeGuard when rain dey usually start.",
        "Yorùbá": "Ọjọ́ yìí kò sí nínú àsìkò òjò (Oṣù Kẹta sí Oṣù Kẹwàá) tí ìkìlọ̀ ti ń ṣiṣẹ́. Ó jẹ́ àdáyébá kí ojú ọjọ́ gbẹ ní àkókò yìí, nítorí náà a kò fi ìkìlọ̀ kankan hàn. Ẹ lè béèrè lọ́wọ́ MaizeGuard ìgbà tí òjò máa ń bẹ̀rẹ̀.",
    },
    "guide_note": {  # NEW
        "English": "This is a guide, not a promise. Weather can change, so check again in a few days.",
        "Pidgin": "Na guide e be, no be promise. Weather fit change, so check again after few days.",
        "Yorùbá": "Ìtọ́sọ́nà ni èyí, kì í ṣe ìlérí. Ojú ọjọ́ lè yí padà, nítorí náà ẹ tún ṣàyẹ̀wò lẹ́yìn ọjọ́ díẹ̀.",
    },
    "listen": {
        "English": "Listen",
        "Pidgin": "Hear am",
        "Yorùbá": "Gbọ́",
    },
}

# ---------- sidebar: language only ----------
with st.sidebar:
    lang = st.selectbox("Language / Èdè", LANGS)


def t(key):
    return TEXT[key][lang]


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


# ---------- read-aloud ----------
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


# ---------- assistant ----------
def ask_assistant(history, question, lang_label):
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        return "The assistant is not set up yet (missing API key)."

    r = st.session_state.get("last_result")
    if r:
        result_text = (
            f"Latest dry-spell check: risk score {r['score']:.2f}, cut-off {r['threshold']:.2f}, "
            f"{'WARNING' if r['warn'] else 'no warning'}. Last weather day: {r['last_day']}. "
            f"Rain in the 14 days: {r['rain_total']:.0f} mm. "
            f"Dry days in a row now: {r['dry_run']}."
        )
    else:
        result_text = "The farmer has not run a dry-spell check yet."

    onset = load_onset()
    calendar_text = (
        f"History (1981-2025): the rains in Ilorin started on median {doy_to_text(onset.median())}, "
        f"earliest {doy_to_text(onset.min())}, latest {doy_to_text(onset.max())}. "
        f"Middle half of years: {doy_to_text(onset.quantile(0.25))} to {doy_to_text(onset.quantile(0.75))}."
    )

    system = (
        "You are the MaizeGuard Assistant, helping maize farmers in Ilorin, Nigeria understand "
        "rainfall guidance in simple words. "
        f"Reply in {lang_label} (if Pidgin, use simple Nigerian Pidgin; if Yoruba, use simple Yoruba). "
        "Keep answers short and clear. "
        "Only explain the information given below and general maize planting practice. "
        "Never invent weather forecasts or numbers. "
        "Never describe a warning as certain, and never say a dry spell will definitely happen. "
        "Do not add disclaimers about the model or repeat accuracy figures unless the farmer asks. "
        "If the farmer asks how reliable it is, answer honestly: the model catches about 6 in 10 "
        "dry spells and gives many false alarms. "
        "The app covers Ilorin only. If you are unsure, or the question is about something else "
        "(pesticides, loans, other places), say so and suggest talking to a local agricultural extension officer. "
        "Do not give medical or legal advice.\n\n"
        f"{result_text}\n{calendar_text}"
    )

    contents = [
        types.Content(
            role="user" if m["role"] == "user" else "model",
            parts=[types.Part(text=m["text"])],
        )
        for m in history
    ]
    contents.append(types.Content(role="user", parts=[types.Part(text=question)]))

    friendly = {
        "English": "The assistant is busy right now. Please try again in a minute.",
        "Pidgin": "The assistant dey busy now. Abeg try again after small time.",
        "Yorùbá": "Olùrànlọ́wọ́ náà ń ṣiṣẹ́ lọ́wọ́ báyìí. Ẹ jọ̀wọ́, ẹ gbìyànjú lẹ́ẹ̀kan sí i láìpẹ́.",
    }

    client = genai.Client(api_key=api_key)
    for model_name in GEMINI_MODELS:
        for attempt in range(2):
            try:
                resp = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(system_instruction=system),
                )
                return resp.text or "Sorry, I could not answer that."
            except Exception as e:
                print("Assistant error:", model_name, e)
                if ("503" in str(e) or "UNAVAILABLE" in str(e)) and attempt == 0:
                    time.sleep(2)
                    continue
                break
    return friendly.get(lang_label, friendly["English"])


# ---------- state ----------
def set_example(name):
    st.session_state["days"] = pd.DataFrame(EXAMPLES[name]["data"])
    st.session_state["last_day"] = EXAMPLES[name]["date"]


if "days" not in st.session_state:
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
      <h1>🌽 MaizeGuard</h1>
      <p>{t("tagline")}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- 1. weather information ----------
st.subheader("1. Weather information")

c1, c2, c3, c4 = st.columns([2, 1, 1, 1.2])
with c1:
    last_day = st.date_input(
        t("date_label"),
        key="last_day",
        max_value=(pd.Timestamp.today() - pd.Timedelta(days=3)).date(),
    )
with c2:
    st.write("")
    st.button("Dry March example", on_click=set_example, args=("dry",))
with c3:
    st.write("")
    st.button("Wet June example", on_click=set_example, args=("wet",))
with c4:
    st.write("")
    load_clicked = st.button(t("load_btn"))

if load_clicked:
    try:
        with st.spinner("..."):
            live = fetch_weather(last_day)
        if len(live) != 14 or live.isna().any().any():
            st.error(t("not_ready"))
        else:
            st.session_state["days"] = live
            st.success(t("loaded"))
    except Exception as e:
        print("Weather error:", e)
        st.error(t("load_failed"))

# ---------- 2. weather summary ----------
days = st.session_state["days"]
rain = days["PRECTOTCORR"].values
dates = pd.date_range(end=last_day, periods=14)

dry_run = 0
for r in reversed(rain):
    if r < 1.0:
        dry_run += 1
    else:
        break

st.subheader("2. Weather summary")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Rain in 14 days", f"{np.nansum(rain):.0f} mm")
m2.metric("Rainy days", f"{int((rain >= 1.0).sum())} of 14")
m3.metric("Dry days in a row now", dry_run)
m4.metric("Average max temperature", f"{days['T2M_MAX'].mean():.1f} °C")

# ---------- 3. guidance ----------
st.subheader("3. Your guidance")
tab1, tab2 = st.tabs([t("tab_warning"), t("tab_ask")])

with tab1:
    if st.button(t("predict"), type="primary"):
        if days.isna().any().any():
            st.error(t("load_failed"))
        elif last_day.month < 3 or last_day.month > 10:
            st.session_state["last_result"] = None
            st.info("🌤️ " + t("off_season"))
        else:
            score = get_score(days, last_day)
            warn = score >= THRESHOLD

            st.session_state["last_result"] = {
                "score": score,
                "threshold": THRESHOLD,
                "warn": warn,
                "last_day": str(last_day),
                "rain_total": float(np.nansum(rain)),
                "dry_run": dry_run,
            }

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
            if lang == "Yorùbá":
                st.caption("🔇 Read-aloud is not available in Yorùbá yet.")
            else:
                listen_button(speech, lang, t("listen"))

            signal_date = None
            for i in range(len(rain) - 2):
                if np.nansum(rain[i:i + 3]) >= 20:
                    signal_date = dates[i + 2]
            if signal_date is not None:
                st.info(
                    f"🌱 Good rain: 20 mm or more fell over 3 days ending "
                    f"{signal_date.strftime('%d %b %Y')}."
                )

            st.caption(t("guide_note"))

with tab2:
    st.subheader("MaizeGuard Assistant")
    st.caption(
        "Ask about your result or maize planting in Ilorin. "
        "For example: \"When do the rains usually start?\" "
        "The assistant can make mistakes, so use it together with the warning. "
        "Please do not type personal details."
    )

    if "chat" not in st.session_state:
        st.session_state["chat"] = []

    for m in st.session_state["chat"]:
        with st.chat_message(m["role"]):
            st.write(m["text"])

    question = st.chat_input("Ask MaizeGuard...")
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.spinner("Thinking..."):
            answer = ask_assistant(st.session_state["chat"], question, lang)
        with st.chat_message("assistant"):
            st.write(answer)
        st.session_state["chat"].append({"role": "user", "text": question})
        st.session_state["chat"].append({"role": "assistant", "text": answer})

# ---------- about ----------
st.divider()
st.markdown("**About MaizeGuard**")
st.write(
    "MaizeGuard helps maize farmers in Ilorin plan around dry spells. "
    "It looks at the last 14 days of weather and checks whether 8 or more days with almost no rain "
    "may start in the next 7 days."
)
st.write(
    "It learned from weather records for Ilorin from 1981 to 2025 (NASA POWER data) and was tested "
    "on the years 2013 to 2025."
)
st.write(
    "It is a guide, not a promise. It catches about 6 in 10 dry spells, and sometimes it gives a "
    "warning when no dry spell comes. It works for Ilorin only, from March to October."
)