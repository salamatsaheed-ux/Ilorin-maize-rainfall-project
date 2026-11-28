<p align="center">
    <img src="maizeguard_logo.jpg" alt="MaizeGuard logo" width="520">
</p>

# MaizeGuard

Rainfall-based planting and dry-spell guidance for maize farmers in Ilorin, Kwara State, Nigeria.

**Live app:** https://maizeguard.streamlit.app

## The problem

Maize farmers in Ilorin plant when the rains start, but the start varies a lot from year to year, and a long dry spell after planting can destroy young plants. Farmers need simple, local guidance on when dry spells are likely.

## What MaizeGuard does

- **Dry-spell warning:** checks the last 14 days of weather and warns if a dry spell (8 or more days in a row with less than 1 mm of rain) may start in the next 7 days.
- **Ask MaizeGuard:** an assistant that explains the result and answers questions about maize planting in Ilorin, using a written knowledge base.
- **Local languages:** English, Pidgin and Yorùbá. Read-aloud is available for English and Pidgin text using the browser's voice.
- **Weather summary:** rain in the last 14 days, rainy days, dry days in a row, and average maximum temperature.

## How it works

- **Data:** NASA POWER daily weather for Ilorin (8.5°N, 4.55°E), 1981 to 2025: rainfall, humidity, and maximum and minimum temperature.
- **Target:** will a dry spell start in the next 7 days? The input is the previous 14 days of weather plus the time of year.
- **Model:** an LSTM neural network built with PyTorch, trained on rainy-season days (March to October).
- **Split by time, with no peeking:** training 1981 to 2006, validation 2007 to 2012, testing 2013 to 2025.
- **Alert level:** a warning is shown when the model's risk is 0.70 or higher. We chose 0.70 because a missed dry spell usually costs a farmer more than an unnecessary precaution.
- **Rains-start history:** the start of the rains is defined as at least 20 mm of rain over 3 days with no dry stretch longer than 7 days in the following 30 days. In 1981 to 2025 the median start was about 17 April, the earliest was 20 March and the latest was 9 June. In half of the years it fell between 8 April and 6 May.

## Results (test years 2013 to 2025)

- Test AUC of about 0.82 to 0.84 (0.5 is guessing, 1.0 is perfect).
- At the 0.70 alert level the model caught about 6 in 10 dry spells (126 of 204 test days), with 78 missed and 332 false-alarm days.
- Performance varies by year: strong in 2016, 2017, 2019 and 2020, and weak in 2013, 2014 and 2024.
- A simple baseline using only the time of year reached a test AUC of about 0.81. Most of the skill therefore comes from the season, and recent weather adds a small gain.
- We also tried predicting the start of the rains with the same approach. It did not beat the time-of-year baseline, so the rains-start information is shown as history and not as a forecast.

## Limitations

- It is a short-range warning: the 7 days after the last observed day. It cannot forecast months ahead.
- It catches about 6 in 10 dry spells and sometimes warns when none comes.
- It was trained on Ilorin only and on March to October only.
- It does not predict how much rain will fall.
- The assistant can make mistakes. It is told to stay within the knowledge base and never to invent forecasts.
- Pidgin and Yorùbá read-aloud depend on the voices installed on the user's device.

## Run it locally

```
pip install -r requirements.txt
streamlit run app.py
```

The assistant needs a Gemini API key. Create `.streamlit/secrets.toml` containing the line below and never commit it to GitHub:

```
GEMINI_API_KEY = "your-key-here"
```

## Project files

```
app.py                  the Streamlit app
knowledge_base.md       facts the assistant uses
Ilorin-rainfall.ipynb   data work, model training and evaluation
Ilorin_rain_lstm.pt     trained model weights
scaling_stats.csv       training mean and spread used to prepare inputs
onset_history.csv       start of the rains, 1981 to 2025
data/                   NASA POWER data for Ilorin
.streamlit/config.toml  app theme
maizeguard_logo.png     logo
```

## Data and credits

- Weather data: NASA POWER (https://power.larc.nasa.gov)
- Built with PyTorch, Streamlit and the Google Gemini API

## Roadmap

- Sign-in and a database to record usage (with user consent) so the app can improve
- SMS alerts for farmers
- Weather-forecast and regional data to extend the warning range
- Prediction of the start of the rains
- More local-language support and voice input
- Testing with real farmers in Ilorin

## Team

- Salamat Saheed Togun
- Owolabi Zainab 
- Olasehinde Jesutomi 
- Falusi Omolola