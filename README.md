ILORIN DRY SPELL PREDICTION - SIMPLE SUMMARY

GOAL
Warn farmers in Ilorin if a damaging dry spell (8 or more days in a row
with less than 1 mm of rain) will start in the next 7 days.

DATA
NASA POWER daily weather for Ilorin, 1981-2025 (rainfall, humidity,
temperature) plus time of year. Only the rainy season (March-October) is used.

MODEL
An LSTM (a neural network that reads sequences). It looks at the last 14 days
of weather and gives each day a score from 0 to 1: how likely a dry spell is.

HOW THE DATA WAS SPLIT (past to future, no peeking)
- Train: 1981-2006
- Validation: 2007-2012 (used to pick the best model and check it)
- Test: 2013-2025 (never used for training; the final check)

WHAT THE THRESHOLD IS
The model gives a score between 0 and 1. The threshold is the cut-off.
If the score is 0.70 or higher, the model gives a warning.
- Lower threshold = more warnings (catches more dry spells, more false alarms)
- Higher threshold = fewer warnings (fewer false alarms, misses more dry spells)
Note: 0.70 does NOT mean a 70% chance. Dry spells are rare, so the model was
trained to take them extra seriously, which pushes its scores up.

CHOSEN THRESHOLD: 0.70
A missed dry spell can cost a farmer a crop and replanting. A false alarm
usually only means waiting a few days or a small precaution. So we chose a
threshold that catches more dry spells, even with more false alarms.
(The automatic best-F1 choice was 0.80. We deliberately chose 0.70.
This is our judgement about costs, not something the data proves.)

TEST RESULTS (threshold 0.70, 3,185 test days, 204 followed by a dry spell)
- Test AUC: about 0.82 (0.5 = guessing, 1.0 = perfect)
- Dry spells warned about: 126 of 204 (about 62%)
- Dry spells missed: 78
- False alarms: 332
- When it warns, it is right about 1 time in 4
(Numbers can change slightly if the notebook is rerun.)

WHAT WE TRIED
1. Fixed the class weighting (it was upside down), so the model now takes
   dry spells seriously.
2. Added "days since last rain" and "rain in last 7 days": no real gain,
   because the model already sees the last 14 days.
3. Tried a 30-day window instead of 14: no real gain either.
So we kept the simplest version (7 inputs, 14-day window).

STRENGTHS AND WEAKNESSES
- Works well in many years (2016, 2017, 2019, 2020: AUC above 0.94).
- Fails in some years (2013 and 2014 are close to guessing; 2024 missed
  most dry spells).
- Gives false alarms even in years with no dry spells (2023, 2025).
- Uses only local weather, so it cannot see the larger weather systems
  that cause dry spells.

NEXT STEPS
- Add wider weather information (for example regional or satellite data).
- Use two alert levels: "watch" and "strong warning".
- Test with real farmer decisions to set the threshold properly.