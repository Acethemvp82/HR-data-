
# MLB HR Mobile Dashboard

A phone-friendly Streamlit dashboard for the MLB HR Savant Model.

## What you get

- Responsive mobile layout
- Large "Build Today's HR Board" button
- Card-based hitter rankings
- Expandable full breakdowns
- Last 10 BBE + Last 15 BBE
- Barrel, Hard-Hit, EV, Pull-Air, Pull-Barrel, Sweet Spot
- Opposing starter pitch mix
- Pitch-type matchup score
- Pitcher vulnerability
- CSV export

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Put it online for phone use

The simplest path is Streamlit Community Cloud:

1. Create a GitHub repository.
2. Upload `app.py` and `requirements.txt`.
3. Sign into Streamlit Community Cloud.
4. Create a new app from that GitHub repository.
5. Choose `app.py` as the entry point.
6. Deploy.
7. Open the generated HTTPS address on your Android phone.
8. In Chrome/Samsung Internet, use "Add to Home screen" if you want it to feel like an app.

No `.bat` file is needed once it is deployed.

## Model formula

Final HR Score:
- 50% recent hitter contact
- 25% pitch-type matchup
- 25% opposing starter vulnerability

Recent contact:
- Last 10 BBE: 60%
- Last 15 BBE: 40%

## Next recommended modules

- Confirmed batting order
- Park factors
- Weather
- Sportsbook HR odds
- Implied probability
- Model fair odds
- Expected value / edge
- Backtest and probability calibration
