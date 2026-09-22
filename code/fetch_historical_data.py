"""
Fetch real historical exchange rate data for the triangular arbitrage pipeline.

This is a standalone addition to the project: it pulls hourly EUR/USD,
USD/JPY and EUR/JPY quotes from the Twelve Data API, all in UTC, so the
three legs are genuinely simultaneous market observations rather than
reference rates fixed by different institutions at different times of day.
Each pair is independently quoted by the market (none is derived from the
other two), which is what makes a triangular-arbitrage check meaningful.

Requires a free Twelve Data API key (https://twelvedata.com/pricing):
set the TWELVE_DATA_API_KEY environment variable, or put it in a .env file
in the project root as TWELVE_DATA_API_KEY=your_key (see .env.example).
The key is never written into this file or into the output CSV.
"""

import io
import json
import os
import urllib.request
import urllib.parse

import pandas as pd

API_URL = "https://api.twelvedata.com/time_series"
PAIRS = ["EUR/USD", "USD/JPY", "EUR/JPY"]


def _load_dotenv(path=".env"):
    """Minimal .env loader: sets os.environ from KEY=VALUE lines, without
    overriding a variable that is already set in the environment."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def get_api_key():
    _load_dotenv()
    key = os.environ.get("TWELVE_DATA_API_KEY")
    if not key:
        raise RuntimeError(
            "TWELVE_DATA_API_KEY not set. Get a free key at "
            "https://twelvedata.com/pricing and set it as an environment "
            "variable, or put it in a .env file (see .env.example)."
        )
    return key


def fetch_pair(symbol, api_key, interval="1h", outputsize=5000):
    """Fetch one pair's hourly series (UTC) as a Date-indexed Series of closes."""
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "timezone": "UTC",
        "apikey": api_key,
    }
    url = f"{API_URL}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    if payload.get("status") != "ok":
        raise RuntimeError(f"Twelve Data error for {symbol}: {payload}")

    df = pd.DataFrame(payload["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.set_index("datetime")["close"].rename(symbol).sort_index()


def build_historical_dataset(pairs=PAIRS, interval="1h", outputsize=5000):
    api_key = get_api_key()
    series = [fetch_pair(pair, api_key, interval=interval, outputsize=outputsize) for pair in pairs]
    combined = pd.concat(series, axis=1, join="inner").dropna()
    combined.index.name = "Date"
    return combined.reset_index()


if __name__ == "__main__":
    data = build_historical_dataset()
    out_path = "data/historical_fx_rates.csv"
    os.makedirs("data", exist_ok=True)
    data.to_csv(out_path, index=False)
    print(f"Wrote {len(data)} rows to {out_path}")
    print(data.head())
    print(data.tail())
