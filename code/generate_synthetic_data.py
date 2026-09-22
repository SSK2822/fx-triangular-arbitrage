"""
Generate a synthetic exchange rate dataset and save it as a CSV.

SYNTHETIC DATA - NOT REAL MARKET DATA. This uses the existing, unmodified
ExchangeRateData.generate_synthetic_data() method (the same code path
main.py runs live) with main.py's own default parameters, and writes the
result to data/synthetic_fx_rates.csv in the same Date, EUR/USD, USD/JPY,
EUR/JPY shape as the real dataset from fetch_historical_data.py. It exists
so the two cases - a model with a designed-in mispricing, and real market
data with none large enough to trade - can be inspected and compared
side by side. See README "Known issues and limitations".
"""

import os

from triangular_arbitrage_implementation import ExchangeRateData

# Same defaults as main.py's run_triangular_arbitrage_simulation()
NUM_PERIODS = 1000
INITIAL_RATES = {'EUR/USD': 1.13, 'USD/JPY': 143.0, 'EUR/JPY': 161.5}
VOLATILITIES = {'EUR/USD': 0.004, 'USD/JPY': 0.0043}
CORRELATION = -0.84


def build_synthetic_dataset(num_periods=NUM_PERIODS):
    exchange_data = ExchangeRateData(
        data_source='synthetic',
        num_periods=num_periods,
        initial_rates=INITIAL_RATES,
        volatilities=VOLATILITIES,
        correlation=CORRELATION,
    )
    rates_df = exchange_data.get_rates()
    output = rates_df[['EUR/USD', 'USD/JPY', 'EUR/JPY']].reset_index()
    # generate_synthetic_data() builds dates from datetime.now(), which carries
    # today's wall-clock time and microseconds through every row; trim to the
    # date part here (an export-formatting choice in this new script, not a
    # change to ExchangeRateData itself).
    output['Date'] = output['Date'].dt.strftime('%Y-%m-%d')
    return output


if __name__ == "__main__":
    data = build_synthetic_dataset()
    out_path = "data/synthetic_fx_rates.csv"
    os.makedirs("data", exist_ok=True)
    data.to_csv(out_path, index=False)
    print(f"SYNTHETIC DATA (not real market data). Wrote {len(data)} rows to {out_path}")
    print(data.head())
    print(data.tail())
