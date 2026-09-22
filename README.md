# FX Triangular Arbitrage Simulation

Simulates and tests triangular arbitrage across EUR/USD, USD/JPY and EUR/JPY.

## The idea

- Law of one price: EUR/JPY should equal EUR/USD x USD/JPY
- If it doesn't, a EUR to USD to JPY to EUR loop is theoretically risk-free
- Profit factor: `max(actual/implied, implied/actual) - 1`
- Rates modelled as correlated log-returns, plus a mispricing term
- Parameters estimated from data, transaction costs applied per leg

## Code

| File | What it does |
|---|---|
| `triangular_arbitrage_implementation.py` | `ExchangeRateData`: rates and returns |
| `parameter_estimation.py` | `ParameterEstimator`: volatility, correlation |
| `mispricing_calculation.py` | `MispricingSimulator`: the mispricing term |
| `arbitrage_detection.py` | `ArbitrageDetector`: opportunities, profit |
| `visualization.py` | `ArbitrageVisualizer`: dashboards, PDF report |
| `main.py` | Runs the full synthetic pipeline |
| `triangular_arbitrage_colab.py` | Single-file Colab version |
| `fetch_historical_data.py`, `run_historical.py` | Real data, via Twelve Data |
| `generate_synthetic_data.py`, `run_synthetic.py` | Synthetic data, saved as CSV |

## Running it

Synthetic, live:
```
pip install -r requirements.txt
python code/main.py
```

Real data (needs a free Twelve Data key in `.env`, see `.env.example`):
```
python code/fetch_historical_data.py
python code/run_historical.py
```

Synthetic data, as a CSV:
```
python code/generate_synthetic_data.py
python code/run_synthetic.py
```

## Results

**Synthetic** (1,000 rows, built to contain arbitrage, not real data):

- 12.7% of opportunities feasible after costs
- avg net profit 0.043, max 0.243
- estimator recovers the true parameters (volatility ~0.004, correlation ~-0.84)

**Real data** (5,000 hourly bars, all three pairs at the same timestamp):

- 0 of 5,000 opportunities survive transaction costs
- mispricing volatility is ~20x smaller than the synthetic model assumes
- expected: these pairs are too liquid and too fast-arbitraged to show a gap at hourly resolution

## Known issues

- Synthetic mispricing (0.1%) is a stress scenario, not a realistic base rate
- Real-data run uses hourly closes, not tick data, so it can't see sub-hour deviations
- Free data tier caps history at about 7 months

## Next steps

- Try 1-minute bars on real data
- Model execution latency, not just a flat transaction cost
- Extend past 3 currencies to a bigger arbitrage graph
