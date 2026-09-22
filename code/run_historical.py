"""
Run the triangular arbitrage pipeline on real historical exchange rate data.

This is a standalone addition alongside main.py (which runs the pipeline on
synthetic data). It reuses the existing, unmodified pipeline classes:
ExchangeRateData.load_historical_data() already computes the implied EUR/JPY
rate and the real mispricing (Epsilon = actual/implied - 1) from the loaded
rates, and ArbitrageDetector / ArbitrageVisualizer both work without a
MispricingSimulator, falling back to the Epsilon column already present in
the data. No code in the original modules is changed to make this work.

Usage:
    python fetch_historical_data.py   # writes data/historical_fx_rates.csv
    python run_historical.py
"""

import matplotlib
matplotlib.use("Agg")

from triangular_arbitrage_implementation import ExchangeRateData
from parameter_estimation import ParameterEstimator
from arbitrage_detection import ArbitrageDetector
from visualization import ArbitrageVisualizer

DATA_PATH = "data/historical_fx_rates.csv"
TRANSACTION_COST = 0.0005  # 0.05% per leg, same default as main.py


def run_historical_arbitrage_analysis(data_path=DATA_PATH, transaction_cost=TRANSACTION_COST):
    print("Loading real historical exchange rate data...")
    # data_source='synthetic' just satisfies the constructor; load_historical_data
    # immediately overwrites rates_df with the real data below.
    exchange_data = ExchangeRateData(data_source='synthetic', num_periods=5)
    exchange_data.load_historical_data(data_path)
    print(f"Loaded {len(exchange_data.rates_df)} rows: "
          f"{exchange_data.rates_df.index.min()} to {exchange_data.rates_df.index.max()}")

    print("\nEstimating statistical parameters from real data...")
    estimator = ParameterEstimator(exchange_data)
    parameters = estimator.estimate_all_parameters()
    for name, value in parameters.items():
        print(f"  {name}: {value}")

    print("\nDetecting arbitrage opportunities in real data...")
    detector = ArbitrageDetector(
        exchange_data,
        mispricing_simulator=None,
        transaction_costs=transaction_cost,
    )
    detector.detect_opportunities()
    detector.calculate_profits()
    detector.analyze_feasibility()
    results_df, feasibility_summary = detector.get_results()

    print("Arbitrage feasibility summary (real data):")
    for key, value in feasibility_summary.items():
        print(f"  {key}: {value}")

    print("\nCreating visualizations...")
    visualizer = ArbitrageVisualizer(
        exchange_data,
        parameter_estimator=estimator,
        mispricing_simulator=None,
        arbitrage_detector=detector,
    )
    visualizer.create_comprehensive_report(output_file="historical_arbitrage_report.pdf")
    print("Wrote historical_arbitrage_report.pdf")

    return exchange_data, estimator, detector, visualizer


if __name__ == "__main__":
    run_historical_arbitrage_analysis()
