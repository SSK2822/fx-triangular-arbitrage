"""
Main Script for Triangular Arbitrage System

This script integrates all components of the triangular arbitrage system
and provides a complete implementation for Google Colab.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os

# Import all system components
from triangular_arbitrage_implementation import ExchangeRateData
from parameter_estimation import ParameterEstimator
from mispricing_calculation import MispricingSimulator
from arbitrage_detection import ArbitrageDetector
from visualization import ArbitrageVisualizer

# Set random seed for reproducibility
np.random.seed(42)

def run_triangular_arbitrage_simulation(num_periods=1000, transaction_cost=0.0005, 
                                        volatility_epsilon=0.001, correlation=-0.84,
                                        initial_rates=None, volatilities=None,
                                        output_pdf=None):
    """
    Run a complete triangular arbitrage simulation
    
    Parameters:
    -----------
    num_periods : int
        Number of periods for simulation
    transaction_cost : float
        Transaction cost as a percentage (e.g., 0.0005 for 0.05%)
    volatility_epsilon : float
        Volatility of the mispricing term
    correlation : float
        Correlation between EUR/USD and USD/JPY returns
    initial_rates : dict, optional
        Initial exchange rates
    volatilities : dict, optional
        Volatility parameters for exchange rates
    output_pdf : str, optional
        Path to save the report (PDF format)
        
    Returns:
    --------
    tuple
        Tuple containing (exchange_data, estimator, simulator, detector, visualizer)
    """
    print("Starting Triangular Arbitrage Simulation...")
    print(f"Parameters: periods={num_periods}, transaction_cost={transaction_cost*100}%, "
          f"volatility_epsilon={volatility_epsilon}, correlation={correlation}")
    
    # Default initial rates if not provided
    if initial_rates is None:
        initial_rates = {
            'EUR/USD': 1.13, 
            'USD/JPY': 143.0, 
            'EUR/JPY': 161.5
        }
    
    # Default volatilities if not provided
    if volatilities is None:
        volatilities = {
            'EUR/USD': 0.004,  # σEU
            'USD/JPY': 0.0043  # σUJ
        }
    
    # Step 1: Create exchange rate data
    print("\n1. Generating exchange rate data...")
    exchange_data = ExchangeRateData(
        data_source='synthetic', 
        num_periods=num_periods,
        initial_rates=initial_rates,
        volatilities=volatilities,
        correlation=correlation
    )
    
    # Step 2: Estimate parameters
    print("\n2. Estimating statistical parameters...")
    estimator = ParameterEstimator(exchange_data)
    parameters = estimator.estimate_all_parameters()
    
    print("Estimated Parameters:")
    for param_name, param_value in parameters.items():
        print(f"  {param_name}: {param_value}")
    
    # Step 3: Simulate mispricing
    print("\n3. Simulating mispricing...")
    simulator = MispricingSimulator(
        exchange_data,
        parameters={'volatility_epsilon': volatility_epsilon}
    )
    
    # Calculate implied rates
    simulator.calculate_implied_rates()
    
    # Generate mispricing
    simulator.generate_mispricing()
    
    # Calculate actual rates
    simulator.calculate_actual_rates()
    
    # Analyze mispricing distribution
    mispricing_stats = simulator.analyze_mispricing_distribution()
    print("Mispricing Distribution Statistics:")
    for stat_name, stat_value in mispricing_stats.items():
        print(f"  {stat_name}: {stat_value}")
    
    # Step 4: Detect arbitrage opportunities
    print("\n4. Detecting arbitrage opportunities...")
    detector = ArbitrageDetector(
        exchange_data,
        mispricing_simulator=simulator,
        transaction_costs=transaction_cost
    )
    
    # Detect opportunities
    detector.detect_opportunities()
    
    # Calculate profits
    detector.calculate_profits()
    
    # Analyze feasibility
    detector.analyze_feasibility()
    
    # Get results
    results_df, feasibility_summary = detector.get_results()
    
    print("Arbitrage Feasibility Summary:")
    for key, value in feasibility_summary.items():
        print(f"  {key}: {value}")
    
    # Step 5: Create visualizations
    print("\n5. Creating visualizations...")
    visualizer = ArbitrageVisualizer(
        exchange_data,
        parameter_estimator=estimator,
        mispricing_simulator=simulator,
        arbitrage_detector=detector
    )
    
    # Create comprehensive report if output file specified
    if output_pdf is not None:
        print(f"Generating comprehensive report: {output_pdf}")
        visualizer.create_comprehensive_report(output_file=output_pdf)
    
    print("\nSimulation completed successfully!")
    
    return exchange_data, estimator, simulator, detector, visualizer

def test_with_different_parameters():
    """
    Test the system with different parameter settings
    """
    print("Running tests with different parameter settings...")
    
    # Test different transaction costs
    transaction_costs = [0.0001, 0.0005, 0.001, 0.002]
    
    for cost in transaction_costs:
        print(f"\n\nTesting with transaction cost: {cost*100}%")
        run_triangular_arbitrage_simulation(
            num_periods=500,  # Shorter period for testing
            transaction_cost=cost,
            output_pdf=f"arbitrage_report_cost_{cost*100:.2f}pct.pdf"
        )
    
    # Test different mispricing volatilities
    volatilities = [0.0005, 0.001, 0.002, 0.005]
    
    for vol in volatilities:
        print(f"\n\nTesting with mispricing volatility: {vol}")
        run_triangular_arbitrage_simulation(
            num_periods=500,  # Shorter period for testing
            volatility_epsilon=vol,
            output_pdf=f"arbitrage_report_vol_{vol:.4f}.pdf"
        )
    
    # Test different correlations
    correlations = [-0.9, -0.5, 0, 0.5, 0.9]
    
    for corr in correlations:
        print(f"\n\nTesting with correlation: {corr}")
        run_triangular_arbitrage_simulation(
            num_periods=500,  # Shorter period for testing
            correlation=corr,
            output_pdf=f"arbitrage_report_corr_{corr:.1f}.pdf"
        )
    
    print("\nAll parameter tests completed!")

def validate_with_excel_example():
    """
    Validate the implementation against the Excel example
    """
    print("Validating implementation against Excel example...")
    
    # The Excel example showed:
    # - Volatility EUR/USD: ~0.004
    # - Volatility USD/JPY: ~0.0043
    # - Correlation: ~-0.84
    # - Volatility Epsilon: ~0.12
    
    # Create a simulation with these parameters
    exchange_data, estimator, simulator, detector, visualizer = run_triangular_arbitrage_simulation(
        num_periods=1000,
        volatility_epsilon=0.12,
        correlation=-0.84,
        volatilities={
            'EUR/USD': 0.004,
            'USD/JPY': 0.0043
        },
        output_pdf="excel_validation_report.pdf"
    )
    
    print("\nValidation completed!")
    
    return exchange_data, estimator, simulator, detector, visualizer

# Main execution
if __name__ == "__main__":
    # Run a standard simulation
    exchange_data, estimator, simulator, detector, visualizer = run_triangular_arbitrage_simulation(
        num_periods=1000,
        transaction_cost=0.0005,  # 0.05% per trade
        output_pdf="triangular_arbitrage_report.pdf"
    )
    
    # Display some visualizations
    print("\nDisplaying visualizations...")
    
    # Exchange rate dashboard
    fig1 = visualizer.create_exchange_rate_dashboard()
    plt.figure(fig1.number)
    plt.show()
    
    # Arbitrage dashboard
    fig2 = visualizer.create_arbitrage_dashboard()
    plt.figure(fig2.number)
    plt.show()
    
    # Parameter dashboard
    fig3 = visualizer.create_parameter_dashboard()
    plt.figure(fig3.number)
    plt.show()
    
    # Transaction cost analysis
    fig4 = visualizer.create_transaction_cost_analysis()
    plt.figure(fig4.number)
    plt.show()
    
    # Uncomment to run additional tests
    # test_with_different_parameters()
    
    # Uncomment to validate against Excel example
    # validate_with_excel_example()
