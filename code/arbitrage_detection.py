"""
Arbitrage Detection and Profit Calculation for Triangular Arbitrage

This module implements the arbitrage detection and profit calculation component
for the triangular arbitrage system. It identifies arbitrage opportunities,
calculates theoretical profits, and analyzes feasibility considering transaction costs.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union

class ArbitrageDetector:
    """
    Detects arbitrage opportunities and calculates profits
    
    This class provides functionality to:
    1. Detect arbitrage opportunities based on rate discrepancies
    2. Calculate theoretical profits from identified opportunities
    3. Analyze feasibility considering transaction costs
    4. Visualize arbitrage opportunities and profit distributions
    """
    
    def __init__(self, exchange_rate_data, mispricing_simulator=None, transaction_costs=0.0):
        """
        Initialize the arbitrage detector
        
        Parameters:
        -----------
        exchange_rate_data : ExchangeRateData
            Exchange rate data object containing rates and returns
        mispricing_simulator : MispricingSimulator, optional
            Mispricing simulator object for accessing mispricing data
        transaction_costs : float or dict, optional
            Transaction costs as a percentage (e.g., 0.001 for 0.1%)
            Can be a single value for all pairs or a dictionary with costs per pair
            Format: {
                'EUR/USD': float,
                'USD/JPY': float,
                'EUR/JPY': float
            }
        """
        self.exchange_data = exchange_rate_data
        self.mispricing_simulator = mispricing_simulator
        
        # Get rates DataFrame
        self.rates_df = exchange_rate_data.get_rates().copy()
        
        # Set transaction costs
        if isinstance(transaction_costs, dict):
            self.transaction_costs = transaction_costs
        else:
            # Use the same cost for all pairs
            self.transaction_costs = {
                'EUR/USD': transaction_costs,
                'USD/JPY': transaction_costs,
                'EUR/JPY': transaction_costs
            }
        
        # Initialize storage for results
        self.results_df = None
    
    def detect_opportunities(self):
        """
        Detect arbitrage opportunities based on rate discrepancies
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing detected opportunities
        """
        # Ensure required columns exist
        required_columns = ['EUR/USD', 'USD/JPY', 'EUR/JPY', 'EUR/JPY_implied']
        for col in required_columns:
            if col not in self.rates_df.columns:
                if col == 'EUR/JPY_implied':
                    # Calculate implied EUR/JPY if not available
                    self.rates_df['EUR/JPY_implied'] = self.rates_df['EUR/USD'] * self.rates_df['USD/JPY']
                else:
                    raise ValueError(f"Required column '{col}' not found in exchange rate data.")
        
        # Create results DataFrame
        self.results_df = pd.DataFrame(index=self.rates_df.index)
        
        # Copy exchange rates to results
        for col in required_columns:
            self.results_df[col] = self.rates_df[col]
        
        # Calculate mispricing (epsilon)
        if 'Epsilon' not in self.rates_df.columns:
            self.results_df['Epsilon'] = (self.rates_df['EUR/JPY'] / self.rates_df['EUR/JPY_implied']) - 1
        else:
            self.results_df['Epsilon'] = self.rates_df['Epsilon']
        
        # Detect arbitrage opportunities
        # Case 1: EUR/JPY is overpriced (actual > implied)
        self.results_df['Case_1_Opportunity'] = self.results_df['EUR/JPY'] > self.results_df['EUR/JPY_implied']
        
        # Case 2: EUR/JPY is underpriced (actual < implied)
        self.results_df['Case_2_Opportunity'] = self.results_df['EUR/JPY'] < self.results_df['EUR/JPY_implied']
        
        # Any opportunity
        self.results_df['Any_Opportunity'] = (
            self.results_df['Case_1_Opportunity'] | 
            self.results_df['Case_2_Opportunity']
        )
        
        return self.results_df
    
    def calculate_profits(self):
        """
        Calculate theoretical profits from identified opportunities
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing profit calculations
        """
        if self.results_df is None:
            self.detect_opportunities()
        
        # Calculate profit factors
        # Case 1: EUR/JPY is overpriced
        self.results_df['Profit_Factor_Case_1'] = (
            self.results_df['EUR/JPY'] / 
            (self.results_df['EUR/USD'] * self.results_df['USD/JPY'])
        )
        
        # Case 2: EUR/JPY is underpriced
        self.results_df['Profit_Factor_Case_2'] = (
            (self.results_df['EUR/USD'] * self.results_df['USD/JPY']) / 
            self.results_df['EUR/JPY']
        )
        
        # General profit factor (max of the two cases)
        self.results_df['Profit_Factor'] = np.maximum(
            self.results_df['Profit_Factor_Case_1'],
            self.results_df['Profit_Factor_Case_2']
        )
        
        # Calculate percentage profit
        self.results_df['Percentage_Profit'] = (self.results_df['Profit_Factor'] - 1) * 100
        
        # Simplified profit factor based on epsilon
        self.results_df['Profit_Factor_Epsilon'] = np.maximum(
            1 + self.results_df['Epsilon'],
            1 / (1 + self.results_df['Epsilon'])
        )
        
        # Calculate percentage profit based on epsilon
        self.results_df['Percentage_Profit_Epsilon'] = (self.results_df['Profit_Factor_Epsilon'] - 1) * 100
        
        return self.results_df
    
    def analyze_feasibility(self):
        """
        Analyze feasibility of opportunities considering transaction costs
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing feasibility analysis
        """
        if self.results_df is None or 'Percentage_Profit' not in self.results_df.columns:
            self.calculate_profits()
        
        # Calculate total transaction costs for each arbitrage loop
        # Each loop involves 3 trades, one for each currency pair
        total_costs = sum(self.transaction_costs.values()) * 100  # Convert to percentage
        
        # Determine feasible opportunities (profit > costs)
        self.results_df['Feasible_Opportunity'] = self.results_df['Percentage_Profit'] > total_costs
        
        # Calculate net profit after costs
        self.results_df['Net_Profit'] = self.results_df['Percentage_Profit'] - total_costs
        
        # Count feasible opportunities
        feasible_count = self.results_df['Feasible_Opportunity'].sum()
        total_count = len(self.results_df)
        feasible_percentage = (feasible_count / total_count) * 100 if total_count > 0 else 0
        
        # Store summary statistics
        self.feasibility_summary = {
            'total_opportunities': total_count,
            'feasible_opportunities': feasible_count,
            'feasible_percentage': feasible_percentage,
            'average_profit': self.results_df['Percentage_Profit'].mean(),
            'average_net_profit': self.results_df['Net_Profit'][self.results_df['Feasible_Opportunity']].mean() 
                if feasible_count > 0 else 0,
            'max_profit': self.results_df['Percentage_Profit'].max(),
            'max_net_profit': self.results_df['Net_Profit'].max(),
            'transaction_costs': total_costs
        }
        
        return self.results_df
    
    def get_results(self):
        """
        Return detection and analysis results
        
        Returns:
        --------
        tuple
            Tuple containing (results_df, feasibility_summary)
        """
        if self.results_df is None:
            self.analyze_feasibility()
        
        return self.results_df, self.feasibility_summary
    
    def plot_profit_distribution(self):
        """
        Plot distribution of arbitrage profits
        
        Returns:
        --------
        matplotlib.figure.Figure
            Figure containing the profit distribution plot
        """
        if self.results_df is None or 'Percentage_Profit' not in self.results_df.columns:
            self.calculate_profits()
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot histogram of percentage profits
        self.results_df['Percentage_Profit'].hist(
            ax=axes[0], bins=30, color='green', alpha=0.7
        )
        axes[0].set_title('Distribution of Percentage Profits')
        axes[0].set_xlabel('Profit (%)')
        axes[0].set_ylabel('Frequency')
        axes[0].grid(True)
        
        # Add vertical line for transaction costs if available
        if hasattr(self, 'feasibility_summary'):
            axes[0].axvline(
                x=self.feasibility_summary['transaction_costs'],
                color='red', linestyle='--', linewidth=2,
                label=f"Transaction Costs: {self.feasibility_summary['transaction_costs']:.4f}%"
            )
            axes[0].legend()
        
        # Plot histogram of net profits (only feasible opportunities)
        if 'Net_Profit' in self.results_df.columns and 'Feasible_Opportunity' in self.results_df.columns:
            feasible_profits = self.results_df['Net_Profit'][self.results_df['Feasible_Opportunity']]
            if len(feasible_profits) > 0:
                feasible_profits.hist(ax=axes[1], bins=30, color='blue', alpha=0.7)
                axes[1].set_title('Distribution of Net Profits (Feasible Opportunities)')
                axes[1].set_xlabel('Net Profit (%)')
                axes[1].set_ylabel('Frequency')
                axes[1].grid(True)
            else:
                axes[1].set_title('No Feasible Opportunities')
                axes[1].set_xlabel('Net Profit (%)')
                axes[1].set_ylabel('Frequency')
                axes[1].grid(True)
        
        plt.tight_layout()
        return fig
    
    def plot_opportunity_frequency(self):
        """
        Plot frequency of arbitrage opportunities
        
        Returns:
        --------
        matplotlib.figure.Figure
            Figure containing the opportunity frequency plot
        """
        if self.results_df is None:
            self.detect_opportunities()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate opportunity counts
        opportunity_counts = {
            'Case 1 (EUR/JPY Overpriced)': self.results_df['Case_1_Opportunity'].sum(),
            'Case 2 (EUR/JPY Underpriced)': self.results_df['Case_2_Opportunity'].sum(),
            'Any Opportunity': self.results_df['Any_Opportunity'].sum()
        }
        
        if 'Feasible_Opportunity' in self.results_df.columns:
            opportunity_counts['Feasible Opportunities'] = self.results_df['Feasible_Opportunity'].sum()
        
        # Calculate percentages
        total_periods = len(self.results_df)
        opportunity_percentages = {
            k: (v / total_periods) * 100 for k, v in opportunity_counts.items()
        }
        
        # Plot bar chart
        bars = ax.bar(
            opportunity_percentages.keys(),
            opportunity_percentages.values(),
            color=['blue', 'green', 'orange', 'red'][:len(opportunity_percentages)]
        )
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height + 0.5,
                f'{height:.2f}%',
                ha='center', va='bottom'
            )
        
        ax.set_title('Frequency of Arbitrage Opportunities')
        ax.set_ylabel('Percentage of Periods (%)')
        ax.set_ylim(0, max(opportunity_percentages.values()) * 1.2)  # Add some space for labels
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig
    
    def plot_profit_time_series(self):
        """
        Plot arbitrage profits over time
        
        Returns:
        --------
        matplotlib.figure.Figure
            Figure containing the profit time series plot
        """
        if self.results_df is None or 'Percentage_Profit' not in self.results_df.columns:
            self.calculate_profits()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot percentage profit time series
        self.results_df['Percentage_Profit'].plot(ax=ax, color='green')
        
        # Add horizontal line for transaction costs if available
        if hasattr(self, 'feasibility_summary'):
            ax.axhline(
                y=self.feasibility_summary['transaction_costs'],
                color='red', linestyle='--', linewidth=2,
                label=f"Transaction Costs: {self.feasibility_summary['transaction_costs']:.4f}%"
            )
            ax.legend()
        
        ax.set_title('Arbitrage Profit Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Profit (%)')
        ax.grid(True)
        
        plt.tight_layout()
        return fig
    
    def plot_epsilon_vs_profit(self):
        """
        Plot relationship between epsilon (mispricing) and profit
        
        Returns:
        --------
        matplotlib.figure.Figure
            Figure containing the scatter plot
        """
        if self.results_df is None or 'Epsilon' not in self.results_df.columns:
            self.calculate_profits()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot scatter of epsilon vs profit
        ax.scatter(
            self.results_df['Epsilon'].abs(),  # Use absolute value of epsilon
            self.results_df['Percentage_Profit'],
            alpha=0.5, color='purple'
        )
        
        # Add trend line
        z = np.polyfit(self.results_df['Epsilon'].abs(), self.results_df['Percentage_Profit'], 1)
        p = np.poly1d(z)
        ax.plot(
            sorted(self.results_df['Epsilon'].abs()),
            p(sorted(self.results_df['Epsilon'].abs())),
            "r--", linewidth=2
        )
        
        ax.set_title('Relationship Between |Epsilon| and Profit')
        ax.set_xlabel('|Epsilon| (Absolute Mispricing)')
        ax.set_ylabel('Profit (%)')
        ax.grid(True)
        
        plt.tight_layout()
        return fig


# Example usage (requires previous modules)
if __name__ == "__main__":
    from triangular_arbitrage_implementation import ExchangeRateData
    from parameter_estimation import ParameterEstimator
    from mispricing_calculation import MispricingSimulator
    
    # Create exchange rate data
    exchange_data = ExchangeRateData(data_source='synthetic', num_periods=1000)
    
    # Create parameter estimator and estimate parameters
    estimator = ParameterEstimator(exchange_data)
    parameters = estimator.estimate_all_parameters()
    
    # Create mispricing simulator
    simulator = MispricingSimulator(
        exchange_data,
        parameters={'volatility_epsilon': parameters['volatility_epsilon']}
    )
    
    # Calculate implied rates and actual rates
    simulator.calculate_implied_rates()
    simulator.calculate_actual_rates()
    
    # Create arbitrage detector with transaction costs
    detector = ArbitrageDetector(
        exchange_data,
        mispricing_simulator=simulator,
        transaction_costs=0.0005  # 0.05% per trade
    )
    
    # Detect opportunities and analyze feasibility
    detector.detect_opportunities()
    detector.calculate_profits()
    detector.analyze_feasibility()
    
    # Get results
    results_df, feasibility_summary = detector.get_results()
    
    # Print summary
    print("Arbitrage Feasibility Summary:")
    for key, value in feasibility_summary.items():
        print(f"{key}: {value}")
    
    # Plot profit distribution
    detector.plot_profit_distribution()
    plt.show()
    
    # Plot opportunity frequency
    detector.plot_opportunity_frequency()
    plt.show()
    
    # Plot profit time series
    detector.plot_profit_time_series()
    plt.show()
    
    # Plot epsilon vs profit
    detector.plot_epsilon_vs_profit()
    plt.show()
