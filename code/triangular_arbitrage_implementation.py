"""
Triangular Arbitrage Implementation

This script implements a triangular arbitrage system for forex markets,
focusing on EUR/USD, USD/JPY, and EUR/JPY currency pairs.

The implementation follows the mathematical modeling described in the project documentation
and is designed to run in Google Colab.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

class ExchangeRateData:
    """
    Handles exchange rate data (historical or synthetic)
    
    This class provides functionality to:
    1. Load historical exchange rate data
    2. Generate synthetic exchange rate data
    3. Calculate logarithmic returns
    4. Manage and access exchange rate time series
    """
    
    def __init__(self, data_source='synthetic', num_periods=1000, 
                 initial_rates=None, volatilities=None, correlation=None):
        """
        Initialize the exchange rate data handler
        
        Parameters:
        -----------
        data_source : str
            Source of data ('synthetic' or 'historical')
        num_periods : int
            Number of periods for synthetic data generation
        initial_rates : dict
            Initial exchange rates for synthetic data generation
            Format: {'EUR/USD': float, 'USD/JPY': float, 'EUR/JPY': float}
        volatilities : dict
            Volatility parameters for synthetic data generation
            Format: {'EUR/USD': float, 'USD/JPY': float}
        correlation : float
            Correlation between EUR/USD and USD/JPY returns
        """
        self.data_source = data_source
        self.num_periods = num_periods
        
        # Default initial rates if not provided
        self.initial_rates = initial_rates or {
            'EUR/USD': 1.13, 
            'USD/JPY': 143.0, 
            'EUR/JPY': 161.5
        }
        
        # Default volatilities if not provided
        self.volatilities = volatilities or {
            'EUR/USD': 0.004,  # σEU
            'USD/JPY': 0.0043  # σUJ
        }
        
        # Default correlation if not provided
        self.correlation = -0.84 if correlation is None else correlation
        
        # Initialize data structures
        self.rates_df = None
        self.returns_df = None
        
        # Generate or load data based on source
        if data_source == 'synthetic':
            self.generate_synthetic_data()
        else:
            raise ValueError("Historical data loading not implemented yet. Use 'synthetic' for now.")
    
    def generate_synthetic_data(self):
        """
        Generate synthetic exchange rate data with specified parameters
        
        The generation follows these steps:
        1. Create a time index
        2. Generate correlated normal random variables for log returns
        3. Convert log returns to exchange rates
        4. Calculate the implied EUR/JPY rate
        5. Add mispricing to create actual EUR/JPY rate
        """
        # Create time index
        dates = [datetime.now() - timedelta(days=i) for i in range(self.num_periods)]
        dates.reverse()
        
        # Extract parameters
        sigma_eu = self.volatilities['EUR/USD']
        sigma_uj = self.volatilities['USD/JPY']
        rho = self.correlation
        
        # Generate correlated normal random variables for log returns
        # Using Cholesky decomposition to generate correlated random variables
        cov_matrix = np.array([
            [sigma_eu**2, rho * sigma_eu * sigma_uj],
            [rho * sigma_eu * sigma_uj, sigma_uj**2]
        ])
        
        L = np.linalg.cholesky(cov_matrix)
        uncorrelated_returns = np.random.normal(0, 1, size=(2, self.num_periods))
        correlated_returns = np.dot(L, uncorrelated_returns)
        
        # Extract the correlated returns
        log_returns_eu = correlated_returns[0]
        log_returns_uj = correlated_returns[1]
        
        # Initialize arrays for rates
        rates_eu = np.zeros(self.num_periods)
        rates_uj = np.zeros(self.num_periods)
        
        # Set initial rates
        rates_eu[0] = self.initial_rates['EUR/USD']
        rates_uj[0] = self.initial_rates['USD/JPY']
        
        # Generate rates from log returns
        for i in range(1, self.num_periods):
            rates_eu[i] = rates_eu[i-1] * np.exp(log_returns_eu[i-1])
            rates_uj[i] = rates_uj[i-1] * np.exp(log_returns_uj[i-1])
        
        # Calculate implied EUR/JPY rate
        implied_rates_ej = rates_eu * rates_uj
        
        # Add mispricing to create actual EUR/JPY rate
        # Mispricing is modeled as a normal random variable with mean 0 and std dev sigma_epsilon
        sigma_epsilon = 0.001  # Default mispricing volatility
        epsilon = np.random.normal(0, sigma_epsilon, self.num_periods)
        actual_rates_ej = implied_rates_ej * (1 + epsilon)
        
        # Create DataFrame
        self.rates_df = pd.DataFrame({
            'Date': dates,
            'EUR/USD': rates_eu,
            'USD/JPY': rates_uj,
            'EUR/JPY_implied': implied_rates_ej,
            'EUR/JPY': actual_rates_ej,
            'Epsilon': epsilon
        })
        
        self.rates_df.set_index('Date', inplace=True)
        
        # Calculate log returns for analysis
        self.calculate_log_returns()
        
        return self.rates_df
    
    def load_historical_data(self, file_path):
        """
        Load historical exchange rate data from CSV or Excel
        
        Parameters:
        -----------
        file_path : str
            Path to the data file
        """
        # Determine file type from extension
        if file_path.endswith('.csv'):
            self.rates_df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            self.rates_df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or Excel files.")
        
        # Ensure required columns exist
        required_columns = ['Date', 'EUR/USD', 'USD/JPY', 'EUR/JPY']
        for col in required_columns:
            if col not in self.rates_df.columns:
                raise ValueError(f"Required column '{col}' not found in the data file.")
        
        # Set Date as index if it exists
        if 'Date' in self.rates_df.columns:
            self.rates_df.set_index('Date', inplace=True)
        
        # Calculate implied EUR/JPY and epsilon if EUR/JPY exists
        if 'EUR/JPY' in self.rates_df.columns:
            self.rates_df['EUR/JPY_implied'] = self.rates_df['EUR/USD'] * self.rates_df['USD/JPY']
            self.rates_df['Epsilon'] = (self.rates_df['EUR/JPY'] / self.rates_df['EUR/JPY_implied']) - 1
        
        # Calculate log returns
        self.calculate_log_returns()
        
        return self.rates_df
    
    def calculate_log_returns(self):
        """
        Calculate logarithmic returns for all currency pairs
        
        For each currency pair, the log return at time t is:
        r(t) = ln(Rate(t) / Rate(t-1))
        """
        if self.rates_df is None:
            raise ValueError("No exchange rate data available. Generate or load data first.")
        
        # Create a new DataFrame for returns
        self.returns_df = pd.DataFrame(index=self.rates_df.index)
        
        # Calculate log returns for each currency pair
        for pair in ['EUR/USD', 'USD/JPY', 'EUR/JPY', 'EUR/JPY_implied']:
            if pair in self.rates_df.columns:
                self.returns_df[f'Log_r_{pair}'] = np.log(
                    self.rates_df[pair] / self.rates_df[pair].shift(1)
                )
        
        # Drop the first row with NaN values
        self.returns_df = self.returns_df.dropna()
        
        return self.returns_df
    
    def get_rates(self):
        """Return the exchange rates DataFrame"""
        return self.rates_df
    
    def get_returns(self):
        """Return the log returns DataFrame"""
        return self.returns_df
    
    def get_latest_rates(self):
        """Return the latest exchange rates"""
        if self.rates_df is None:
            raise ValueError("No exchange rate data available.")
        
        return self.rates_df.iloc[-1].to_dict()
    
    def plot_rates(self):
        """Plot the exchange rates over time"""
        if self.rates_df is None:
            raise ValueError("No exchange rate data available.")
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Plot EUR/USD
        self.rates_df['EUR/USD'].plot(ax=axes[0], color='blue')
        axes[0].set_title('EUR/USD Exchange Rate')
        axes[0].set_ylabel('Rate')
        axes[0].grid(True)
        
        # Plot USD/JPY
        self.rates_df['USD/JPY'].plot(ax=axes[1], color='green')
        axes[1].set_title('USD/JPY Exchange Rate')
        axes[1].set_ylabel('Rate')
        axes[1].grid(True)
        
        # Plot EUR/JPY (actual and implied)
        self.rates_df['EUR/JPY'].plot(ax=axes[2], color='red', label='Actual')
        self.rates_df['EUR/JPY_implied'].plot(ax=axes[2], color='orange', linestyle='--', label='Implied')
        axes[2].set_title('EUR/JPY Exchange Rate')
        axes[2].set_ylabel('Rate')
        axes[2].set_xlabel('Date')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        return fig
    
    def plot_mispricing(self):
        """Plot the mispricing (epsilon) over time"""
        if self.rates_df is None or 'Epsilon' not in self.rates_df.columns:
            raise ValueError("No mispricing data available.")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        self.rates_df['Epsilon'].plot(ax=ax, color='purple')
        ax.set_title('EUR/JPY Mispricing (Epsilon)')
        ax.set_ylabel('Epsilon')
        ax.set_xlabel('Date')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True)
        
        plt.tight_layout()
        return fig


# Example usage
if __name__ == "__main__":
    # Create exchange rate data with default parameters
    exchange_data = ExchangeRateData(data_source='synthetic', num_periods=1000)
    
    # Get the generated rates
    rates_df = exchange_data.get_rates()
    
    # Print the first few rows
    print(rates_df.head())
    
    # Plot the exchange rates
    exchange_data.plot_rates()
    plt.show()
    
    # Plot the mispricing
    exchange_data.plot_mispricing()
    plt.show()
