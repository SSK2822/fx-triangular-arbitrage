"""
Parameter Estimation for Triangular Arbitrage

This module implements the statistical parameter estimation component for the triangular arbitrage system.
It estimates volatility parameters, correlation coefficients, and mispricing volatility from exchange rate data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union

class ParameterEstimator:
    """
    Estimates statistical parameters from exchange rate data
    
    This class provides functionality to:
    1. Estimate volatility parameters for EUR/USD and USD/JPY
    2. Estimate correlation between EUR/USD and USD/JPY returns
    3. Estimate volatility of the mispricing term
    4. Visualize parameter distributions and stability
    """
    
    def __init__(self, exchange_rate_data):
        """
        Initialize the parameter estimator
        
        Parameters:
        -----------
        exchange_rate_data : ExchangeRateData
            Exchange rate data object containing rates and returns
        """
        self.exchange_data = exchange_rate_data
        self.rates_df = exchange_rate_data.get_rates()
        self.returns_df = exchange_rate_data.get_returns()
        
        # Initialize parameter storage
        self.parameters = {
            'volatility_EUR/USD': None,
            'volatility_USD/JPY': None,
            'correlation': None,
            'volatility_epsilon': None
        }
        
        # Validate data
        self._validate_data()
    
    def _validate_data(self):
        """Validate that the required data is available for parameter estimation"""
        if self.returns_df is None:
            raise ValueError("No returns data available. Calculate returns first.")
        
        required_columns = ['Log_r_EUR/USD', 'Log_r_USD/JPY']
        for col in required_columns:
            if col not in self.returns_df.columns:
                raise ValueError(f"Required column '{col}' not found in returns data.")
        
        if self.rates_df is None:
            raise ValueError("No exchange rate data available.")
        
        if 'Epsilon' not in self.rates_df.columns:
            raise ValueError("Mispricing (Epsilon) data not found in exchange rate data.")
    
    def estimate_volatilities(self, window_size=None):
        """
        Estimate volatility parameters for EUR/USD and USD/JPY
        
        Parameters:
        -----------
        window_size : int, optional
            Size of the rolling window for volatility estimation.
            If None, uses the entire dataset.
            
        Returns:
        --------
        dict
            Dictionary containing the estimated volatilities
        """
        # If window_size is None, use the entire dataset
        if window_size is None:
            # Calculate volatility for EUR/USD
            self.parameters['volatility_EUR/USD'] = np.std(
                self.returns_df['Log_r_EUR/USD'], ddof=1
            )
            
            # Calculate volatility for USD/JPY
            self.parameters['volatility_USD/JPY'] = np.std(
                self.returns_df['Log_r_USD/JPY'], ddof=1
            )
        else:
            # Calculate rolling volatility
            rolling_vol_eu = self.returns_df['Log_r_EUR/USD'].rolling(
                window=window_size
            ).std(ddof=1)
            
            rolling_vol_uj = self.returns_df['Log_r_USD/JPY'].rolling(
                window=window_size
            ).std(ddof=1)
            
            # Use the latest values
            self.parameters['volatility_EUR/USD'] = rolling_vol_eu.iloc[-1]
            self.parameters['volatility_USD/JPY'] = rolling_vol_uj.iloc[-1]
        
        return {
            'volatility_EUR/USD': self.parameters['volatility_EUR/USD'],
            'volatility_USD/JPY': self.parameters['volatility_USD/JPY']
        }
    
    def estimate_correlation(self, window_size=None):
        """
        Estimate correlation between EUR/USD and USD/JPY returns
        
        Parameters:
        -----------
        window_size : int, optional
            Size of the rolling window for correlation estimation.
            If None, uses the entire dataset.
            
        Returns:
        --------
        float
            Estimated correlation coefficient
        """
        # If window_size is None, use the entire dataset
        if window_size is None:
            # Calculate correlation
            self.parameters['correlation'] = np.corrcoef(
                self.returns_df['Log_r_EUR/USD'],
                self.returns_df['Log_r_USD/JPY']
            )[0, 1]
        else:
            # Calculate rolling correlation
            rolling_corr = self.returns_df['Log_r_EUR/USD'].rolling(
                window=window_size
            ).corr(self.returns_df['Log_r_USD/JPY'])
            
            # Use the latest value
            self.parameters['correlation'] = rolling_corr.iloc[-1]
        
        return self.parameters['correlation']
    
    def estimate_mispricing_volatility(self, window_size=None):
        """
        Estimate volatility of the mispricing term
        
        Parameters:
        -----------
        window_size : int, optional
            Size of the rolling window for volatility estimation.
            If None, uses the entire dataset.
            
        Returns:
        --------
        float
            Estimated mispricing volatility
        """
        # If window_size is None, use the entire dataset
        if window_size is None:
            # Calculate mispricing volatility
            self.parameters['volatility_epsilon'] = np.std(
                self.rates_df['Epsilon'], ddof=1
            )
        else:
            # Calculate rolling volatility
            rolling_vol_eps = self.rates_df['Epsilon'].rolling(
                window=window_size
            ).std(ddof=1)
            
            # Use the latest value
            self.parameters['volatility_epsilon'] = rolling_vol_eps.iloc[-1]
        
        return self.parameters['volatility_epsilon']
    
    def estimate_all_parameters(self, window_size=None):
        """
        Estimate all parameters at once
        
        Parameters:
        -----------
        window_size : int, optional
            Size of the rolling window for parameter estimation.
            If None, uses the entire dataset.
            
        Returns:
        --------
        dict
            Dictionary containing all estimated parameters
        """
        self.estimate_volatilities(window_size)
        self.estimate_correlation(window_size)
        self.estimate_mispricing_volatility(window_size)
        
        return self.get_parameters()
    
    def get_parameters(self):
        """
        Return all estimated parameters
        
        Returns:
        --------
        dict
            Dictionary containing all estimated parameters
        """
        return self.parameters
    
    def plot_parameter_stability(self, window_sizes=[30, 60, 90, 120]):
        """
        Plot parameter stability across different window sizes
        
        Parameters:
        -----------
        window_sizes : list
            List of window sizes to test
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure containing the parameter stability plots
        """
        # Initialize storage for parameters at different window sizes
        volatility_eu = []
        volatility_uj = []
        correlation_values = []
        volatility_eps = []
        
        # Calculate parameters for each window size
        for window in window_sizes:
            self.estimate_all_parameters(window)
            volatility_eu.append(self.parameters['volatility_EUR/USD'])
            volatility_uj.append(self.parameters['volatility_USD/JPY'])
            correlation_values.append(self.parameters['correlation'])
            volatility_eps.append(self.parameters['volatility_epsilon'])
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot volatility EUR/USD
        axes[0, 0].plot(window_sizes, volatility_eu, 'o-', color='blue')
        axes[0, 0].set_title('Volatility EUR/USD vs Window Size')
        axes[0, 0].set_xlabel('Window Size')
        axes[0, 0].set_ylabel('Volatility')
        axes[0, 0].grid(True)
        
        # Plot volatility USD/JPY
        axes[0, 1].plot(window_sizes, volatility_uj, 'o-', color='green')
        axes[0, 1].set_title('Volatility USD/JPY vs Window Size')
        axes[0, 1].set_xlabel('Window Size')
        axes[0, 1].set_ylabel('Volatility')
        axes[0, 1].grid(True)
        
        # Plot correlation
        axes[1, 0].plot(window_sizes, correlation_values, 'o-', color='red')
        axes[1, 0].set_title('Correlation vs Window Size')
        axes[1, 0].set_xlabel('Window Size')
        axes[1, 0].set_ylabel('Correlation')
        axes[1, 0].grid(True)
        
        # Plot volatility epsilon
        axes[1, 1].plot(window_sizes, volatility_eps, 'o-', color='purple')
        axes[1, 1].set_title('Volatility Epsilon vs Window Size')
        axes[1, 1].set_xlabel('Window Size')
        axes[1, 1].set_ylabel('Volatility')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        return fig
    
    def plot_rolling_parameters(self, window_size=30):
        """
        Plot rolling parameters over time
        
        Parameters:
        -----------
        window_size : int
            Size of the rolling window
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure containing the rolling parameter plots
        """
        # Calculate rolling volatilities
        rolling_vol_eu = self.returns_df['Log_r_EUR/USD'].rolling(
            window=window_size
        ).std(ddof=1)
        
        rolling_vol_uj = self.returns_df['Log_r_USD/JPY'].rolling(
            window=window_size
        ).std(ddof=1)
        
        # Calculate rolling correlation
        rolling_corr = self.returns_df['Log_r_EUR/USD'].rolling(
            window=window_size
        ).corr(self.returns_df['Log_r_USD/JPY'])
        
        # Calculate rolling mispricing volatility
        rolling_vol_eps = self.rates_df['Epsilon'].rolling(
            window=window_size
        ).std(ddof=1)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
        
        # Plot rolling volatility EUR/USD
        rolling_vol_eu.plot(ax=axes[0, 0], color='blue')
        axes[0, 0].set_title(f'Rolling Volatility EUR/USD (Window={window_size})')
        axes[0, 0].set_ylabel('Volatility')
        axes[0, 0].grid(True)
        
        # Plot rolling volatility USD/JPY
        rolling_vol_uj.plot(ax=axes[0, 1], color='green')
        axes[0, 1].set_title(f'Rolling Volatility USD/JPY (Window={window_size})')
        axes[0, 1].set_ylabel('Volatility')
        axes[0, 1].grid(True)
        
        # Plot rolling correlation
        rolling_corr.plot(ax=axes[1, 0], color='red')
        axes[1, 0].set_title(f'Rolling Correlation (Window={window_size})')
        axes[1, 0].set_ylabel('Correlation')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].grid(True)
        
        # Plot rolling mispricing volatility
        rolling_vol_eps.plot(ax=axes[1, 1], color='purple')
        axes[1, 1].set_title(f'Rolling Volatility Epsilon (Window={window_size})')
        axes[1, 1].set_ylabel('Volatility')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        return fig


# Example usage (requires ExchangeRateData from triangular_arbitrage_implementation.py)
if __name__ == "__main__":
    from triangular_arbitrage_implementation import ExchangeRateData
    
    # Create exchange rate data
    exchange_data = ExchangeRateData(data_source='synthetic', num_periods=1000)
    
    # Create parameter estimator
    estimator = ParameterEstimator(exchange_data)
    
    # Estimate all parameters
    params = estimator.estimate_all_parameters()
    
    # Print the estimated parameters
    print("Estimated Parameters:")
    for param_name, param_value in params.items():
        print(f"{param_name}: {param_value}")
    
    # Plot parameter stability
    estimator.plot_parameter_stability()
    plt.show()
    
    # Plot rolling parameters
    estimator.plot_rolling_parameters()
    plt.show()
