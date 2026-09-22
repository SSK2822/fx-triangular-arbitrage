"""
Mispricing Calculation for Triangular Arbitrage

This module implements the mispricing calculation component for the triangular arbitrage system.
It calculates implied cross rates, generates mispricing terms, and determines actual rates.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union

class MispricingSimulator:
    """
    Simulates mispricing in cross rates
    
    This class provides functionality to:
    1. Calculate implied cross rates
    2. Generate mispricing terms based on normal distribution
    3. Calculate actual rates incorporating mispricing
    4. Analyze mispricing distribution and characteristics
    """
    
    def __init__(self, exchange_rate_data, parameters=None):
        """
        Initialize the mispricing simulator
        
        Parameters:
        -----------
        exchange_rate_data : ExchangeRateData
            Exchange rate data object containing rates and returns
        parameters : dict, optional
            Statistical parameters for mispricing simulation
            Format: {
                'volatility_epsilon': float,
                'mean_epsilon': float (default: 0)
            }
        """
        self.exchange_data = exchange_rate_data
        self.rates_df = exchange_rate_data.get_rates().copy()
        
        # Default parameters if not provided
        if parameters is None:
            parameters = {'volatility_epsilon': 0.001}
            
        # Ensure mean_epsilon exists in parameters with default value 0.0
        if 'mean_epsilon' not in parameters:
            parameters['mean_epsilon'] = 0.0
            
        self.parameters = parameters
    
    def calculate_implied_rates(self, rates_df=None):
        """
        Calculate implied cross rates
        
        Parameters:
        -----------
        rates_df : pandas.DataFrame, optional
            Exchange rate data to use for calculation
            If None, uses the data from exchange_rate_data
            
        Returns:
        --------
        pandas.Series
            Series containing the implied EUR/JPY rates
        """
        if rates_df is None:
            rates_df = self.rates_df
        
        # Calculate implied EUR/JPY rate
        implied_rates = rates_df['EUR/USD'] * rates_df['USD/JPY']
        
        # Store in the rates DataFrame
        rates_df['EUR/JPY_implied'] = implied_rates
        
        return implied_rates
    
    def generate_mispricing(self, num_periods=None, custom_volatility=None):
        """
        Generate mispricing terms based on normal distribution
        
        Parameters:
        -----------
        num_periods : int, optional
            Number of periods to generate mispricing for
            If None, uses the length of the rates DataFrame
        custom_volatility : float, optional
            Custom volatility parameter for mispricing generation
            If None, uses the volatility_epsilon from parameters
            
        Returns:
        --------
        pandas.Series
            Series containing the generated mispricing terms
        """
        if num_periods is None:
            num_periods = len(self.rates_df)
        
        # Use custom volatility if provided, otherwise use from parameters
        volatility = custom_volatility or self.parameters['volatility_epsilon']
        mean = self.parameters.get('mean_epsilon', 0.0)  # Default to 0.0 if not present
        
        # Generate mispricing terms
        epsilon = np.random.normal(mean, volatility, num_periods)
        
        # Store in the rates DataFrame
        self.rates_df['Epsilon'] = epsilon
        
        return pd.Series(epsilon, index=self.rates_df.index)
    
    def calculate_actual_rates(self, implied_rates=None, epsilon=None):
        """
        Calculate actual rates incorporating mispricing
        
        Parameters:
        -----------
        implied_rates : pandas.Series, optional
            Implied EUR/JPY rates
            If None, calculates from the rates DataFrame
        epsilon : pandas.Series, optional
            Mispricing terms
            If None, uses the Epsilon column from the rates DataFrame
            
        Returns:
        --------
        pandas.Series
            Series containing the actual EUR/JPY rates
        """
        # Calculate implied rates if not provided
        if implied_rates is None:
            implied_rates = self.calculate_implied_rates()
        
        # Use existing epsilon if not provided
        if epsilon is None:
            if 'Epsilon' not in self.rates_df.columns:
                epsilon = self.generate_mispricing()
            else:
                epsilon = self.rates_df['Epsilon']
        
        # Calculate actual rates
        actual_rates = implied_rates * (1 + epsilon)
        
        # Store in the rates DataFrame
        self.rates_df['EUR/JPY'] = actual_rates
        
        return actual_rates
    
    def simulate_mispricing(self, num_simulations=1, custom_parameters=None):
        """
        Run multiple mispricing simulations
        
        Parameters:
        -----------
        num_simulations : int
            Number of simulations to run
        custom_parameters : dict, optional
            Custom parameters for the simulations
            If None, uses the current parameters
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the simulation results
        """
        # Use custom parameters if provided
        params = custom_parameters or self.parameters
        
        # Ensure mean_epsilon exists in params
        if 'mean_epsilon' not in params:
            params['mean_epsilon'] = 0.0
        
        # Initialize storage for simulation results
        self.simulation_results = pd.DataFrame(
            index=self.rates_df.index,
            columns=[f'Simulation_{i+1}' for i in range(num_simulations)]
        )
        
        # Run simulations
        for i in range(num_simulations):
            # Generate new mispricing terms
            epsilon = np.random.normal(
                params['mean_epsilon'],
                params['volatility_epsilon'],
                len(self.rates_df)
            )
            
            # Calculate implied rates
            implied_rates = self.rates_df['EUR/USD'] * self.rates_df['USD/JPY']
            
            # Calculate actual rates
            actual_rates = implied_rates * (1 + epsilon)
            
            # Store results
            self.simulation_results[f'Simulation_{i+1}'] = epsilon
        
        return self.simulation_results
    
    def analyze_mispricing_distribution(self):
        """
        Analyze the distribution of mispricing terms
        
        Returns:
        --------
        dict
            Dictionary containing distribution statistics
        """
        if 'Epsilon' not in self.rates_df.columns:
            raise ValueError("No mispricing data available. Generate mispricing first.")
        
        # Calculate statistics
        epsilon = self.rates_df['Epsilon'].dropna()
        stats = {
            'mean': epsilon.mean(),
            'median': epsilon.median(),
            'std_dev': epsilon.std(),
            'min': epsilon.min(),
            'max': epsilon.max(),
            'skewness': epsilon.skew(),
            'kurtosis': epsilon.kurt(),
            'q1': epsilon.quantile(0.25),
            'q3': epsilon.quantile(0.75)
        }
        
        return stats
    
    def plot_mispricing_distribution(self):
        """
        Plot the distribution of mispricing terms
        
        Returns:
        --------
        matplotlib.figure.Figure
            Figure containing the distribution plot
        """
        if 'Epsilon' not in self.rates_df.columns:
            raise ValueError("No mispricing data available. Generate mispricing first.")
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot histogram
        self.rates_df['Epsilon'].hist(ax=axes[0], bins=30, color='purple', alpha=0.7)
        axes[0].set_title('Mispricing (Epsilon) Distribution')
        axes[0].set_xlabel('Epsilon')
        axes[0].set_ylabel('Frequency')
        axes[0].grid(True)
        
        # Add normal distribution fit
        epsilon = self.rates_df['Epsilon'].dropna()
        x = np.linspace(epsilon.min(), epsilon.max(), 100)
        mean = epsilon.mean()
        std = epsilon.std()
        pdf = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
        axes[0].plot(x, pdf * len(epsilon) * (x[1] - x[0]), 'r-', linewidth=2)
        
        # Plot QQ plot
        from scipy import stats
        stats.probplot(epsilon, dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot')
        axes[1].grid(True)
        
        plt.tight_layout()
        return fig
    
    def plot_mispricing_time_series(self):
        """
        Plot the mispricing terms over time
        
        Returns:
        --------
        matplotlib.figure.Figure
            Figure containing the time series plot
        """
        if 'Epsilon' not in self.rates_df.columns:
            raise ValueError("No mispricing data available. Generate mispricing first.")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot mispricing time series
        self.rates_df['Epsilon'].plot(ax=ax, color='purple')
        ax.set_title('Mispricing (Epsilon) Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Epsilon')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True)
        
        plt.tight_layout()
        return fig
    
    def plot_actual_vs_implied(self):
        """
        Plot actual vs implied EUR/JPY rates
        
        Returns:
        --------
        matplotlib.figure.Figure
            Figure containing the comparison plot
        """
        if 'EUR/JPY' not in self.rates_df.columns or 'EUR/JPY_implied' not in self.rates_df.columns:
            raise ValueError("Actual or implied EUR/JPY rates not available.")
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot rates
        self.rates_df['EUR/JPY'].plot(ax=axes[0], color='red', label='Actual')
        self.rates_df['EUR/JPY_implied'].plot(ax=axes[0], color='blue', linestyle='--', label='Implied')
        axes[0].set_title('Actual vs Implied EUR/JPY Rates')
        axes[0].set_ylabel('Rate')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot difference
        difference = self.rates_df['EUR/JPY'] - self.rates_df['EUR/JPY_implied']
        difference.plot(ax=axes[1], color='green')
        axes[1].set_title('Difference (Actual - Implied)')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Difference')
        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1].grid(True)
        
        plt.tight_layout()
        return fig
    
    def get_mispricing_series(self):
        """
        Return the mispricing series
        
        Returns:
        --------
        pandas.Series
            Series containing the mispricing terms
        """
        if 'Epsilon' not in self.rates_df.columns:
            raise ValueError("No mispricing data available. Generate mispricing first.")
        
        return self.rates_df['Epsilon']


# Example usage (requires ExchangeRateData from triangular_arbitrage_implementation.py)
if __name__ == "__main__":
    from triangular_arbitrage_implementation import ExchangeRateData
    from parameter_estimation import ParameterEstimator
    
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
    
    # Calculate implied rates
    implied_rates = simulator.calculate_implied_rates()
    
    # Generate mispricing
    epsilon = simulator.generate_mispricing()
    
    # Calculate actual rates
    actual_rates = simulator.calculate_actual_rates()
    
    # Analyze mispricing distribution
    stats = simulator.analyze_mispricing_distribution()
    print("Mispricing Distribution Statistics:")
    for stat_name, stat_value in stats.items():
        print(f"{stat_name}: {stat_value}")
    
    # Plot mispricing distribution
    simulator.plot_mispricing_distribution()
    plt.show()
    
    # Plot mispricing time series
    simulator.plot_mispricing_time_series()
    plt.show()
    
    # Plot actual vs implied rates
    simulator.plot_actual_vs_implied()
    plt.show()
