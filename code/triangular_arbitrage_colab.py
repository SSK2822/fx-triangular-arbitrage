"""
Google Colab Implementation of Triangular Arbitrage System

This notebook implements a complete triangular arbitrage system for forex markets,
focusing on EUR/USD, USD/JPY, and EUR/JPY currency pairs.

The implementation follows the mathematical modeling described in the project documentation
and includes components for exchange rate modeling, parameter estimation, mispricing calculation,
arbitrage detection, and visualization.
"""

# Install required packages
!pip install numpy pandas matplotlib seaborn

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages

# Set random seed for reproducibility
np.random.seed(42)

# Configure matplotlib
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

#############################################
# Part 1: Exchange Rate Data Implementation #
#############################################

class ExchangeRateData:
    """
    Handles exchange rate data (historical or synthetic)
    
    This class provides functionality to:
    1. Generate synthetic exchange rate data
    2. Calculate logarithmic returns
    3. Manage and access exchange rate time series
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

#############################################
# Part 2: Parameter Estimation             #
#############################################

class ParameterEstimator:
    """
    Estimates statistical parameters from exchange rate data
    
    This class provides functionality to:
    1. Estimate volatility parameters for EUR/USD and USD/JPY
    2. Estimate correlation between EUR/USD and USD/JPY returns
    3. Estimate volatility of the mispricing term
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
        """
        self.estimate_volatilities(window_size)
        self.estimate_correlation(window_size)
        self.estimate_mispricing_volatility(window_size)
        
        return self.get_parameters()
    
    def get_parameters(self):
        """Return all estimated parameters"""
        return self.parameters

#############################################
# Part 3: Mispricing Calculation           #
#############################################

class MispricingSimulator:
    """
    Simulates mispricing in cross rates
    
    This class provides functionality to:
    1. Calculate implied cross rates
    2. Generate mispricing terms based on normal distribution
    3. Calculate actual rates incorporating mispricing
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
        self.parameters = parameters or {
            'volatility_epsilon': 0.001,
            'mean_epsilon': 0.0
        }
    
    def calculate_implied_rates(self, rates_df=None):
        """
        Calculate implied cross rates
        
        Parameters:
        -----------
        rates_df : pandas.DataFrame, optional
            Exchange rate data to use for calculation
            If None, uses the data from exchange_rate_data
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
        """
        if num_periods is None:
            num_periods = len(self.rates_df)
        
        # Use custom volatility if provided, otherwise use from parameters
        volatility = custom_volatility or self.parameters['volatility_epsilon']
        mean = self.parameters['mean_epsilon']
        
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
    
    def analyze_mispricing_distribution(self):
        """Analyze the distribution of mispricing terms"""
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

#############################################
# Part 4: Arbitrage Detection              #
#############################################

class ArbitrageDetector:
    """
    Detects arbitrage opportunities and calculates profits
    
    This class provides functionality to:
    1. Detect arbitrage opportunities based on rate discrepancies
    2. Calculate theoretical profits from identified opportunities
    3. Analyze feasibility considering transaction costs
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
        """Detect arbitrage opportunities based on rate discrepancies"""
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
        """Calculate theoretical profits from identified opportunities"""
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
        """Analyze feasibility of opportunities considering transaction costs"""
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
        """Return detection and analysis results"""
        if self.results_df is None:
            self.analyze_feasibility()
        
        return self.results_df, self.feasibility_summary

#############################################
# Part 5: Visualization                    #
#############################################

class ArbitrageVisualizer:
    """
    Comprehensive visualization tools for triangular arbitrage analysis
    
    This class provides functionality to:
    1. Create integrated dashboards of arbitrage results
    2. Generate publication-quality figures for reports
    3. Visualize relationships between different parameters
    """
    
    def __init__(self, exchange_data, parameter_estimator=None, 
                 mispricing_simulator=None, arbitrage_detector=None):
        """
        Initialize the visualizer with data from all components
        
        Parameters:
        -----------
        exchange_data : ExchangeRateData
            Exchange rate data object
        parameter_estimator : ParameterEstimator, optional
            Parameter estimation object
        mispricing_simulator : MispricingSimulator, optional
            Mispricing simulation object
        arbitrage_detector : ArbitrageDetector, optional
            Arbitrage detection object
        """
        self.exchange_data = exchange_data
        self.parameter_estimator = parameter_estimator
        self.mispricing_simulator = mispricing_simulator
        self.arbitrage_detector = arbitrage_detector
    
    def create_exchange_rate_dashboard(self, time_period=None):
        """
        Create a dashboard of exchange rate data
        
        Parameters:
        -----------
        time_period : tuple, optional
            Time period to display (start_date, end_date)
        """
        rates_df = self.exchange_data.get_rates()
        
        # Filter by time period if specified
        if time_period is not None:
            start_date, end_date = time_period
            rates_df = rates_df.loc[start_date:end_date]
        
        # Create figure
        fig = plt.figure(figsize=(15, 12))
        gs = GridSpec(3, 2, figure=fig)
        
        # Plot EUR/USD
        ax1 = fig.add_subplot(gs[0, 0])
        rates_df['EUR/USD'].plot(ax=ax1, color='blue', linewidth=2)
        ax1.set_title('EUR/USD Exchange Rate', fontsize=14)
        ax1.set_ylabel('Rate', fontsize=12)
        ax1.grid(True)
        
        # Plot USD/JPY
        ax2 = fig.add_subplot(gs[0, 1])
        rates_df['USD/JPY'].plot(ax=ax2, color='green', linewidth=2)
        ax2.set_title('USD/JPY Exchange Rate', fontsize=14)
        ax2.set_ylabel('Rate', fontsize=12)
        ax2.grid(True)
        
        # Plot EUR/JPY (actual and implied)
        ax3 = fig.add_subplot(gs[1, :])
        rates_df['EUR/JPY'].plot(ax=ax3, color='red', linewidth=2, label='Actual')
        rates_df['EUR/JPY_implied'].plot(ax=ax3, color='orange', linestyle='--', 
                                         linewidth=2, label='Implied')
        ax3.set_title('EUR/JPY Exchange Rate (Actual vs Implied)', fontsize=14)
        ax3.set_ylabel('Rate', fontsize=12)
        ax3.legend(fontsize=12)
        ax3.grid(True)
        
        # Plot mispricing (epsilon)
        ax4 = fig.add_subplot(gs[2, 0])
        rates_df['Epsilon'].plot(ax=ax4, color='purple', linewidth=2)
        ax4.set_title('Mispricing (Epsilon)', fontsize=14)
        ax4.set_ylabel('Epsilon', fontsize=12)
        ax4.set_xlabel('Date', fontsize=12)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.grid(True)
        
        # Plot histogram of mispricing
        ax5 = fig.add_subplot(gs[2, 1])
        rates_df['Epsilon'].hist(ax=ax5, bins=30, color='purple', alpha=0.7)
        ax5.set_title('Distribution of Mispricing', fontsize=14)
        ax5.set_xlabel('Epsilon', fontsize=12)
        ax5.set_ylabel('Frequency', fontsize=12)
        ax5.grid(True)
        
        # Format x-axis dates
        for ax in [ax1, ax2, ax3, ax4]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        return fig
    
    def create_arbitrage_dashboard(self, time_period=None):
        """
        Create a dashboard of arbitrage detection results
        
        Parameters:
        -----------
        time_period : tuple, optional
            Time period to display (start_date, end_date)
        """
        if self.arbitrage_detector is None:
            raise ValueError("Arbitrage detector not provided. Cannot create dashboard.")
        
        results_df, feasibility_summary = self.arbitrage_detector.get_results()
        
        # Filter by time period if specified
        if time_period is not None:
            start_date, end_date = time_period
            results_df = results_df.loc[start_date:end_date]
        
        # Create figure
        fig = plt.figure(figsize=(15, 15))
        gs = GridSpec(4, 2, figure=fig)
        
        # Plot profit time series
        ax1 = fig.add_subplot(gs[0, :])
        results_df['Percentage_Profit'].plot(ax=ax1, color='green', linewidth=2)
        if 'transaction_costs' in feasibility_summary:
            ax1.axhline(
                y=feasibility_summary['transaction_costs'],
                color='red', linestyle='--', linewidth=2,
                label=f"Transaction Costs: {feasibility_summary['transaction_costs']:.4f}%"
            )
        ax1.set_title('Arbitrage Profit Over Time', fontsize=14)
        ax1.set_ylabel('Profit (%)', fontsize=12)
        ax1.legend(fontsize=12)
        ax1.grid(True)
        
        # Plot profit distribution
        ax2 = fig.add_subplot(gs[1, 0])
        results_df['Percentage_Profit'].hist(ax=ax2, bins=30, color='green', alpha=0.7)
        if 'transaction_costs' in feasibility_summary:
            ax2.axvline(
                x=feasibility_summary['transaction_costs'],
                color='red', linestyle='--', linewidth=2,
                label=f"Transaction Costs: {feasibility_summary['transaction_costs']:.4f}%"
            )
        ax2.set_title('Distribution of Percentage Profits', fontsize=14)
        ax2.set_xlabel('Profit (%)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.legend(fontsize=12)
        ax2.grid(True)
        
        # Plot net profit distribution (feasible opportunities)
        ax3 = fig.add_subplot(gs[1, 1])
        if 'Net_Profit' in results_df.columns and 'Feasible_Opportunity' in results_df.columns:
            feasible_profits = results_df['Net_Profit'][results_df['Feasible_Opportunity']]
            if len(feasible_profits) > 0:
                feasible_profits.hist(ax=ax3, bins=30, color='blue', alpha=0.7)
                ax3.set_title('Distribution of Net Profits (Feasible Opportunities)', fontsize=14)
            else:
                ax3.set_title('No Feasible Opportunities', fontsize=14)
        else:
            ax3.set_title('Net Profit Data Not Available', fontsize=14)
        ax3.set_xlabel('Net Profit (%)', fontsize=12)
        ax3.set_ylabel('Frequency', fontsize=12)
        ax3.grid(True)
        
        # Plot opportunity frequency
        ax4 = fig.add_subplot(gs[2, 0])
        opportunity_counts = {
            'Case 1\n(EUR/JPY Overpriced)': results_df['Case_1_Opportunity'].sum(),
            'Case 2\n(EUR/JPY Underpriced)': results_df['Case_2_Opportunity'].sum(),
            'Any\nOpportunity': results_df['Any_Opportunity'].sum()
        }
        if 'Feasible_Opportunity' in results_df.columns:
            opportunity_counts['Feasible\nOpportunities'] = results_df['Feasible_Opportunity'].sum()
        
        total_periods = len(results_df)
        opportunity_percentages = {
            k: (v / total_periods) * 100 for k, v in opportunity_counts.items()
        }
        
        bars = ax4.bar(
            opportunity_percentages.keys(),
            opportunity_percentages.values(),
            color=['blue', 'green', 'orange', 'red'][:len(opportunity_percentages)]
        )
        
        for bar in bars:
            height = bar.get_height()
            ax4.text(
                bar.get_x() + bar.get_width() / 2.,
                height + 0.5,
                f'{height:.2f}%',
                ha='center', va='bottom'
            )
        
        ax4.set_title('Frequency of Arbitrage Opportunities', fontsize=14)
        ax4.set_ylabel('Percentage of Periods (%)', fontsize=12)
        ax4.set_ylim(0, max(opportunity_percentages.values()) * 1.2)
        ax4.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot epsilon vs profit scatter
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.scatter(
            results_df['Epsilon'].abs(),
            results_df['Percentage_Profit'],
            alpha=0.5, color='purple'
        )
        
        z = np.polyfit(results_df['Epsilon'].abs(), results_df['Percentage_Profit'], 1)
        p = np.poly1d(z)
        ax5.plot(
            sorted(results_df['Epsilon'].abs()),
            p(sorted(results_df['Epsilon'].abs())),
            "r--", linewidth=2
        )
        
        ax5.set_title('Relationship Between |Epsilon| and Profit', fontsize=14)
        ax5.set_xlabel('|Epsilon| (Absolute Mispricing)', fontsize=12)
        ax5.set_ylabel('Profit (%)', fontsize=12)
        ax5.grid(True)
        
        # Plot feasibility summary table
        ax6 = fig.add_subplot(gs[3, :])
        
        # Create summary table
        summary_data = {
            'Metric': [
                'Total Opportunities',
                'Feasible Opportunities',
                'Feasible Percentage',
                'Average Profit',
                'Average Net Profit (Feasible)',
                'Max Profit',
                'Max Net Profit',
                'Transaction Costs'
            ],
            'Value': [
                feasibility_summary.get('total_opportunities', 'N/A'),
                feasibility_summary.get('feasible_opportunities', 'N/A'),
                f"{feasibility_summary.get('feasible_percentage', 'N/A'):.2f}%",
                f"{feasibility_summary.get('average_profit', 'N/A'):.4f}%",
                f"{feasibility_summary.get('average_net_profit', 'N/A'):.4f}%",
                f"{feasibility_summary.get('max_profit', 'N/A'):.4f}%",
                f"{feasibility_summary.get('max_net_profit', 'N/A'):.4f}%",
                f"{feasibility_summary.get('transaction_costs', 'N/A'):.4f}%"
            ]
        }
        
        # Hide axes
        ax6.axis('off')
        
        # Create table
        table = ax6.table(
            cellText=[[m, v] for m, v in zip(summary_data['Metric'], summary_data['Value'])],
            colLabels=['Metric', 'Value'],
            loc='center',
            cellLoc='center'
        )
        
        # Style table
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)
        
        # Format x-axis dates for time series plots
        for ax in [ax1]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        return fig

#############################################
# Part 6: Main Execution                   #
#############################################

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
    
    # Create exchange rate dashboard
    exchange_dashboard = visualizer.create_exchange_rate_dashboard()
    plt.figure(exchange_dashboard.number)
    plt.show()
    
    # Create arbitrage dashboard
    arbitrage_dashboard = visualizer.create_arbitrage_dashboard()
    plt.figure(arbitrage_dashboard.number)
    plt.show()
    
    print("\nSimulation completed successfully!")
    
    return exchange_data, estimator, simulator, detector, visualizer

# Run the simulation
exchange_data, estimator, simulator, detector, visualizer = run_triangular_arbitrage_simulation(
    num_periods=1000,
    transaction_cost=0.0005,  # 0.05% per trade
    volatility_epsilon=0.001,
    correlation=-0.84
)
