"""
Visualization Components for Triangular Arbitrage

This module provides comprehensive visualization tools for the triangular arbitrage system.
It integrates data from all components to create insightful visualizations for analysis and presentation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.dates as mdates

class ArbitrageVisualizer:
    """
    Comprehensive visualization tools for triangular arbitrage analysis
    
    This class provides functionality to:
    1. Create integrated dashboards of arbitrage results
    2. Generate publication-quality figures for reports
    3. Visualize relationships between different parameters
    4. Create animated visualizations of arbitrage opportunities over time
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
        
        # Set default style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Configure default figure settings
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        
    def create_exchange_rate_dashboard(self, time_period=None):
        """
        Create a dashboard of exchange rate data
        
        Parameters:
        -----------
        time_period : tuple, optional
            Time period to display (start_date, end_date)
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure containing the dashboard
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
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure containing the dashboard
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
        
        # Plot feasibility heatmap
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
    
    def create_parameter_dashboard(self):
        """
        Create a dashboard of parameter estimation results
        
        Returns:
        --------
        matplotlib.figure.Figure
            Figure containing the dashboard
        """
        if self.parameter_estimator is None:
            raise ValueError("Parameter estimator not provided. Cannot create dashboard.")
        
        # Get parameters
        parameters = self.parameter_estimator.get_parameters()
        
        # Create figure
        fig = plt.figure(figsize=(15, 12))
        gs = GridSpec(3, 2, figure=fig)
        
        # Plot rolling volatilities
        window_size = 30  # Default window size
        
        # Get returns data
        returns_df = self.exchange_data.get_returns()
        
        # Calculate rolling volatilities
        rolling_vol_eu = returns_df['Log_r_EUR/USD'].rolling(window=window_size).std(ddof=1)
        rolling_vol_uj = returns_df['Log_r_USD/JPY'].rolling(window=window_size).std(ddof=1)
        
        # Calculate rolling correlation
        rolling_corr = returns_df['Log_r_EUR/USD'].rolling(window=window_size).corr(
            returns_df['Log_r_USD/JPY']
        )
        
        # Calculate rolling mispricing volatility
        rates_df = self.exchange_data.get_rates()
        rolling_vol_eps = rates_df['Epsilon'].rolling(window=window_size).std(ddof=1)
        
        # Plot rolling volatility EUR/USD
        ax1 = fig.add_subplot(gs[0, 0])
        rolling_vol_eu.plot(ax=ax1, color='blue', linewidth=2)
        ax1.axhline(
            y=parameters['volatility_EUR/USD'],
            color='red', linestyle='--', linewidth=2,
            label=f"Overall: {parameters['volatility_EUR/USD']:.6f}"
        )
        ax1.set_title(f'Rolling Volatility EUR/USD (Window={window_size})', fontsize=14)
        ax1.set_ylabel('Volatility', fontsize=12)
        ax1.legend(fontsize=12)
        ax1.grid(True)
        
        # Plot rolling volatility USD/JPY
        ax2 = fig.add_subplot(gs[0, 1])
        rolling_vol_uj.plot(ax=ax2, color='green', linewidth=2)
        ax2.axhline(
            y=parameters['volatility_USD/JPY'],
            color='red', linestyle='--', linewidth=2,
            label=f"Overall: {parameters['volatility_USD/JPY']:.6f}"
        )
        ax2.set_title(f'Rolling Volatility USD/JPY (Window={window_size})', fontsize=14)
        ax2.set_ylabel('Volatility', fontsize=12)
        ax2.legend(fontsize=12)
        ax2.grid(True)
        
        # Plot rolling correlation
        ax3 = fig.add_subplot(gs[1, 0])
        rolling_corr.plot(ax=ax3, color='red', linewidth=2)
        ax3.axhline(
            y=parameters['correlation'],
            color='blue', linestyle='--', linewidth=2,
            label=f"Overall: {parameters['correlation']:.6f}"
        )
        ax3.set_title(f'Rolling Correlation (Window={window_size})', fontsize=14)
        ax3.set_ylabel('Correlation', fontsize=12)
        ax3.legend(fontsize=12)
        ax3.grid(True)
        
        # Plot rolling mispricing volatility
        ax4 = fig.add_subplot(gs[1, 1])
        rolling_vol_eps.plot(ax=ax4, color='purple', linewidth=2)
        ax4.axhline(
            y=parameters['volatility_epsilon'],
            color='orange', linestyle='--', linewidth=2,
            label=f"Overall: {parameters['volatility_epsilon']:.6f}"
        )
        ax4.set_title(f'Rolling Volatility Epsilon (Window={window_size})', fontsize=14)
        ax4.set_ylabel('Volatility', fontsize=12)
        ax4.legend(fontsize=12)
        ax4.grid(True)
        
        # Plot return distributions
        ax5 = fig.add_subplot(gs[2, 0])
        sns.histplot(returns_df['Log_r_EUR/USD'], ax=ax5, kde=True, color='blue', stat='density')
        sns.histplot(returns_df['Log_r_USD/JPY'], ax=ax5, kde=True, color='green', stat='density')
        ax5.set_title('Distribution of Log Returns', fontsize=14)
        ax5.set_xlabel('Log Return', fontsize=12)
        ax5.set_ylabel('Density', fontsize=12)
        ax5.legend(['EUR/USD', 'USD/JPY'], fontsize=12)
        ax5.grid(True)
        
        # Plot parameter summary
        ax6 = fig.add_subplot(gs[2, 1])
        
        # Create summary table
        summary_data = {
            'Parameter': [
                'Volatility EUR/USD',
                'Volatility USD/JPY',
                'Correlation',
                'Volatility Epsilon'
            ],
            'Value': [
                f"{parameters.get('volatility_EUR/USD', 'N/A'):.6f}",
                f"{parameters.get('volatility_USD/JPY', 'N/A'):.6f}",
                f"{parameters.get('correlation', 'N/A'):.6f}",
                f"{parameters.get('volatility_epsilon', 'N/A'):.6f}"
            ]
        }
        
        # Hide axes
        ax6.axis('off')
        
        # Create table
        table = ax6.table(
            cellText=[[p, v] for p, v in zip(summary_data['Parameter'], summary_data['Value'])],
            colLabels=['Parameter', 'Value'],
            loc='center',
            cellLoc='center'
        )
        
        # Style table
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)
        
        # Format x-axis dates for time series plots
        for ax in [ax1, ax2, ax3, ax4]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        return fig
    
    def create_transaction_cost_analysis(self, cost_range=None):
        """
        Create analysis of arbitrage feasibility across different transaction costs
        
        Parameters:
        -----------
        cost_range : list, optional
            List of transaction cost percentages to analyze
            If None, uses default range
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure containing the analysis
        """
        if self.arbitrage_detector is None:
            raise ValueError("Arbitrage detector not provided. Cannot create analysis.")
        
        # Default cost range if not provided
        if cost_range is None:
            cost_range = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01]
        
        # Get original results
        original_results, _ = self.arbitrage_detector.get_results()
        
        # Initialize storage for results
        feasible_percentages = []
        avg_net_profits = []
        
        # Calculate feasibility for each cost
        for cost in cost_range:
            # Calculate total costs (3 trades)
            total_cost = cost * 3 * 100  # Convert to percentage
            
            # Calculate net profit
            net_profit = original_results['Percentage_Profit'] - total_cost
            
            # Determine feasible opportunities
            feasible = net_profit > 0
            
            # Calculate metrics
            feasible_count = feasible.sum()
            total_count = len(original_results)
            feasible_pct = (feasible_count / total_count) * 100 if total_count > 0 else 0
            avg_net = net_profit[feasible].mean() if feasible_count > 0 else 0
            
            # Store results
            feasible_percentages.append(feasible_pct)
            avg_net_profits.append(avg_net)
        
        # Convert cost range to percentage for display
        cost_pct = [c * 100 for c in cost_range]
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot feasible percentage vs transaction cost
        axes[0].plot(cost_pct, feasible_percentages, 'o-', color='blue', linewidth=2)
        axes[0].set_title('Feasible Opportunities vs Transaction Cost', fontsize=14)
        axes[0].set_xlabel('Transaction Cost per Trade (%)', fontsize=12)
        axes[0].set_ylabel('Feasible Opportunities (%)', fontsize=12)
        axes[0].grid(True)
        
        # Plot average net profit vs transaction cost
        axes[1].plot(cost_pct, avg_net_profits, 'o-', color='green', linewidth=2)
        axes[1].set_title('Average Net Profit vs Transaction Cost', fontsize=14)
        axes[1].set_xlabel('Transaction Cost per Trade (%)', fontsize=12)
        axes[1].set_ylabel('Average Net Profit (%)', fontsize=12)
        axes[1].grid(True)
        
        plt.tight_layout()
        return fig
    
    def create_comprehensive_report(self, output_file=None):
        """
        Create a comprehensive report with all visualizations
        
        Parameters:
        -----------
        output_file : str, optional
            Path to save the report (PDF format)
            If None, displays the figures but doesn't save
            
        Returns:
        --------
        list
            List of figures created for the report
        """
        # Create all dashboards
        figures = []
        
        try:
            # Exchange rate dashboard
            fig1 = self.create_exchange_rate_dashboard()
            figures.append(fig1)
            
            # Parameter dashboard
            if self.parameter_estimator is not None:
                fig2 = self.create_parameter_dashboard()
                figures.append(fig2)
            
            # Arbitrage dashboard
            if self.arbitrage_detector is not None:
                fig3 = self.create_arbitrage_dashboard()
                figures.append(fig3)
                
                # Transaction cost analysis
                fig4 = self.create_transaction_cost_analysis()
                figures.append(fig4)
        
        except Exception as e:
            print(f"Error creating report: {e}")
        
        # Save to PDF if output file specified
        if output_file is not None:
            from matplotlib.backends.backend_pdf import PdfPages
            
            with PdfPages(output_file) as pdf:
                for fig in figures:
                    pdf.savefig(fig)
                    plt.close(fig)
                
                # Add metadata
                d = pdf.infodict()
                d['Title'] = 'Triangular Arbitrage Analysis Report'
                d['Author'] = 'Triangular Arbitrage System'
                d['Subject'] = 'Forex Triangular Arbitrage Analysis'
                d['Keywords'] = 'forex, arbitrage, analysis, visualization'
                d['CreationDate'] = pd.Timestamp.now()
        
        return figures


# Example usage (requires all previous modules)
if __name__ == "__main__":
    from triangular_arbitrage_implementation import ExchangeRateData
    from parameter_estimation import ParameterEstimator
    from mispricing_calculation import MispricingSimulator
    from arbitrage_detection import ArbitrageDetector
    
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
    
    # Create visualizer
    visualizer = ArbitrageVisualizer(
        exchange_data,
        parameter_estimator=estimator,
        mispricing_simulator=simulator,
        arbitrage_detector=detector
    )
    
    # Create exchange rate dashboard
    visualizer.create_exchange_rate_dashboard()
    plt.show()
    
    # Create parameter dashboard
    visualizer.create_parameter_dashboard()
    plt.show()
    
    # Create arbitrage dashboard
    visualizer.create_arbitrage_dashboard()
    plt.show()
    
    # Create transaction cost analysis
    visualizer.create_transaction_cost_analysis()
    plt.show()
    
    # Create comprehensive report
    visualizer.create_comprehensive_report(output_file="triangular_arbitrage_report.pdf")
