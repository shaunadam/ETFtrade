#!/usr/bin/env python3
"""
Parameter Optimizer for Backtest Engine

Extracted from BacktestEngine to handle walk-forward parameter optimization.
Provides grid search optimization for trading parameters using various fitness functions.

Usage:
    from parameter_optimizer import ParameterOptimizer
    from backtest_config import BacktestConfiguration
    
    config = BacktestConfiguration.default()
    optimizer = ParameterOptimizer(config)
    
    optimal_params = optimizer.optimize_parameters(
        training_days=training_days,
        setup_types=setup_types,
        regime_aware=True
    )
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Callable, Dict, Any
from datetime import datetime
from itertools import product

from backtest_config import BacktestConfiguration
from trade_setups import SetupType


@dataclass
class OptimizationParameters:
    """Parameters to optimize during walk-forward validation."""
    stop_loss_pct: float = 0.05  # 5% stop loss
    profit_target_r: float = 2.0  # 2R profit target
    confidence_threshold: float = 0.6  # Minimum confidence for signals
    max_holding_days: int = 60  # Maximum days to hold position
    position_size_method: str = "fixed_risk"  # fixed_risk, kelly, equal_weight


class ParameterOptimizer:
    """
    Handles parameter optimization for backtesting using grid search.
    
    Supports multiple fitness functions and walk-forward validation.
    """
    
    def __init__(self, config: BacktestConfiguration):
        """
        Initialize parameter optimizer.
        
        Args:
            config: Backtest configuration containing optimization parameters
        """
        self.config = config
        self.logger = logging.getLogger('backtest.optimizer')
        
        # Fitness function mapping
        self.fitness_functions = {
            'sharpe_ratio': self._fitness_sharpe_ratio,
            'calmar_ratio': self._fitness_calmar_ratio,
            'profit_factor': self._fitness_profit_factor
        }
        
        self.logger.debug(f"Initialized ParameterOptimizer with fitness function: {config.optimization.fitness_function}")
    
    def optimize_parameters(self, 
                          training_days: List[datetime],
                          setup_types: List[SetupType],
                          regime_aware: bool,
                          selected_instruments: Optional[List[str]] = None,
                          backtest_runner: Optional[Callable] = None) -> OptimizationParameters:
        """
        Optimize parameters using grid search on training data.
        
        Args:
            training_days: List of trading days for training
            setup_types: List of setup types to test
            regime_aware: Whether to use regime-aware filtering
            selected_instruments: Optional list of specific instruments
            backtest_runner: Function to run backtest with given parameters
            
        Returns:
            OptimizationParameters with best parameter combination
        """
        if not backtest_runner:
            raise ValueError("backtest_runner function is required for optimization")
        
        self.logger.info(f"Starting parameter optimization on {len(training_days)} training days")
        
        # Get parameter ranges from config
        stop_loss_range = self.config.optimization.stop_loss_range
        profit_target_range = self.config.optimization.profit_target_range
        confidence_range = self.config.optimization.confidence_range
        
        total_combinations = len(stop_loss_range) * len(profit_target_range) * len(confidence_range)
        
        if self.config.debug.verbose_optimization:
            self.logger.debug(f"Testing {total_combinations} parameter combinations")
            self.logger.debug(f"Stop loss range: {stop_loss_range}")
            self.logger.debug(f"Profit target range: {profit_target_range}")
            self.logger.debug(f"Confidence range: {confidence_range}")
        
        best_params = self._get_default_parameters()
        best_score = float('-inf')
        optimization_results = []
        
        # Grid search over parameter combinations
        for i, (stop_loss, target_r, confidence) in enumerate(product(stop_loss_range, profit_target_range, confidence_range)):
            # Create test parameters
            test_params = OptimizationParameters(
                stop_loss_pct=stop_loss,
                profit_target_r=target_r,
                confidence_threshold=confidence,
                max_holding_days=self.config.trading.max_holding_days,
                position_size_method=self.config.risk_management.position_size_method
            )
            
            try:
                # Run backtest with these parameters
                result = backtest_runner(
                    test_params, 
                    training_days, 
                    setup_types, 
                    regime_aware, 
                    selected_instruments
                )
                
                # Calculate fitness score
                score = self._calculate_fitness_score(result)
                
                optimization_results.append({
                    'params': test_params,
                    'score': score,
                    'result': result
                })
                
                if self.config.debug.verbose_optimization:
                    self.logger.debug(f"Combo {i+1}/{total_combinations}: "
                                    f"stop_loss={stop_loss:.1%}, target={target_r:.1f}R, "
                                    f"conf={confidence:.2f}, score={score:.3f}")
                
                if score > best_score:
                    best_score = score
                    best_params = test_params
                    
                    if self.config.debug.verbose_optimization:
                        self.logger.debug(f"New best score: {best_score:.3f}")
                
            except Exception as e:
                self.logger.warning(f"Error evaluating parameter combination {i+1}: {e}")
                continue
        
        # Log final results
        self.logger.info(f"Optimization complete. Best parameters: "
                        f"stop_loss={best_params.stop_loss_pct:.1%}, "
                        f"target={best_params.profit_target_r:.1f}R, "
                        f"confidence={best_params.confidence_threshold:.2f}, "
                        f"score={best_score:.3f}")
        
        if self.config.debug.verbose_optimization:
            self._log_optimization_summary(optimization_results)
        
        return best_params
    
    def _calculate_fitness_score(self, backtest_result: Dict[str, Any]) -> float:
        """
        Calculate fitness score for optimization result.
        
        Args:
            backtest_result: Result dictionary from backtest
            
        Returns:
            Fitness score (higher is better)
        """
        if 'performance' not in backtest_result:
            return float('-inf')
        
        performance = backtest_result['performance']
        trades = backtest_result.get('trades', [])
        
        # Check minimum trades requirement
        if len(trades) < self.config.optimization.min_trades_required:
            self.logger.debug(f"Insufficient trades ({len(trades)}) for valid optimization")
            return float('-inf')
        
        # Get fitness function
        fitness_func = self.fitness_functions.get(
            self.config.optimization.fitness_function,
            self._fitness_sharpe_ratio
        )
        
        # Calculate base fitness score
        base_score = fitness_func(performance)
        
        # Apply trade frequency penalty
        trade_frequency_penalty = self._calculate_trade_frequency_penalty(len(trades))
        
        final_score = base_score - trade_frequency_penalty
        
        if self.config.debug.verbose_optimization:
            self.logger.debug(f"Fitness calculation: base={base_score:.3f}, "
                            f"penalty={trade_frequency_penalty:.3f}, "
                            f"final={final_score:.3f}")
        
        return final_score
    
    def _fitness_sharpe_ratio(self, performance) -> float:
        """Fitness function based on Sharpe ratio."""
        return getattr(performance, 'sharpe_ratio', 0.0)
    
    def _fitness_calmar_ratio(self, performance) -> float:
        """Fitness function based on Calmar ratio."""
        return getattr(performance, 'calmar_ratio', 0.0)
    
    def _fitness_profit_factor(self, performance) -> float:
        """Fitness function based on profit factor."""
        profit_factor = getattr(performance, 'profit_factor', 1.0)
        # Convert profit factor to a more reasonable scale for optimization
        return min(profit_factor - 1.0, 5.0)  # Cap at 5.0 to prevent extreme values
    
    def _calculate_trade_frequency_penalty(self, trade_count: int) -> float:
        """
        Calculate penalty for excessive trading frequency.
        
        Args:
            trade_count: Number of trades executed
            
        Returns:
            Penalty value to subtract from fitness score
        """
        penalty_threshold = 50  # Trades before penalty kicks in
        penalty_rate = self.config.optimization.trade_frequency_penalty
        
        if trade_count <= penalty_threshold:
            return 0.0
        
        excess_trades = trade_count - penalty_threshold
        penalty = excess_trades * penalty_rate
        
        return penalty
    
    def _get_default_parameters(self) -> OptimizationParameters:
        """Get default optimization parameters from config."""
        return OptimizationParameters(
            stop_loss_pct=self.config.trading.default_stop_loss_pct,
            profit_target_r=self.config.trading.default_profit_target_r,
            confidence_threshold=self.config.trading.default_confidence_threshold,
            max_holding_days=self.config.trading.max_holding_days,
            position_size_method=self.config.risk_management.position_size_method
        )
    
    def _log_optimization_summary(self, optimization_results: List[Dict]) -> None:
        """Log summary of optimization results."""
        if not optimization_results:
            return
        
        # Sort by score
        sorted_results = sorted(optimization_results, key=lambda x: x['score'], reverse=True)
        
        self.logger.debug("Top 5 parameter combinations:")
        for i, result in enumerate(sorted_results[:5]):
            params = result['params']
            score = result['score']
            self.logger.debug(f"  {i+1}. stop_loss={params.stop_loss_pct:.1%}, "
                            f"target={params.profit_target_r:.1f}R, "
                            f"conf={params.confidence_threshold:.2f}, "
                            f"score={score:.3f}")
        
        # Statistics
        scores = [r['score'] for r in optimization_results if r['score'] != float('-inf')]
        if scores:
            self.logger.debug(f"Score statistics: "
                            f"mean={sum(scores)/len(scores):.3f}, "
                            f"min={min(scores):.3f}, "
                            f"max={max(scores):.3f}")
    
    def validate_parameter_ranges(self) -> List[str]:
        """
        Validate that parameter ranges are reasonable.
        
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Check stop loss range
        stop_loss_range = self.config.optimization.stop_loss_range
        if not stop_loss_range or not all(0.01 <= sl <= 0.20 for sl in stop_loss_range):
            errors.append("Stop loss range must contain values between 1% and 20%")
        
        # Check profit target range
        profit_target_range = self.config.optimization.profit_target_range
        if not profit_target_range or not all(0.5 <= pt <= 10.0 for pt in profit_target_range):
            errors.append("Profit target range must contain values between 0.5R and 10R")
        
        # Check confidence range
        confidence_range = self.config.optimization.confidence_range
        if not confidence_range or not all(0.0 <= conf <= 1.0 for conf in confidence_range):
            errors.append("Confidence range must contain values between 0.0 and 1.0")
        
        # Check fitness function
        if self.config.optimization.fitness_function not in self.fitness_functions:
            errors.append(f"Unknown fitness function: {self.config.optimization.fitness_function}")
        
        return errors


def create_parameter_grid(config: BacktestConfiguration) -> List[OptimizationParameters]:
    """
    Create a list of all parameter combinations for grid search.
    
    Args:
        config: Backtest configuration
        
    Returns:
        List of OptimizationParameters for grid search
    """
    stop_loss_range = config.optimization.stop_loss_range
    profit_target_range = config.optimization.profit_target_range
    confidence_range = config.optimization.confidence_range
    
    parameter_grid = []
    
    for stop_loss, target_r, confidence in product(stop_loss_range, profit_target_range, confidence_range):
        params = OptimizationParameters(
            stop_loss_pct=stop_loss,
            profit_target_r=target_r,
            confidence_threshold=confidence,
            max_holding_days=config.trading.max_holding_days,
            position_size_method=config.risk_management.position_size_method
        )
        parameter_grid.append(params)
    
    return parameter_grid


if __name__ == "__main__":
    # Example usage
    from backtest_config import BacktestConfiguration
    
    config = BacktestConfiguration.default()
    optimizer = ParameterOptimizer(config)
    
    # Validate parameter ranges
    errors = optimizer.validate_parameter_ranges()
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("✅ Parameter ranges are valid")
        
        # Show parameter grid size
        param_grid = create_parameter_grid(config)
        print(f"Parameter grid contains {len(param_grid)} combinations")