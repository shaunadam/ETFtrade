#!/usr/bin/env python3
"""
Walk-forward backtesting engine for ETF trading system.

This module implements comprehensive backtesting with:
- Walk-forward validation to prevent overfitting
- Regime-aware performance analysis
- Portfolio management with risk controls
- Performance analytics and benchmark comparison

Usage:
    # Backtest from CSV file exported by screener
    python backtest.py --csv-file etf_signals_20250630.csv
    
    # Traditional setup-based backtesting
    python backtest.py --setup trend_pullback --walk-forward
    python backtest.py --all-setups --regime-aware
    
    # In-memory pipeline usage:
    from screener import ETFScreener
    from backtest import BacktestEngine
    from datetime import datetime
    
    # Get signals from screener
    screener = ETFScreener()
    signals = screener.screen_etfs(regime_filter=True)
    
    # Backtest those specific signals
    engine = BacktestEngine()
    results = engine.backtest_from_signals(
        signals=signals,
        start_date=datetime(2025, 1, 1),
        end_date=datetime(2025, 6, 15)
    )
"""

import argparse
import json
import sqlite3
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, NamedTuple
from enum import Enum
import pandas as pd
import numpy as np
from itertools import product

from data_cache import DataCache
from regime_detection import RegimeDetector, RegimeData
from trade_setups import SetupManager, SetupType, TradeSignal, SignalStrength


class TradeStatus(Enum):
    OPEN = "open"
    CLOSED_PROFIT = "closed_profit"
    CLOSED_LOSS = "closed_loss"
    CLOSED_STOP = "closed_stop"
    CLOSED_TARGET = "closed_target"


@dataclass
class BacktestTrade:
    """Individual trade record for backtesting."""
    trade_id: int
    symbol: str
    setup_type: SetupType
    entry_date: datetime
    entry_price: float
    position_size: float
    stop_loss: float
    target_price: float
    risk_per_share: float
    confidence: float
    regime_at_entry: RegimeData
    
    # Exit information
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    status: TradeStatus = TradeStatus.OPEN
    r_multiple: Optional[float] = None
    pnl: Optional[float] = None
    days_held: Optional[int] = None
    
    # Professional metrics tracking
    mae: float = 0.0  # Maximum Adverse Excursion
    mfe: float = 0.0  # Maximum Favorable Excursion


@dataclass
class OptimizationParameters:
    """Parameters to optimize during walk-forward validation."""
    stop_loss_pct: float = 0.05  # 5% stop loss
    profit_target_r: float = 2.0  # 2R profit target
    confidence_threshold: float = 0.6  # Minimum confidence for signals
    max_holding_days: int = 60  # Maximum days to hold position
    position_size_method: str = "fixed_risk"  # fixed_risk, kelly, equal_weight


@dataclass
class RegimePerformance:
    """Performance metrics broken down by market regime."""
    volatility_low: Optional['PerformanceMetrics'] = None
    volatility_medium: Optional['PerformanceMetrics'] = None  
    volatility_high: Optional['PerformanceMetrics'] = None
    trend_up: Optional['PerformanceMetrics'] = None
    trend_neutral: Optional['PerformanceMetrics'] = None
    trend_down: Optional['PerformanceMetrics'] = None
    risk_on: Optional['PerformanceMetrics'] = None
    risk_off: Optional['PerformanceMetrics'] = None


@dataclass
class PerformanceMetrics:
    """Performance metrics for backtesting results."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_r_multiple: float
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    calmar_ratio: float
    avg_days_held: float
    largest_winner: float
    largest_loser: float
    consecutive_wins: int
    consecutive_losses: int
    profit_factor: float
    
    # Professional metrics (Phase 1 enhancements)
    sortino_ratio: float = 0.0
    information_ratio: float = 0.0
    max_adverse_excursion: float = 0.0
    max_favorable_excursion: float = 0.0
    avg_mae: float = 0.0
    avg_mfe: float = 0.0
    annual_return: float = 0.0
    downside_deviation: float = 0.0
    tracking_error: float = 0.0
    benchmark_return: float = 0.0
    
    # Instrument type breakdown
    etf_trades: int = 0
    stock_trades: int = 0
    etf_win_rate: float = 0.0
    stock_win_rate: float = 0.0
    etf_avg_return: float = 0.0
    stock_avg_return: float = 0.0
    etf_avg_r_multiple: float = 0.0
    stock_avg_r_multiple: float = 0.0


class BacktestEngine:
    """Walk-forward backtesting engine with regime analysis."""
    
    def __init__(self, db_path: str = "journal.db", initial_capital: float = 100000):
        self.db_path = db_path
        self.initial_capital = initial_capital
        self.data_cache = DataCache(db_path)
        self.regime_detector = RegimeDetector(db_path)
        self.setup_manager = SetupManager(db_path)
        
        # Risk management parameters (optimized for ETF diversification)
        self.max_risk_per_trade = 0.02  # 2% per trade
        self.max_concurrent_positions = 6  # Increased from 4 for better ETF diversification
        self.max_sector_allocation = 0.30  # 30% max in correlated sectors
        self.max_similar_etfs = 2  # Max 2 ETFs from same category
        
        # Trade tracking
        self.trades: List[BacktestTrade] = []
        self.current_positions: List[BacktestTrade] = []
        self.trade_id_counter = 1
        
        # Performance tracking
        self.daily_equity = []
        self.daily_dates = []
        
        # Optimization parameters
        self.current_params = OptimizationParameters()
        self.regime_performance_tracking = []
    
    def backtest_from_signals(self,
                             signals: List[TradeSignal],
                             start_date: datetime,
                             end_date: datetime) -> Dict:
        """
        Backtest specific trade signals from screener results.
        
        Args:
            signals: List of TradeSignal objects from screener
            start_date: Backtest start date
            end_date: Backtest end date
            
        Returns:
            Dictionary with backtest results
        """
        print(f"Backtesting {len(signals)} signals from {start_date.date()} to {end_date.date()}")
        
        # Initialize tracking
        self.trades = []
        self.current_positions = []
        self.trade_id_counter = 1
        self.daily_equity = [self.initial_capital]
        self.daily_dates = []
        
        # Group signals by symbol and setup for targeted backtesting
        signal_map = {}
        for signal in signals:
            key = (signal.symbol, signal.setup_type)
            if key not in signal_map:
                signal_map[key] = []
            signal_map[key].append(signal)
        
        # Get trading days
        trading_days = self._get_trading_days(start_date, end_date)
        
        # Run backtest using only the provided signals
        for current_date in trading_days:
            self.daily_dates.append(current_date)
            
            # Update existing positions
            self._update_positions(current_date)
            
            # Look for entry opportunities from provided signals
            if len(self.current_positions) < self.max_concurrent_positions:
                available_signals = self._get_signals_for_date_from_provided(
                    current_date, signal_map
                )
                
                for signal in available_signals:
                    if self._can_enter_trade(signal, current_date):
                        self._enter_trade(signal, current_date)
                        
                        if len(self.current_positions) >= self.max_concurrent_positions:
                            break
            
            # Calculate daily equity
            current_equity = self._calculate_current_equity(current_date)
            self.daily_equity.append(current_equity)
        
        # Calculate performance metrics
        performance = self._calculate_performance_metrics()
        
        return {
            'performance': performance,
            'trades': [asdict(trade) for trade in self.trades],
            'signals_tested': len(signals),
            'unique_symbols': len(set(s.symbol for s in signals)),
            'setups_tested': list(set(s.setup_type.value for s in signals)),
            'daily_equity': self.daily_equity,
            'daily_dates': [d.isoformat() for d in self.daily_dates]
        }
    
    def backtest_from_csv(self,
                         csv_file: str,
                         start_date: datetime,
                         end_date: datetime) -> Dict:
        """
        Backtest signals from a CSV file exported by screener.
        
        Args:
            csv_file: Path to CSV file with screener results
            start_date: Backtest start date
            end_date: Backtest end date
            
        Returns:
            Dictionary with backtest results
        """
        signals = self._load_signals_from_csv(csv_file)
        return self.backtest_from_signals(signals, start_date, end_date)

    def run_backtest(self, 
                     start_date: datetime,
                     end_date: datetime,
                     setup_types: Optional[List[SetupType]] = None,
                     walk_forward: bool = False,
                     regime_aware: bool = True,
                     instrument_types: Optional[List[str]] = None) -> Dict:
        """
        Run comprehensive backtest with walk-forward validation.
        
        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            setup_types: List of setups to test (None for all)
            walk_forward: Enable walk-forward validation
            regime_aware: Include regime analysis
            
        Returns:
            Dictionary with backtest results
        """
        print(f"Starting backtest from {start_date.date()} to {end_date.date()}")
        
        if setup_types is None:
            setup_types = list(SetupType)
        
        # Initialize tracking variables
        self.trades = []
        self.current_positions = []
        self.trade_id_counter = 1
        self.daily_equity = [self.initial_capital]
        self.daily_dates = []
        
        # Get trading days
        trading_days = self._get_trading_days(start_date, end_date)
        
        if walk_forward:
            return self._run_walk_forward_backtest(
                trading_days, setup_types, regime_aware, instrument_types
            )
        else:
            return self._run_standard_backtest(
                trading_days, setup_types, regime_aware, instrument_types
            )
    
    def _run_standard_backtest(self, 
                              trading_days: List[datetime],
                              setup_types: List[SetupType],
                              regime_aware: bool,
                              instrument_types: Optional[List[str]] = None) -> Dict:
        """Run standard backtesting without walk-forward validation."""
        
        for current_date in trading_days:
            self.daily_dates.append(current_date)
            
            # Update existing positions
            self._update_positions(current_date)
            
            # Look for new trade opportunities
            if len(self.current_positions) < self.max_concurrent_positions:
                signals = self._get_signals_for_date(current_date, setup_types, regime_aware, instrument_types)
                
                for signal in signals:
                    if self._can_enter_trade(signal, current_date):
                        self._enter_trade(signal, current_date)
                        
                        if len(self.current_positions) >= self.max_concurrent_positions:
                            break
            
            # Calculate daily equity
            current_equity = self._calculate_current_equity(current_date)
            self.daily_equity.append(current_equity)
        
        # Calculate final performance metrics
        performance = self._calculate_performance_metrics()
        
        return {
            'performance': performance,
            'trades': [asdict(trade) for trade in self.trades],
            'daily_equity': self.daily_equity,
            'daily_dates': [d.isoformat() for d in self.daily_dates]
        }
    
    def _run_walk_forward_backtest(self,
                                  trading_days: List[datetime],
                                  setup_types: List[SetupType],
                                  regime_aware: bool,
                                  instrument_types: Optional[List[str]] = None) -> Dict:
        """Run walk-forward backtesting with parameter optimization."""
        
        # Walk-forward parameters
        training_period_days = 252  # 1 year training
        test_period_days = 63      # 3 months testing
        
        results = []
        total_trades = []
        optimization_history = []
        
        # Split into walk-forward periods
        for i in range(0, len(trading_days), test_period_days):
            training_start = max(0, i - training_period_days)
            training_end = i
            test_start = i
            test_end = min(len(trading_days), i + test_period_days)
            
            if test_start >= len(trading_days):
                break
            
            training_days = trading_days[training_start:training_end] if training_end > training_start else []
            test_days = trading_days[test_start:test_end]
                
            print(f"Walk-forward period: {trading_days[test_start].date()} to {trading_days[test_end-1].date()}")
            
            # Optimize parameters on training data
            if training_days:
                print(f"  Optimizing on training data: {training_days[0].date()} to {training_days[-1].date()}")
                optimal_params = self._optimize_parameters(training_days, setup_types, regime_aware)
                optimization_history.append({
                    'period': f"{training_days[0].date()}_{training_days[-1].date()}",
                    'optimal_params': asdict(optimal_params)
                })
            else:
                # Use default parameters for first period
                optimal_params = OptimizationParameters()
                optimization_history.append({
                    'period': 'default_first_period',
                    'optimal_params': asdict(optimal_params)
                })
            
            # Apply optimized parameters to test period
            self.current_params = optimal_params
            print(f"  Testing with: stop_loss={optimal_params.stop_loss_pct:.1%}, target={optimal_params.profit_target_r:.1f}R, conf={optimal_params.confidence_threshold:.2f}")
            
            # Test on out-of-sample period with optimized parameters
            period_result = self._run_standard_backtest(test_days, setup_types, regime_aware, instrument_types)
            period_result['optimization_params'] = asdict(optimal_params)
            
            results.append(period_result)
            total_trades.extend(period_result['trades'])
        
        # Combine results
        combined_performance = self._combine_walk_forward_results(results)
        
        return {
            'performance': combined_performance,
            'trades': total_trades,
            'walk_forward_results': results,
            'optimization_history': optimization_history,
            'regime_performance': self._calculate_regime_performance()
        }
    
    def _get_signals_for_date_from_provided(self,
                                           current_date: datetime,
                                           signal_map: Dict) -> List[TradeSignal]:
        """Get signals for a specific date from provided signal map."""
        available_signals = []
        
        for (symbol, setup_type), signals in signal_map.items():
            # For backtesting, we simulate that signals are available on any date
            # In reality, you'd want more sophisticated logic here
            if signals and len(signals) > 0:
                # Use the first signal for this symbol/setup combination
                signal = signals[0]
                
                # Update the signal's entry conditions for the current date
                current_price = self._get_price_for_date(symbol, current_date)
                if current_price is not None:
                    # Create a new signal instance for this date
                    updated_signal = TradeSignal(
                        symbol=signal.symbol,
                        setup_type=signal.setup_type,
                        signal_strength=signal.signal_strength,
                        confidence=signal.confidence,
                        entry_price=current_price,
                        stop_loss=current_price * 0.95,  # 5% stop loss
                        target_price=current_price * 1.10,  # 10% target
                        risk_per_share=current_price * 0.05,
                        position_size=signal.position_size,
                        regime_context=signal.regime_context,
                        notes=f"Backtesting signal from {signal.notes}"
                    )
                    available_signals.append(updated_signal)
        
        return available_signals
    
    def _load_signals_from_csv(self, csv_file: str) -> List[TradeSignal]:
        """Load trade signals from CSV file exported by screener."""
        signals = []
        
        try:
            import csv
            with open(csv_file, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                
                for row in reader:
                    try:
                        # Parse setup type
                        setup_type = SetupType(row['setup_type'])
                        
                        # Parse signal strength
                        signal_strength = SignalStrength(row['signal_strength'])
                        
                        # Create regime context (simplified for CSV)
                        from regime_detection import VolatilityRegime, TrendRegime, SectorRotation, RiskSentiment
                        regime_context = RegimeData(
                            date=datetime.now(),
                            volatility_regime=VolatilityRegime.MEDIUM,
                            trend_regime=TrendRegime.NEUTRAL,
                            sector_rotation=SectorRotation.NEUTRAL,
                            risk_sentiment=RiskSentiment.NEUTRAL,
                            vix_level=20.0,
                            spy_vs_sma200=0.0,
                            growth_value_ratio=1.0,
                            risk_on_off_ratio=1.0
                        )
                        
                        # Create TradeSignal object
                        signal = TradeSignal(
                            symbol=row['symbol'],
                            setup_type=setup_type,
                            signal_strength=signal_strength,
                            confidence=float(row['confidence']),
                            entry_price=float(row['entry_price']),
                            stop_loss=float(row['stop_loss']),
                            target_price=float(row['target_price']),
                            risk_per_share=float(row['risk_per_share']),
                            position_size=float(row['position_size']),
                            regime_context=regime_context,
                            notes=row['notes']
                        )
                        
                        signals.append(signal)
                        
                    except (ValueError, KeyError) as e:
                        print(f"Error parsing row: {row}, Error: {e}")
                        continue
                        
        except FileNotFoundError:
            print(f"CSV file not found: {csv_file}")
        except Exception as e:
            print(f"Error loading signals from CSV: {e}")
        
        print(f"Loaded {len(signals)} signals from {csv_file}")
        return signals

    def _get_signals_for_date(self, 
                             current_date: datetime,
                             setup_types: List[SetupType],
                             regime_aware: bool,
                             instrument_types: Optional[List[str]] = None) -> List[TradeSignal]:
        """Get trade signals for a specific date."""
        
        # Get symbols from setup manager with instrument type filtering
        symbols = self.setup_manager.get_all_symbols(instrument_types)
        
        # Get current regime if regime-aware
        current_regime = None
        if regime_aware:
            current_regime = self.regime_detector.detect_current_regime()
        
        # Scan for signals using each setup
        all_signals = []
        for setup_type in setup_types:
            try:
                setup = self.setup_manager.setups[setup_type]
                signals = setup.scan_for_signals(symbols)
                
                # Filter by regime if enabled
                if regime_aware and current_regime:
                    signals = [s for s in signals if self._validate_regime_signal(s, current_regime)]
                
                all_signals.extend(signals)
            except Exception as e:
                print(f"Error scanning {setup_type}: {e}")
                continue
        
        # Sort by confidence and return top signals
        all_signals.sort(key=lambda x: x.confidence, reverse=True)
        return all_signals[:5]  # Limit to top 5 signals per day
    
    def _can_enter_trade(self, signal: TradeSignal, current_date: datetime) -> bool:
        """Check if we can enter a new trade based on enhanced risk management rules."""
        
        # Check position limits
        if len(self.current_positions) >= self.max_concurrent_positions:
            return False
        
        # Check if already holding this symbol
        if any(pos.symbol == signal.symbol for pos in self.current_positions):
            return False
        
        # Enhanced sector correlation checking
        if not self._check_sector_allocation(signal.symbol):
            return False
        
        # Check portfolio heat (total risk exposure)
        if not self._check_portfolio_heat(signal):
            return False
        
        return True
    
    def _check_sector_allocation(self, symbol: str) -> bool:
        """Check if adding this symbol would violate sector allocation limits."""
        try:
            # Get sector for the new symbol
            new_sector = self._get_symbol_sector(symbol)
            if new_sector == "Unknown":
                return True  # Allow unknown sectors
            
            # Count current positions in same sector
            sector_positions = 0
            for pos in self.current_positions:
                pos_sector = self._get_symbol_sector(pos.symbol)
                if pos_sector == new_sector:
                    sector_positions += 1
            
            # Check if we'd exceed max similar ETFs
            return sector_positions < self.max_similar_etfs
            
        except Exception:
            return True  # Allow trade if sector check fails
    
    def _check_portfolio_heat(self, signal: TradeSignal) -> bool:
        """Check if total portfolio risk exposure is within limits."""
        try:
            # Calculate current total risk exposure
            current_equity = self.daily_equity[-1] if self.daily_equity else self.initial_capital
            total_current_risk = 0
            
            for pos in self.current_positions:
                position_risk = pos.position_size * pos.risk_per_share
                total_current_risk += position_risk
            
            # Calculate risk for new position
            new_position_risk = current_equity * self.max_risk_per_trade
            
            # Check if total risk would exceed 8% of capital (4 positions * 2% each)
            max_total_risk = current_equity * 0.08
            
            return (total_current_risk + new_position_risk) <= max_total_risk
            
        except Exception:
            return True  # Allow trade if heat check fails
    
    def _get_symbol_sector(self, symbol: str) -> str:
        """Get sector for a symbol from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT sector, theme FROM instruments WHERE symbol = ?", (symbol,))
                result = cursor.fetchone()
                
                if result and result[0]:
                    return result[0]
                elif result and result[1]:
                    return result[1]
                
                return "Unknown"
        except Exception:
            return "Unknown"
    
    def _is_stock(self, symbol: str) -> bool:
        """Check if symbol is an individual stock."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT type FROM instruments WHERE symbol = ?", (symbol,))
                result = cursor.fetchone()
                return result and result[0] == 'Stock'
        except Exception:
            return False
    
    def _enter_trade(self, signal: TradeSignal, entry_date: datetime):
        """Enter a new trade position using optimized parameters."""
        
        # Calculate position size based on risk
        current_capital = self.daily_equity[-1] if self.daily_equity else self.initial_capital
        max_risk_amount = current_capital * self.max_risk_per_trade
        
        # Apply optimized stop loss and target parameters
        entry_price = signal.entry_price
        optimized_stop_loss = entry_price * (1 - self.current_params.stop_loss_pct)
        optimized_target = entry_price * (1 + self.current_params.stop_loss_pct * self.current_params.profit_target_r)
        
        # Calculate position size based on optimized stop loss using dynamic equity
        risk_per_share = entry_price - optimized_stop_loss
        position_size = max_risk_amount / risk_per_share if risk_per_share > 0 else signal.position_size
        
        # Create trade record with optimized parameters
        trade = BacktestTrade(
            trade_id=self.trade_id_counter,
            symbol=signal.symbol,
            setup_type=signal.setup_type,
            entry_date=entry_date,
            entry_price=entry_price,
            position_size=position_size,
            stop_loss=optimized_stop_loss,
            target_price=optimized_target,
            risk_per_share=risk_per_share,
            confidence=signal.confidence,
            regime_at_entry=signal.regime_context
        )
        
        self.current_positions.append(trade)
        self.trade_id_counter += 1
        
        print(f"Entered {signal.setup_type.value} trade: {signal.symbol} @ ${entry_price:.2f} (stop: ${optimized_stop_loss:.2f}, target: ${optimized_target:.2f})")
    
    def _update_positions(self, current_date: datetime):
        """Update all open positions and check for exits."""
        
        positions_to_close = []
        
        for position in self.current_positions:
            # Get current price
            try:
                current_price = self._get_price_for_date(position.symbol, current_date)
                if current_price is None:
                    continue
                
                # Update MAE/MFE tracking
                price_change_pct = (current_price - position.entry_price) / position.entry_price
                
                # Track Maximum Adverse Excursion (MAE) - worst move against position
                adverse_excursion = min(0, price_change_pct) * 100  # Convert to percentage
                if adverse_excursion < position.mae:
                    position.mae = adverse_excursion
                
                # Track Maximum Favorable Excursion (MFE) - best move in favor of position
                favorable_excursion = max(0, price_change_pct) * 100  # Convert to percentage
                if favorable_excursion > position.mfe:
                    position.mfe = favorable_excursion
                
                # Check for exit conditions
                exit_triggered = False
                exit_reason = TradeStatus.OPEN
                
                # Check stop loss
                if current_price <= position.stop_loss:
                    exit_triggered = True
                    exit_reason = TradeStatus.CLOSED_STOP
                
                # Check target
                elif current_price >= position.target_price:
                    exit_triggered = True
                    exit_reason = TradeStatus.CLOSED_TARGET
                
                # Check maximum holding period (use optimized parameter)
                days_held = (current_date - position.entry_date).days
                if days_held >= self.current_params.max_holding_days:
                    exit_triggered = True
                    exit_reason = TradeStatus.CLOSED_PROFIT if current_price > position.entry_price else TradeStatus.CLOSED_LOSS
                
                if exit_triggered:
                    self._close_position(position, current_date, current_price, exit_reason)
                    positions_to_close.append(position)
                
            except Exception as e:
                print(f"Error updating position {position.symbol}: {e}")
                continue
        
        # Remove closed positions
        for position in positions_to_close:
            self.current_positions.remove(position)
    
    def _close_position(self, 
                       position: BacktestTrade,
                       exit_date: datetime,
                       exit_price: float,
                       exit_reason: TradeStatus):
        """Close a position and calculate P&L."""
        
        position.exit_date = exit_date
        position.exit_price = exit_price
        position.status = exit_reason
        position.days_held = (exit_date - position.entry_date).days
        
        # Calculate P&L
        price_change = exit_price - position.entry_price
        position.pnl = position.position_size * price_change
        
        # Calculate R-multiple
        risk_amount = position.position_size * position.risk_per_share
        if risk_amount > 0:
            position.r_multiple = position.pnl / risk_amount
        else:
            position.r_multiple = 0
        
        self.trades.append(position)
        
        print(f"Closed {position.symbol}: ${position.entry_price:.2f} -> ${exit_price:.2f} "
              f"(R: {position.r_multiple:.2f}, P&L: ${position.pnl:.2f})")
    
    def _get_price_for_date(self, symbol: str, date: datetime) -> Optional[float]:
        """Get price for a symbol on a specific date."""
        try:
            # Try to get from cache first
            data = self.data_cache.get_cached_data(symbol, "1y")
            if data is not None and not data.empty:
                # Find closest date
                date_str = date.strftime('%Y-%m-%d')
                if date_str in data.index:
                    return data.loc[date_str, 'Close']
                else:
                    # Find nearest date
                    nearest_idx = data.index.get_indexer([date_str], method='nearest')[0]
                    if nearest_idx >= 0 and nearest_idx < len(data):
                        return data.iloc[nearest_idx]['Close']
            return None
        except Exception:
            return None
    
    def _calculate_current_equity(self, current_date: datetime) -> float:
        """Calculate current total equity including open positions."""
        
        cash = self.initial_capital
        
        # Subtract cost of all trades
        for trade in self.trades:
            cash += trade.pnl or 0
        
        # Add unrealized P&L from open positions
        for position in self.current_positions:
            current_price = self._get_price_for_date(position.symbol, current_date)
            if current_price:
                unrealized_pnl = position.position_size * (current_price - position.entry_price)
                cash += unrealized_pnl
        
        return cash
    
    def _calculate_performance_metrics(self) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        
        if not self.trades:
            return PerformanceMetrics(
                total_trades=0, winning_trades=0, losing_trades=0,
                win_rate=0, avg_r_multiple=0, total_return=0,
                max_drawdown=0, sharpe_ratio=0, calmar_ratio=0,
                avg_days_held=0, largest_winner=0, largest_loser=0,
                consecutive_wins=0, consecutive_losses=0, profit_factor=0,
                # Professional metrics defaults
                sortino_ratio=0.0, information_ratio=0.0,
                max_adverse_excursion=0.0, max_favorable_excursion=0.0,
                avg_mae=0.0, avg_mfe=0.0, annual_return=0.0,
                downside_deviation=0.0, tracking_error=0.0, benchmark_return=0.0
            )
        
        # Basic trade statistics
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if (t.pnl or 0) > 0])
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # R-multiple analysis
        r_multiples = [t.r_multiple for t in self.trades if t.r_multiple is not None]
        avg_r_multiple = np.mean(r_multiples) if r_multiples else 0
        
        # Return calculations
        total_pnl = sum(t.pnl for t in self.trades if t.pnl is not None)
        total_return = total_pnl / self.initial_capital if self.initial_capital > 0 else 0
        
        # Drawdown calculation
        max_drawdown = self._calculate_max_drawdown()
        
        # Risk-adjusted returns
        returns_series = pd.Series(self.daily_equity).pct_change().dropna()
        
        # Calculate Sharpe ratio with risk-free rate
        if len(returns_series) > 1 and self.daily_dates:
            # Get risk-free rate for the backtest period
            start_date = self.daily_dates[0] if self.daily_dates else datetime.now() - timedelta(days=365)
            end_date = self.daily_dates[-1] if self.daily_dates else datetime.now()
            risk_free_rate = self.data_cache.get_risk_free_rate(start_date, end_date)
            
            # Convert annual risk-free rate to daily
            daily_rf_rate = risk_free_rate / 252
            
            # Calculate excess returns
            excess_returns = returns_series - daily_rf_rate
            
            # Sharpe ratio: (mean excess return * 252) / (std of returns * sqrt(252))
            sharpe_ratio = (excess_returns.mean() * 252) / (returns_series.std() * np.sqrt(252))
        else:
            sharpe_ratio = 0
        
        calmar_ratio = (total_return * 100) / (abs(max_drawdown) * 100) if max_drawdown != 0 else 0
        
        # Other metrics
        days_held = [t.days_held for t in self.trades if t.days_held is not None]
        avg_days_held = np.mean(days_held) if days_held else 0
        
        pnls = [t.pnl for t in self.trades if t.pnl is not None]
        largest_winner = max(pnls) if pnls else 0
        largest_loser = min(pnls) if pnls else 0
        
        # Consecutive wins/losses
        consecutive_wins, consecutive_losses = self._calculate_consecutive_streaks()
        
        # Profit factor
        gross_profit = sum(pnl for pnl in pnls if pnl > 0)
        gross_loss = abs(sum(pnl for pnl in pnls if pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate instrument type breakdown
        etf_trades = sum(1 for t in self.trades if not self._is_stock(t.symbol))
        stock_trades = sum(1 for t in self.trades if self._is_stock(t.symbol))
        
        etf_wins = sum(1 for t in self.trades if not self._is_stock(t.symbol) and (t.pnl or 0) > 0)
        stock_wins = sum(1 for t in self.trades if self._is_stock(t.symbol) and (t.pnl or 0) > 0)
        
        etf_win_rate = etf_wins / etf_trades if etf_trades > 0 else 0
        stock_win_rate = stock_wins / stock_trades if stock_trades > 0 else 0
        
        etf_r_multiples = [t.r_multiple for t in self.trades if not self._is_stock(t.symbol) and t.r_multiple is not None]
        stock_r_multiples = [t.r_multiple for t in self.trades if self._is_stock(t.symbol) and t.r_multiple is not None]
        
        etf_avg_r = np.mean(etf_r_multiples) if etf_r_multiples else 0
        stock_avg_r = np.mean(stock_r_multiples) if stock_r_multiples else 0
        
        etf_returns = [t.pnl / (t.entry_price * t.position_size) for t in self.trades 
                      if not self._is_stock(t.symbol) and t.pnl is not None and t.entry_price and t.position_size]
        stock_returns = [t.pnl / (t.entry_price * t.position_size) for t in self.trades 
                        if self._is_stock(t.symbol) and t.pnl is not None and t.entry_price and t.position_size]
        
        etf_avg_return = np.mean(etf_returns) if etf_returns else 0
        stock_avg_return = np.mean(stock_returns) if stock_returns else 0
        
        # Professional metrics calculations (Phase 1)
        # Calculate annual return
        if len(self.daily_dates) > 1:
            days_elapsed = (self.daily_dates[-1] - self.daily_dates[0]).days
            annual_return = (1 + total_return) ** (365 / days_elapsed) - 1 if days_elapsed > 0 else 0
        else:
            annual_return = 0
        
        # Calculate downside deviation for Sortino ratio
        downside_returns = [r for r in returns_series if r < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252) if downside_returns else 0
        
        # Calculate Sortino ratio
        if downside_deviation > 0 and len(returns_series) > 1:
            if len(self.daily_dates) > 1:
                daily_rf_rate = risk_free_rate / 252
                excess_returns = returns_series - daily_rf_rate
                sortino_ratio = (excess_returns.mean() * 252) / downside_deviation
            else:
                sortino_ratio = 0
        else:
            sortino_ratio = 0
        
        # Calculate MAE and MFE metrics
        mae_values = [t.mae for t in self.trades if t.mae != 0]
        mfe_values = [t.mfe for t in self.trades if t.mfe != 0]
        
        max_adverse_excursion = max(mae_values) if mae_values else 0
        max_favorable_excursion = max(mfe_values) if mfe_values else 0
        avg_mae = np.mean(mae_values) if mae_values else 0
        avg_mfe = np.mean(mfe_values) if mfe_values else 0
        
        # Benchmark comparison (SPY as benchmark)
        benchmark_return = 0.0
        information_ratio = 0.0
        tracking_error = 0.0
        
        # Try to get SPY benchmark data for comparison
        if len(self.daily_dates) > 1:
            try:
                spy_data = self.data_cache.get_price_data('SPY', self.daily_dates[0], self.daily_dates[-1])
                if spy_data is not None and len(spy_data) > 1:
                    spy_start = spy_data['close'].iloc[0]
                    spy_end = spy_data['close'].iloc[-1]
                    benchmark_return = (spy_end - spy_start) / spy_start
                    
                    # Calculate tracking error and information ratio
                    if len(spy_data) > 1:
                        spy_returns = spy_data['close'].pct_change().dropna()
                        
                        # Align portfolio and benchmark returns
                        if len(spy_returns) > 0 and len(returns_series) > 0:
                            min_length = min(len(spy_returns), len(returns_series))
                            if min_length > 1:
                                aligned_portfolio = returns_series.iloc[-min_length:]
                                aligned_benchmark = spy_returns.iloc[-min_length:]
                                
                                excess_vs_benchmark = aligned_portfolio - aligned_benchmark
                                tracking_error = excess_vs_benchmark.std() * np.sqrt(252)
                                
                                if tracking_error > 0:
                                    information_ratio = (excess_vs_benchmark.mean() * 252) / tracking_error
            except Exception:
                # If benchmark data fails, use defaults
                pass
        
        return PerformanceMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_r_multiple=avg_r_multiple,
            total_return=total_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            calmar_ratio=calmar_ratio,
            avg_days_held=avg_days_held,
            largest_winner=largest_winner,
            largest_loser=largest_loser,
            consecutive_wins=consecutive_wins,
            consecutive_losses=consecutive_losses,
            profit_factor=profit_factor,
            etf_trades=etf_trades,
            stock_trades=stock_trades,
            etf_win_rate=etf_win_rate,
            stock_win_rate=stock_win_rate,
            etf_avg_return=etf_avg_return,
            stock_avg_return=stock_avg_return,
            etf_avg_r_multiple=etf_avg_r,
            stock_avg_r_multiple=stock_avg_r,
            # Professional metrics
            sortino_ratio=sortino_ratio,
            information_ratio=information_ratio,
            max_adverse_excursion=max_adverse_excursion,
            max_favorable_excursion=max_favorable_excursion,
            avg_mae=avg_mae,
            avg_mfe=avg_mfe,
            annual_return=annual_return,
            downside_deviation=downside_deviation,
            tracking_error=tracking_error,
            benchmark_return=benchmark_return
        )
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from equity curve."""
        if len(self.daily_equity) < 2:
            return 0
        
        equity_series = pd.Series(self.daily_equity)
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max
        return drawdown.min()
    
    def _calculate_consecutive_streaks(self) -> Tuple[int, int]:
        """Calculate maximum consecutive wins and losses."""
        if not self.trades:
            return 0, 0
        
        current_wins = 0
        current_losses = 0
        max_wins = 0
        max_losses = 0
        
        for trade in self.trades:
            if (trade.pnl or 0) > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
        
        return max_wins, max_losses
    
    def _get_trading_days(self, start_date: datetime, end_date: datetime) -> List[datetime]:
        """Get list of trading days between start and end dates."""
        # Simple implementation - could be enhanced with actual trading calendar
        current = start_date
        trading_days = []
        
        while current <= end_date:
            # Skip weekends (Saturday=5, Sunday=6)
            if current.weekday() < 5:
                trading_days.append(current)
            current += timedelta(days=1)
        
        return trading_days
    
    
    def _validate_regime_signal(self, signal: TradeSignal, current_regime: RegimeData) -> bool:
        """Enhanced regime validation for setup appropriateness."""
        
        # Volatility-based filtering
        vix_level = current_regime.vix_level
        
        # High volatility regime (VIX > 30) - avoid momentum strategies
        if vix_level > 30:
            if signal.setup_type in [SetupType.BREAKOUT_CONTINUATION, 
                                   SetupType.RELATIVE_STRENGTH_MOMENTUM]:
                return False
        
        # Low volatility regime (VIX < 20) - favor specific setups
        elif vix_level < 20:
            if signal.setup_type == SetupType.VOLATILITY_CONTRACTION:
                return True  # Strong preference for vol contraction in low vol
        
        # Trend-based filtering
        spy_trend = current_regime.spy_vs_sma200
        
        # Strong uptrend - avoid mean reversion, favor momentum
        if spy_trend > 0.10:  # SPY > 10% above SMA200
            if signal.setup_type == SetupType.OVERSOLD_MEAN_REVERSION:
                return False
            if signal.setup_type in [SetupType.TREND_PULLBACK, 
                                   SetupType.RELATIVE_STRENGTH_MOMENTUM]:
                return True
        
        # Strong downtrend - favor mean reversion, avoid momentum
        elif spy_trend < -0.10:  # SPY > 10% below SMA200
            if signal.setup_type in [SetupType.BREAKOUT_CONTINUATION,
                                   SetupType.RELATIVE_STRENGTH_MOMENTUM]:
                return False
            if signal.setup_type in [SetupType.OVERSOLD_MEAN_REVERSION,
                                   SetupType.GAP_FILL_REVERSAL]:
                return True
        
        # Risk sentiment filtering
        risk_ratio = current_regime.risk_on_off_ratio
        
        # Risk-off environment - favor defensive setups
        if risk_ratio < 0.95:
            if signal.setup_type == SetupType.DIVIDEND_DISTRIBUTION_PLAY:
                return True  # Favor dividend plays in risk-off
            if signal.setup_type == SetupType.RELATIVE_STRENGTH_MOMENTUM:
                return False  # Avoid momentum in risk-off
        
        # Risk-on environment - favor growth setups
        elif risk_ratio > 1.05:
            if signal.setup_type in [SetupType.BREAKOUT_CONTINUATION,
                                   SetupType.RELATIVE_STRENGTH_MOMENTUM]:
                return True
        
        # Sector rotation consideration
        growth_value_ratio = current_regime.growth_value_ratio
        
        # Growth outperforming - favor tech/growth oriented setups
        if growth_value_ratio > 1.10:
            # Would need ETF sector mapping for full implementation
            # For now, accept most momentum setups
            if signal.setup_type in [SetupType.BREAKOUT_CONTINUATION,
                                   SetupType.RELATIVE_STRENGTH_MOMENTUM]:
                return True
        
        # Value outperforming - favor defensive setups
        elif growth_value_ratio < 0.90:
            if signal.setup_type in [SetupType.DIVIDEND_DISTRIBUTION_PLAY,
                                   SetupType.OVERSOLD_MEAN_REVERSION]:
                return True
        
        return True  # Default to accepting signal if no specific rules apply
    
    def _combine_walk_forward_results(self, results: List[Dict]) -> PerformanceMetrics:
        """Combine multiple walk-forward period results."""
        all_trades = []
        all_equity = []
        
        for result in results:
            all_trades.extend(result['trades'])
            if all_equity:
                # Continue equity curve from last value
                start_equity = all_equity[-1]
                period_equity = result['daily_equity']
                if period_equity:
                    period_returns = [(e / period_equity[0] - 1) for e in period_equity[1:]]
                    for ret in period_returns:
                        all_equity.append(all_equity[-1] * (1 + ret))
            else:
                all_equity = result['daily_equity']
        
        # Recalculate combined metrics
        self.trades = [BacktestTrade(**trade) for trade in all_trades]
        self.daily_equity = all_equity
        
        return self._calculate_performance_metrics()
    
    def _optimize_parameters(self, 
                           training_days: List[datetime],
                           setup_types: List[SetupType],
                           regime_aware: bool) -> OptimizationParameters:
        """Optimize parameters using grid search on training data."""
        
        # Parameter ranges to test
        stop_loss_range = [0.03, 0.04, 0.05, 0.06, 0.08]  # 3-8%
        profit_target_range = [1.5, 2.0, 2.5, 3.0]  # R multiples
        confidence_range = [0.5, 0.6, 0.7, 0.8]  # Confidence thresholds
        
        best_params = OptimizationParameters()
        best_score = float('-inf')
        
        print(f"    Testing {len(stop_loss_range) * len(profit_target_range) * len(confidence_range)} parameter combinations...")
        
        # Grid search over parameter combinations
        for stop_loss, target_r, confidence in product(stop_loss_range, profit_target_range, confidence_range):
            # Create test parameters
            test_params = OptimizationParameters(
                stop_loss_pct=stop_loss,
                profit_target_r=target_r,
                confidence_threshold=confidence
            )
            
            # Backtest with these parameters
            try:
                score = self._evaluate_parameters(test_params, training_days, setup_types, regime_aware)
                
                if score > best_score:
                    best_score = score
                    best_params = test_params
            except Exception as e:
                # Skip parameter combinations that cause errors
                continue
        
        print(f"    Best parameters: stop_loss={best_params.stop_loss_pct:.1%}, target={best_params.profit_target_r:.1f}R, score={best_score:.3f}")
        return best_params
    
    def _evaluate_parameters(self,
                           params: OptimizationParameters,
                           training_days: List[datetime],
                           setup_types: List[SetupType],
                           regime_aware: bool) -> float:
        """Evaluate parameter set and return fitness score."""
        
        # Save current state
        original_params = self.current_params
        original_trades = self.trades.copy()
        original_positions = self.current_positions.copy()
        original_counter = self.trade_id_counter
        original_equity = self.daily_equity.copy()
        original_dates = self.daily_dates.copy()
        
        try:
            # Apply test parameters
            self.current_params = params
            self.trades = []
            self.current_positions = []
            self.trade_id_counter = 1
            self.daily_equity = [self.initial_capital]
            self.daily_dates = []
            
            # Run mini-backtest on training period
            for current_date in training_days:
                self.daily_dates.append(current_date)
                self._update_positions(current_date)
                
                if len(self.current_positions) < self.max_concurrent_positions:
                    signals = self._get_signals_for_date(current_date, setup_types, regime_aware)
                    
                    # Filter by confidence threshold
                    signals = [s for s in signals if s.confidence >= params.confidence_threshold]
                    
                    for signal in signals:
                        if self._can_enter_trade(signal, current_date):
                            self._enter_trade(signal, current_date)
                            if len(self.current_positions) >= self.max_concurrent_positions:
                                break
                
                current_equity = self._calculate_current_equity(current_date)
                self.daily_equity.append(current_equity)
            
            # Calculate fitness score (Sharpe ratio with penalty for excessive trades)
            if len(self.trades) < 3:  # Minimum trades required
                return float('-inf')
            
            perf = self._calculate_performance_metrics()
            
            # Fitness function: Sharpe ratio adjusted for trade frequency
            trade_frequency_penalty = max(0, (len(self.trades) - 50) * 0.01)  # Penalize >50 trades
            fitness_score = perf.sharpe_ratio - trade_frequency_penalty
            
            return fitness_score
            
        finally:
            # Restore original state
            self.current_params = original_params
            self.trades = original_trades
            self.current_positions = original_positions
            self.trade_id_counter = original_counter
            self.daily_equity = original_equity
            self.daily_dates = original_dates
    
    def _calculate_regime_performance(self) -> RegimePerformance:
        """Calculate performance metrics broken down by market regime."""
        
        if not self.trades:
            return RegimePerformance()
        
        # Group trades by regime characteristics
        volatility_low_trades = []
        volatility_medium_trades = []
        volatility_high_trades = []
        trend_up_trades = []
        trend_neutral_trades = []
        trend_down_trades = []
        risk_on_trades = []
        risk_off_trades = []
        
        for trade in self.trades:
            if trade.regime_at_entry:
                regime = trade.regime_at_entry
                
                # Categorize by volatility regime
                if regime.vix_level < 20:
                    volatility_low_trades.append(trade)
                elif regime.vix_level > 30:
                    volatility_high_trades.append(trade)
                else:
                    volatility_medium_trades.append(trade)
                
                # Categorize by trend regime
                if regime.spy_vs_sma200 > 0.05:  # SPY > 5% above SMA200
                    trend_up_trades.append(trade)
                elif regime.spy_vs_sma200 < -0.05:  # SPY > 5% below SMA200
                    trend_down_trades.append(trade)
                else:
                    trend_neutral_trades.append(trade)
                
                # Categorize by risk sentiment
                if regime.risk_on_off_ratio > 1.02:  # Risk-on environment
                    risk_on_trades.append(trade)
                elif regime.risk_on_off_ratio < 0.98:  # Risk-off environment
                    risk_off_trades.append(trade)
        
        # Calculate metrics for each regime category
        regime_performance = RegimePerformance()
        
        if volatility_low_trades:
            regime_performance.volatility_low = self._calculate_metrics_for_trades(volatility_low_trades)
        if volatility_medium_trades:
            regime_performance.volatility_medium = self._calculate_metrics_for_trades(volatility_medium_trades)
        if volatility_high_trades:
            regime_performance.volatility_high = self._calculate_metrics_for_trades(volatility_high_trades)
        if trend_up_trades:
            regime_performance.trend_up = self._calculate_metrics_for_trades(trend_up_trades)
        if trend_neutral_trades:
            regime_performance.trend_neutral = self._calculate_metrics_for_trades(trend_neutral_trades)
        if trend_down_trades:
            regime_performance.trend_down = self._calculate_metrics_for_trades(trend_down_trades)
        if risk_on_trades:
            regime_performance.risk_on = self._calculate_metrics_for_trades(risk_on_trades)
        if risk_off_trades:
            regime_performance.risk_off = self._calculate_metrics_for_trades(risk_off_trades)
        
        return regime_performance
    
    def _calculate_metrics_for_trades(self, trades: List[BacktestTrade]) -> PerformanceMetrics:
        """Calculate performance metrics for a subset of trades."""
        
        if not trades:
            return PerformanceMetrics(
                total_trades=0, winning_trades=0, losing_trades=0,
                win_rate=0, avg_r_multiple=0, total_return=0,
                max_drawdown=0, sharpe_ratio=0, calmar_ratio=0,
                avg_days_held=0, largest_winner=0, largest_loser=0,
                consecutive_wins=0, consecutive_losses=0, profit_factor=0
            )
        
        # Basic statistics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if (t.pnl or 0) > 0])
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # R-multiple analysis
        r_multiples = [t.r_multiple for t in trades if t.r_multiple is not None]
        avg_r_multiple = np.mean(r_multiples) if r_multiples else 0
        
        # Return calculations
        total_pnl = sum(t.pnl for t in trades if t.pnl is not None)
        total_return = total_pnl / self.initial_capital if self.initial_capital > 0 else 0
        
        # Other metrics
        days_held = [t.days_held for t in trades if t.days_held is not None]
        avg_days_held = np.mean(days_held) if days_held else 0
        
        pnls = [t.pnl for t in trades if t.pnl is not None]
        largest_winner = max(pnls) if pnls else 0
        largest_loser = min(pnls) if pnls else 0
        
        # Profit factor
        gross_profit = sum(pnl for pnl in pnls if pnl > 0)
        gross_loss = abs(sum(pnl for pnl in pnls if pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return PerformanceMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_r_multiple=avg_r_multiple,
            total_return=total_return,
            max_drawdown=0,  # Simplified for regime breakdown
            sharpe_ratio=0,   # Simplified for regime breakdown
            calmar_ratio=0,   # Simplified for regime breakdown
            avg_days_held=avg_days_held,
            largest_winner=largest_winner,
            largest_loser=largest_loser,
            consecutive_wins=0,  # Simplified for regime breakdown
            consecutive_losses=0,  # Simplified for regime breakdown
            profit_factor=profit_factor
        )


def main():
    """Main CLI interface for backtesting."""
    parser = argparse.ArgumentParser(
        description="ETF Trading System Backtester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Two distinct backtesting modes:

1. SIGNAL-BASED BACKTESTING (recommended for triggered signals):
   Tests only specific signals that have already triggered
   --csv-file signals.csv        # Test signals from screener export
   
2. SETUP-BASED BACKTESTING (traditional approach):
   Scans all symbols daily for setup patterns during backtest period
   --setup trend_pullback        # Test one setup across all symbols
   --all-setups                  # Test all setups across all symbols

Examples:
  # Test specific triggered signals from screener
  python screener.py --export-csv
  python backtest.py --csv-file etf_signals_20250630.csv
  
  # Traditional setup backtesting (scans all symbols)
  python backtest.py --setup trend_pullback --start-date 2020-01-01
        """
    )
    
    # Input method selection
    parser.add_argument("--csv-file", type=str,
                       help="SIGNAL-BASED: CSV file with screener signals to backtest")
    parser.add_argument("--setup", type=str, 
                       help="SETUP-BASED: Test specific setup across all symbols (e.g., trend_pullback)")
    parser.add_argument("--all-setups", action="store_true",
                       help="SETUP-BASED: Test all available setups across all symbols")
    
    # Backtest configuration
    parser.add_argument("--walk-forward", action="store_true",
                       help="Enable walk-forward validation")
    parser.add_argument("--regime-aware", action="store_true", default=True,
                       help="Include regime analysis")
    parser.add_argument("--type", type=str, 
                       choices=['etf', 'stock', 'all'],
                       default='etf',
                       help="Instrument type to backtest (default: etf)")
    parser.add_argument("--start-date", type=str, default="2025-01-01",
                       help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default="2025-06-15",
                       help="Backtest end date (YYYY-MM-DD)")
    
    # Output options
    parser.add_argument("--export-results", action="store_true",
                       help="Export results to JSON file")
    parser.add_argument("--optimize", action="store_true",
                       help="Run parameter optimization (future enhancement)")
    
    args = parser.parse_args()
    
    # Parse dates
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    
    # Initialize backtesting engine
    engine = BacktestEngine()
    
    # Determine backtest method
    if args.csv_file:
        # Backtest from CSV file
        print(f"Running backtest from CSV file: {args.csv_file}")
        results = engine.backtest_from_csv(args.csv_file, start_date, end_date)
    else:
        # Traditional backtest by setup types
        setup_types = None
        if args.setup:
            try:
                setup_types = [SetupType(args.setup)]
            except ValueError:
                print(f"Invalid setup type: {args.setup}")
                print(f"Available setups: {[s.value for s in SetupType]}")
                return
        elif args.all_setups:
            setup_types = list(SetupType)
        else:
            # Default to trend_pullback
            setup_types = [SetupType.TREND_PULLBACK]
        
        print(f"Running backtest for setups: {[s.value for s in setup_types]}")
        
        # Determine instrument types based on user selection
        if args.type == 'etf':
            instrument_types = ['ETF', 'ETN']
        elif args.type == 'stock':
            instrument_types = ['Stock']
        elif args.type == 'all':
            instrument_types = ['ETF', 'ETN', 'Stock']
        else:
            instrument_types = ['ETF', 'ETN']  # Default fallback
        
        results = engine.run_backtest(
            start_date=start_date,
            end_date=end_date,
            setup_types=setup_types,
            walk_forward=args.walk_forward,
            regime_aware=args.regime_aware,
            instrument_types=instrument_types
        )
    
    # Display results
    perf = results['performance']
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    
    # Show different info based on backtest type
    if args.csv_file:
        print(f"CSV File: {args.csv_file}")
        print(f"Signals Tested: {results.get('signals_tested', 'N/A')}")
        print(f"Unique Symbols: {results.get('unique_symbols', 'N/A')}")
        print(f"Setups Tested: {', '.join(results.get('setups_tested', []))}")
    else:
        print(f"Setups: {[s.value for s in setup_types]}")
        print(f"Walk-Forward: {args.walk_forward}")
    
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print("-" * 60)
    print(f"Total Trades: {perf.total_trades}")
    print(f"Win Rate: {perf.win_rate:.1%}")
    print(f"Average R-Multiple: {perf.avg_r_multiple:.2f}")
    print(f"Total Return: {perf.total_return:.1%}")
    print(f"Maximum Drawdown: {perf.max_drawdown:.1%}")
    print(f"Sharpe Ratio: {perf.sharpe_ratio:.2f}")
    print(f"Calmar Ratio: {perf.calmar_ratio:.2f}")
    print(f"Profit Factor: {perf.profit_factor:.2f}")
    print(f"Average Days Held: {perf.avg_days_held:.1f}")
    print(f"Largest Winner: ${perf.largest_winner:.2f}")
    print(f"Largest Loser: ${perf.largest_loser:.2f}")
    
    # Show instrument type breakdown if both ETFs and stocks were traded
    if perf.etf_trades > 0 and perf.stock_trades > 0:
        print("-" * 60)
        print("INSTRUMENT TYPE BREAKDOWN")
        print(f"ETF Trades: {perf.etf_trades} ({perf.etf_win_rate:.1%} win rate, {perf.etf_avg_r_multiple:.2f}R avg)")
        print(f"Stock Trades: {perf.stock_trades} ({perf.stock_win_rate:.1%} win rate, {perf.stock_avg_r_multiple:.2f}R avg)")
    elif perf.stock_trades > 0:
        print(f"Instrument Type: Stocks only ({perf.stock_trades} trades)")
    else:
        print(f"Instrument Type: ETFs only ({perf.etf_trades} trades)")
    
    print("="*60)
    
    # Display regime performance analysis if available
    if 'regime_performance' in results:
        regime_perf = results['regime_performance']
        print("\nREGIME PERFORMANCE ANALYSIS")
        print("="*60)
        
        # Volatility regimes
        if regime_perf.volatility_low:
            vol_low = regime_perf.volatility_low
            print(f"Low Volatility (VIX<20): {vol_low.total_trades} trades, "
                  f"{vol_low.win_rate:.1%} win rate, "
                  f"{vol_low.avg_r_multiple:.2f}R avg")
        
        if regime_perf.volatility_medium:
            vol_med = regime_perf.volatility_medium
            print(f"Medium Volatility (VIX 20-30): {vol_med.total_trades} trades, "
                  f"{vol_med.win_rate:.1%} win rate, "
                  f"{vol_med.avg_r_multiple:.2f}R avg")
        
        if regime_perf.volatility_high:
            vol_high = regime_perf.volatility_high
            print(f"High Volatility (VIX>30): {vol_high.total_trades} trades, "
                  f"{vol_high.win_rate:.1%} win rate, "
                  f"{vol_high.avg_r_multiple:.2f}R avg")
        
        print("-" * 40)
        
        # Trend regimes
        if regime_perf.trend_up:
            trend_up = regime_perf.trend_up
            print(f"Uptrend (SPY>SMA200+5%): {trend_up.total_trades} trades, "
                  f"{trend_up.win_rate:.1%} win rate, "
                  f"{trend_up.avg_r_multiple:.2f}R avg")
        
        if regime_perf.trend_neutral:
            trend_neu = regime_perf.trend_neutral
            print(f"Neutral Trend: {trend_neu.total_trades} trades, "
                  f"{trend_neu.win_rate:.1%} win rate, "
                  f"{trend_neu.avg_r_multiple:.2f}R avg")
        
        if regime_perf.trend_down:
            trend_down = regime_perf.trend_down
            print(f"Downtrend (SPY<SMA200-5%): {trend_down.total_trades} trades, "
                  f"{trend_down.win_rate:.1%} win rate, "
                  f"{trend_down.avg_r_multiple:.2f}R avg")
        
        print("-" * 40)
        
        # Risk sentiment regimes
        if regime_perf.risk_on:
            risk_on = regime_perf.risk_on
            print(f"Risk-On Environment: {risk_on.total_trades} trades, "
                  f"{risk_on.win_rate:.1%} win rate, "
                  f"{risk_on.avg_r_multiple:.2f}R avg")
        
        if regime_perf.risk_off:
            risk_off = regime_perf.risk_off
            print(f"Risk-Off Environment: {risk_off.total_trades} trades, "
                  f"{risk_off.win_rate:.1%} win rate, "
                  f"{risk_off.avg_r_multiple:.2f}R avg")
        
        print("="*60)
    
    # Display walk-forward optimization history if available
    if 'optimization_history' in results:
        opt_history = results['optimization_history']
        print("\nWALK-FORWARD OPTIMIZATION HISTORY")
        print("="*60)
        for i, period in enumerate(opt_history):
            params = period['optimal_params']
            print(f"Period {i+1}: stop_loss={params['stop_loss_pct']:.1%}, "
                  f"target={params['profit_target_r']:.1f}R, "
                  f"confidence={params['confidence_threshold']:.2f}")
        print("="*60)
    
    # Export results if requested
    if args.export_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"backtest_results_{timestamp}.json"
        
        # Convert results to JSON-serializable format
        export_data = {
            'parameters': {
                'setups': [s.value for s in setup_types],
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'walk_forward': args.walk_forward,
                'regime_aware': args.regime_aware
            },
            'performance': asdict(perf),
            'trades': results['trades'],
            'daily_equity': results['daily_equity'],
            'daily_dates': results['daily_dates']
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"Results exported to {filename}")
    
    if args.optimize:
        print("Parameter optimization not yet implemented")


if __name__ == "__main__":
    main()