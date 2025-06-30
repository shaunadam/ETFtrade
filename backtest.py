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


class BacktestEngine:
    """Walk-forward backtesting engine with regime analysis."""
    
    def __init__(self, db_path: str = "journal.db", initial_capital: float = 100000):
        self.db_path = db_path
        self.initial_capital = initial_capital
        self.data_cache = DataCache(db_path)
        self.regime_detector = RegimeDetector(db_path)
        self.setup_manager = SetupManager(db_path)
        
        # Risk management parameters
        self.max_risk_per_trade = 0.02  # 2% per trade
        self.max_concurrent_positions = 4
        self.max_sector_allocation = 0.30  # 30% max in correlated sectors
        
        # Trade tracking
        self.trades: List[BacktestTrade] = []
        self.current_positions: List[BacktestTrade] = []
        self.trade_id_counter = 1
        
        # Performance tracking
        self.daily_equity = []
        self.daily_dates = []
    
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
                     regime_aware: bool = True) -> Dict:
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
                trading_days, setup_types, regime_aware
            )
        else:
            return self._run_standard_backtest(
                trading_days, setup_types, regime_aware
            )
    
    def _run_standard_backtest(self, 
                              trading_days: List[datetime],
                              setup_types: List[SetupType],
                              regime_aware: bool) -> Dict:
        """Run standard backtesting without walk-forward validation."""
        
        for current_date in trading_days:
            self.daily_dates.append(current_date)
            
            # Update existing positions
            self._update_positions(current_date)
            
            # Look for new trade opportunities
            if len(self.current_positions) < self.max_concurrent_positions:
                signals = self._get_signals_for_date(current_date, setup_types, regime_aware)
                
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
                                  regime_aware: bool) -> Dict:
        """Run walk-forward backtesting to prevent overfitting."""
        
        # Walk-forward parameters
        training_period_days = 252  # 1 year training
        test_period_days = 63      # 3 months testing
        
        results = []
        total_trades = []
        
        # Split into walk-forward periods
        for i in range(0, len(trading_days), test_period_days):
            training_start = max(0, i - training_period_days)
            training_end = i
            test_start = i
            test_end = min(len(trading_days), i + test_period_days)
            
            if test_start >= len(trading_days):
                break
                
            print(f"Walk-forward period: {trading_days[test_start].date()} to {trading_days[test_end-1].date()}")
                
            # Train on historical data (for parameter optimization - future enhancement)
            # For now, use standard parameters
            
            # Test on out-of-sample period
            test_days = trading_days[test_start:test_end]
            period_result = self._run_standard_backtest(test_days, setup_types, regime_aware)
            
            results.append(period_result)
            total_trades.extend(period_result['trades'])
        
        # Combine results
        combined_performance = self._combine_walk_forward_results(results)
        
        return {
            'performance': combined_performance,
            'trades': total_trades,
            'walk_forward_results': results
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
                             regime_aware: bool) -> List[TradeSignal]:
        """Get trade signals for a specific date."""
        
        # Get ETF symbols from cache
        symbols = self._get_etf_symbols()
        
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
        """Check if we can enter a new trade based on risk management rules."""
        
        # Check position limits
        if len(self.current_positions) >= self.max_concurrent_positions:
            return False
        
        # Check if already holding this symbol
        if any(pos.symbol == signal.symbol for pos in self.current_positions):
            return False
        
        # Check sector allocation (simplified - could be enhanced)
        # For now, just check we don't have too many similar ETFs
        similar_symbols = [pos.symbol for pos in self.current_positions 
                          if pos.symbol.startswith(signal.symbol[:2])]
        if len(similar_symbols) >= 2:  # Max 2 similar ETFs
            return False
        
        return True
    
    def _enter_trade(self, signal: TradeSignal, entry_date: datetime):
        """Enter a new trade position."""
        
        # Calculate position size based on risk
        current_capital = self.daily_equity[-1] if self.daily_equity else self.initial_capital
        max_risk_amount = current_capital * self.max_risk_per_trade
        
        # Use signal's position size if within risk limits
        risk_amount = signal.position_size * signal.risk_per_share
        if risk_amount > max_risk_amount:
            # Scale down position to fit risk limits
            signal.position_size = max_risk_amount / signal.risk_per_share
        
        # Create trade record
        trade = BacktestTrade(
            trade_id=self.trade_id_counter,
            symbol=signal.symbol,
            setup_type=signal.setup_type,
            entry_date=entry_date,
            entry_price=signal.entry_price,
            position_size=signal.position_size,
            stop_loss=signal.stop_loss,
            target_price=signal.target_price,
            risk_per_share=signal.risk_per_share,
            confidence=signal.confidence,
            regime_at_entry=signal.regime_context
        )
        
        self.current_positions.append(trade)
        self.trade_id_counter += 1
        
        print(f"Entered {signal.setup_type.value} trade: {signal.symbol} @ ${signal.entry_price:.2f}")
    
    def _update_positions(self, current_date: datetime):
        """Update all open positions and check for exits."""
        
        positions_to_close = []
        
        for position in self.current_positions:
            # Get current price
            try:
                current_price = self._get_price_for_date(position.symbol, current_date)
                if current_price is None:
                    continue
                
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
                
                # Check maximum holding period (60 days)
                days_held = (current_date - position.entry_date).days
                if days_held >= 60:
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
                consecutive_wins=0, consecutive_losses=0, profit_factor=0
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
        sharpe_ratio = (returns_series.mean() * 252) / (returns_series.std() * np.sqrt(252)) if len(returns_series) > 1 else 0
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
            profit_factor=profit_factor
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
    
    def _get_etf_symbols(self) -> List[str]:
        """Get list of ETF symbols for backtesting."""
        # Read from etf_list.csv
        try:
            import csv
            symbols = []
            with open('etf_list.csv', 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    symbols.append(row['Symbol'])
            return symbols
        except Exception:
            # Fallback to hardcoded list
            return ['SPY', 'QQQ', 'IWM', 'XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLU', 'XLP']
    
    def _validate_regime_signal(self, signal: TradeSignal, current_regime: RegimeData) -> bool:
        """Validate if signal is appropriate for current market regime."""
        # Basic regime validation - could be enhanced
        
        # High volatility regime - avoid breakout continuation
        if (current_regime.volatility_regime.value > 30 and 
            signal.setup_type == SetupType.BREAKOUT_CONTINUATION):
            return False
        
        # Low volatility - favor volatility contraction setups
        if (current_regime.volatility_regime.value < 20 and
            signal.setup_type == SetupType.VOLATILITY_CONTRACTION):
            return True
        
        return True  # Default to accepting signal
    
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
        
        results = engine.run_backtest(
            start_date=start_date,
            end_date=end_date,
            setup_types=setup_types,
            walk_forward=args.walk_forward,
            regime_aware=args.regime_aware
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