"""
Trade management system for backtesting engine.

This module handles all trade-related operations including:
- Trade entry validation and execution
- Position sizing and risk management
- Position tracking and updates
- Trade exit logic and performance calculation
- Portfolio equity calculation
"""

import sqlite3
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from data_cache import DataCache
from backtest_config import RiskManagementConfig, OptimizationConfig


@dataclass
class BacktestTrade:
    """Represents a single trade in the backtest."""
    symbol: str
    setup: str
    entry_date: datetime
    entry_price: float
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    quantity: float = 0.0
    stop_loss: float = 0.0
    target: float = 0.0
    max_holding_days: int = 30
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    mae: float = 0.0  # Maximum Adverse Excursion
    mfe: float = 0.0  # Maximum Favorable Excursion
    r_multiple: float = 0.0
    status: str = "open"
    regime_at_entry: Optional[str] = None
    confidence: float = 0.0
    
    def __post_init__(self):
        if self.quantity == 0.0:
            self.quantity = 100.0  # Default quantity
        if self.stop_loss == 0.0:
            self.stop_loss = self.entry_price * 0.95  # 5% stop loss
        if self.target == 0.0:
            self.target = self.entry_price * 1.10  # 10% target


class TradeManager:
    """
    Manages all trade operations for backtesting.
    
    This class handles:
    - Trade entry validation and execution
    - Position sizing based on risk parameters
    - Real-time position tracking with MAE/MFE monitoring
    - Multiple exit conditions (stop loss, target, time-based)
    - Portfolio equity calculation with unrealized P&L
    - Performance metrics calculation
    """
    
    def __init__(self, 
                 db_path: str,
                 data_cache: DataCache,
                 risk_config: RiskManagementConfig,
                 optimization_config: OptimizationConfig,
                 initial_capital: float = 100000.0):
        """
        Initialize TradeManager.
        
        Args:
            db_path: Path to SQLite database
            data_cache: Data cache instance for price data
            risk_config: Risk management configuration
            optimization_config: Optimization configuration
            initial_capital: Starting capital for backtesting
        """
        self.db_path = db_path
        self.data_cache = data_cache
        self.risk_config = risk_config
        self.optimization_config = optimization_config
        self.initial_capital = initial_capital
        
        # Trade tracking
        self.trades: List[BacktestTrade] = []
        self.current_positions: List[BacktestTrade] = []
        self.daily_equity: List[float] = []
        
        # Current optimization parameters (set externally)
        self.current_params = None
        
        # Logger
        self.logger = logging.getLogger(__name__)
    
    def can_enter_trade(self, symbol: str, setup: str, date: datetime) -> bool:
        """
        Check if a new trade can be entered based on risk management rules.
        
        Args:
            symbol: Symbol to trade
            setup: Trade setup name
            date: Trade date
            
        Returns:
            bool: True if trade can be entered, False otherwise
        """
        # Check maximum concurrent positions
        if len(self.current_positions) >= self.risk_config.max_concurrent_positions:
            self.logger.debug(f"Cannot enter {symbol}: max concurrent positions reached ({len(self.current_positions)})")
            return False
        
        # Check if symbol is already held
        if any(pos.symbol == symbol for pos in self.current_positions):
            self.logger.debug(f"Cannot enter {symbol}: already holding position")
            return False
        
        # Check sector allocation limits
        if not self._check_sector_allocation(symbol):
            self.logger.debug(f"Cannot enter {symbol}: sector allocation limit exceeded")
            return False
        
        # Check portfolio heat (total risk exposure)
        if not self._check_portfolio_heat():
            self.logger.debug(f"Cannot enter {symbol}: portfolio heat limit exceeded")
            return False
        
        return True
    
    def _check_sector_allocation(self, symbol: str) -> bool:
        """
        Check if adding this symbol would exceed sector allocation limits.
        
        Args:
            symbol: Symbol to check
            
        Returns:
            bool: True if within limits, False otherwise
        """
        symbol_sector = self._get_symbol_sector(symbol)
        if not symbol_sector:
            return True  # No sector info, allow trade
        
        # Count current positions in same sector
        same_sector_count = sum(
            1 for pos in self.current_positions 
            if self._get_symbol_sector(pos.symbol) == symbol_sector
        )
        
        # Check max similar ETFs limit
        if same_sector_count >= self.risk_config.max_similar_etfs:
            self.logger.debug(f"Sector allocation limit: {same_sector_count} positions in {symbol_sector}")
            return False
        
        return True
    
    def _check_portfolio_heat(self) -> bool:
        """
        Check if total portfolio risk exposure is within limits.
        
        Returns:
            bool: True if within limits, False otherwise
        """
        current_equity = self._calculate_current_equity()
        
        # Calculate current total risk exposure
        current_heat = 0.0
        for pos in self.current_positions:
            risk_amount = abs(pos.entry_price - pos.stop_loss) * pos.quantity
            current_heat += risk_amount
        
        # Add risk from new position
        new_position_risk = current_equity * self.risk_config.max_risk_per_trade
        total_heat = current_heat + new_position_risk
        
        # Check if total heat exceeds 8% of capital
        max_heat = current_equity * 0.08
        if total_heat > max_heat:
            self.logger.debug(f"Portfolio heat limit: {total_heat:.2f} > {max_heat:.2f}")
            return False
        
        return True
    
    def enter_trade(self, symbol: str, setup: str, date: datetime, 
                   entry_price: float, signal_data: Dict) -> BacktestTrade:
        """
        Execute trade entry with optimized parameters.
        
        Args:
            symbol: Symbol to trade
            setup: Trade setup name
            date: Entry date
            entry_price: Entry price
            signal_data: Signal data dictionary
            
        Returns:
            BacktestTrade: Created trade object
        """
        # Calculate position size based on risk
        current_equity = self._calculate_current_equity()
        risk_amount = current_equity * self.risk_config.max_risk_per_trade
        
        # Use optimized parameters if available
        if self.current_params:
            stop_loss_pct = self.current_params.stop_loss / 100.0
            target_pct = self.current_params.profit_target / 100.0
            max_days = self.current_params.max_holding_days
        else:
            stop_loss_pct = 0.05  # 5% default
            target_pct = 0.10     # 10% default
            max_days = 30         # 30 days default
        
        # Calculate stop loss and target prices
        stop_loss_price = entry_price * (1 - stop_loss_pct)
        target_price = entry_price * (1 + target_pct)
        
        # Calculate position size
        risk_per_share = entry_price - stop_loss_price
        quantity = risk_amount / risk_per_share if risk_per_share > 0 else 100
        
        # Create trade object
        trade = BacktestTrade(
            symbol=symbol,
            setup=setup,
            entry_date=date,
            entry_price=entry_price,
            quantity=quantity,
            stop_loss=stop_loss_price,
            target=target_price,
            max_holding_days=max_days,
            confidence=signal_data.get('confidence', 0.0),
            regime_at_entry=signal_data.get('regime', 'unknown')
        )
        
        # Add to current positions
        self.current_positions.append(trade)
        
        self.logger.info(f"Entered {symbol} at {entry_price:.2f}, quantity: {quantity:.0f}, "
                        f"stop: {stop_loss_price:.2f}, target: {target_price:.2f}")
        
        return trade
    
    def update_positions(self, date: datetime) -> None:
        """
        Update all open positions and check for exit conditions.
        
        Args:
            date: Current date
        """
        positions_to_close = []
        
        for trade in self.current_positions:
            # Get current price
            current_price = self._get_price_for_date(trade.symbol, date)
            if current_price is None:
                continue
            
            # Update MAE (Maximum Adverse Excursion) and MFE (Maximum Favorable Excursion)
            unrealized_pnl = (current_price - trade.entry_price) * trade.quantity
            trade.unrealized_pnl = unrealized_pnl
            
            # Update MAE (worst point)
            if unrealized_pnl < trade.mae:
                trade.mae = unrealized_pnl
            
            # Update MFE (best point)
            if unrealized_pnl > trade.mfe:
                trade.mfe = unrealized_pnl
            
            # Check exit conditions
            should_exit = False
            exit_reason = ""
            
            # Stop loss check
            if current_price <= trade.stop_loss:
                should_exit = True
                exit_reason = "stop_loss"
            
            # Target check
            elif current_price >= trade.target:
                should_exit = True
                exit_reason = "target"
            
            # Maximum holding period check
            elif (date - trade.entry_date).days >= trade.max_holding_days:
                should_exit = True
                exit_reason = "max_holding_days"
            
            if should_exit:
                positions_to_close.append((trade, current_price, exit_reason))
        
        # Close positions that met exit conditions
        for trade, exit_price, exit_reason in positions_to_close:
            self._close_position(trade, date, exit_price, exit_reason)
    
    def _close_position(self, trade: BacktestTrade, exit_date: datetime, 
                       exit_price: float, exit_reason: str) -> None:
        """
        Close a position and calculate P&L.
        
        Args:
            trade: Trade to close
            exit_date: Exit date
            exit_price: Exit price
            exit_reason: Reason for exit
        """
        # Update trade with exit information
        trade.exit_date = exit_date
        trade.exit_price = exit_price
        trade.status = "closed"
        
        # Calculate P&L
        trade.realized_pnl = (exit_price - trade.entry_price) * trade.quantity
        trade.unrealized_pnl = 0.0
        
        # Calculate R-multiple
        risk_amount = abs(trade.entry_price - trade.stop_loss) * trade.quantity
        if risk_amount > 0:
            trade.r_multiple = trade.realized_pnl / risk_amount
        
        # Move from current positions to completed trades
        self.current_positions.remove(trade)
        self.trades.append(trade)
        
        self.logger.info(f"Closed {trade.symbol} at {exit_price:.2f}, "
                        f"P&L: {trade.realized_pnl:.2f}, R: {trade.r_multiple:.2f}, "
                        f"reason: {exit_reason}")
    
    def _calculate_current_equity(self) -> float:
        """
        Calculate current portfolio equity including unrealized P&L.
        
        Returns:
            float: Current equity value
        """
        # Start with initial capital
        equity = self.initial_capital
        
        # Add realized P&L from closed trades
        for trade in self.trades:
            equity += trade.realized_pnl
        
        # Add unrealized P&L from open positions
        for trade in self.current_positions:
            equity += trade.unrealized_pnl
        
        return equity
    
    def _get_symbol_sector(self, symbol: str) -> Optional[str]:
        """
        Get sector information for a symbol from database.
        
        Args:
            symbol: Symbol to look up
            
        Returns:
            Optional[str]: Sector name or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT sector FROM instruments WHERE symbol = ?",
                    (symbol,)
                )
                result = cursor.fetchone()
                return result[0] if result else None
        except Exception as e:
            self.logger.warning(f"Error getting sector for {symbol}: {e}")
            return None
    
    def _is_stock(self, symbol: str) -> bool:
        """
        Determine if symbol is an individual stock vs ETF.
        
        Args:
            symbol: Symbol to check
            
        Returns:
            bool: True if stock, False if ETF
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT type FROM instruments WHERE symbol = ?",
                    (symbol,)
                )
                result = cursor.fetchone()
                return result[0] == "stock" if result else False
        except Exception as e:
            self.logger.warning(f"Error checking type for {symbol}: {e}")
            return False
    
    def _get_price_for_date(self, symbol: str, date: datetime) -> Optional[float]:
        """
        Get price data for a symbol on a specific date.
        
        Args:
            symbol: Symbol to get price for
            date: Date to get price for
            
        Returns:
            Optional[float]: Close price or None if not found
        """
        try:
            # Use data cache to get price data
            data = self.data_cache.get_price_data(symbol, days_back=5)
            if data is None or data.empty:
                return None
            
            # Find nearest date
            target_date = pd.to_datetime(date).normalize()
            data['date'] = pd.to_datetime(data['date']).dt.normalize()
            
            # Get exact or nearest earlier date
            available_dates = data[data['date'] <= target_date]
            if available_dates.empty:
                return None
            
            # Get the most recent available date
            latest_date = available_dates['date'].max()
            price_row = available_dates[available_dates['date'] == latest_date]
            
            if not price_row.empty:
                return float(price_row['close'].iloc[0])
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Error getting price for {symbol} on {date}: {e}")
            return None
    
    def calculate_performance_metrics(self) -> Dict:
        """
        Calculate comprehensive performance metrics for closed trades.
        
        Returns:
            Dict: Performance metrics dictionary
        """
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_r_multiple': 0.0,
                'total_return': 0.0,
                'max_drawdown': 0.0
            }
        
        # Basic statistics
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t.realized_pnl > 0]
        losing_trades = [t for t in self.trades if t.realized_pnl < 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0
        
        # R-multiple statistics
        r_multiples = [t.r_multiple for t in self.trades if t.r_multiple != 0]
        avg_r_multiple = np.mean(r_multiples) if r_multiples else 0.0
        
        # Total return
        total_pnl = sum(t.realized_pnl for t in self.trades)
        total_return = total_pnl / self.initial_capital if self.initial_capital > 0 else 0.0
        
        # Calculate equity curve for drawdown
        equity_curve = [self.initial_capital]
        for trade in self.trades:
            equity_curve.append(equity_curve[-1] + trade.realized_pnl)
        
        # Calculate maximum drawdown
        max_drawdown = 0.0
        peak = equity_curve[0]
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak if peak > 0 else 0.0
            max_drawdown = max(max_drawdown, drawdown)
        
        # MAE/MFE statistics
        mae_values = [t.mae for t in self.trades if t.mae != 0]
        mfe_values = [t.mfe for t in self.trades if t.mfe != 0]
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_r_multiple': avg_r_multiple,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'total_pnl': total_pnl,
            'avg_mae': np.mean(mae_values) if mae_values else 0.0,
            'avg_mfe': np.mean(mfe_values) if mfe_values else 0.0,
            'equity_curve': equity_curve
        }
    
    def get_trade_summary(self) -> Dict:
        """
        Get summary of all trades (open and closed).
        
        Returns:
            Dict: Trade summary
        """
        return {
            'total_trades': len(self.trades),
            'open_positions': len(self.current_positions),
            'closed_trades': len(self.trades),
            'current_equity': self._calculate_current_equity(),
            'unrealized_pnl': sum(t.unrealized_pnl for t in self.current_positions),
            'realized_pnl': sum(t.realized_pnl for t in self.trades)
        }
    
    def reset(self) -> None:
        """Reset all trade tracking for new backtest."""
        self.trades.clear()
        self.current_positions.clear()
        self.daily_equity.clear()


# Test the TradeManager class
if __name__ == "__main__":
    from backtest_config import BacktestConfiguration
    from data_cache import DataCache
    
    # Create test configuration
    config = BacktestConfiguration()
    
    # Initialize components
    data_cache = DataCache("journal.db")
    trade_manager = TradeManager(
        db_path="journal.db",
        data_cache=data_cache,
        risk_config=config.risk_management,
        optimization_config=config.optimization
    )
    
    # Test basic functionality
    print("TradeManager initialized successfully")
    print(f"Initial equity: ${trade_manager._calculate_current_equity():,.2f}")
    
    # Test trade entry validation
    can_enter = trade_manager.can_enter_trade("SPY", "momentum", datetime.now())
    print(f"Can enter SPY trade: {can_enter}")
    
    # Print trade summary
    summary = trade_manager.get_trade_summary()
    print(f"Trade summary: {summary}")