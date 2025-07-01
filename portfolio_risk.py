#!/usr/bin/env python3
"""
Portfolio-level risk management for stock concentration and correlation controls.

This module provides advanced risk management specifically for portfolios containing
both individual stocks and ETFs, with focus on concentration limits and correlation monitoring.
"""

import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

from data_cache import DataCache


@dataclass
class PositionInfo:
    """Information about a current portfolio position."""
    symbol: str
    instrument_type: str  # 'ETF', 'Stock', 'ETN'
    sector: str
    position_size: float
    entry_price: float
    current_price: float
    market_value: float
    weight: float  # Percentage of portfolio
    days_held: int


@dataclass
class RiskLimits:
    """Portfolio risk limits configuration."""
    # Position concentration limits
    max_single_position_pct: float = 5.0  # 5% max per individual stock
    max_sector_allocation_pct: float = 20.0  # 20% max per sector across stocks
    max_stock_allocation_pct: float = 50.0  # 50% max total allocation to individual stocks
    
    # Correlation limits
    max_correlated_positions: int = 3  # Max positions with >0.7 correlation
    max_correlation_threshold: float = 0.7  # Correlation threshold
    
    # Risk exposure limits
    max_total_risk_pct: float = 8.0  # 8% max total risk exposure (4 positions * 2% each)
    max_single_risk_pct: float = 2.0  # 2% max risk per position
    
    # Stock-specific limits
    max_concurrent_stocks: int = 8  # Max individual stock positions
    min_stock_liquidity_volume: float = 1000000  # Min daily volume for stocks


@dataclass
class RiskMetrics:
    """Current portfolio risk metrics."""
    total_positions: int
    stock_positions: int
    etf_positions: int
    
    # Concentration metrics
    largest_position_pct: float
    stock_allocation_pct: float
    sector_concentrations: Dict[str, float]
    
    # Risk exposure metrics
    total_risk_exposure_pct: float
    largest_risk_exposure_pct: float
    
    # Correlation metrics
    avg_correlation: float
    max_correlation: float
    high_correlation_pairs: int
    
    # Compliance
    risk_limit_violations: List[str]
    concentration_violations: List[str]


class PortfolioRiskManager:
    """Advanced portfolio risk management with stock concentration controls."""
    
    def __init__(self, db_path: str = "journal.db", risk_limits: Optional[RiskLimits] = None):
        self.db_path = db_path
        self.data_cache = DataCache(db_path)
        self.risk_limits = risk_limits or RiskLimits()
    
    def get_current_positions(self, portfolio_value: float) -> List[PositionInfo]:
        """Get current portfolio positions from the trading journal."""
        positions = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Query open trades from the journal
                cursor = conn.execute("""
                    SELECT t.symbol, t.entry_price, t.size, t.entry_date,
                           i.type, i.sector
                    FROM trades t
                    JOIN instruments i ON t.instrument_id = i.id
                    WHERE t.status = 'open'
                    ORDER BY t.entry_date
                """)
                
                for row in cursor.fetchall():
                    symbol, entry_price, size, entry_date, instrument_type, sector = row
                    
                    # Get current price
                    try:
                        data = self.data_cache.get_cached_data(symbol, "5d")
                        if not data.empty:
                            current_price = data['Close'].iloc[-1]
                        else:
                            current_price = entry_price  # Fallback
                    except Exception:
                        current_price = entry_price  # Fallback
                    
                    # Calculate position metrics
                    market_value = size * current_price
                    weight = (market_value / portfolio_value) * 100 if portfolio_value > 0 else 0
                    
                    # Calculate days held
                    entry_dt = datetime.strptime(entry_date, "%Y-%m-%d")
                    days_held = (datetime.now() - entry_dt).days
                    
                    position = PositionInfo(
                        symbol=symbol,
                        instrument_type=instrument_type,
                        sector=sector or "Unknown",
                        position_size=size,
                        entry_price=entry_price,
                        current_price=current_price,
                        market_value=market_value,
                        weight=weight,
                        days_held=days_held
                    )
                    
                    positions.append(position)
                    
        except Exception as e:
            print(f"Error getting positions: {e}")
        
        return positions
    
    def calculate_portfolio_risk_metrics(self, positions: List[PositionInfo], 
                                       portfolio_value: float) -> RiskMetrics:
        """Calculate comprehensive portfolio risk metrics."""
        if not positions:
            return RiskMetrics(
                total_positions=0, stock_positions=0, etf_positions=0,
                largest_position_pct=0, stock_allocation_pct=0,
                sector_concentrations={}, total_risk_exposure_pct=0,
                largest_risk_exposure_pct=0, avg_correlation=0,
                max_correlation=0, high_correlation_pairs=0,
                risk_limit_violations=[], concentration_violations=[]
            )
        
        # Basic position counts
        total_positions = len(positions)
        stock_positions = sum(1 for p in positions if p.instrument_type == 'Stock')
        etf_positions = sum(1 for p in positions if p.instrument_type in ['ETF', 'ETN'])
        
        # Concentration analysis
        weights = [p.weight for p in positions]
        largest_position_pct = max(weights) if weights else 0
        
        stock_allocation = sum(p.weight for p in positions if p.instrument_type == 'Stock')
        
        # Sector concentration
        sector_concentrations = {}
        for position in positions:
            sector = position.sector
            if sector not in sector_concentrations:
                sector_concentrations[sector] = 0
            sector_concentrations[sector] += position.weight
        
        # Risk exposure calculation (simplified)
        # Assume 2% risk per position for now
        total_risk_exposure = len(positions) * 2.0  # Simplified
        largest_risk_exposure = 2.0  # Simplified
        
        # Correlation analysis
        correlation_metrics = self._calculate_correlation_metrics(positions)
        
        # Check for violations
        violations = self._check_risk_violations(
            positions, stock_allocation, sector_concentrations,
            total_risk_exposure, largest_position_pct
        )
        
        return RiskMetrics(
            total_positions=total_positions,
            stock_positions=stock_positions,
            etf_positions=etf_positions,
            largest_position_pct=largest_position_pct,
            stock_allocation_pct=stock_allocation,
            sector_concentrations=sector_concentrations,
            total_risk_exposure_pct=total_risk_exposure,
            largest_risk_exposure_pct=largest_risk_exposure,
            avg_correlation=correlation_metrics['avg_correlation'],
            max_correlation=correlation_metrics['max_correlation'],
            high_correlation_pairs=correlation_metrics['high_correlation_pairs'],
            risk_limit_violations=violations['risk_violations'],
            concentration_violations=violations['concentration_violations']
        )
    
    def _calculate_correlation_metrics(self, positions: List[PositionInfo]) -> Dict:
        """Calculate correlation metrics for current positions."""
        if len(positions) < 2:
            return {
                'avg_correlation': 0.0,
                'max_correlation': 0.0,
                'high_correlation_pairs': 0
            }
        
        symbols = [p.symbol for p in positions]
        
        try:
            # Get price data for correlation calculation
            price_data = {}
            for symbol in symbols:
                try:
                    data = self.data_cache.get_cached_data(symbol, "3mo")
                    if len(data) > 50:  # Need reasonable amount of data
                        price_data[symbol] = data['Close']
                except Exception:
                    continue
            
            if len(price_data) < 2:
                return {'avg_correlation': 0.0, 'max_correlation': 0.0, 'high_correlation_pairs': 0}
            
            # Create price DataFrame and calculate returns
            price_df = pd.DataFrame(price_data)
            returns_df = price_df.pct_change().dropna()
            
            # Calculate correlation matrix
            correlation_matrix = returns_df.corr()
            
            # Extract correlation values (excluding diagonal)
            correlations = []
            high_correlation_pairs = 0
            
            for i in range(len(correlation_matrix.columns)):
                for j in range(i + 1, len(correlation_matrix.columns)):
                    corr = correlation_matrix.iloc[i, j]
                    if not pd.isna(corr):
                        correlations.append(abs(corr))
                        if abs(corr) > self.risk_limits.max_correlation_threshold:
                            high_correlation_pairs += 1
            
            avg_correlation = np.mean(correlations) if correlations else 0.0
            max_correlation = max(correlations) if correlations else 0.0
            
            return {
                'avg_correlation': avg_correlation,
                'max_correlation': max_correlation,
                'high_correlation_pairs': high_correlation_pairs
            }
            
        except Exception as e:
            print(f"Error calculating correlations: {e}")
            return {'avg_correlation': 0.0, 'max_correlation': 0.0, 'high_correlation_pairs': 0}
    
    def _check_risk_violations(self, positions: List[PositionInfo], stock_allocation: float,
                             sector_concentrations: Dict[str, float], total_risk_exposure: float,
                             largest_position: float) -> Dict[str, List[str]]:
        """Check for risk limit violations."""
        risk_violations = []
        concentration_violations = []
        
        # Check total risk exposure
        if total_risk_exposure > self.risk_limits.max_total_risk_pct:
            risk_violations.append(f"Total risk exposure {total_risk_exposure:.1f}% > {self.risk_limits.max_total_risk_pct:.1f}%")
        
        # Check stock allocation
        if stock_allocation > self.risk_limits.max_stock_allocation_pct:
            concentration_violations.append(f"Stock allocation {stock_allocation:.1f}% > {self.risk_limits.max_stock_allocation_pct:.1f}%")
        
        # Check single position size
        if largest_position > self.risk_limits.max_single_position_pct:
            concentration_violations.append(f"Largest position {largest_position:.1f}% > {self.risk_limits.max_single_position_pct:.1f}%")
        
        # Check sector concentration
        for sector, allocation in sector_concentrations.items():
            if allocation > self.risk_limits.max_sector_allocation_pct:
                concentration_violations.append(f"Sector {sector} allocation {allocation:.1f}% > {self.risk_limits.max_sector_allocation_pct:.1f}%")
        
        # Check number of stock positions
        stock_count = sum(1 for p in positions if p.instrument_type == 'Stock')
        if stock_count > self.risk_limits.max_concurrent_stocks:
            concentration_violations.append(f"Stock positions {stock_count} > {self.risk_limits.max_concurrent_stocks}")
        
        return {
            'risk_violations': risk_violations,
            'concentration_violations': concentration_violations
        }
    
    def can_add_position(self, symbol: str, position_size_pct: float, 
                        current_positions: List[PositionInfo]) -> Tuple[bool, List[str]]:
        """Check if a new position can be added within risk limits."""
        reasons = []
        
        try:
            # Get instrument info
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT type, sector FROM instruments WHERE symbol = ?", (symbol,)
                )
                result = cursor.fetchone()
                if not result:
                    return False, ["Symbol not found in instruments database"]
                
                instrument_type, sector = result
        except Exception as e:
            return False, [f"Database error: {e}"]
        
        # Check single position size limit
        if instrument_type == 'Stock' and position_size_pct > self.risk_limits.max_single_position_pct:
            reasons.append(f"Position size {position_size_pct:.1f}% > {self.risk_limits.max_single_position_pct:.1f}% limit for stocks")
        
        # Check if already holding this symbol
        if any(p.symbol == symbol for p in current_positions):
            reasons.append(f"Already holding position in {symbol}")
        
        # Check stock count limit
        stock_count = sum(1 for p in current_positions if p.instrument_type == 'Stock')
        if instrument_type == 'Stock' and stock_count >= self.risk_limits.max_concurrent_stocks:
            reasons.append(f"Already at max stock positions ({self.risk_limits.max_concurrent_stocks})")
        
        # Check sector concentration
        current_sector_allocation = sum(
            p.weight for p in current_positions if p.sector == sector
        )
        if current_sector_allocation + position_size_pct > self.risk_limits.max_sector_allocation_pct:
            reasons.append(f"Sector {sector} would exceed {self.risk_limits.max_sector_allocation_pct:.1f}% limit")
        
        # Check total stock allocation
        if instrument_type == 'Stock':
            current_stock_allocation = sum(
                p.weight for p in current_positions if p.instrument_type == 'Stock'
            )
            if current_stock_allocation + position_size_pct > self.risk_limits.max_stock_allocation_pct:
                reasons.append(f"Total stock allocation would exceed {self.risk_limits.max_stock_allocation_pct:.1f}% limit")
        
        # Check liquidity for stocks
        if instrument_type == 'Stock':
            try:
                data = self.data_cache.get_cached_data(symbol, "1mo")
                if not data.empty:
                    avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
                    if avg_volume < self.risk_limits.min_stock_liquidity_volume:
                        reasons.append(f"Stock liquidity too low: {avg_volume:,.0f} < {self.risk_limits.min_stock_liquidity_volume:,.0f}")
            except Exception:
                reasons.append(f"Could not verify liquidity for {symbol}")
        
        can_add = len(reasons) == 0
        return can_add, reasons
    
    def get_portfolio_risk_report(self, portfolio_value: float) -> str:
        """Generate a comprehensive portfolio risk report."""
        positions = self.get_current_positions(portfolio_value)
        metrics = self.calculate_portfolio_risk_metrics(positions, portfolio_value)
        
        report = []
        report.append("PORTFOLIO RISK ANALYSIS")
        report.append("=" * 50)
        report.append(f"Portfolio Value: ${portfolio_value:,.2f}")
        report.append(f"Total Positions: {metrics.total_positions}")
        report.append(f"  • ETF Positions: {metrics.etf_positions}")
        report.append(f"  • Stock Positions: {metrics.stock_positions}")
        report.append("")
        
        # Concentration Analysis
        report.append("CONCENTRATION ANALYSIS")
        report.append("-" * 30)
        report.append(f"Largest Position: {metrics.largest_position_pct:.1f}%")
        report.append(f"Stock Allocation: {metrics.stock_allocation_pct:.1f}%")
        report.append("")
        
        if metrics.sector_concentrations:
            report.append("Sector Allocations:")
            for sector, allocation in sorted(metrics.sector_concentrations.items(), 
                                           key=lambda x: x[1], reverse=True):
                report.append(f"  • {sector}: {allocation:.1f}%")
            report.append("")
        
        # Risk Exposure
        report.append("RISK EXPOSURE")
        report.append("-" * 30)
        report.append(f"Total Risk Exposure: {metrics.total_risk_exposure_pct:.1f}%")
        report.append(f"Largest Risk Exposure: {metrics.largest_risk_exposure_pct:.1f}%")
        report.append("")
        
        # Correlation Analysis
        report.append("CORRELATION ANALYSIS")
        report.append("-" * 30)
        report.append(f"Average Correlation: {metrics.avg_correlation:.3f}")
        report.append(f"Maximum Correlation: {metrics.max_correlation:.3f}")
        report.append(f"High Correlation Pairs: {metrics.high_correlation_pairs}")
        report.append("")
        
        # Violations
        if metrics.risk_limit_violations or metrics.concentration_violations:
            report.append("⚠️  RISK LIMIT VIOLATIONS")
            report.append("-" * 30)
            
            for violation in metrics.risk_limit_violations:
                report.append(f"  • {violation}")
            
            for violation in metrics.concentration_violations:
                report.append(f"  • {violation}")
            
            report.append("")
        else:
            report.append("✅ No risk limit violations")
            report.append("")
        
        # Current Positions Detail
        if positions:
            report.append("CURRENT POSITIONS")
            report.append("-" * 30)
            
            # Sort by weight descending
            sorted_positions = sorted(positions, key=lambda x: x.weight, reverse=True)
            
            for position in sorted_positions:
                pnl_pct = ((position.current_price - position.entry_price) / position.entry_price) * 100
                report.append(f"{position.symbol} ({position.instrument_type}):")
                report.append(f"  Weight: {position.weight:.1f}%, Sector: {position.sector}")
                report.append(f"  P&L: {pnl_pct:+.1f}%, Days: {position.days_held}")
            
            report.append("")
        
        return "\n".join(report)


def main():
    """CLI interface for portfolio risk analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Portfolio Risk Analysis")
    parser.add_argument("--portfolio-value", type=float, default=100000,
                       help="Current portfolio value (default: 100000)")
    parser.add_argument("--export-csv", type=str,
                       help="Export risk metrics to CSV file")
    
    args = parser.parse_args()
    
    # Initialize risk manager
    risk_manager = PortfolioRiskManager()
    
    # Generate risk report
    print(risk_manager.get_portfolio_risk_report(args.portfolio_value))
    
    # Export if requested
    if args.export_csv:
        positions = risk_manager.get_current_positions(args.portfolio_value)
        if positions:
            df = pd.DataFrame([
                {
                    'symbol': p.symbol,
                    'type': p.instrument_type,
                    'sector': p.sector,
                    'weight_pct': p.weight,
                    'market_value': p.market_value,
                    'days_held': p.days_held
                }
                for p in positions
            ])
            df.to_csv(args.export_csv, index=False)
            print(f"Risk metrics exported to {args.export_csv}")


if __name__ == "__main__":
    main()