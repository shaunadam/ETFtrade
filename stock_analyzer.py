#!/usr/bin/env python3
"""
Stock-specific analysis tools for the trading system.

This module provides analysis capabilities specifically designed for individual stocks,
including sector analysis, correlation tracking, and stock-specific risk metrics.

Usage:
    python stock_analyzer.py --sector Technology --correlation-matrix
    python stock_analyzer.py --stock AAPL --detailed-analysis
    python stock_analyzer.py --sector-rotation --export-csv
"""

import argparse
import sqlite3
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from data_cache import DataCache
from regime_detection import RegimeDetector


class StockAnalyzer:
    """Stock-specific analysis and research tools."""
    
    def __init__(self, db_path: str = "journal.db"):
        self.db_path = db_path
        self.data_cache = DataCache(db_path)
        self.regime_detector = RegimeDetector(db_path)
    
    def get_stocks_by_sector(self, sector: Optional[str] = None) -> Dict[str, List[str]]:
        """Get stocks grouped by sector."""
        query = "SELECT symbol, sector FROM instruments WHERE type = 'Stock'"
        if sector:
            query += " AND sector = ?"
            params = (sector,)
        else:
            params = ()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(query, params)
                results = cursor.fetchall()
            
            if sector:
                return {sector: [row[0] for row in results]}
            else:
                sectors = {}
                for symbol, stock_sector in results:
                    if stock_sector not in sectors:
                        sectors[stock_sector] = []
                    sectors[stock_sector].append(symbol)
                return sectors
                
        except Exception as e:
            print(f"❌ Error querying stocks: {e}")
            return {}
    
    def analyze_sector_performance(self, period: str = "3mo") -> pd.DataFrame:
        """Analyze performance of all sectors."""
        sectors = self.get_stocks_by_sector()
        
        if not sectors:
            print("❌ No stocks found in database")
            return pd.DataFrame()
        
        sector_performance = []
        
        for sector, symbols in sectors.items():
            if not symbols:
                continue
                
            print(f"📊 Analyzing {sector} sector ({len(symbols)} stocks)...")
            
            # Calculate average sector performance
            sector_returns = []
            sector_volatility = []
            valid_stocks = 0
            
            for symbol in symbols:
                try:
                    data = self.data_cache.get_cached_data(symbol, period)
                    if len(data) < 20:  # Need minimum data
                        continue
                    
                    # Calculate return and volatility
                    returns = data['Close'].pct_change().dropna()
                    if len(returns) > 0:
                        total_return = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
                        volatility = returns.std() * np.sqrt(252) * 100  # Annualized
                        
                        sector_returns.append(total_return)
                        sector_volatility.append(volatility)
                        valid_stocks += 1
                        
                except Exception as e:
                    print(f"   ⚠️  Error analyzing {symbol}: {e}")
                    continue
            
            if sector_returns:
                avg_return = np.mean(sector_returns)
                avg_volatility = np.mean(sector_volatility)
                median_return = np.median(sector_returns)
                best_stock_return = max(sector_returns)
                worst_stock_return = min(sector_returns)
                
                sector_performance.append({
                    'sector': sector,
                    'num_stocks': len(symbols),
                    'valid_stocks': valid_stocks,
                    'avg_return_pct': avg_return,
                    'median_return_pct': median_return,
                    'avg_volatility_pct': avg_volatility,
                    'best_stock_return_pct': best_stock_return,
                    'worst_stock_return_pct': worst_stock_return,
                    'return_spread_pct': best_stock_return - worst_stock_return
                })
        
        if sector_performance:
            df = pd.DataFrame(sector_performance)
            df = df.sort_values('avg_return_pct', ascending=False)
            return df
        else:
            return pd.DataFrame()
    
    def calculate_stock_correlations(self, symbols: List[str], period: str = "6mo") -> pd.DataFrame:
        """Calculate correlation matrix for given stocks."""
        if len(symbols) < 2:
            print("❌ Need at least 2 symbols for correlation analysis")
            return pd.DataFrame()
        
        print(f"📊 Calculating correlations for {len(symbols)} stocks...")
        
        # Collect price data for all symbols
        price_data = {}
        
        for symbol in symbols:
            try:
                data = self.data_cache.get_cached_data(symbol, period)
                if len(data) > 50:  # Need reasonable amount of data
                    price_data[symbol] = data['Close']
                else:
                    print(f"   ⚠️  Insufficient data for {symbol}")
            except Exception as e:
                print(f"   ⚠️  Error getting data for {symbol}: {e}")
        
        if len(price_data) < 2:
            print("❌ Not enough valid symbols for correlation analysis")
            return pd.DataFrame()
        
        # Create price DataFrame
        price_df = pd.DataFrame(price_data)
        
        # Calculate returns
        returns_df = price_df.pct_change().dropna()
        
        # Calculate correlation matrix
        correlation_matrix = returns_df.corr()
        
        return correlation_matrix
    
    def analyze_single_stock(self, symbol: str, period: str = "1y") -> Dict:
        """Perform detailed analysis of a single stock."""
        print(f"📊 Analyzing {symbol}...")
        
        try:
            data = self.data_cache.get_cached_data(symbol, period)
            if data.empty:
                return {"error": f"No data available for {symbol}"}
            
            # Get stock info from database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT name, sector, theme, notes FROM instruments WHERE symbol = ?", 
                    (symbol,)
                )
                result = cursor.fetchone()
                if result:
                    name, sector, theme, notes = result
                else:
                    name = sector = theme = notes = "Unknown"
            
            # Calculate basic metrics
            current_price = data['Close'].iloc[-1]
            start_price = data['Close'].iloc[0]
            total_return = (current_price / start_price - 1) * 100
            
            # Volatility metrics
            returns = data['Close'].pct_change().dropna()
            daily_vol = returns.std() * 100
            annual_vol = daily_vol * np.sqrt(252)
            
            # Technical indicators
            sma20 = data['SMA20'].iloc[-1] if 'SMA20' in data.columns else None
            sma50 = data['SMA50'].iloc[-1] if 'SMA50' in data.columns else None
            sma200 = data['SMA200'].iloc[-1] if 'SMA200' in data.columns else None
            rsi = data['RSI'].iloc[-1] if 'RSI' in data.columns else None
            atr = data['ATR'].iloc[-1] if 'ATR' in data.columns else None
            
            # Price relative to moving averages
            price_vs_sma20 = (current_price / sma20 - 1) * 100 if sma20 else None
            price_vs_sma50 = (current_price / sma50 - 1) * 100 if sma50 else None
            price_vs_sma200 = (current_price / sma200 - 1) * 100 if sma200 else None
            
            # Volume analysis
            current_volume = data['Volume'].iloc[-1]
            avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else None
            
            # Recent performance
            performance_1w = (current_price / data['Close'].iloc[-5] - 1) * 100 if len(data) >= 5 else None
            performance_1m = (current_price / data['Close'].iloc[-21] - 1) * 100 if len(data) >= 21 else None
            
            # Risk metrics
            max_drawdown = self._calculate_max_drawdown(data['Close'])
            
            analysis = {
                "symbol": symbol,
                "name": name,
                "sector": sector,
                "theme": theme,
                "notes": notes,
                "current_price": current_price,
                "total_return_pct": total_return,
                "performance_1w_pct": performance_1w,
                "performance_1m_pct": performance_1m,
                "daily_volatility_pct": daily_vol,
                "annual_volatility_pct": annual_vol,
                "max_drawdown_pct": max_drawdown,
                "current_volume": current_volume,
                "avg_volume_20d": avg_volume,
                "volume_ratio": volume_ratio,
                "rsi": rsi,
                "atr": atr,
                "price_vs_sma20_pct": price_vs_sma20,
                "price_vs_sma50_pct": price_vs_sma50,
                "price_vs_sma200_pct": price_vs_sma200,
                "data_points": len(data)
            }
            
            return analysis
            
        except Exception as e:
            return {"error": f"Error analyzing {symbol}: {e}"}
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown percentage."""
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        max_drawdown = drawdown.min() * 100
        return max_drawdown
    
    def compare_stock_to_sector_etf(self, stock_symbol: str, period: str = "6mo") -> Dict:
        """Compare individual stock performance to its sector ETF."""
        # Get stock sector
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT sector FROM instruments WHERE symbol = ?", 
                    (stock_symbol,)
                )
                result = cursor.fetchone()
                if not result:
                    return {"error": f"Stock {stock_symbol} not found"}
                
                sector = result[0]
        except Exception as e:
            return {"error": f"Error querying stock sector: {e}"}
        
        # Map sectors to ETFs (simplified mapping)
        sector_etf_map = {
            "Technology": "XLK",
            "Financial": "XLF", 
            "Healthcare": "XLV",
            "Energy": "XLE",
            "Consumer Discretionary": "XLY",
            "Consumer Staples": "XLP",
            "Industrial": "XLI",
            "Materials": "XLB",
            "Utilities": "XLU",
            "Real Estate": "XLRE",
            "Communication Services": "XLC"
        }
        
        sector_etf = sector_etf_map.get(sector)
        if not sector_etf:
            return {"error": f"No sector ETF mapping found for {sector}"}
        
        try:
            # Get data for both
            stock_data = self.data_cache.get_cached_data(stock_symbol, period)
            etf_data = self.data_cache.get_cached_data(sector_etf, period)
            
            if stock_data.empty or etf_data.empty:
                return {"error": "Insufficient data for comparison"}
            
            # Calculate returns
            stock_return = (stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[0] - 1) * 100
            etf_return = (etf_data['Close'].iloc[-1] / etf_data['Close'].iloc[0] - 1) * 100
            
            # Calculate relative performance
            relative_performance = stock_return - etf_return
            
            # Calculate volatilities
            stock_vol = stock_data['Close'].pct_change().std() * np.sqrt(252) * 100
            etf_vol = etf_data['Close'].pct_change().std() * np.sqrt(252) * 100
            
            return {
                "stock_symbol": stock_symbol,
                "sector": sector,
                "sector_etf": sector_etf,
                "stock_return_pct": stock_return,
                "etf_return_pct": etf_return,
                "relative_performance_pct": relative_performance,
                "stock_volatility_pct": stock_vol,
                "etf_volatility_pct": etf_vol,
                "outperformed_sector": relative_performance > 0
            }
            
        except Exception as e:
            return {"error": f"Error comparing {stock_symbol} to {sector_etf}: {e}"}
    
    def export_analysis_csv(self, data: pd.DataFrame, filename: str) -> None:
        """Export analysis results to CSV."""
        try:
            data.to_csv(filename, index=False)
            print(f"📄 Analysis exported to {filename}")
        except Exception as e:
            print(f"❌ Error exporting to CSV: {e}")


def main():
    """Main CLI interface for stock analysis."""
    parser = argparse.ArgumentParser(
        description="Stock Analysis Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python stock_analyzer.py --sector Technology
  python stock_analyzer.py --stock AAPL --detailed
  python stock_analyzer.py --correlation-matrix --sector Technology
  python stock_analyzer.py --sector-performance --export-csv sector_analysis.csv
        """
    )
    
    # Analysis type
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--sector-performance', action='store_true',
                      help='Analyze performance across all sectors')
    group.add_argument('--sector', type=str,
                      help='Analyze specific sector')
    group.add_argument('--stock', type=str,
                      help='Analyze specific stock')
    group.add_argument('--correlation-matrix', action='store_true',
                      help='Calculate correlation matrix')
    
    # Options
    parser.add_argument('--period', type=str, default='3mo',
                       choices=['1mo', '3mo', '6mo', '1y', '2y'],
                       help='Analysis period (default: 3mo)')
    parser.add_argument('--detailed', action='store_true',
                       help='Show detailed analysis (for single stock)')
    parser.add_argument('--compare-sector', action='store_true',
                       help='Compare stock to sector ETF (for single stock)')
    parser.add_argument('--export-csv', type=str,
                       help='Export results to CSV file')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = StockAnalyzer()
    
    if args.sector_performance:
        print("🔍 Analyzing sector performance...")
        results = analyzer.analyze_sector_performance(args.period)
        
        if not results.empty:
            print(f"\n📊 Sector Performance Analysis ({args.period}):")
            print("=" * 80)
            for _, row in results.iterrows():
                print(f"{row['sector']:<25} "
                      f"Avg: {row['avg_return_pct']:+6.1f}% "
                      f"Med: {row['median_return_pct']:+6.1f}% "
                      f"Vol: {row['avg_volatility_pct']:5.1f}% "
                      f"({row['valid_stocks']}/{row['num_stocks']} stocks)")
            
            if args.export_csv:
                analyzer.export_analysis_csv(results, args.export_csv)
        else:
            print("❌ No sector performance data available")
    
    elif args.sector:
        print(f"🔍 Analyzing {args.sector} sector...")
        stocks = analyzer.get_stocks_by_sector(args.sector)
        
        if args.sector in stocks:
            symbols = stocks[args.sector]
            print(f"📊 Found {len(symbols)} stocks in {args.sector}:")
            
            if args.correlation_matrix and len(symbols) > 1:
                corr_matrix = analyzer.calculate_stock_correlations(symbols, args.period)
                if not corr_matrix.empty:
                    print(f"\n📈 Correlation Matrix ({args.period}):")
                    print(corr_matrix.round(3))
                    
                    if args.export_csv:
                        corr_matrix.to_csv(args.export_csv)
                        print(f"📄 Correlation matrix exported to {args.export_csv}")
            else:
                for symbol in symbols:
                    analysis = analyzer.analyze_single_stock(symbol, args.period)
                    if 'error' not in analysis:
                        print(f"\n{symbol}: {analysis['total_return_pct']:+.1f}% "
                              f"(Vol: {analysis['annual_volatility_pct']:.1f}%)")
        else:
            print(f"❌ No stocks found in {args.sector} sector")
    
    elif args.stock:
        symbol = args.stock.upper()
        print(f"🔍 Analyzing {symbol}...")
        
        analysis = analyzer.analyze_single_stock(symbol, args.period)
        
        if 'error' in analysis:
            print(f"❌ {analysis['error']}")
        else:
            print(f"\n📊 {symbol} - {analysis['name']}")
            print("=" * 60)
            print(f"Sector:           {analysis['sector']}")
            print(f"Current Price:    ${analysis['current_price']:.2f}")
            print(f"Total Return:     {analysis['total_return_pct']:+.1f}%")
            if analysis['performance_1w_pct']:
                print(f"1-Week Return:    {analysis['performance_1w_pct']:+.1f}%")
            if analysis['performance_1m_pct']:
                print(f"1-Month Return:   {analysis['performance_1m_pct']:+.1f}%")
            print(f"Annual Volatility: {analysis['annual_volatility_pct']:.1f}%")
            print(f"Max Drawdown:     {analysis['max_drawdown_pct']:.1f}%")
            
            if args.detailed:
                print(f"\nTechnical Indicators:")
                if analysis['rsi']:
                    print(f"RSI:              {analysis['rsi']:.1f}")
                if analysis['price_vs_sma20_pct'] is not None:
                    print(f"vs SMA20:         {analysis['price_vs_sma20_pct']:+.1f}%")
                if analysis['price_vs_sma50_pct'] is not None:
                    print(f"vs SMA50:         {analysis['price_vs_sma50_pct']:+.1f}%")
                if analysis['price_vs_sma200_pct'] is not None:
                    print(f"vs SMA200:        {analysis['price_vs_sma200_pct']:+.1f}%")
                if analysis['volume_ratio']:
                    print(f"Volume Ratio:     {analysis['volume_ratio']:.1f}x")
            
            if args.compare_sector:
                comparison = analyzer.compare_stock_to_sector_etf(symbol, args.period)
                if 'error' not in comparison:
                    print(f"\nSector Comparison ({comparison['sector_etf']}):")
                    print(f"Stock Return:     {comparison['stock_return_pct']:+.1f}%")
                    print(f"Sector ETF:       {comparison['etf_return_pct']:+.1f}%")
                    print(f"Relative Perf:    {comparison['relative_performance_pct']:+.1f}%")
                    status = "✅ OUTPERFORMED" if comparison['outperformed_sector'] else "❌ UNDERPERFORMED"
                    print(f"Status:           {status}")
    
    elif args.correlation_matrix:
        # Get all stocks for correlation matrix
        all_stocks = analyzer.get_stocks_by_sector()
        all_symbols = []
        for symbols in all_stocks.values():
            all_symbols.extend(symbols)
        
        if len(all_symbols) > 1:
            print(f"🔍 Calculating correlation matrix for {len(all_symbols)} stocks...")
            corr_matrix = analyzer.calculate_stock_correlations(all_symbols[:20], args.period)  # Limit to first 20
            
            if not corr_matrix.empty:
                print(f"\n📈 Stock Correlation Matrix ({args.period}):")
                print(corr_matrix.round(3))
                
                if args.export_csv:
                    corr_matrix.to_csv(args.export_csv)
                    print(f"📄 Correlation matrix exported to {args.export_csv}")
        else:
            print("❌ Not enough stocks for correlation analysis")


if __name__ == "__main__":
    main()