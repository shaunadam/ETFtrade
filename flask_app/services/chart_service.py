"""
Chart Service

Service for generating chart data for Plotly.js visualizations.
Integrates with existing data cache and provides formatted data for:
- Price charts with technical indicators
- Volume analysis
- Trade setup visualization
- Regime overlays
"""

import sys
import os
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data_cache import DataCache
from regime_detection import RegimeDetector
try:
    from trade_setups import TradeSetupEngine
except ImportError:
    TradeSetupEngine = None

class ChartService:
    """Service for generating chart data and configurations"""
    
    def __init__(self):
        # Use the correct database path - same as other services
        db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'journal.db')
        self.data_cache = DataCache(db_path)
        self.regime_detector = RegimeDetector(db_path)
        self.setup_engine = TradeSetupEngine() if TradeSetupEngine else None
        
    def get_price_chart_data(self, symbol: str, period_days: int = 90) -> Dict[str, Any]:
        """
        Get price chart data with technical indicators for a symbol
        
        Args:
            symbol: Stock/ETF symbol
            period_days: Number of days to include in chart
            
        Returns:
            Dictionary with chart data and configuration
        """
        try:
            # Get price data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period_days)
            
            # Get price data using get_cached_data with appropriate period
            period_str = f"{period_days}d" if period_days < 365 else "1y"
            price_data = self.data_cache.get_cached_data(symbol, period_str)
            if price_data is None or price_data.empty:
                return {'error': f'No price data available for {symbol}'}
            
            # Get technical indicators from cached data
            indicators = self._extract_indicators_from_cached_data(price_data)
            
            # Get current regime data
            regime_data = self._get_regime_overlay_data(start_date, end_date)
            
            # Format data for Plotly
            chart_data = {
                'price_data': self._format_price_data(price_data),
                'indicators': indicators,
                'regime_overlay': regime_data,
                'volume_data': self._format_volume_data(price_data),
                'symbol': symbol,
                'period_days': period_days
            }
            
            return chart_data
            
        except Exception as e:
            return {'error': f'Failed to get chart data for {symbol}: {str(e)}'}
    
    def get_trade_setup_chart_data(self, symbol: str, setup_name: str, period_days: int = 90) -> Dict[str, Any]:
        """
        Get chart data with trade setup visualization
        
        Args:
            symbol: Stock/ETF symbol
            setup_name: Name of the trade setup
            period_days: Number of days to include
            
        Returns:
            Dictionary with chart data including setup signals
        """
        try:
            # Get base chart data
            chart_data = self.get_price_chart_data(symbol, period_days)
            if 'error' in chart_data:
                return chart_data
            
            # Get trade setup analysis
            if self.setup_engine:
                setup_analysis = self.setup_engine.analyze_symbol(symbol, setup_name)
                if setup_analysis and 'error' not in setup_analysis:
                    chart_data['setup_analysis'] = setup_analysis
                    chart_data['setup_signals'] = self._format_setup_signals(setup_analysis)
            
            return chart_data
            
        except Exception as e:
            return {'error': f'Failed to get trade setup chart data: {str(e)}'}
    
    def _extract_indicators_from_cached_data(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Extract technical indicators from cached data DataFrame"""
        indicators = {}
        
        try:
            # List of indicators to extract
            indicator_columns = ['SMA_20', 'SMA_50', 'SMA_200', 'RSI', 'BB_UPPER', 'BB_LOWER', 'EMA_13']
            
            for indicator_name in indicator_columns:
                if indicator_name in price_data.columns:
                    # Filter out NaN values
                    indicator_data = price_data[indicator_name].dropna()
                    if not indicator_data.empty:
                        indicators[indicator_name] = {
                            'dates': indicator_data.index.strftime('%Y-%m-%d').tolist(),
                            'values': indicator_data.tolist()
                        }
            
            return indicators
            
        except Exception as e:
            print(f"Error extracting indicators from cached data: {e}")
            return {}

    def _get_technical_indicators(self, symbol: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get technical indicators for the symbol"""
        indicators = {}
        
        try:
            # Get cached indicators
            db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'journal.db')
            
            with sqlite3.connect(db_path) as conn:
                # Get SMA indicators
                sma_query = """
                    SELECT date, indicator_name, value 
                    FROM indicators 
                    WHERE symbol = ? AND date >= ? AND date <= ?
                    AND indicator_name IN ('SMA_20', 'SMA_50', 'SMA_200', 'RSI', 'BB_UPPER', 'BB_LOWER', 'EMA_13')
                    ORDER BY date
                """
                
                sma_df = pd.read_sql_query(sma_query, conn, params=(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))
                
                if not sma_df.empty:
                    sma_df['date'] = pd.to_datetime(sma_df['date'])
                    
                    # Group by indicator name
                    for indicator_name in sma_df['indicator_name'].unique():
                        indicator_data = sma_df[sma_df['indicator_name'] == indicator_name]
                        indicators[indicator_name] = {
                            'dates': indicator_data['date'].dt.strftime('%Y-%m-%d').tolist(),
                            'values': indicator_data['value'].tolist()
                        }
            
            return indicators
            
        except Exception as e:
            print(f"Error getting technical indicators: {e}")
            return {}
    
    def _get_regime_overlay_data(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get market regime data for overlay"""
        try:
            db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'journal.db')
            
            with sqlite3.connect(db_path) as conn:
                regime_query = """
                    SELECT date, volatility_regime, trend_regime, risk_on_off, vix_level
                    FROM market_regimes 
                    WHERE date >= ? AND date <= ?
                    ORDER BY date
                """
                
                regime_df = pd.read_sql_query(regime_query, conn, params=(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))
                
                if not regime_df.empty:
                    regime_df['date'] = pd.to_datetime(regime_df['date'])
                    
                    return {
                        'dates': regime_df['date'].dt.strftime('%Y-%m-%d').tolist(),
                        'volatility_regime': regime_df['volatility_regime'].tolist(),
                        'trend_regime': regime_df['trend_regime'].tolist(),
                        'risk_on_off': regime_df['risk_on_off'].tolist(),
                        'vix_levels': regime_df['vix_level'].tolist()
                    }
            
            return {}
            
        except Exception as e:
            print(f"Error getting regime overlay data: {e}")
            return {}
    
    def _format_price_data(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Format price data for Plotly candlestick chart"""
        price_data = price_data.copy()
        
        return {
            'dates': price_data.index.strftime('%Y-%m-%d').tolist(),
            'open': price_data['Open'].tolist(),
            'high': price_data['High'].tolist(),
            'low': price_data['Low'].tolist(),
            'close': price_data['Close'].tolist()
        }
    
    def _format_volume_data(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Format volume data for chart"""
        price_data = price_data.copy()
        
        return {
            'dates': price_data.index.strftime('%Y-%m-%d').tolist(),
            'volume': price_data['Volume'].tolist()
        }
    
    def _format_setup_signals(self, setup_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Format trade setup signals for chart overlay"""
        signals = {}
        
        try:
            if 'entry_price' in setup_analysis:
                signals['entry_price'] = setup_analysis['entry_price']
            
            if 'target_price' in setup_analysis:
                signals['target_price'] = setup_analysis['target_price']
            
            if 'stop_loss' in setup_analysis:
                signals['stop_loss'] = setup_analysis['stop_loss']
            
            if 'signal' in setup_analysis:
                signals['signal'] = setup_analysis['signal']
                
        except Exception as e:
            print(f"Error formatting setup signals: {e}")
        
        return signals
    
    def get_chart_config(self, chart_type: str = 'price') -> Dict[str, Any]:
        """
        Get Plotly chart configuration for different chart types
        
        Args:
            chart_type: Type of chart ('price', 'volume', 'indicators')
            
        Returns:
            Plotly configuration dictionary
        """
        base_config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
            'responsive': True
        }
        
        if chart_type == 'price':
            return {
                **base_config,
                'scrollZoom': True,
                'doubleClick': 'reset+autosize'
            }
        
        return base_config
    
    def get_chart_layout(self, symbol: str, chart_type: str = 'price') -> Dict[str, Any]:
        """
        Get Plotly layout configuration for different chart types
        
        Args:
            symbol: Stock/ETF symbol for title
            chart_type: Type of chart
            
        Returns:
            Plotly layout dictionary
        """
        base_layout = {
            'template': 'plotly_dark',
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'font': {'color': 'white'},
            'margin': {'l': 50, 'r': 50, 't': 50, 'b': 50},
            'height': 500
        }
        
        if chart_type == 'price':
            return {
                **base_layout,
                'title': f'{symbol} - Price Chart',
                'xaxis': {
                    'title': 'Date',
                    'gridcolor': 'rgba(128,128,128,0.2)',
                    'showgrid': True
                },
                'yaxis': {
                    'title': 'Price ($)',
                    'gridcolor': 'rgba(128,128,128,0.2)',
                    'showgrid': True
                },
                'showlegend': True,
                'legend': {
                    'x': 0.01,
                    'y': 0.99,
                    'bgcolor': 'rgba(0,0,0,0.5)'
                }
            }
        
        return base_layout