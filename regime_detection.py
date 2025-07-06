"""
Market regime detection system for ETF trading strategy.

This module implements detection of various market regimes:
- Volatility regimes (VIX-based)
- Trend regimes (SPY vs 200-day SMA)
- Sector rotation (Growth vs Value)
- Risk-on/Risk-off sentiment
"""

import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import yfinance as yf
import pandas as pd
import numpy as np

from data_cache import DataCache


class VolatilityRegime(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TrendRegime(Enum):
    STRONG_UPTREND = "strong_uptrend"
    MILD_UPTREND = "mild_uptrend"
    RANGING = "ranging"
    DOWNTREND = "downtrend"


class SectorRotation(Enum):
    GROWTH_FAVORED = "growth_favored"
    BALANCED = "balanced"
    VALUE_FAVORED = "value_favored"


class RiskSentiment(Enum):
    RISK_ON = "risk_on"
    NEUTRAL = "neutral"
    RISK_OFF = "risk_off"


@dataclass
class RegimeData:
    date: datetime
    volatility_regime: VolatilityRegime
    trend_regime: TrendRegime
    sector_rotation: SectorRotation
    risk_sentiment: RiskSentiment
    vix_level: float
    spy_vs_sma200: float
    growth_value_ratio: float
    risk_on_off_ratio: float


class RegimeDetector:
    """Detects and stores market regime information."""
    
    def __init__(self, db_path: str = "journal.db"):
        self.db_path = db_path
        self.data_cache = DataCache(db_path)
        self._ensure_tables()
    
    def _ensure_tables(self) -> None:
        """Ensure market_regimes table exists in database."""
        with sqlite3.connect(self.db_path) as conn:
            # Check if table exists and has the columns we need
            cursor = conn.execute("PRAGMA table_info(market_regimes)")
            columns = [row[1] for row in cursor.fetchall()]
            
            if not columns:
                # Create new table with full schema
                conn.execute("""
                    CREATE TABLE market_regimes (
                        date TEXT PRIMARY KEY,
                        volatility_regime TEXT NOT NULL,
                        trend_regime TEXT NOT NULL,
                        sector_rotation TEXT NOT NULL,
                        risk_on_off TEXT NOT NULL,
                        vix_level REAL,
                        spy_vs_sma200 REAL,
                        growth_value_ratio REAL,
                        risk_on_off_ratio REAL,
                        notes TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
            else:
                # Add missing columns to existing table
                if 'vix_level' not in columns:
                    conn.execute("ALTER TABLE market_regimes ADD COLUMN vix_level REAL")
                if 'spy_vs_sma200' not in columns:
                    conn.execute("ALTER TABLE market_regimes ADD COLUMN spy_vs_sma200 REAL")
                if 'growth_value_ratio' not in columns:
                    conn.execute("ALTER TABLE market_regimes ADD COLUMN growth_value_ratio REAL")
                if 'risk_on_off_ratio' not in columns:
                    conn.execute("ALTER TABLE market_regimes ADD COLUMN risk_on_off_ratio REAL")
            
            conn.commit()
    
    def detect_current_regime(self) -> RegimeData:
        """Detect current market regime across all dimensions."""
        today = datetime.now().date()
        
        # Get market data for regime detection
        vix_data = self._get_vix_data()
        spy_data = self._get_spy_data()
        sector_data = self._get_sector_rotation_data()
        risk_data = self._get_risk_sentiment_data()
        
        # Detect each regime component
        volatility_regime, vix_level = self._detect_volatility_regime(vix_data)
        trend_regime, spy_vs_sma = self._detect_trend_regime(spy_data)
        sector_rotation, gv_ratio = self._detect_sector_rotation(sector_data)
        risk_sentiment, risk_ratio = self._detect_risk_sentiment(risk_data)
        
        return RegimeData(
            date=today,
            volatility_regime=volatility_regime,
            trend_regime=trend_regime,
            sector_rotation=sector_rotation,
            risk_sentiment=risk_sentiment,
            vix_level=vix_level,
            spy_vs_sma200=spy_vs_sma,
            growth_value_ratio=gv_ratio,
            risk_on_off_ratio=risk_ratio
        )
    
    def _get_vix_data(self, period: str = "3mo") -> pd.DataFrame:
        """Get VIX data for volatility regime detection."""
        # VIX is not in our cache, use yfinance directly
        vix = yf.Ticker("^VIX")
        return vix.history(period=period)
    
    def _get_spy_data(self, period: str = "1y") -> pd.DataFrame:
        """Get SPY data for trend regime detection."""
        return self.data_cache.get_cached_data("SPY", period)
    
    def _get_sector_rotation_data(self, period: str = "3mo") -> Dict[str, pd.DataFrame]:
        """Get sector ETF data for rotation detection."""
        tickers = ["QQQ", "IWM", "XLK", "XLF"]
        data = {}
        for ticker in tickers:
            data[ticker] = self.data_cache.get_cached_data(ticker, period)
        return data
    
    def _get_risk_sentiment_data(self, period: str = "3mo") -> Dict[str, pd.DataFrame]:
        """Get ETF data for risk-on/risk-off detection."""
        # Defensive: Utilities, Consumer Staples, Treasuries
        # Aggressive: Tech, Growth, Small-cap
        tickers = ["XLU", "XLP", "TLT", "QQQ", "XLF", "IWM"]
        data = {}
        for ticker in tickers:
            data[ticker] = self.data_cache.get_cached_data(ticker, period)
        return data
    
    def _detect_volatility_regime(self, vix_data: pd.DataFrame) -> Tuple[VolatilityRegime, float]:
        """Detect volatility regime based on VIX levels."""
        if vix_data.empty or 'Close' not in vix_data.columns:
            # Default to medium volatility when no data available
            return VolatilityRegime.MEDIUM, 25.0
        
        current_vix = vix_data['Close'].iloc[-1]
        
        if current_vix < 20:
            regime = VolatilityRegime.LOW
        elif current_vix <= 30:
            regime = VolatilityRegime.MEDIUM
        else:
            regime = VolatilityRegime.HIGH
        
        return regime, current_vix
    
    def _detect_trend_regime(self, spy_data: pd.DataFrame) -> Tuple[TrendRegime, float]:
        """Detect trend regime based on SPY vs 200-day SMA."""
        if spy_data.empty or 'Close' not in spy_data.columns:
            # Default to ranging when no data available
            return TrendRegime.RANGING, 1.0
        
        current_price = spy_data['Close'].iloc[-1]
        
        # Use cached SMA200 indicator if available, otherwise calculate on-the-fly
        if 'SMA200' in spy_data.columns and not pd.isna(spy_data['SMA200'].iloc[-1]):
            current_sma200 = spy_data['SMA200'].iloc[-1]
        else:
            # Fallback: calculate on-the-fly if cached indicator unavailable
            if len(spy_data) >= 200:
                current_sma200 = spy_data['Close'].rolling(window=200).mean().iloc[-1]
            else:
                # Insufficient data - return neutral regime with ratio 1.0
                return TrendRegime.RANGING, 1.0
        
        # Handle case where SMA200 calculation failed
        if pd.isna(current_sma200) or current_sma200 == 0:
            return TrendRegime.RANGING, 1.0
            
        ratio = current_price / current_sma200
        
        if ratio > 1.05:
            regime = TrendRegime.STRONG_UPTREND
        elif ratio > 1.00:
            regime = TrendRegime.MILD_UPTREND
        elif ratio > 0.95:
            regime = TrendRegime.RANGING
        else:
            regime = TrendRegime.DOWNTREND
        
        return regime, ratio
    
    def _detect_sector_rotation(self, sector_data: Dict[str, pd.DataFrame]) -> Tuple[SectorRotation, float]:
        """Detect sector rotation based on Growth vs Value performance."""
        # Check if we have the required data
        required_symbols = ["QQQ", "IWM", "XLK", "XLF"]
        for symbol in required_symbols:
            if symbol not in sector_data or sector_data[symbol].empty or 'Close' not in sector_data[symbol].columns:
                # Default to balanced when data is missing
                return SectorRotation.BALANCED, 1.0
        
        # Calculate QQQ/IWM and XLK/XLF ratios
        qqq_iwm_ratio = sector_data["QQQ"]['Close'].iloc[-1] / sector_data["IWM"]['Close'].iloc[-1]
        xlk_xlf_ratio = sector_data["XLK"]['Close'].iloc[-1] / sector_data["XLF"]['Close'].iloc[-1]
        
        # Use 20-day moving average of ratios to smooth signals
        qqq_iwm_ma = (sector_data["QQQ"]['Close'] / sector_data["IWM"]['Close']).rolling(20).mean().iloc[-1]
        xlk_xlf_ma = (sector_data["XLK"]['Close'] / sector_data["XLF"]['Close']).rolling(20).mean().iloc[-1]
        
        # Handle case where moving averages are NaN
        if pd.isna(qqq_iwm_ma) or pd.isna(xlk_xlf_ma):
            return SectorRotation.BALANCED, 1.0
        
        combined_ratio = (qqq_iwm_ratio / qqq_iwm_ma + xlk_xlf_ratio / xlk_xlf_ma) / 2
        
        if combined_ratio > 1.02:
            regime = SectorRotation.GROWTH_FAVORED
        elif combined_ratio < 0.98:
            regime = SectorRotation.VALUE_FAVORED
        else:
            regime = SectorRotation.BALANCED
        
        return regime, combined_ratio
    
    def _detect_risk_sentiment(self, risk_data: Dict[str, pd.DataFrame]) -> Tuple[RiskSentiment, float]:
        """Detect risk sentiment based on defensive vs aggressive ETF performance."""
        # Check if we have the required data
        required_symbols = ["XLU", "XLP", "TLT", "QQQ", "XLF", "IWM"]
        for symbol in required_symbols:
            if symbol not in risk_data or risk_data[symbol].empty or 'Close' not in risk_data[symbol].columns:
                # Default to neutral when data is missing
                return RiskSentiment.NEUTRAL, 1.0
            # Check if we have enough data for 21-day lookback
            if len(risk_data[symbol]) < 21:
                return RiskSentiment.NEUTRAL, 1.0
        
        # Defensive performance (average of XLU, XLP, TLT)
        defensive_perf = (
            risk_data["XLU"]['Close'].iloc[-1] / risk_data["XLU"]['Close'].iloc[-21] +
            risk_data["XLP"]['Close'].iloc[-1] / risk_data["XLP"]['Close'].iloc[-21] +
            risk_data["TLT"]['Close'].iloc[-1] / risk_data["TLT"]['Close'].iloc[-21]
        ) / 3
        
        # Aggressive performance (average of QQQ, XLF, IWM)
        aggressive_perf = (
            risk_data["QQQ"]['Close'].iloc[-1] / risk_data["QQQ"]['Close'].iloc[-21] +
            risk_data["XLF"]['Close'].iloc[-1] / risk_data["XLF"]['Close'].iloc[-21] +
            risk_data["IWM"]['Close'].iloc[-1] / risk_data["IWM"]['Close'].iloc[-21]
        ) / 3
        
        # Handle case where calculations result in NaN or infinite values
        if pd.isna(defensive_perf) or pd.isna(aggressive_perf) or defensive_perf == 0:
            return RiskSentiment.NEUTRAL, 1.0
        
        ratio = aggressive_perf / defensive_perf
        
        if ratio > 1.05:
            regime = RiskSentiment.RISK_ON
        elif ratio < 0.95:
            regime = RiskSentiment.RISK_OFF
        else:
            regime = RiskSentiment.NEUTRAL
        
        return regime, ratio
    
    def store_regime_data(self, regime_data: RegimeData) -> None:
        """Store regime data in SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO market_regimes 
                (date, volatility_regime, trend_regime, sector_rotation, risk_on_off,
                 vix_level, spy_vs_sma200, growth_value_ratio, risk_on_off_ratio)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                regime_data.date.isoformat(),
                regime_data.volatility_regime.value,
                regime_data.trend_regime.value,
                regime_data.sector_rotation.value,
                regime_data.risk_sentiment.value,
                regime_data.vix_level,
                regime_data.spy_vs_sma200,
                regime_data.growth_value_ratio,
                regime_data.risk_on_off_ratio
            ))
            conn.commit()
    
    def get_regime_history(self, days: int = 30) -> List[RegimeData]:
        """Get historical regime data."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT date, volatility_regime, trend_regime, sector_rotation, risk_on_off,
                       vix_level, spy_vs_sma200, growth_value_ratio, risk_on_off_ratio
                FROM market_regimes 
                ORDER BY date DESC 
                LIMIT ?
            """, (days,))
            
            results = []
            for row in cursor.fetchall():
                results.append(RegimeData(
                    date=datetime.fromisoformat(row[0]).date(),
                    volatility_regime=VolatilityRegime(row[1]),
                    trend_regime=TrendRegime(row[2]),
                    sector_rotation=SectorRotation(row[3]),
                    risk_sentiment=RiskSentiment(row[4]),
                    vix_level=row[5],
                    spy_vs_sma200=row[6],
                    growth_value_ratio=row[7],
                    risk_on_off_ratio=row[8]
                ))
            
            return results
    
    def update_daily_regime(self) -> RegimeData:
        """Update daily regime data and store in database."""
        regime_data = self.detect_current_regime()
        self.store_regime_data(regime_data)
        return regime_data


if __name__ == "__main__":
    detector = RegimeDetector()
    current_regime = detector.update_daily_regime()
    
    print(f"Current Market Regime ({current_regime.date}):")
    print(f"  Volatility: {current_regime.volatility_regime.value} (VIX: {current_regime.vix_level:.2f})")
    print(f"  Trend: {current_regime.trend_regime.value} (SPY/SMA200: {current_regime.spy_vs_sma200:.3f})")
    print(f"  Sector Rotation: {current_regime.sector_rotation.value} (Ratio: {current_regime.growth_value_ratio:.3f})")
    print(f"  Risk Sentiment: {current_regime.risk_sentiment.value} (Ratio: {current_regime.risk_on_off_ratio:.3f})")