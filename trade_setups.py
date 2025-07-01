"""
Trade setup implementations for ETF swing trading system.

This module implements the core trade setups:
- Trend Pullback: Pullback entries in trending markets
- Breakout Continuation: Volume-confirmed breakouts
- Oversold Mean Reversion: RSI/Bollinger Band reversals
- Regime Rotation: Sector rotation based on regime changes
"""

import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
import yfinance as yf

from regime_detection import RegimeDetector, RegimeData, VolatilityRegime, TrendRegime
from data_cache import DataCache


class SignalStrength(Enum):
    STRONG = "strong"
    MEDIUM = "medium"
    WEAK = "weak"
    NONE = "none"


class SetupType(Enum):
    TREND_PULLBACK = "trend_pullback"
    BREAKOUT_CONTINUATION = "breakout_continuation"
    OVERSOLD_MEAN_REVERSION = "oversold_mean_reversion"
    REGIME_ROTATION = "regime_rotation"
    GAP_FILL_REVERSAL = "gap_fill_reversal"
    RELATIVE_STRENGTH_MOMENTUM = "relative_strength_momentum"
    VOLATILITY_CONTRACTION = "volatility_contraction"
    DIVIDEND_DISTRIBUTION_PLAY = "dividend_distribution_play"
    ELDER_TRIPLE_SCREEN = "elder_triple_screen"
    INSTITUTIONAL_VOLUME_CLIMAX = "institutional_volume_climax"
    FAILED_BREAKDOWN_REVERSAL = "failed_breakdown_reversal"
    EARNINGS_EXPECTATION_RESET = "earnings_expectation_reset"


@dataclass
class TradeSignal:
    symbol: str
    setup_type: SetupType
    signal_strength: SignalStrength
    entry_price: float
    stop_loss: float
    target_price: float
    position_size: float
    risk_per_share: float
    confidence: float
    regime_context: RegimeData
    notes: str


class BaseSetup(ABC):
    """Base class for all trade setups."""
    
    def __init__(self, db_path: str = "journal.db"):
        self.db_path = db_path
        self.regime_detector = RegimeDetector(db_path)
        self.data_cache = DataCache(db_path)
    
    @abstractmethod
    def scan_for_signals(self, symbols: List[str]) -> List[TradeSignal]:
        """Scan list of symbols for trade signals."""
        pass
    
    @abstractmethod
    def validate_signal(self, symbol: str, regime: RegimeData) -> Tuple[bool, float]:
        """Validate if signal is suitable given current regime."""
        pass
    
    def get_market_data(self, symbol: str, period: str = "6mo") -> pd.DataFrame:
        """Get market data for analysis."""
        return self.data_cache.get_cached_data(symbol, period)
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_atr(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(window=window).mean()
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        
        return upper, lower, sma
    
    def calculate_position_size(self, entry_price: float, stop_loss: float, risk_per_trade: float = 0.02, 
                              account_size: float = 100000) -> float:
        """Calculate position size based on risk management."""
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share == 0:
            return 0
        
        risk_amount = account_size * risk_per_trade
        return risk_amount / risk_per_share


class TrendPullbackSetup(BaseSetup):
    """Trend pullback setup - buy pullbacks in uptrending markets."""
    
    def scan_for_signals(self, symbols: List[str]) -> List[TradeSignal]:
        signals = []
        current_regime = self.regime_detector.detect_current_regime()
        
        # Only trade pullbacks in trending markets
        if current_regime.trend_regime not in [TrendRegime.STRONG_UPTREND, TrendRegime.MILD_UPTREND]:
            return signals
        
        for symbol in symbols:
            try:
                data = self.get_market_data(symbol)
                if len(data) < 200:  # Need enough data for 200-day SMA
                    continue
                
                signal = self._analyze_pullback(symbol, data, current_regime)
                if signal:
                    signals.append(signal)
                    
            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")
        
        return signals
    
    def _analyze_pullback(self, symbol: str, data: pd.DataFrame, regime: RegimeData) -> Optional[TradeSignal]:
        """Analyze for pullback entry opportunity."""
        current_price = data['Close'].iloc[-1]
        sma200 = data['SMA200'].iloc[-1]
        sma20 = data['SMA20'].iloc[-1]
        atr = data['ATR'].iloc[-1]
        
        # Confirm uptrend: price above 200-day SMA
        if current_price < sma200:
            return None
        
        # Look for pullback: current price below 20-day SMA
        if current_price >= sma20:
            return None
        
        # Calculate pullback percentage from recent high
        recent_high = data['High'].rolling(20).max().iloc[-1]
        pullback_pct = (recent_high - current_price) / recent_high
        
        # Look for 3-8% pullback
        if not (0.03 <= pullback_pct <= 0.08):
            return None
        
        # Check for bounce signal (price above yesterday's high)
        if len(data) > 1 and current_price <= data['High'].iloc[-2]:
            return None
        
        # Position sizing and risk management
        stop_loss = current_price - (2 * atr)  # 2 ATR stop
        target_price = current_price + (3 * atr)  # 3:2 reward/risk
        
        # Regime-based confidence adjustment
        confidence = self._calculate_confidence(regime, pullback_pct)
        if confidence < 0.5:
            return None
        
        position_size = self.calculate_position_size(current_price, stop_loss)
        
        return TradeSignal(
            symbol=symbol,
            setup_type=SetupType.TREND_PULLBACK,
            signal_strength=SignalStrength.MEDIUM if confidence > 0.7 else SignalStrength.WEAK,
            entry_price=current_price,
            stop_loss=stop_loss,
            target_price=target_price,
            position_size=position_size,
            risk_per_share=current_price - stop_loss,
            confidence=confidence,
            regime_context=regime,
            notes=f"Pullback {pullback_pct:.1%} from high, trending market"
        )
    
    def _calculate_confidence(self, regime: RegimeData, pullback_pct: float) -> float:
        """Calculate confidence based on regime and pullback characteristics."""
        confidence = 0.6  # Base confidence
        
        # Adjust for trend strength
        if regime.trend_regime == TrendRegime.STRONG_UPTREND:
            confidence += 0.2
        
        # Adjust for volatility
        if regime.volatility_regime == VolatilityRegime.LOW:
            confidence += 0.15
        elif regime.volatility_regime == VolatilityRegime.HIGH:
            confidence -= 0.2
        
        # Adjust for pullback size (sweet spot is 4-6%)
        if 0.04 <= pullback_pct <= 0.06:
            confidence += 0.1
        
        return min(1.0, max(0.0, confidence))
    
    def validate_signal(self, symbol: str, regime: RegimeData) -> Tuple[bool, float]:
        """Validate if pullback signal is suitable for current regime."""
        if regime.trend_regime not in [TrendRegime.STRONG_UPTREND, TrendRegime.MILD_UPTREND]:
            return False, 0.0
        
        confidence = 0.7
        if regime.volatility_regime == VolatilityRegime.HIGH:
            confidence -= 0.3
        
        return confidence > 0.4, confidence


class BreakoutContinuationSetup(BaseSetup):
    """Breakout continuation setup - buy volume-confirmed breakouts."""
    
    def scan_for_signals(self, symbols: List[str]) -> List[TradeSignal]:
        signals = []
        current_regime = self.regime_detector.detect_current_regime()
        
        # Avoid breakouts in high volatility
        if current_regime.volatility_regime == VolatilityRegime.HIGH:
            return signals
        
        for symbol in symbols:
            try:
                data = self.get_market_data(symbol)
                if len(data) < 50:
                    continue
                
                signal = self._analyze_breakout(symbol, data, current_regime)
                if signal:
                    signals.append(signal)
                    
            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")
        
        return signals
    
    def _analyze_breakout(self, symbol: str, data: pd.DataFrame, regime: RegimeData) -> Optional[TradeSignal]:
        """Analyze for breakout continuation opportunity."""
        current_price = data['Close'].iloc[-1]
        current_volume = data['Volume'].iloc[-1]
        
        # Calculate 20-day high and average volume
        high_20d = data['High'].rolling(20).max().iloc[-2]  # Exclude today
        avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
        atr = data['ATR'].iloc[-1]
        
        # Check for breakout above 20-day high
        if current_price <= high_20d:
            return None
        
        # Require volume confirmation (1.5x average)
        volume_ratio = current_volume / avg_volume
        if volume_ratio < 1.5:
            return None
        
        # Calculate breakout strength
        breakout_pct = (current_price - high_20d) / high_20d
        
        # Position sizing and risk management
        stop_loss = high_20d * 0.98  # Just below breakout level
        target_price = current_price + (2 * atr)  # 2 ATR target
        
        # Regime-based confidence
        confidence = self._calculate_breakout_confidence(regime, volume_ratio, breakout_pct)
        if confidence < 0.5:
            return None
        
        position_size = self.calculate_position_size(current_price, stop_loss)
        
        return TradeSignal(
            symbol=symbol,
            setup_type=SetupType.BREAKOUT_CONTINUATION,
            signal_strength=SignalStrength.STRONG if volume_ratio > 2.0 else SignalStrength.MEDIUM,
            entry_price=current_price,
            stop_loss=stop_loss,
            target_price=target_price,
            position_size=position_size,
            risk_per_share=current_price - stop_loss,
            confidence=confidence,
            regime_context=regime,
            notes=f"Breakout with {volume_ratio:.1f}x volume"
        )
    
    def _calculate_breakout_confidence(self, regime: RegimeData, volume_ratio: float, breakout_pct: float) -> float:
        """Calculate confidence for breakout signal."""
        confidence = 0.5  # Base confidence
        
        # Volume confirmation boosts confidence
        if volume_ratio > 2.0:
            confidence += 0.2
        elif volume_ratio > 1.8:
            confidence += 0.1
        
        # Trend regime matters
        if regime.trend_regime == TrendRegime.STRONG_UPTREND:
            confidence += 0.2
        elif regime.trend_regime == TrendRegime.DOWNTREND:
            confidence -= 0.3
        
        # Low volatility is better for breakouts
        if regime.volatility_regime == VolatilityRegime.LOW:
            confidence += 0.1
        
        return min(1.0, max(0.0, confidence))
    
    def validate_signal(self, symbol: str, regime: RegimeData) -> Tuple[bool, float]:
        """Validate if breakout signal is suitable for current regime."""
        if regime.volatility_regime == VolatilityRegime.HIGH:
            return False, 0.0
        
        confidence = 0.6
        if regime.trend_regime == TrendRegime.STRONG_UPTREND:
            confidence += 0.2
        
        return True, confidence


class OversoldMeanReversionSetup(BaseSetup):
    """Oversold mean reversion setup - buy oversold conditions in ranging markets."""
    
    def scan_for_signals(self, symbols: List[str]) -> List[TradeSignal]:
        signals = []
        current_regime = self.regime_detector.detect_current_regime()
        
        # Best in ranging/neutral markets
        for symbol in symbols:
            try:
                data = self.get_market_data(symbol)
                if len(data) < 50:
                    continue
                
                signal = self._analyze_oversold(symbol, data, current_regime)
                if signal:
                    signals.append(signal)
                    
            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")
        
        return signals
    
    def _analyze_oversold(self, symbol: str, data: pd.DataFrame, regime: RegimeData) -> Optional[TradeSignal]:
        """Analyze for oversold mean reversion opportunity."""
        current_price = data['Close'].iloc[-1]
        rsi = data['RSI'].iloc[-1]
        bb_lower = data['BB_Lower'].iloc[-1]
        bb_middle = data['BB_Middle'].iloc[-1]
        atr = data['ATR'].iloc[-1]
        
        # Check oversold conditions
        if rsi > 30:  # RSI not oversold
            return None
        
        if current_price > bb_lower:  # Not below lower Bollinger Band
            return None
        
        # Look for reversal signal (current price > previous low)
        if len(data) > 1 and current_price <= data['Low'].iloc[-2]:
            return None
        
        # Position sizing and risk management
        stop_loss = current_price - (1.5 * atr)  # Tight stop for mean reversion
        target_price = bb_middle  # Target middle Bollinger Band
        
        # Regime-based confidence
        confidence = self._calculate_reversion_confidence(regime, rsi, current_price, bb_lower)
        if confidence < 0.4:
            return None
        
        position_size = self.calculate_position_size(current_price, stop_loss)
        
        return TradeSignal(
            symbol=symbol,
            setup_type=SetupType.OVERSOLD_MEAN_REVERSION,
            signal_strength=SignalStrength.WEAK if rsi > 25 else SignalStrength.MEDIUM,
            entry_price=current_price,
            stop_loss=stop_loss,
            target_price=target_price,
            position_size=position_size,
            risk_per_share=current_price - stop_loss,
            confidence=confidence,
            regime_context=regime,
            notes=f"RSI {rsi:.1f}, below lower BB"
        )
    
    def _calculate_reversion_confidence(self, regime: RegimeData, rsi: float, price: float, bb_lower: float) -> float:
        """Calculate confidence for mean reversion signal."""
        confidence = 0.4  # Base confidence (mean reversion is harder)
        
        # More oversold = higher confidence
        if rsi < 20:
            confidence += 0.2
        elif rsi < 25:
            confidence += 0.1
        
        # Distance below BB matters
        bb_distance = (bb_lower - price) / price
        if bb_distance > 0.02:  # More than 2% below
            confidence += 0.15
        
        # Better in ranging markets
        if regime.trend_regime == TrendRegime.RANGING:
            confidence += 0.2
        elif regime.trend_regime == TrendRegime.DOWNTREND:
            confidence -= 0.3
        
        return min(1.0, max(0.0, confidence))
    
    def validate_signal(self, symbol: str, regime: RegimeData) -> Tuple[bool, float]:
        """Validate if mean reversion signal is suitable for current regime."""
        confidence = 0.5
        
        # Prefer ranging markets
        if regime.trend_regime == TrendRegime.RANGING:
            confidence += 0.2
        elif regime.trend_regime == TrendRegime.DOWNTREND:
            confidence -= 0.3
        
        return confidence > 0.3, confidence


class RegimeRotationSetup(BaseSetup):
    """Regime rotation setup - sector rotation based on regime changes."""
    
    def scan_for_signals(self, symbols: List[str]) -> List[TradeSignal]:
        signals = []
        current_regime = self.regime_detector.detect_current_regime()
        
        # Get recent regime history to detect changes
        regime_history = self.regime_detector.get_regime_history(days=5)
        if len(regime_history) < 2:
            return signals
        
        regime_changed = self._detect_regime_change(regime_history)
        if not regime_changed:
            return signals
        
        # Get sector preferences for current regime
        preferred_sectors = self._get_preferred_sectors(current_regime)
        
        for symbol in symbols:
            try:
                sector = self._get_symbol_sector(symbol)
                if sector in preferred_sectors:
                    signal = self._analyze_rotation_entry(symbol, current_regime, sector)
                    if signal:
                        signals.append(signal)
                        
            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")
        
        return signals
    
    def _detect_regime_change(self, regime_history: List[RegimeData]) -> bool:
        """Detect if there's been a significant regime change."""
        if len(regime_history) < 2:
            return False
        
        current = regime_history[0]
        previous = regime_history[1]
        
        # Check for volatility regime change
        if current.volatility_regime != previous.volatility_regime:
            return True
        
        # Check for trend regime change
        if current.trend_regime != previous.trend_regime:
            return True
        
        # Check for sector rotation change
        if current.sector_rotation != previous.sector_rotation:
            return True
        
        return False
    
    def _get_preferred_sectors(self, regime: RegimeData) -> List[str]:
        """Get preferred sectors for current regime."""
        sectors = []
        
        # Based on volatility regime
        if regime.volatility_regime == VolatilityRegime.LOW:
            sectors.extend(["Technology", "Consumer Discretionary", "Growth"])
        elif regime.volatility_regime == VolatilityRegime.HIGH:
            sectors.extend(["Consumer Staples", "Utilities", "Healthcare"])
        
        # Based on trend regime
        if regime.trend_regime in [TrendRegime.STRONG_UPTREND, TrendRegime.MILD_UPTREND]:
            sectors.extend(["Technology", "Financial"])
        elif regime.trend_regime == TrendRegime.DOWNTREND:
            sectors.extend(["Fixed Income", "Commodities", "Utilities"])
        
        return list(set(sectors))  # Remove duplicates
    
    def _get_symbol_sector(self, symbol: str) -> str:
        """Get sector for a symbol from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT sector, theme FROM instruments WHERE symbol = ?", (symbol,))
            result = cursor.fetchone()
            
            if result and result[0]:
                return result[0]
            elif result and result[1]:
                return result[1]
            
            return "Unknown"
    
    def _analyze_rotation_entry(self, symbol: str, regime: RegimeData, sector: str) -> Optional[TradeSignal]:
        """Analyze entry for regime rotation play."""
        data = self.get_market_data(symbol, period="3mo")
        if len(data) < 20:
            return None
        
        current_price = data['Close'].iloc[-1]
        sma20 = data['SMA20'].iloc[-1]
        atr = data['ATR'].iloc[-1]
        
        # Simple momentum filter - price above 20-day SMA
        if current_price < sma20:
            return None
        
        # Position sizing and risk management
        stop_loss = current_price - (2 * atr)
        target_price = current_price + (2 * atr)
        
        confidence = 0.6  # Base confidence for rotation plays
        position_size = self.calculate_position_size(current_price, stop_loss, risk_per_trade=0.015)  # Smaller risk
        
        return TradeSignal(
            symbol=symbol,
            setup_type=SetupType.REGIME_ROTATION,
            signal_strength=SignalStrength.MEDIUM,
            entry_price=current_price,
            stop_loss=stop_loss,
            target_price=target_price,
            position_size=position_size,
            risk_per_share=current_price - stop_loss,
            confidence=confidence,
            regime_context=regime,
            notes=f"Regime rotation to {sector}"
        )
    
    def validate_signal(self, symbol: str, regime: RegimeData) -> Tuple[bool, float]:
        """Validate if rotation signal is suitable."""
        return True, 0.6  # Regime rotation signals are generally valid


class GapFillReversalSetup(BaseSetup):
    """Gap fill reversal setup - trade ETFs that gap down with reversal signals."""
    
    def scan_for_signals(self, symbols: List[str]) -> List[TradeSignal]:
        signals = []
        current_regime = self.regime_detector.detect_current_regime()
        
        for symbol in symbols:
            try:
                data = self.get_market_data(symbol, period="3mo")
                if len(data) < 20:
                    continue
                
                signal = self._analyze_gap_reversal(symbol, data, current_regime)
                if signal:
                    signals.append(signal)
                    
            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")
        
        return signals
    
    def _analyze_gap_reversal(self, symbol: str, data: pd.DataFrame, regime: RegimeData) -> Optional[TradeSignal]:
        """Analyze for gap fill reversal opportunity."""
        if len(data) < 2:
            return None
            
        current_price = data['Close'].iloc[-1]
        current_volume = data['Volume'].iloc[-1]
        prev_close = data['Close'].iloc[-2]
        today_low = data['Low'].iloc[-1]
        today_high = data['High'].iloc[-1]
        
        # Calculate gap percentage
        gap_pct = (current_price - prev_close) / prev_close
        
        # Look for gap down of at least 2%
        if gap_pct > -0.02:
            return None
        
        # Check for reversal signal - price recovered from low
        recovery_pct = (current_price - today_low) / today_low
        if recovery_pct < 0.01:  # Need at least 1% recovery from low
            return None
        
        # Volume confirmation - above average
        avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
        volume_ratio = current_volume / avg_volume
        if volume_ratio < 1.2:
            return None
        
        # Calculate technical levels
        atr = data['ATR'].iloc[-1]
        gap_fill_level = prev_close
        
        # Position sizing and risk management
        stop_loss = today_low * 0.98  # Just below today's low
        target_price = min(gap_fill_level, current_price + (2 * atr))  # Gap fill or 2 ATR
        
        # Confidence calculation
        confidence = self._calculate_gap_confidence(regime, abs(gap_pct), recovery_pct, volume_ratio)
        if confidence < 0.5:
            return None
        
        position_size = self.calculate_position_size(current_price, stop_loss)
        
        return TradeSignal(
            symbol=symbol,
            setup_type=SetupType.GAP_FILL_REVERSAL,
            signal_strength=SignalStrength.STRONG if abs(gap_pct) > 0.03 else SignalStrength.MEDIUM,
            entry_price=current_price,
            stop_loss=stop_loss,
            target_price=target_price,
            position_size=position_size,
            risk_per_share=current_price - stop_loss,
            confidence=confidence,
            regime_context=regime,
            notes=f"Gap down {gap_pct:.1%}, recovered {recovery_pct:.1%} from low"
        )
    
    def _calculate_gap_confidence(self, regime: RegimeData, gap_pct: float, recovery_pct: float, volume_ratio: float) -> float:
        """Calculate confidence for gap reversal signal."""
        confidence = 0.5  # Base confidence
        
        # Larger gaps have better reversal potential
        if gap_pct > 0.04:
            confidence += 0.2
        elif gap_pct > 0.03:
            confidence += 0.1
        
        # Strong recovery boosts confidence
        if recovery_pct > 0.03:
            confidence += 0.15
        elif recovery_pct > 0.02:
            confidence += 0.1
        
        # Volume confirmation
        if volume_ratio > 2.0:
            confidence += 0.15
        elif volume_ratio > 1.5:
            confidence += 0.1
        
        # Better in stable regimes
        if regime.volatility_regime == VolatilityRegime.LOW:
            confidence += 0.1
        elif regime.volatility_regime == VolatilityRegime.HIGH:
            confidence -= 0.1
        
        return min(1.0, max(0.0, confidence))
    
    def validate_signal(self, symbol: str, regime: RegimeData) -> Tuple[bool, float]:
        """Validate if gap reversal signal is suitable for current regime."""
        confidence = 0.6
        
        if regime.volatility_regime == VolatilityRegime.HIGH:
            confidence -= 0.2
        
        return True, confidence


class RelativeStrengthMomentumSetup(BaseSetup):
    """Relative strength momentum setup - buy ETFs outperforming SPY during weakness."""
    
    def scan_for_signals(self, symbols: List[str]) -> List[TradeSignal]:
        signals = []
        current_regime = self.regime_detector.detect_current_regime()
        
        # Get SPY data for comparison
        spy_data = self.get_market_data("SPY", period="3mo")
        if len(spy_data) < 20:
            return signals
        
        for symbol in symbols:
            if symbol == "SPY":  # Skip SPY itself
                continue
                
            try:
                data = self.get_market_data(symbol, period="3mo")
                if len(data) < 20:
                    continue
                
                signal = self._analyze_relative_strength(symbol, data, spy_data, current_regime)
                if signal:
                    signals.append(signal)
                    
            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")
        
        return signals
    
    def _analyze_relative_strength(self, symbol: str, data: pd.DataFrame, spy_data: pd.DataFrame, regime: RegimeData) -> Optional[TradeSignal]:
        """Analyze for relative strength momentum opportunity."""
        # Calculate recent performance vs SPY
        lookback_days = 10
        
        if len(data) < lookback_days or len(spy_data) < lookback_days:
            return None
        
        symbol_return = (data['Close'].iloc[-1] / data['Close'].iloc[-lookback_days] - 1)
        spy_return = (spy_data['Close'].iloc[-1] / spy_data['Close'].iloc[-lookback_days] - 1)
        
        relative_performance = symbol_return - spy_return
        
        # Look for strong outperformance (>3%) while SPY is weak/flat
        if relative_performance < 0.03:
            return None
        
        # SPY should be weak or flat
        if spy_return > 0.02:
            return None
        
        current_price = data['Close'].iloc[-1]
        sma20 = data['SMA20'].iloc[-1]
        atr = data['ATR'].iloc[-1]
        
        # Price should be above 20-day SMA (momentum confirmation)
        if current_price < sma20:
            return None
        
        # Position sizing and risk management
        stop_loss = current_price - (2 * atr)
        target_price = current_price + (2.5 * atr)  # Slightly higher target for momentum
        
        # Confidence calculation
        confidence = self._calculate_strength_confidence(regime, relative_performance, symbol_return)
        if confidence < 0.5:
            return None
        
        position_size = self.calculate_position_size(current_price, stop_loss)
        
        return TradeSignal(
            symbol=symbol,
            setup_type=SetupType.RELATIVE_STRENGTH_MOMENTUM,
            signal_strength=SignalStrength.STRONG if relative_performance > 0.05 else SignalStrength.MEDIUM,
            entry_price=current_price,
            stop_loss=stop_loss,
            target_price=target_price,
            position_size=position_size,
            risk_per_share=current_price - stop_loss,
            confidence=confidence,
            regime_context=regime,
            notes=f"Outperforming SPY by {relative_performance:.1%} over {lookback_days} days"
        )
    
    def _calculate_strength_confidence(self, regime: RegimeData, relative_perf: float, absolute_return: float) -> float:
        """Calculate confidence for relative strength signal."""
        confidence = 0.5  # Base confidence
        
        # Stronger relative performance = higher confidence
        if relative_perf > 0.06:
            confidence += 0.2
        elif relative_perf > 0.04:
            confidence += 0.1
        
        # Positive absolute return while market is weak
        if absolute_return > 0.03:
            confidence += 0.15
        elif absolute_return > 0.01:
            confidence += 0.1
        
        # Works well in trending markets
        if regime.trend_regime in [TrendRegime.STRONG_UPTREND, TrendRegime.MILD_UPTREND]:
            confidence += 0.1
        elif regime.trend_regime == TrendRegime.DOWNTREND:
            confidence += 0.2  # Even better when market is down
        
        return min(1.0, max(0.0, confidence))
    
    def validate_signal(self, symbol: str, regime: RegimeData) -> Tuple[bool, float]:
        """Validate if relative strength signal is suitable for current regime."""
        confidence = 0.6
        
        # Works across most regimes
        if regime.trend_regime == TrendRegime.DOWNTREND:
            confidence += 0.2
        
        return True, confidence


class VolatilityContractionSetup(BaseSetup):
    """Volatility contraction setup - trade after periods of low volatility before expansion."""
    
    def scan_for_signals(self, symbols: List[str]) -> List[TradeSignal]:
        signals = []
        current_regime = self.regime_detector.detect_current_regime()
        
        for symbol in symbols:
            try:
                data = self.get_market_data(symbol, period="6mo")
                if len(data) < 50:
                    continue
                
                signal = self._analyze_volatility_contraction(symbol, data, current_regime)
                if signal:
                    signals.append(signal)
                    
            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")
        
        return signals
    
    def _analyze_volatility_contraction(self, symbol: str, data: pd.DataFrame, regime: RegimeData) -> Optional[TradeSignal]:
        """Analyze for volatility contraction setup."""
        current_atr = data['ATR'].iloc[-1]
        atr_20_avg = data['ATR'].rolling(20).mean().iloc[-1]
        
        # ATR should be significantly below 20-day average (contraction)
        atr_ratio = current_atr / atr_20_avg
        if atr_ratio > 0.8:  # ATR not contracted enough
            return None
        
        current_price = data['Close'].iloc[-1]
        sma20 = data['SMA20'].iloc[-1]
        sma50 = data['SMA50'].iloc[-1]
        
        # Price should be coiling near moving averages
        sma_distance = abs(current_price - sma20) / current_price
        if sma_distance > 0.02:  # Too far from SMA20
            return None
        
        # Look for recent price compression (narrow range)
        recent_range = data['High'].rolling(5).max().iloc[-1] - data['Low'].rolling(5).min().iloc[-1]
        range_pct = recent_range / current_price
        
        if range_pct > 0.03:  # Range too wide
            return None
        
        # Determine direction bias based on SMA alignment
        direction_bullish = current_price > sma50 and sma20 > sma50
        
        # Position sizing and risk management
        if direction_bullish:
            stop_loss = current_price - (1.5 * current_atr)  # Tighter stop due to low volatility
            target_price = current_price + (3 * current_atr)  # Expect volatility expansion
        else:
            return None  # Only trade bullish setups for now
        
        # Confidence calculation
        confidence = self._calculate_contraction_confidence(regime, atr_ratio, range_pct)
        if confidence < 0.5:
            return None
        
        position_size = self.calculate_position_size(current_price, stop_loss)
        
        return TradeSignal(
            symbol=symbol,
            setup_type=SetupType.VOLATILITY_CONTRACTION,
            signal_strength=SignalStrength.MEDIUM if atr_ratio < 0.7 else SignalStrength.WEAK,
            entry_price=current_price,
            stop_loss=stop_loss,
            target_price=target_price,
            position_size=position_size,
            risk_per_share=current_price - stop_loss,
            confidence=confidence,
            regime_context=regime,
            notes=f"ATR contracted to {atr_ratio:.1%} of 20-day avg, range {range_pct:.1%}"
        )
    
    def _calculate_contraction_confidence(self, regime: RegimeData, atr_ratio: float, range_pct: float) -> float:
        """Calculate confidence for volatility contraction signal."""
        confidence = 0.5  # Base confidence
        
        # Lower ATR ratio = higher confidence
        if atr_ratio < 0.6:
            confidence += 0.2
        elif atr_ratio < 0.7:
            confidence += 0.1
        
        # Tighter range = higher confidence
        if range_pct < 0.02:
            confidence += 0.15
        elif range_pct < 0.025:
            confidence += 0.1
        
        # Better after low volatility periods
        if regime.volatility_regime == VolatilityRegime.LOW:
            confidence += 0.1
        
        return min(1.0, max(0.0, confidence))
    
    def validate_signal(self, symbol: str, regime: RegimeData) -> Tuple[bool, float]:
        """Validate if volatility contraction signal is suitable for current regime."""
        confidence = 0.6
        
        # Works better in low volatility regimes
        if regime.volatility_regime == VolatilityRegime.LOW:
            confidence += 0.2
        elif regime.volatility_regime == VolatilityRegime.HIGH:
            confidence -= 0.3
        
        return confidence > 0.4, confidence


class DividendDistributionPlaySetup(BaseSetup):
    """Dividend/distribution play setup - combine technical setups with ex-dividend timing."""
    
    def scan_for_signals(self, symbols: List[str]) -> List[TradeSignal]:
        signals = []
        current_regime = self.regime_detector.detect_current_regime()
        
        # This setup works better in stable regimes
        if current_regime.volatility_regime == VolatilityRegime.HIGH:
            return signals
        
        for symbol in symbols:
            try:
                data = self.get_market_data(symbol, period="3mo")
                if len(data) < 20:
                    continue
                
                signal = self._analyze_dividend_play(symbol, data, current_regime)
                if signal:
                    signals.append(signal)
                    
            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")
        
        return signals
    
    def _analyze_dividend_play(self, symbol: str, data: pd.DataFrame, regime: RegimeData) -> Optional[TradeSignal]:
        """Analyze for dividend/distribution play opportunity."""
        # Note: This is a simplified implementation
        # In practice, you'd want to integrate with a dividend calendar API
        
        current_price = data['Close'].iloc[-1]
        sma20 = data['SMA20'].iloc[-1]
        sma50 = data['SMA50'].iloc[-1]
        atr = data['ATR'].iloc[-1]
        
        # Basic technical requirements
        # Price above both SMAs (uptrend)
        if not (current_price > sma20 > sma50):
            return None
        
        # Low volatility preferred for dividend plays
        current_atr = data['ATR'].iloc[-1]
        atr_20_avg = data['ATR'].rolling(20).mean().iloc[-1]
        if current_atr > atr_20_avg * 1.2:  # High volatility
            return None
        
        # Check if ETF typically pays dividends (simplified check)
        sector = self._get_symbol_sector(symbol)
        dividend_sectors = ["Utilities", "Consumer Staples", "Real Estate", "Fixed Income"]
        
        if sector not in dividend_sectors:
            return None
        
        # Position sizing and risk management
        stop_loss = current_price - (1.5 * atr)  # Tighter stop for dividend plays
        target_price = current_price + (1 * atr)  # Conservative target
        
        # Confidence calculation
        confidence = self._calculate_dividend_confidence(regime, sector)
        if confidence < 0.4:
            return None
        
        position_size = self.calculate_position_size(current_price, stop_loss, risk_per_trade=0.015)  # Lower risk
        
        return TradeSignal(
            symbol=symbol,
            setup_type=SetupType.DIVIDEND_DISTRIBUTION_PLAY,
            signal_strength=SignalStrength.WEAK,  # Conservative approach
            entry_price=current_price,
            stop_loss=stop_loss,
            target_price=target_price,
            position_size=position_size,
            risk_per_share=current_price - stop_loss,
            confidence=confidence,
            regime_context=regime,
            notes=f"Dividend play in {sector} sector"
        )
    
    def _get_symbol_sector(self, symbol: str) -> str:
        """Get sector for a symbol from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT sector, theme FROM instruments WHERE symbol = ?", (symbol,))
            result = cursor.fetchone()
            
            if result and result[0]:
                return result[0]
            elif result and result[1]:
                return result[1]
            
            return "Unknown"
    
    def _calculate_dividend_confidence(self, regime: RegimeData, sector: str) -> float:
        """Calculate confidence for dividend play signal."""
        confidence = 0.4  # Base confidence (conservative)
        
        # Better in stable regimes
        if regime.volatility_regime == VolatilityRegime.LOW:
            confidence += 0.2
        elif regime.volatility_regime == VolatilityRegime.MEDIUM:
            confidence += 0.1
        
        # Better in certain market conditions
        if regime.trend_regime in [TrendRegime.MILD_UPTREND, TrendRegime.RANGING]:
            confidence += 0.15
        
        # Sector preferences during defensive periods
        defensive_sectors = ["Utilities", "Consumer Staples"]
        if sector in defensive_sectors and regime.volatility_regime != VolatilityRegime.LOW:
            confidence += 0.1
        
        return min(1.0, max(0.0, confidence))
    
    def validate_signal(self, symbol: str, regime: RegimeData) -> Tuple[bool, float]:
        """Validate if dividend play signal is suitable for current regime."""
        confidence = 0.5
        
        # Avoid high volatility periods
        if regime.volatility_regime == VolatilityRegime.HIGH:
            return False, 0.0
        
        # Better in stable conditions
        if regime.volatility_regime == VolatilityRegime.LOW:
            confidence += 0.2
        
        return True, confidence


class ElderTripleScreenSetup(BaseSetup):
    """Elder's Triple Screen setup - multi-timeframe trend following with precise entry timing."""
    
    def scan_for_signals(self, symbols: List[str]) -> List[TradeSignal]:
        signals = []
        current_regime = self.regime_detector.detect_current_regime()
        
        for symbol in symbols:
            try:
                # Get longer period for weekly analysis
                data = self.get_market_data(symbol, period="1y")
                if len(data) < 100:
                    continue
                
                signal = self._analyze_triple_screen(symbol, data, current_regime)
                if signal:
                    signals.append(signal)
                    
            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")
        
        return signals
    
    def _analyze_triple_screen(self, symbol: str, data: pd.DataFrame, regime: RegimeData) -> Optional[TradeSignal]:
        """Analyze using Elder's Triple Screen methodology."""
        # Screen 1: Weekly trend filter (using 13-week EMA approximation)
        weekly_ema = data['Close'].ewm(span=65).mean()  # 13 weeks * 5 days
        current_price = data['Close'].iloc[-1]
        
        # Only trade long positions when above weekly EMA
        if current_price < weekly_ema.iloc[-1]:
            return None
        
        # Screen 2: Daily oscillator timing (RSI oversold)
        rsi = data['RSI'].iloc[-1]
        stoch_k = self._calculate_stochastic(data)
        
        # Look for oversold conditions
        if not (rsi < 30 or stoch_k < 20):
            return None
        
        # Screen 3: Intraday entry trigger (break above previous day's high)
        if len(data) < 2:
            return None
            
        prev_high = data['High'].iloc[-2]
        current_volume = data['Volume'].iloc[-1]
        avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
        
        # Entry condition: current price above yesterday's high with volume
        if current_price <= prev_high:
            return None
        
        # Volume confirmation (1.5x+ average)
        volume_ratio = current_volume / avg_volume
        if volume_ratio < 1.5:
            return None
        
        # Position sizing and risk management
        atr = data['ATR'].iloc[-1]
        recent_swing_low = data['Low'].rolling(10).min().iloc[-1]
        stop_loss = max(recent_swing_low, current_price - (2 * atr))
        target_price = current_price + (3 * atr)  # 3:2 reward/risk
        
        # Confidence calculation
        confidence = self._calculate_triple_screen_confidence(regime, rsi, stoch_k, volume_ratio)
        if confidence < 0.5:
            return None
        
        position_size = self.calculate_position_size(current_price, stop_loss)
        
        return TradeSignal(
            symbol=symbol,
            setup_type=SetupType.ELDER_TRIPLE_SCREEN,
            signal_strength=SignalStrength.STRONG if volume_ratio > 2.0 else SignalStrength.MEDIUM,
            entry_price=current_price,
            stop_loss=stop_loss,
            target_price=target_price,
            position_size=position_size,
            risk_per_share=current_price - stop_loss,
            confidence=confidence,
            regime_context=regime,
            notes=f"Triple screen: weekly uptrend, RSI {rsi:.1f}, {volume_ratio:.1f}x volume"
        )
    
    def _calculate_stochastic(self, data: pd.DataFrame, k_period: int = 14) -> float:
        """Calculate %K Stochastic oscillator."""
        if len(data) < k_period:
            return 50.0
        
        recent_data = data.tail(k_period)
        highest_high = recent_data['High'].max()
        lowest_low = recent_data['Low'].min()
        current_close = data['Close'].iloc[-1]
        
        if highest_high == lowest_low:
            return 50.0
        
        return 100 * (current_close - lowest_low) / (highest_high - lowest_low)
    
    def _calculate_triple_screen_confidence(self, regime: RegimeData, rsi: float, stoch_k: float, volume_ratio: float) -> float:
        """Calculate confidence for triple screen signal."""
        confidence = 0.6  # Base confidence
        
        # More oversold = higher confidence
        if rsi < 25 or stoch_k < 15:
            confidence += 0.2
        elif rsi < 30 or stoch_k < 20:
            confidence += 0.1
        
        # Volume confirmation
        if volume_ratio > 2.5:
            confidence += 0.2
        elif volume_ratio > 2.0:
            confidence += 0.15
        elif volume_ratio > 1.8:
            confidence += 0.1
        
        # Trending regimes are better
        if regime.trend_regime == TrendRegime.STRONG_UPTREND:
            confidence += 0.15
        elif regime.trend_regime == TrendRegime.MILD_UPTREND:
            confidence += 0.1
        
        return min(1.0, max(0.0, confidence))
    
    def validate_signal(self, symbol: str, regime: RegimeData) -> Tuple[bool, float]:
        """Validate if triple screen signal is suitable for current regime."""
        confidence = 0.7
        
        # Works best in trending markets
        if regime.trend_regime in [TrendRegime.STRONG_UPTREND, TrendRegime.MILD_UPTREND]:
            confidence += 0.2
        elif regime.trend_regime == TrendRegime.DOWNTREND:
            confidence -= 0.4
        
        return confidence > 0.5, confidence


class InstitutionalVolumeClimaxSetup(BaseSetup):
    """Institutional volume climax setup - detect accumulation during retail panic selling."""
    
    def scan_for_signals(self, symbols: List[str]) -> List[TradeSignal]:
        signals = []
        current_regime = self.regime_detector.detect_current_regime()
        
        for symbol in symbols:
            try:
                data = self.get_market_data(symbol, period="6mo")
                if len(data) < 50:
                    continue
                
                signal = self._analyze_volume_climax(symbol, data, current_regime)
                if signal:
                    signals.append(signal)
                    
            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")
        
        return signals
    
    def _analyze_volume_climax(self, symbol: str, data: pd.DataFrame, regime: RegimeData) -> Optional[TradeSignal]:
        """Analyze for institutional volume climax opportunity."""
        if len(data) < 5:
            return None
        
        current_price = data['Close'].iloc[-1]
        current_volume = data['Volume'].iloc[-1]
        avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
        
        # Check for recent high volume selling (2-3 days)
        recent_volumes = data['Volume'].tail(3)
        recent_closes = data['Close'].tail(3)
        volume_ratios = recent_volumes / avg_volume
        
        # Look for 2+ days of high volume (3x+ average)
        high_volume_days = sum(1 for ratio in volume_ratios if ratio >= 3.0)
        if high_volume_days < 2:
            return None
        
        # Check for price decline during high volume
        price_decline = (recent_closes.iloc[-1] - recent_closes.iloc[0]) / recent_closes.iloc[0]
        if price_decline > -0.02:  # Need at least 2% decline
            return None
        
        # Check for support level holding
        sma50 = data['SMA50'].iloc[-1]
        prev_low = data['Low'].rolling(20).min().iloc[-2]  # Exclude today
        support_level = max(sma50, prev_low)
        
        # Price should hold above support despite selling pressure
        if current_price < support_level * 0.98:  # 2% buffer
            return None
        
        # Look for absorption signal (volume drying up, price bouncing)
        if current_volume > avg_volume * 1.5:  # Volume should be below average
            return None
        
        # Price should close higher than previous day
        if len(data) > 1 and current_price <= data['Close'].iloc[-2]:
            return None
        
        # Position sizing and risk management
        atr = data['ATR'].iloc[-1]
        stop_loss = support_level * 0.96  # Below support level
        target_price = current_price + (2.5 * atr)  # Conservative target
        
        # Confidence calculation
        max_volume_ratio = max(volume_ratios)
        confidence = self._calculate_climax_confidence(regime, high_volume_days, max_volume_ratio, price_decline)
        if confidence < 0.5:
            return None
        
        position_size = self.calculate_position_size(current_price, stop_loss)
        
        return TradeSignal(
            symbol=symbol,
            setup_type=SetupType.INSTITUTIONAL_VOLUME_CLIMAX,
            signal_strength=SignalStrength.STRONG if max_volume_ratio > 4.0 else SignalStrength.MEDIUM,
            entry_price=current_price,
            stop_loss=stop_loss,
            target_price=target_price,
            position_size=position_size,
            risk_per_share=current_price - stop_loss,
            confidence=confidence,
            regime_context=regime,
            notes=f"Volume climax: {high_volume_days} high vol days, max {max_volume_ratio:.1f}x, decline {price_decline:.1%}"
        )
    
    def _calculate_climax_confidence(self, regime: RegimeData, high_vol_days: int, max_vol_ratio: float, price_decline: float) -> float:
        """Calculate confidence for volume climax signal."""
        confidence = 0.5  # Base confidence
        
        # More high volume days = higher confidence
        if high_vol_days >= 3:
            confidence += 0.2
        
        # Higher volume spikes = higher confidence
        if max_vol_ratio > 5.0:
            confidence += 0.2
        elif max_vol_ratio > 4.0:
            confidence += 0.15
        elif max_vol_ratio > 3.5:
            confidence += 0.1
        
        # Larger decline = better reversal potential
        abs_decline = abs(price_decline)
        if abs_decline > 0.05:
            confidence += 0.15
        elif abs_decline > 0.03:
            confidence += 0.1
        
        # Works better in volatile markets
        if regime.volatility_regime == VolatilityRegime.HIGH:
            confidence += 0.1
        
        return min(1.0, max(0.0, confidence))
    
    def validate_signal(self, symbol: str, regime: RegimeData) -> Tuple[bool, float]:
        """Validate if volume climax signal is suitable for current regime."""
        confidence = 0.6
        
        # Works across most regimes, but better in volatile periods
        if regime.volatility_regime == VolatilityRegime.HIGH:
            confidence += 0.2
        
        return True, confidence


class FailedBreakdownReversalSetup(BaseSetup):
    """Failed breakdown reversal setup - capitalize on bear traps and quick reversals."""
    
    def scan_for_signals(self, symbols: List[str]) -> List[TradeSignal]:
        signals = []
        current_regime = self.regime_detector.detect_current_regime()
        
        for symbol in symbols:
            try:
                data = self.get_market_data(symbol, period="6mo")
                if len(data) < 50:
                    continue
                
                signal = self._analyze_failed_breakdown(symbol, data, current_regime)
                if signal:
                    signals.append(signal)
                    
            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")
        
        return signals
    
    def _analyze_failed_breakdown(self, symbol: str, data: pd.DataFrame, regime: RegimeData) -> Optional[TradeSignal]:
        """Analyze for failed breakdown reversal opportunity."""
        if len(data) < 10:
            return None
        
        current_price = data['Close'].iloc[-1]
        current_volume = data['Volume'].iloc[-1]
        sma50 = data['SMA50'].iloc[-1]
        
        # Identify key support levels
        support_levels = []
        support_levels.append(sma50)
        
        # Add trend line support (simplified)
        recent_lows = data['Low'].rolling(20).min()
        support_levels.append(recent_lows.iloc[-1])
        
        # Previous significant low
        prev_low = data['Low'].rolling(50).min().iloc[-10]  # Exclude recent data
        support_levels.append(prev_low)
        
        # Find the most relevant support level
        key_support = max([s for s in support_levels if s < current_price * 1.02])
        
        # Look for recent breakdown within last 5 days
        breakdown_found = False
        breakdown_low = None
        breakdown_volume = None
        
        for i in range(1, min(6, len(data))):
            past_close = data['Close'].iloc[-i]
            past_low = data['Low'].iloc[-i]
            past_volume = data['Volume'].iloc[-i]
            
            # Check if price broke below support on volume
            if past_low < key_support and past_volume > data['Volume'].rolling(20).mean().iloc[-i] * 1.3:
                breakdown_found = True
                breakdown_low = past_low
                breakdown_volume = past_volume
                break
        
        if not breakdown_found:
            return None
        
        # Check for reversal - price back above breakdown level
        if current_price <= key_support:
            return None
        
        # Volume confirmation on reversal day
        avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
        volume_ratio = current_volume / avg_volume
        if volume_ratio < 1.2:
            return None
        
        # Calculate time since breakdown (should be recent)
        days_since_breakdown = 1
        for i in range(1, min(6, len(data))):
            if data['Low'].iloc[-i] == breakdown_low:
                days_since_breakdown = i
                break
        
        if days_since_breakdown > 3:  # Too long ago
            return None
        
        # Position sizing and risk management
        atr = data['ATR'].iloc[-1]
        stop_loss = breakdown_low * 0.98  # Below failed breakdown low
        
        # Target previous resistance or 2.5 ATR
        recent_high = data['High'].rolling(20).max().iloc[-1]
        target_price = min(recent_high, current_price + (2.5 * atr))
        
        # Confidence calculation
        confidence = self._calculate_breakdown_confidence(regime, volume_ratio, days_since_breakdown, current_price, key_support)
        if confidence < 0.5:
            return None
        
        position_size = self.calculate_position_size(current_price, stop_loss)
        
        return TradeSignal(
            symbol=symbol,
            setup_type=SetupType.FAILED_BREAKDOWN_REVERSAL,
            signal_strength=SignalStrength.STRONG if volume_ratio > 2.0 else SignalStrength.MEDIUM,
            entry_price=current_price,
            stop_loss=stop_loss,
            target_price=target_price,
            position_size=position_size,
            risk_per_share=current_price - stop_loss,
            confidence=confidence,
            regime_context=regime,
            notes=f"Failed breakdown reversal: support ${key_support:.2f}, {days_since_breakdown}d ago, {volume_ratio:.1f}x vol"
        )
    
    def _calculate_breakdown_confidence(self, regime: RegimeData, volume_ratio: float, days_since: int, current_price: float, support_level: float) -> float:
        """Calculate confidence for failed breakdown signal."""
        confidence = 0.5  # Base confidence
        
        # Volume confirmation
        if volume_ratio > 2.5:
            confidence += 0.2
        elif volume_ratio > 2.0:
            confidence += 0.15
        elif volume_ratio > 1.5:
            confidence += 0.1
        
        # Quicker reversal = higher confidence
        if days_since == 1:
            confidence += 0.2
        elif days_since == 2:
            confidence += 0.1
        
        # Distance above support
        recovery_pct = (current_price - support_level) / support_level
        if recovery_pct > 0.02:
            confidence += 0.15
        elif recovery_pct > 0.01:
            confidence += 0.1
        
        # Works well in trending markets (bear traps)
        if regime.trend_regime in [TrendRegime.STRONG_UPTREND, TrendRegime.MILD_UPTREND]:
            confidence += 0.15
        
        return min(1.0, max(0.0, confidence))
    
    def validate_signal(self, symbol: str, regime: RegimeData) -> Tuple[bool, float]:
        """Validate if failed breakdown signal is suitable for current regime."""
        confidence = 0.6
        
        # Works well in trending markets
        if regime.trend_regime in [TrendRegime.STRONG_UPTREND, TrendRegime.MILD_UPTREND]:
            confidence += 0.2
        
        return True, confidence


class EarningsExpectationResetSetup(BaseSetup):
    """Earnings expectation reset setup - trade technical patterns after earnings uncertainty is removed."""
    
    def scan_for_signals(self, symbols: List[str]) -> List[TradeSignal]:
        signals = []
        current_regime = self.regime_detector.detect_current_regime()
        
        for symbol in symbols:
            try:
                data = self.get_market_data(symbol, period="3mo")
                if len(data) < 30:
                    continue
                
                signal = self._analyze_earnings_reset(symbol, data, current_regime)
                if signal:
                    signals.append(signal)
                    
            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")
        
        return signals
    
    def _analyze_earnings_reset(self, symbol: str, data: pd.DataFrame, regime: RegimeData) -> Optional[TradeSignal]:
        """Analyze for post-earnings technical setup opportunity."""
        # Simplified implementation - detect recent volatility spike and subsequent calm
        current_price = data['Close'].iloc[-1]
        atr = data['ATR'].iloc[-1]
        
        # Look for recent volatility spike (earnings-like pattern)
        atr_history = data['ATR'].rolling(20).mean()
        recent_atr_spike = False
        
        # Check last 10 days for ATR spike above 20-day average
        for i in range(1, min(11, len(data))):
            daily_atr = data['ATR'].iloc[-i]
            avg_atr = atr_history.iloc[-i]
            
            if daily_atr > avg_atr * 1.5:  # 50% above average
                recent_atr_spike = True
                break
        
        if not recent_atr_spike:
            return None
        
        # Current ATR should be normalized (no longer spiking)
        current_avg_atr = atr_history.iloc[-1]
        if atr > current_avg_atr * 1.3:  # Still elevated
            return None
        
        # Look for one of the existing technical setups to emerge
        # This is a meta-setup that enhances other setups after earnings
        
        # Check for trend pullback pattern
        sma20 = data['SMA20'].iloc[-1]
        sma50 = data['SMA50'].iloc[-1]
        
        setup_pattern = None
        confidence_boost = 0.0
        
        # Pattern 1: Trend continuation after pullback
        if current_price > sma50 and current_price < sma20:
            recent_high = data['High'].rolling(10).max().iloc[-1]
            pullback_pct = (recent_high - current_price) / recent_high
            
            if 0.02 <= pullback_pct <= 0.06:  # 2-6% pullback
                setup_pattern = "post_earnings_pullback"
                confidence_boost = 0.15
        
        # Pattern 2: Breakout above consolidation
        elif current_price > sma20:
            recent_range = data['High'].rolling(5).max().iloc[-1] - data['Low'].rolling(5).min().iloc[-1]
            range_pct = recent_range / current_price
            
            if range_pct < 0.03:  # Tight consolidation
                setup_pattern = "post_earnings_breakout"
                confidence_boost = 0.1
        
        if not setup_pattern:
            return None
        
        # Position sizing and risk management
        if setup_pattern == "post_earnings_pullback":
            stop_loss = current_price - (2 * atr)
            target_price = current_price + (2.5 * atr)
        else:  # breakout
            stop_loss = sma20 * 0.98
            target_price = current_price + (2 * atr)
        
        # Base confidence from reduced uncertainty
        confidence = 0.6 + confidence_boost
        
        # Regime adjustments
        if regime.volatility_regime == VolatilityRegime.LOW:
            confidence += 0.1
        elif regime.volatility_regime == VolatilityRegime.HIGH:
            confidence -= 0.15
        
        if confidence < 0.5:
            return None
        
        position_size = self.calculate_position_size(current_price, stop_loss)
        
        return TradeSignal(
            symbol=symbol,
            setup_type=SetupType.EARNINGS_EXPECTATION_RESET,
            signal_strength=SignalStrength.MEDIUM,
            entry_price=current_price,
            stop_loss=stop_loss,
            target_price=target_price,
            position_size=position_size,
            risk_per_share=current_price - stop_loss,
            confidence=confidence,
            regime_context=regime,
            notes=f"Post-earnings {setup_pattern}, volatility normalized"
        )
    
    def validate_signal(self, symbol: str, regime: RegimeData) -> Tuple[bool, float]:
        """Validate if earnings reset signal is suitable for current regime."""
        confidence = 0.6
        
        # Works better in stable regimes
        if regime.volatility_regime == VolatilityRegime.LOW:
            confidence += 0.2
        elif regime.volatility_regime == VolatilityRegime.HIGH:
            confidence -= 0.2
        
        return confidence > 0.4, confidence


class SetupManager:
    """Manages all trade setups and scanning."""
    
    def __init__(self, db_path: str = "journal.db"):
        self.db_path = db_path
        self.setups = {
            SetupType.TREND_PULLBACK: TrendPullbackSetup(db_path),
            SetupType.BREAKOUT_CONTINUATION: BreakoutContinuationSetup(db_path),
            SetupType.OVERSOLD_MEAN_REVERSION: OversoldMeanReversionSetup(db_path),
            SetupType.REGIME_ROTATION: RegimeRotationSetup(db_path),
            SetupType.GAP_FILL_REVERSAL: GapFillReversalSetup(db_path),
            SetupType.RELATIVE_STRENGTH_MOMENTUM: RelativeStrengthMomentumSetup(db_path),
            SetupType.VOLATILITY_CONTRACTION: VolatilityContractionSetup(db_path),
            SetupType.DIVIDEND_DISTRIBUTION_PLAY: DividendDistributionPlaySetup(db_path),
            SetupType.ELDER_TRIPLE_SCREEN: ElderTripleScreenSetup(db_path),
            SetupType.INSTITUTIONAL_VOLUME_CLIMAX: InstitutionalVolumeClimaxSetup(db_path),
            SetupType.FAILED_BREAKDOWN_REVERSAL: FailedBreakdownReversalSetup(db_path),
            SetupType.EARNINGS_EXPECTATION_RESET: EarningsExpectationResetSetup(db_path)
        }
    
    def get_all_symbols(self) -> List[str]:
        """Get all ETF symbols from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT symbol FROM instruments WHERE type = 'ETF'")
            return [row[0] for row in cursor.fetchall()]
    
    def scan_all_setups(self, max_signals_per_setup: int = 5) -> Dict[SetupType, List[TradeSignal]]:
        """Scan all setups for signals."""
        symbols = self.get_all_symbols()
        all_signals = {}
        
        for setup_type, setup in self.setups.items():
            try:
                signals = setup.scan_for_signals(symbols)
                # Sort by confidence and take top signals
                signals.sort(key=lambda x: x.confidence, reverse=True)
                all_signals[setup_type] = signals[:max_signals_per_setup]
            except Exception as e:
                print(f"Error scanning {setup_type.value}: {e}")
                all_signals[setup_type] = []
        
        return all_signals
    
    def get_top_signals(self, max_signals: int = 10) -> List[TradeSignal]:
        """Get top signals across all setups."""
        all_signals = self.scan_all_setups()
        combined_signals = []
        
        for signals in all_signals.values():
            combined_signals.extend(signals)
        
        # Sort by confidence and return top signals
        combined_signals.sort(key=lambda x: x.confidence, reverse=True)
        return combined_signals[:max_signals]


if __name__ == "__main__":
    # Test the setup manager
    manager = SetupManager()
    
    print("🔍 Scanning for trade signals...")
    signals = manager.get_top_signals(max_signals=5)
    
    if not signals:
        print("No signals found in current market conditions.")
    else:
        print(f"\n📊 Found {len(signals)} trade signals:")
        for i, signal in enumerate(signals, 1):
            print(f"\n{i}. {signal.symbol} - {signal.setup_type.value}")
            print(f"   Entry: ${signal.entry_price:.2f}")
            print(f"   Stop: ${signal.stop_loss:.2f}")
            print(f"   Target: ${signal.target_price:.2f}")
            print(f"   Risk/Share: ${signal.risk_per_share:.2f}")
            print(f"   Position Size: {signal.position_size:.0f} shares")
            print(f"   Confidence: {signal.confidence:.1%}")
            print(f"   Notes: {signal.notes}")