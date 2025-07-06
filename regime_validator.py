#!/usr/bin/env python3
"""
Regime Validator for Backtest Engine

Extracted from BacktestEngine to handle regime-aware signal filtering.
Provides comprehensive regime validation for setup appropriateness based on market conditions.

Usage:
    from regime_validator import RegimeValidator
    from backtest_config import BacktestConfiguration
    
    config = BacktestConfiguration.default()
    validator = RegimeValidator(config)
    
    # Filter signals based on current regime
    filtered_signals = validator.filter_signals_by_regime(signals, current_regime)
    
    # Validate individual signal
    is_valid = validator.validate_signal(signal, current_regime)
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

from backtest_config import BacktestConfiguration
from trade_setups import TradeSignal, SetupType
from regime_detection import RegimeData


@dataclass
class RegimeFilteringRules:
    """Configuration for regime-based filtering rules."""
    
    # Volatility regime thresholds
    high_volatility_threshold: float = 30.0  # VIX level
    low_volatility_threshold: float = 20.0   # VIX level
    
    # Trend regime thresholds
    strong_uptrend_threshold: float = 0.10   # SPY vs SMA200 (10% above)
    strong_downtrend_threshold: float = -0.10  # SPY vs SMA200 (10% below)
    
    # Risk sentiment thresholds
    risk_off_threshold: float = 0.95   # Risk on/off ratio
    risk_on_threshold: float = 1.05    # Risk on/off ratio
    
    # Sector rotation thresholds
    growth_outperforming_threshold: float = 1.10  # Growth/Value ratio
    value_outperforming_threshold: float = 0.90   # Growth/Value ratio
    
    # Override settings
    enable_confidence_override: bool = True
    confidence_override_threshold: float = 0.8  # Override regime filter if confidence > this


class RegimeValidator:
    """
    Handles regime-aware signal filtering for backtesting.
    
    Provides sophisticated market regime validation that considers:
    - Volatility regimes (low/medium/high VIX)
    - Trend regimes (uptrend/neutral/downtrend)
    - Risk sentiment (risk-on/risk-off)
    - Sector rotation (growth vs value)
    """
    
    def __init__(self, config: BacktestConfiguration):
        """
        Initialize regime validator.
        
        Args:
            config: Backtest configuration containing regime settings
        """
        self.config = config
        self.logger = logging.getLogger('backtest.regime')
        
        # Initialize filtering rules from config
        self.rules = RegimeFilteringRules(
            high_volatility_threshold=30.0,
            low_volatility_threshold=20.0,
            strong_uptrend_threshold=0.10,
            strong_downtrend_threshold=-0.10,
            risk_off_threshold=0.95,
            risk_on_threshold=1.05,
            growth_outperforming_threshold=1.10,
            value_outperforming_threshold=0.90,
            enable_confidence_override=config.trading.regime_override_confidence > 0,
            confidence_override_threshold=config.trading.regime_override_confidence
        )
        
        self.logger.debug(f"Initialized RegimeValidator with regime filtering enabled: {config.trading.regime_aware_trading}")
    
    def filter_signals_by_regime(self, 
                                signals: List[TradeSignal], 
                                current_regime: RegimeData) -> List[TradeSignal]:
        """
        Filter a list of signals based on current market regime.
        
        Args:
            signals: List of trade signals to filter
            current_regime: Current market regime data
            
        Returns:
            List of signals that pass regime validation
        """
        if not self.config.trading.regime_aware_trading:
            self.logger.debug("Regime filtering disabled, returning all signals")
            return signals
        
        if not current_regime:
            self.logger.warning("No regime data available, returning all signals")
            return signals
        
        filtered_signals = []
        rejection_reasons = {}
        
        for signal in signals:
            is_valid, reason = self.validate_signal_with_reason(signal, current_regime)
            
            if is_valid:
                filtered_signals.append(signal)
            else:
                # Track rejection reasons for debugging
                if reason not in rejection_reasons:
                    rejection_reasons[reason] = 0
                rejection_reasons[reason] += 1
        
        # Log filtering results
        self.logger.debug(f"Regime filtering: {len(signals)} -> {len(filtered_signals)} signals")
        
        if self.config.debug.verbose_regime_filtering and rejection_reasons:
            for reason, count in rejection_reasons.items():
                self.logger.debug(f"  Rejected {count} signals: {reason}")
        
        return filtered_signals
    
    def validate_signal(self, signal: TradeSignal, current_regime: RegimeData) -> bool:
        """
        Validate a single signal against current market regime.
        
        Args:
            signal: Trade signal to validate
            current_regime: Current market regime data
            
        Returns:
            True if signal is valid for current regime, False otherwise
        """
        is_valid, _ = self.validate_signal_with_reason(signal, current_regime)
        return is_valid
    
    def validate_signal_with_reason(self, 
                                   signal: TradeSignal, 
                                   current_regime: RegimeData) -> tuple[bool, str]:
        """
        Validate signal and return reason for rejection if invalid.
        
        Args:
            signal: Trade signal to validate
            current_regime: Current market regime data
            
        Returns:
            Tuple of (is_valid, rejection_reason)
        """
        if not self.config.trading.regime_aware_trading:
            return True, "regime_filtering_disabled"
        
        if not current_regime:
            return True, "no_regime_data"
        
        # Confidence override - high confidence signals bypass regime filtering
        if (self.rules.enable_confidence_override and 
            signal.confidence >= self.rules.confidence_override_threshold):
            self.logger.debug(f"Signal {signal.symbol} bypassed regime filter due to high confidence: {signal.confidence:.2f}")
            return True, "confidence_override"
        
        # Volatility regime validation
        volatility_valid, volatility_reason = self._validate_volatility_regime(signal, current_regime)
        if not volatility_valid:
            return False, volatility_reason
        
        # Trend regime validation
        trend_valid, trend_reason = self._validate_trend_regime(signal, current_regime)
        if not trend_valid:
            return False, trend_reason
        
        # Risk sentiment validation
        risk_valid, risk_reason = self._validate_risk_sentiment(signal, current_regime)
        if not risk_valid:
            return False, risk_reason
        
        # Sector rotation validation
        sector_valid, sector_reason = self._validate_sector_rotation(signal, current_regime)
        if not sector_valid:
            return False, sector_reason
        
        return True, "regime_valid"
    
    def _validate_volatility_regime(self, 
                                   signal: TradeSignal, 
                                   current_regime: RegimeData) -> tuple[bool, str]:
        """Validate signal against volatility regime."""
        vix_level = current_regime.vix_level
        
        # High volatility regime (VIX > 30) - avoid momentum strategies
        if vix_level > self.rules.high_volatility_threshold:
            if signal.setup_type in [SetupType.BREAKOUT_CONTINUATION, 
                                   SetupType.RELATIVE_STRENGTH_MOMENTUM]:
                return False, f"high_volatility_avoid_momentum_vix_{vix_level:.1f}"
        
        # Low volatility regime (VIX < 20) - favor volatility contraction
        elif vix_level < self.rules.low_volatility_threshold:
            if signal.setup_type == SetupType.VOLATILITY_CONTRACTION:
                return True, f"low_volatility_favor_vol_contraction_vix_{vix_level:.1f}"
        
        return True, "volatility_regime_ok"
    
    def _validate_trend_regime(self, 
                              signal: TradeSignal, 
                              current_regime: RegimeData) -> tuple[bool, str]:
        """Validate signal against trend regime."""
        spy_trend = current_regime.spy_vs_sma200
        
        # Strong uptrend - avoid mean reversion, favor momentum
        if spy_trend > self.rules.strong_uptrend_threshold:
            if signal.setup_type == SetupType.OVERSOLD_MEAN_REVERSION:
                return False, f"strong_uptrend_avoid_mean_reversion_spy_{spy_trend:.1%}"
            if signal.setup_type in [SetupType.TREND_PULLBACK, 
                                   SetupType.RELATIVE_STRENGTH_MOMENTUM]:
                return True, f"strong_uptrend_favor_momentum_spy_{spy_trend:.1%}"
        
        # Strong downtrend - favor mean reversion, avoid momentum
        elif spy_trend < self.rules.strong_downtrend_threshold:
            if signal.setup_type in [SetupType.BREAKOUT_CONTINUATION,
                                   SetupType.RELATIVE_STRENGTH_MOMENTUM]:
                return False, f"strong_downtrend_avoid_momentum_spy_{spy_trend:.1%}"
            if signal.setup_type in [SetupType.OVERSOLD_MEAN_REVERSION,
                                   SetupType.GAP_FILL_REVERSAL]:
                return True, f"strong_downtrend_favor_mean_reversion_spy_{spy_trend:.1%}"
        
        return True, "trend_regime_ok"
    
    def _validate_risk_sentiment(self, 
                                signal: TradeSignal, 
                                current_regime: RegimeData) -> tuple[bool, str]:
        """Validate signal against risk sentiment."""
        risk_ratio = current_regime.risk_on_off_ratio
        
        # Risk-off environment - favor defensive setups
        if risk_ratio < self.rules.risk_off_threshold:
            if signal.setup_type == SetupType.DIVIDEND_DISTRIBUTION_PLAY:
                return True, f"risk_off_favor_dividend_plays_ratio_{risk_ratio:.2f}"
            if signal.setup_type == SetupType.RELATIVE_STRENGTH_MOMENTUM:
                return False, f"risk_off_avoid_momentum_ratio_{risk_ratio:.2f}"
        
        # Risk-on environment - favor growth setups
        elif risk_ratio > self.rules.risk_on_threshold:
            if signal.setup_type in [SetupType.BREAKOUT_CONTINUATION,
                                   SetupType.RELATIVE_STRENGTH_MOMENTUM]:
                return True, f"risk_on_favor_growth_setups_ratio_{risk_ratio:.2f}"
        
        return True, "risk_sentiment_ok"
    
    def _validate_sector_rotation(self, 
                                 signal: TradeSignal, 
                                 current_regime: RegimeData) -> tuple[bool, str]:
        """Validate signal against sector rotation patterns."""
        growth_value_ratio = current_regime.growth_value_ratio
        
        # Growth outperforming - favor tech/growth oriented setups
        if growth_value_ratio > self.rules.growth_outperforming_threshold:
            if signal.setup_type in [SetupType.BREAKOUT_CONTINUATION,
                                   SetupType.RELATIVE_STRENGTH_MOMENTUM]:
                return True, f"growth_outperforming_favor_momentum_ratio_{growth_value_ratio:.2f}"
        
        # Value outperforming - favor defensive setups
        elif growth_value_ratio < self.rules.value_outperforming_threshold:
            if signal.setup_type in [SetupType.DIVIDEND_DISTRIBUTION_PLAY,
                                   SetupType.OVERSOLD_MEAN_REVERSION]:
                return True, f"value_outperforming_favor_defensive_ratio_{growth_value_ratio:.2f}"
        
        return True, "sector_rotation_ok"
    
    def get_regime_analysis(self, current_regime: RegimeData) -> Dict[str, str]:
        """
        Get human-readable analysis of current regime.
        
        Args:
            current_regime: Current market regime data
            
        Returns:
            Dictionary with regime analysis
        """
        if not current_regime:
            return {"error": "No regime data available"}
        
        analysis = {}
        
        # Volatility analysis
        vix = current_regime.vix_level
        if vix > self.rules.high_volatility_threshold:
            analysis["volatility"] = f"HIGH (VIX: {vix:.1f}) - Avoid momentum strategies"
        elif vix < self.rules.low_volatility_threshold:
            analysis["volatility"] = f"LOW (VIX: {vix:.1f}) - Favor volatility contraction"
        else:
            analysis["volatility"] = f"MEDIUM (VIX: {vix:.1f}) - Normal conditions"
        
        # Trend analysis
        spy_trend = current_regime.spy_vs_sma200
        if spy_trend > self.rules.strong_uptrend_threshold:
            analysis["trend"] = f"STRONG UPTREND (SPY: {spy_trend:.1%} above SMA200) - Favor momentum"
        elif spy_trend < self.rules.strong_downtrend_threshold:
            analysis["trend"] = f"STRONG DOWNTREND (SPY: {spy_trend:.1%} below SMA200) - Favor mean reversion"
        else:
            analysis["trend"] = f"NEUTRAL (SPY: {spy_trend:.1%} vs SMA200) - Normal conditions"
        
        # Risk sentiment analysis
        risk_ratio = current_regime.risk_on_off_ratio
        if risk_ratio < self.rules.risk_off_threshold:
            analysis["risk_sentiment"] = f"RISK-OFF ({risk_ratio:.2f}) - Favor defensive setups"
        elif risk_ratio > self.rules.risk_on_threshold:
            analysis["risk_sentiment"] = f"RISK-ON ({risk_ratio:.2f}) - Favor growth setups"
        else:
            analysis["risk_sentiment"] = f"NEUTRAL ({risk_ratio:.2f}) - Normal conditions"
        
        # Sector rotation analysis
        growth_value = current_regime.growth_value_ratio
        if growth_value > self.rules.growth_outperforming_threshold:
            analysis["sector_rotation"] = f"GROWTH OUTPERFORMING ({growth_value:.2f}) - Favor tech/momentum"
        elif growth_value < self.rules.value_outperforming_threshold:
            analysis["sector_rotation"] = f"VALUE OUTPERFORMING ({growth_value:.2f}) - Favor defensive"
        else:
            analysis["sector_rotation"] = f"BALANCED ({growth_value:.2f}) - Normal rotation"
        
        return analysis
    
    def get_preferred_setups(self, current_regime: RegimeData) -> List[SetupType]:
        """
        Get list of preferred setups for current regime.
        
        Args:
            current_regime: Current market regime data
            
        Returns:
            List of setup types that are favored in current regime
        """
        if not current_regime:
            return list(SetupType)  # Return all setups if no regime data
        
        preferred_setups = set()
        
        # Add setups based on regime conditions
        vix = current_regime.vix_level
        spy_trend = current_regime.spy_vs_sma200
        risk_ratio = current_regime.risk_on_off_ratio
        growth_value = current_regime.growth_value_ratio
        
        # Low volatility - favor volatility contraction
        if vix < self.rules.low_volatility_threshold:
            preferred_setups.add(SetupType.VOLATILITY_CONTRACTION)
        
        # Strong uptrend - favor momentum
        if spy_trend > self.rules.strong_uptrend_threshold:
            preferred_setups.update([SetupType.TREND_PULLBACK, SetupType.RELATIVE_STRENGTH_MOMENTUM])
        
        # Strong downtrend - favor mean reversion
        if spy_trend < self.rules.strong_downtrend_threshold:
            preferred_setups.update([SetupType.OVERSOLD_MEAN_REVERSION, SetupType.GAP_FILL_REVERSAL])
        
        # Risk-off - favor defensive
        if risk_ratio < self.rules.risk_off_threshold:
            preferred_setups.add(SetupType.DIVIDEND_DISTRIBUTION_PLAY)
        
        # Risk-on - favor growth
        if risk_ratio > self.rules.risk_on_threshold:
            preferred_setups.update([SetupType.BREAKOUT_CONTINUATION, SetupType.RELATIVE_STRENGTH_MOMENTUM])
        
        # Growth outperforming - favor momentum
        if growth_value > self.rules.growth_outperforming_threshold:
            preferred_setups.update([SetupType.BREAKOUT_CONTINUATION, SetupType.RELATIVE_STRENGTH_MOMENTUM])
        
        # Value outperforming - favor defensive
        if growth_value < self.rules.value_outperforming_threshold:
            preferred_setups.update([SetupType.DIVIDEND_DISTRIBUTION_PLAY, SetupType.OVERSOLD_MEAN_REVERSION])
        
        # If no specific preferences, return all setups
        if not preferred_setups:
            return list(SetupType)
        
        return list(preferred_setups)


if __name__ == "__main__":
    # Example usage
    from backtest_config import BacktestConfiguration
    from regime_detection import RegimeData, VolatilityRegime, TrendRegime, SectorRotation, RiskSentiment
    from datetime import datetime
    
    # Create test regime data
    test_regime = RegimeData(
        date=datetime.now(),
        volatility_regime=VolatilityRegime.HIGH,
        trend_regime=TrendRegime.STRONG_UPTREND,
        sector_rotation=SectorRotation.GROWTH_FAVORED,
        risk_sentiment=RiskSentiment.RISK_ON,
        vix_level=35.0,  # High volatility
        spy_vs_sma200=0.15,  # Strong uptrend
        growth_value_ratio=1.20,  # Growth outperforming
        risk_on_off_ratio=1.10  # Risk-on
    )
    
    config = BacktestConfiguration.default()
    validator = RegimeValidator(config)
    
    # Get regime analysis
    analysis = validator.get_regime_analysis(test_regime)
    print("Current Regime Analysis:")
    for category, description in analysis.items():
        print(f"  {category.title()}: {description}")
    
    # Get preferred setups
    preferred = validator.get_preferred_setups(test_regime)
    print(f"\nPreferred setups for this regime: {[s.value for s in preferred]}")