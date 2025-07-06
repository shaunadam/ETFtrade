#!/usr/bin/env python3
"""
Backtest Configuration System

Centralized configuration management for the backtesting engine.
Replaces hardcoded parameters with configurable, validated settings.

Usage:
    # Load from file
    config = BacktestConfiguration.from_file('config.json')
    
    # Load from dictionary
    config = BacktestConfiguration.from_dict(config_dict)
    
    # Use default configuration
    config = BacktestConfiguration.default()
    
    # Use preset configurations
    config = BacktestConfiguration.conservative_preset()
    config = BacktestConfiguration.aggressive_preset()
    config = BacktestConfiguration.debug_preset()
"""

import json
import logging
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime


@dataclass
class RiskManagementConfig:
    """Risk management and position sizing configuration."""
    
    # Core risk parameters
    max_risk_per_trade: float = 0.02  # 2% per trade
    max_concurrent_positions: int = 6  # Max open positions
    max_sector_allocation: float = 0.30  # 30% max in correlated sectors
    max_similar_etfs: int = 2  # Max ETFs from same category
    initial_capital: float = 100000.0  # Starting capital
    
    # Position sizing
    position_size_method: str = "fixed_risk"  # fixed_risk, kelly, equal_weight
    min_position_size: float = 100.0  # Minimum position size in dollars
    max_position_size: float = 10000.0  # Maximum position size in dollars
    
    # Portfolio heat limits
    max_total_risk: float = 0.08  # 8% total portfolio risk
    correlation_adjustment: bool = True  # Adjust for sector correlation
    
    def validate(self) -> List[str]:
        """Validate risk management parameters."""
        errors = []
        
        if not 0 < self.max_risk_per_trade <= 0.10:
            errors.append("max_risk_per_trade must be between 0.1% and 10%")
        
        if not 1 <= self.max_concurrent_positions <= 20:
            errors.append("max_concurrent_positions must be between 1 and 20")
        
        if not 0.05 <= self.max_sector_allocation <= 1.0:
            errors.append("max_sector_allocation must be between 5% and 100%")
        
        if not 0 < self.initial_capital <= 10000000:
            errors.append("initial_capital must be between $1 and $10M")
        
        if self.position_size_method not in ["fixed_risk", "kelly", "equal_weight"]:
            errors.append("position_size_method must be 'fixed_risk', 'kelly', or 'equal_weight'")
        
        return errors


@dataclass
class OptimizationConfig:
    """Parameter optimization configuration for walk-forward analysis."""
    
    # Stop loss optimization ranges
    stop_loss_range: List[float] = field(default_factory=lambda: [0.03, 0.04, 0.05, 0.06, 0.08])
    
    # Profit target optimization ranges (R multiples)
    profit_target_range: List[float] = field(default_factory=lambda: [1.5, 2.0, 2.5, 3.0])
    
    # Confidence threshold ranges
    confidence_range: List[float] = field(default_factory=lambda: [0.5, 0.6, 0.7, 0.8])
    
    # Walk-forward parameters
    training_period_days: int = 252  # 1 year training
    test_period_days: int = 63  # 3 months testing
    
    # Optimization scoring
    fitness_function: str = "sharpe_ratio"  # sharpe_ratio, calmar_ratio, profit_factor
    trade_frequency_penalty: float = 0.01  # Penalty for excessive trades
    min_trades_required: int = 3  # Minimum trades for valid optimization
    
    def validate(self) -> List[str]:
        """Validate optimization parameters."""
        errors = []
        
        if not all(0.01 <= sl <= 0.20 for sl in self.stop_loss_range):
            errors.append("stop_loss_range values must be between 1% and 20%")
        
        if not all(0.5 <= pt <= 10.0 for pt in self.profit_target_range):
            errors.append("profit_target_range values must be between 0.5R and 10R")
        
        if not all(0.0 <= conf <= 1.0 for conf in self.confidence_range):
            errors.append("confidence_range values must be between 0 and 1")
        
        if not 50 <= self.training_period_days <= 1000:
            errors.append("training_period_days must be between 50 and 1000")
        
        if not 10 <= self.test_period_days <= 200:
            errors.append("test_period_days must be between 10 and 200")
        
        if self.fitness_function not in ["sharpe_ratio", "calmar_ratio", "profit_factor"]:
            errors.append("fitness_function must be 'sharpe_ratio', 'calmar_ratio', or 'profit_factor'")
        
        return errors


@dataclass
class TradingConfig:
    """Core trading parameters and rules."""
    
    # Default position parameters
    default_stop_loss_pct: float = 0.05  # 5% stop loss
    default_profit_target_r: float = 2.0  # 2R profit target
    default_confidence_threshold: float = 0.6  # Minimum confidence for signals
    max_holding_days: int = 60  # Maximum days to hold position
    
    # Entry/exit rules
    allow_same_day_entry_exit: bool = False
    require_confirmation: bool = True  # Require signal confirmation
    trailing_stop_enabled: bool = True  # Enable trailing stops
    trailing_stop_trigger: float = 1.0  # Trigger trailing stop at 1R profit
    
    # Regime filtering
    regime_aware_trading: bool = True  # Enable regime-based filtering
    regime_override_confidence: float = 0.8  # Override regime filter if confidence > this
    
    # Signal processing
    max_signals_per_day: int = 5  # Limit signals processed per day
    signal_strength_weight: float = 0.3  # Weight for signal strength in ranking
    confidence_weight: float = 0.7  # Weight for confidence in ranking
    
    def validate(self) -> List[str]:
        """Validate trading parameters."""
        errors = []
        
        if not 0.01 <= self.default_stop_loss_pct <= 0.20:
            errors.append("default_stop_loss_pct must be between 1% and 20%")
        
        if not 0.5 <= self.default_profit_target_r <= 10.0:
            errors.append("default_profit_target_r must be between 0.5R and 10R")
        
        if not 0.0 <= self.default_confidence_threshold <= 1.0:
            errors.append("default_confidence_threshold must be between 0 and 1")
        
        if not 1 <= self.max_holding_days <= 500:
            errors.append("max_holding_days must be between 1 and 500")
        
        if not 0.0 <= self.regime_override_confidence <= 1.0:
            errors.append("regime_override_confidence must be between 0 and 1")
        
        if not 1 <= self.max_signals_per_day <= 50:
            errors.append("max_signals_per_day must be between 1 and 50")
        
        if not (0.0 <= self.signal_strength_weight <= 1.0 and 
                0.0 <= self.confidence_weight <= 1.0 and
                abs(self.signal_strength_weight + self.confidence_weight - 1.0) < 0.01):
            errors.append("signal_strength_weight and confidence_weight must sum to 1.0")
        
        return errors


@dataclass
class DebugConfig:
    """Debug and logging configuration."""
    
    # Logging settings
    logging_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    log_to_file: bool = False
    log_file_path: str = "backtest_debug.log"
    log_to_console: bool = True
    
    # Debug features
    enable_debug_mode: bool = False
    debug_max_trades: Optional[int] = None  # Limit trades in debug mode
    debug_date_range_days: Optional[int] = 30  # Limit date range in debug mode
    
    # Verbose output
    verbose_optimization: bool = False  # Detailed optimization logging
    verbose_trade_decisions: bool = False  # Log every trade decision
    verbose_regime_filtering: bool = False  # Log regime filtering decisions
    verbose_performance_calc: bool = False  # Log performance calculations
    
    # Data validation
    validate_data_integrity: bool = True  # Check data quality
    warn_missing_data: bool = True  # Warn about missing price data
    strict_date_validation: bool = True  # Strict date range validation
    
    def validate(self) -> List[str]:
        """Validate debug parameters."""
        errors = []
        
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
        if self.logging_level not in valid_levels:
            errors.append(f"logging_level must be one of {valid_levels}")
        
        if self.debug_max_trades is not None and self.debug_max_trades < 1:
            errors.append("debug_max_trades must be positive if specified")
        
        if self.debug_date_range_days is not None and not 1 <= self.debug_date_range_days <= 1000:
            errors.append("debug_date_range_days must be between 1 and 1000 if specified")
        
        return errors


@dataclass
class BacktestConfiguration:
    """Complete backtest configuration."""
    
    risk_management: RiskManagementConfig = field(default_factory=RiskManagementConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    
    # Metadata
    config_name: str = "default"
    config_version: str = "1.0"
    created_at: Optional[str] = None
    description: str = "Default backtest configuration"
    
    def __post_init__(self):
        """Set creation timestamp if not provided."""
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def validate(self) -> Dict[str, List[str]]:
        """Validate all configuration sections."""
        validation_results = {
            'risk_management': self.risk_management.validate(),
            'optimization': self.optimization.validate(),
            'trading': self.trading.validate(),
            'debug': self.debug.validate()
        }
        
        # Cross-section validation
        cross_validation_errors = []
        
        # Check if max risk per trade * max positions exceeds total risk
        max_theoretical_risk = (self.risk_management.max_risk_per_trade * 
                               self.risk_management.max_concurrent_positions)
        if max_theoretical_risk > self.risk_management.max_total_risk * 1.5:
            cross_validation_errors.append(
                f"Max theoretical risk ({max_theoretical_risk:.1%}) significantly exceeds "
                f"max total risk ({self.risk_management.max_total_risk:.1%})"
            )
        
        # Check optimization ranges are reasonable
        if (max(self.optimization.profit_target_range) < 
            max(self.optimization.stop_loss_range) * 10):  # 10x seems reasonable max
            cross_validation_errors.append(
                "Profit target range should be higher relative to stop loss range"
            )
        
        validation_results['cross_validation'] = cross_validation_errors
        
        return validation_results
    
    def is_valid(self) -> bool:
        """Check if entire configuration is valid."""
        validation_results = self.validate()
        return all(len(errors) == 0 for errors in validation_results.values())
    
    def get_validation_summary(self) -> str:
        """Get human-readable validation summary."""
        validation_results = self.validate()
        
        if self.is_valid():
            return "✅ Configuration is valid"
        
        summary = "❌ Configuration has validation errors:\n"
        for section, errors in validation_results.items():
            if errors:
                summary += f"\n{section.title()}:\n"
                for error in errors:
                    summary += f"  - {error}\n"
        
        return summary
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BacktestConfiguration':
        """Create configuration from dictionary."""
        # Extract nested configurations
        risk_config = RiskManagementConfig(**config_dict.get('risk_management', {}))
        opt_config = OptimizationConfig(**config_dict.get('optimization', {}))
        trading_config = TradingConfig(**config_dict.get('trading', {}))
        debug_config = DebugConfig(**config_dict.get('debug', {}))
        
        # Extract metadata
        metadata = {k: v for k, v in config_dict.items() 
                   if k not in ['risk_management', 'optimization', 'trading', 'debug']}
        
        return cls(
            risk_management=risk_config,
            optimization=opt_config,
            trading=trading_config,
            debug=debug_config,
            **metadata
        )
    
    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> 'BacktestConfiguration':
        """Load configuration from JSON file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                config_dict = json.load(f)
            return cls.from_dict(config_dict)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")
        except Exception as e:
            raise ValueError(f"Error loading configuration file: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def to_file(self, file_path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        file_path = Path(file_path)
        
        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = self.to_dict()
        
        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
    
    @classmethod
    def default(cls) -> 'BacktestConfiguration':
        """Create default configuration."""
        return cls(
            config_name="default",
            description="Default balanced configuration for ETF/stock trading"
        )
    
    @classmethod
    def conservative_preset(cls) -> 'BacktestConfiguration':
        """Create conservative trading configuration."""
        risk_config = RiskManagementConfig(
            max_risk_per_trade=0.01,  # 1% per trade
            max_concurrent_positions=4,  # Fewer positions
            max_sector_allocation=0.25,  # Lower sector concentration
        )
        
        opt_config = OptimizationConfig(
            stop_loss_range=[0.06, 0.08, 0.10],  # Wider stops
            profit_target_range=[2.0, 2.5, 3.0],  # Higher targets
            confidence_range=[0.7, 0.8, 0.9],  # Higher confidence required
        )
        
        trading_config = TradingConfig(
            default_stop_loss_pct=0.08,  # 8% stop loss
            default_profit_target_r=2.5,  # 2.5R target
            default_confidence_threshold=0.8,  # High confidence required
            max_holding_days=90,  # Longer hold periods
        )
        
        return cls(
            risk_management=risk_config,
            optimization=opt_config,
            trading=trading_config,
            config_name="conservative",
            description="Conservative configuration with lower risk and higher confidence requirements"
        )
    
    @classmethod
    def aggressive_preset(cls) -> 'BacktestConfiguration':
        """Create aggressive trading configuration."""
        risk_config = RiskManagementConfig(
            max_risk_per_trade=0.025,  # 2.5% per trade
            max_concurrent_positions=8,  # More positions
            max_sector_allocation=0.35,  # Higher sector concentration
        )
        
        opt_config = OptimizationConfig(
            stop_loss_range=[0.03, 0.04, 0.05],  # Tighter stops
            profit_target_range=[1.5, 2.0, 2.5],  # Lower targets
            confidence_range=[0.4, 0.5, 0.6],  # Lower confidence accepted
        )
        
        trading_config = TradingConfig(
            default_stop_loss_pct=0.04,  # 4% stop loss
            default_profit_target_r=1.5,  # 1.5R target
            default_confidence_threshold=0.5,  # Lower confidence accepted
            max_holding_days=30,  # Shorter hold periods
        )
        
        return cls(
            risk_management=risk_config,
            optimization=opt_config,
            trading=trading_config,
            config_name="aggressive",
            description="Aggressive configuration with higher risk and lower confidence requirements"
        )
    
    @classmethod
    def debug_preset(cls) -> 'BacktestConfiguration':
        """Create debug configuration for testing and development."""
        debug_config = DebugConfig(
            logging_level="DEBUG",
            enable_debug_mode=True,
            debug_max_trades=10,  # Limit trades for fast testing
            debug_date_range_days=30,  # Short date range
            verbose_optimization=True,
            verbose_trade_decisions=True,
            verbose_regime_filtering=True,
            log_to_file=True,
        )
        
        risk_config = RiskManagementConfig(
            max_concurrent_positions=2,  # Fewer positions for easier debugging
        )
        
        opt_config = OptimizationConfig(
            stop_loss_range=[0.05],  # Single value for faster testing
            profit_target_range=[2.0],
            confidence_range=[0.6],
            training_period_days=60,  # Shorter training period
            test_period_days=20,  # Shorter test period
        )
        
        return cls(
            risk_management=risk_config,
            optimization=opt_config,
            debug=debug_config,
            config_name="debug",
            description="Debug configuration with extensive logging and limited scope for testing"
        )
    
    def setup_logging(self) -> None:
        """Setup logging based on debug configuration."""
        # Convert string level to logging constant
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR
        }
        
        level = level_map.get(self.debug.logging_level, logging.INFO)
        
        # Create logger
        logger = logging.getLogger('backtest')
        logger.setLevel(level)
        
        # Clear existing handlers
        logger.handlers = []
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        if self.debug.log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # File handler
        if self.debug.log_to_file:
            file_handler = logging.FileHandler(self.debug.log_file_path)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger


# Convenience functions for common use cases
def load_config(config_path: Optional[str] = None, 
                preset: Optional[str] = None) -> BacktestConfiguration:
    """
    Load configuration from file or preset.
    
    Args:
        config_path: Path to JSON configuration file
        preset: Preset name ('default', 'conservative', 'aggressive', 'debug')
    
    Returns:
        BacktestConfiguration instance
    """
    if config_path:
        return BacktestConfiguration.from_file(config_path)
    elif preset:
        preset_map = {
            'default': BacktestConfiguration.default,
            'conservative': BacktestConfiguration.conservative_preset,
            'aggressive': BacktestConfiguration.aggressive_preset,
            'debug': BacktestConfiguration.debug_preset,
        }
        
        if preset not in preset_map:
            raise ValueError(f"Unknown preset '{preset}'. Available: {list(preset_map.keys())}")
        
        return preset_map[preset]()
    else:
        return BacktestConfiguration.default()


def create_sample_configs() -> None:
    """Create sample configuration files for reference."""
    configs = [
        ('default_config.json', BacktestConfiguration.default()),
        ('conservative_config.json', BacktestConfiguration.conservative_preset()),
        ('aggressive_config.json', BacktestConfiguration.aggressive_preset()),
        ('debug_config.json', BacktestConfiguration.debug_preset()),
    ]
    
    for filename, config in configs:
        config.to_file(filename)
        print(f"Created {filename}")


if __name__ == "__main__":
    # Create sample configurations
    create_sample_configs()
    
    # Test validation
    config = BacktestConfiguration.default()
    print(config.get_validation_summary())