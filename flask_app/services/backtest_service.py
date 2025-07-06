"""
Backtest Service Layer

Integrates backtest.py with Flask application.
Provides web-friendly interface for backtesting operations.
"""

import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

# Add parent directory to import CLI modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from backtest import BacktestEngine, OptimizationParameters
    from trade_setups import SetupManager, SetupType
    from flask_app.models import db, Setup
except ImportError as e:
    print(f"Import error in backtest service: {e}")
    # Fallback for development
    BacktestEngine = None
    OptimizationParameters = None
    SetupManager = None
    SetupType = None

class BacktestService:
    """Service for backtesting operations in Flask app"""
    
    def __init__(self):
        # Use the correct database path - same as Flask config
        db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'journal.db')
        
        if BacktestEngine:
            self.backtest_engine = BacktestEngine(db_path)
        else:
            self.backtest_engine = None
        
        if SetupManager:
            self.setup_manager = SetupManager(db_path)
        else:
            self.setup_manager = None
    
    def get_available_setups(self) -> List[Dict]:
        """Get available trade setups for backtesting"""
        try:
            # Get from database
            setups = Setup.query.all()
            
            return [
                {
                    'id': setup.id,
                    'name': setup.name,
                    'description': setup.description,
                    'parameters': setup.parameters
                }
                for setup in setups
            ]
            
        except Exception as e:
            # Fallback to hardcoded list
            return [
                {'id': 1, 'name': 'trend_pullback', 'description': 'Trend pullback setup'},
                {'id': 2, 'name': 'breakout_continuation', 'description': 'Breakout continuation setup'},
                {'id': 3, 'name': 'oversold_mean_reversion', 'description': 'Oversold mean reversion setup'},
                {'id': 4, 'name': 'regime_rotation', 'description': 'Regime rotation setup'},
                {'id': 5, 'name': 'gap_fill_reversal', 'description': 'Gap fill reversal setup'},
                {'id': 6, 'name': 'relative_strength_momentum', 'description': 'Relative strength momentum setup'},
                {'id': 7, 'name': 'volatility_contraction', 'description': 'Volatility contraction setup'},
                {'id': 8, 'name': 'dividend_distribution_play', 'description': 'Dividend distribution play setup'}
            ]
    
    def run_backtest(self, config: Dict) -> Dict:
        """Run backtest with given configuration"""
        try:
            if not self.backtest_engine:
                return {
                    'success': False,
                    'error': 'Backtest engine not available. Check CLI integration.'
                }
            
            # Extract configuration
            setup_name = config.get('setup_name', 'all')
            start_date = datetime.fromisoformat(config.get('start_date', '2023-01-01'))
            end_date = datetime.fromisoformat(config.get('end_date', datetime.now().strftime('%Y-%m-%d')))
            walk_forward = config.get('walk_forward', False)
            regime_aware = config.get('regime_aware', True)
            selected_instruments = config.get('selected_instruments')  # New parameter
            
            # Create optimization parameters
            optimization_params = OptimizationParameters(
                stop_loss_pct=config.get('stop_loss_pct', 0.05),
                profit_target_r=config.get('profit_target_r', 2.0),
                confidence_threshold=config.get('confidence_threshold', 0.6),
                max_holding_days=config.get('max_holding_days', 60),
                position_size_method=config.get('position_size_method', 'fixed_risk')
            )
            
            # Run backtest with selected instruments if provided
            if setup_name == 'all':
                results = self.backtest_engine.run_backtest(
                    start_date=start_date,
                    end_date=end_date,
                    setup_types=None,  # None means all setups
                    walk_forward=walk_forward,
                    regime_aware=regime_aware,
                    selected_instruments=selected_instruments
                )
            else:
                setup_type = getattr(SetupType, setup_name.upper(), None)
                if not setup_type:
                    return {
                        'success': False,
                        'error': f'Unknown setup type: {setup_name}'
                    }
                
                results = self.backtest_engine.run_backtest(
                    start_date=start_date,
                    end_date=end_date,
                    setup_types=[setup_type],
                    walk_forward=walk_forward,
                    regime_aware=regime_aware,
                    selected_instruments=selected_instruments
                )
            
            # Convert results to web-friendly format
            web_results = self._format_backtest_results(results)
            
            # Add selected instruments info to results
            if selected_instruments:
                web_results['selected_instruments'] = selected_instruments
                web_results['instrument_count'] = len(selected_instruments)
            
            return {
                'success': True,
                'data': web_results,
                'config': config,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Backtest failed: {str(e)}'
            }
    
    def get_default_parameters(self) -> Dict:
        """Get default backtest parameters"""
        return {
            'start_date': (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
            'end_date': datetime.now().strftime('%Y-%m-%d'),
            'setup_name': 'all',
            'walk_forward': True,
            'regime_aware': True,
            'stop_loss_pct': 0.05,
            'profit_target_r': 2.0,
            'confidence_threshold': 0.6,
            'max_holding_days': 60,
            'position_size_method': 'fixed_risk',
            'initial_capital': 100000
        }
    
    def validate_parameters(self, config: Dict) -> Dict:
        """Validate backtest parameters"""
        errors = []
        
        # Validate dates
        try:
            start_date = datetime.fromisoformat(config.get('start_date', ''))
            end_date = datetime.fromisoformat(config.get('end_date', ''))
            
            if start_date >= end_date:
                errors.append('Start date must be before end date')
            
            if end_date > datetime.now():
                errors.append('End date cannot be in the future')
                
        except ValueError:
            errors.append('Invalid date format. Use YYYY-MM-DD')
        
        # Validate numerical parameters
        try:
            stop_loss = float(config.get('stop_loss_pct', 0))
            if not 0 < stop_loss <= 1:
                errors.append('Stop loss must be between 0% and 100%')
        except ValueError:
            errors.append('Invalid stop loss percentage')
        
        try:
            profit_target = float(config.get('profit_target_r', 0))
            if not 0 < profit_target <= 10:
                errors.append('Profit target must be between 0R and 10R')
        except ValueError:
            errors.append('Invalid profit target')
        
        try:
            confidence = float(config.get('confidence_threshold', 0))
            if not 0 <= confidence <= 1:
                errors.append('Confidence threshold must be between 0 and 1')
        except ValueError:
            errors.append('Invalid confidence threshold')
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    def _format_backtest_results(self, results) -> Dict:
        """Format backtest results for web display"""
        if not results:
            return {'error': 'No results available'}
        
        try:
            # Handle different result types
            if hasattr(results, 'performance_metrics'):
                # Single setup results
                return self._format_single_setup_results(results)
            elif isinstance(results, dict):
                # Multiple setup results
                return self._format_multiple_setup_results(results)
            else:
                return {'error': 'Unknown result format'}
                
        except Exception as e:
            return {'error': f'Failed to format results: {str(e)}'}
    
    def _format_single_setup_results(self, results) -> Dict:
        """Format single setup backtest results"""
        try:
            metrics = results.performance_metrics
            
            formatted_results = {
                'setup_name': results.setup_type.value if hasattr(results, 'setup_type') else 'unknown',
                'total_trades': metrics.total_trades,
                'winning_trades': metrics.winning_trades,
                'losing_trades': metrics.losing_trades,
                'win_rate': round(metrics.win_rate * 100, 2),
                'total_return': round(metrics.total_return * 100, 2),
                'annual_return': round(metrics.annual_return * 100, 2) if hasattr(metrics, 'annual_return') else 0,
                'sharpe_ratio': round(metrics.sharpe_ratio, 2) if metrics.sharpe_ratio else None,
                'max_drawdown': round(metrics.max_drawdown * 100, 2),
                'profit_factor': round(metrics.profit_factor, 2) if metrics.profit_factor else None,
                'avg_r_multiple': round(metrics.avg_r_multiple, 2) if metrics.avg_r_multiple else None,
                # Professional metrics
                'sortino_ratio': round(metrics.sortino_ratio, 2) if hasattr(metrics, 'sortino_ratio') and metrics.sortino_ratio else None,
                'information_ratio': round(metrics.information_ratio, 2) if hasattr(metrics, 'information_ratio') and metrics.information_ratio else None,
                'max_adverse_excursion': round(metrics.max_adverse_excursion, 2) if hasattr(metrics, 'max_adverse_excursion') else None,
                'max_favorable_excursion': round(metrics.max_favorable_excursion, 2) if hasattr(metrics, 'max_favorable_excursion') else None,
                'avg_mae': round(metrics.avg_mae, 2) if hasattr(metrics, 'avg_mae') else None,
                'avg_mfe': round(metrics.avg_mfe, 2) if hasattr(metrics, 'avg_mfe') else None,
                'downside_deviation': round(metrics.downside_deviation * 100, 2) if hasattr(metrics, 'downside_deviation') else None,
                'tracking_error': round(metrics.tracking_error * 100, 2) if hasattr(metrics, 'tracking_error') else None,
                'benchmark_return': round(metrics.benchmark_return * 100, 2) if hasattr(metrics, 'benchmark_return') else None,
                'trades': [],
                'equity_curve': []
            }
            
            # Add trade details if available
            if hasattr(results, 'trades'):
                formatted_results['trades'] = [
                    {
                        'symbol': trade.symbol,
                        'entry_date': trade.entry_date.strftime('%Y-%m-%d'),
                        'exit_date': trade.exit_date.strftime('%Y-%m-%d') if trade.exit_date else None,
                        'entry_price': round(trade.entry_price, 2),
                        'exit_price': round(trade.exit_price, 2) if trade.exit_price else None,
                        'r_multiple': round(trade.r_multiple, 2) if trade.r_multiple else None,
                        'pnl': round(trade.pnl, 2) if trade.pnl else None,
                        'status': trade.status.value if trade.status else None
                    }
                    for trade in results.trades[:50]  # Limit to first 50 trades
                ]
            
            return formatted_results
            
        except Exception as e:
            return {'error': f'Failed to format single setup results: {str(e)}'}
    
    def _format_multiple_setup_results(self, results_dict: Dict) -> Dict:
        """Format multiple setup backtest results"""
        try:
            formatted_results = {
                'setups': {},
                'summary': {
                    'total_setups': len(results_dict),
                    'best_setup': None,
                    'best_sharpe': None,
                    'best_return': None
                }
            }
            
            best_sharpe = -999
            best_return = -999
            best_setup_sharpe = None
            best_setup_return = None
            
            for setup_name, setup_results in results_dict.items():
                if setup_results and hasattr(setup_results, 'performance_metrics'):
                    metrics = setup_results.performance_metrics
                    
                    formatted_setup = {
                        'total_trades': metrics.total_trades,
                        'win_rate': round(metrics.win_rate * 100, 2),
                        'total_return': round(metrics.total_return * 100, 2),
                        'annual_return': round(metrics.annual_return * 100, 2),
                        'sharpe_ratio': round(metrics.sharpe_ratio, 2) if metrics.sharpe_ratio else None,
                        'max_drawdown': round(metrics.max_drawdown * 100, 2),
                        'profit_factor': round(metrics.profit_factor, 2) if metrics.profit_factor else None
                    }
                    
                    formatted_results['setups'][setup_name] = formatted_setup
                    
                    # Track best performing setups
                    if metrics.sharpe_ratio and metrics.sharpe_ratio > best_sharpe:
                        best_sharpe = metrics.sharpe_ratio
                        best_setup_sharpe = setup_name
                    
                    if metrics.annual_return and metrics.annual_return > best_return:
                        best_return = metrics.annual_return
                        best_setup_return = setup_name
            
            formatted_results['summary']['best_setup'] = best_setup_sharpe
            formatted_results['summary']['best_sharpe'] = round(best_sharpe, 2) if best_sharpe > -999 else None
            formatted_results['summary']['best_return'] = round(best_return * 100, 2) if best_return > -999 else None
            
            return formatted_results
            
        except Exception as e:
            return {'error': f'Failed to format multiple setup results: {str(e)}'}
    
