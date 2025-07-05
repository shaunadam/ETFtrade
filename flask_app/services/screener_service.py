"""
Screener Service Layer

Integrates screener.py and trade_setups.py with Flask application.
Provides web-friendly interface for ETF/stock screening operations.
"""

import sys
import os
from datetime import datetime
from typing import Dict, List, Optional

# Add parent directory to import CLI modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import trade_setups
from models import db, Instrument, Setup

class ScreenerService:
    """Service for screening operations in Flask app"""
    
    def __init__(self):
        # Import screener module dynamically to avoid import issues
        try:
            import screener
            self.screener_module = screener
            # Initialize ETF Screener instance with correct path to journal.db
            db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'journal.db')
            self.etf_screener = screener.ETFScreener(db_path)
        except ImportError:
            self.screener_module = None
            self.etf_screener = None
    
    def get_available_setups(self) -> List[Dict]:
        """Get all available trade setups"""
        try:
            # Get setups from database
            setups = Setup.query.order_by(Setup.name).all()
            
            return [
                {
                    'id': setup.id,
                    'name': setup.name,
                    'display_name': setup.name.replace('_', ' ').title(),
                    'description': setup.description,
                    'parameters': setup.parameters
                }
                for setup in setups
            ]
            
        except Exception as e:
            return [{'error': f'Failed to get setups: {str(e)}'}]
    
    def run_screening(self, setup_name: str = None, 
                     instrument_type: str = None,
                     regime_filter: bool = True,
                     min_confidence: float = 0.5) -> Dict:
        """Run screening with specified parameters"""
        try:
            if not self.etf_screener:
                return {
                    'results': [],
                    'total_screened': 0,
                    'total_matches': 0,
                    'error': 'ETF Screener not initialized'
                }
            
            # Map instrument types to CLI format
            instrument_types = None
            if instrument_type:
                if instrument_type.lower() == 'etf':
                    instrument_types = ['ETF', 'ETN']
                elif instrument_type.lower() == 'stock':
                    instrument_types = ['Stock']
            
            # Use CLI screener to get signals
            signals = self.etf_screener.screen_instruments(
                setup_filter=setup_name,
                min_confidence=min_confidence,
                max_signals=100,
                regime_filter=regime_filter,
                update_data=False, # Don't update data in web interface
                instrument_types=instrument_types
            )
            
            # Convert TradeSignal objects to web-friendly format
            screening_results = []
            for signal in signals:
                # Get instrument info from database
                instrument = Instrument.query.filter(Instrument.symbol == signal.symbol).first()
                
                result = {
                    'setup_name': signal.setup_type.value,
                    'setup_display_name': signal.setup_type.value.replace('_', ' ').title(),
                    'confidence': signal.confidence,
                    'signal': signal.signal_strength.value,
                    'entry_price': signal.entry_price,
                    'stop_loss': signal.stop_loss,
                    'target_price': signal.target_price,
                    'notes': signal.notes or f'Signal from {signal.setup_type.value}',
                    'instrument': {
                        'id': instrument.id if instrument else None,
                        'symbol': signal.symbol,
                        'name': instrument.name if instrument else signal.symbol,
                        'type': instrument.type if instrument else 'Unknown',
                        'sector': instrument.sector if instrument else None,
                        'theme': instrument.theme if instrument else None
                    },
                    'screened_at': datetime.now().isoformat()
                }
                screening_results.append(result)
            
            return {
                'results': screening_results,
                'total_screened': len(screening_results), # CLI does filtering internally
                'total_matches': len(screening_results),
                'screening_parameters': {
                    'setup_name': setup_name,
                    'instrument_type': instrument_type,
                    'regime_filter': regime_filter,
                    'min_confidence': min_confidence
                },
                'screened_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'results': [],
                'total_screened': 0,
                'total_matches': 0,
                'error': f'Screening failed: {str(e)}'
            }
    
    def analyze_symbol(self, symbol: str, setup_name: str = None) -> Dict:
        """Analyze a specific symbol for trade setups"""
        try:
            if not self.etf_screener:
                return {
                    'symbol': symbol,
                    'error': 'ETF Screener not initialized'
                }
                
            # Get instrument info
            instrument = Instrument.query.filter(Instrument.symbol == symbol).first()
            
            if not instrument:
                return {
                    'symbol': symbol,
                    'error': 'Symbol not found in database'
                }
            
            # Use CLI screener to analyze just this symbol
            instrument_types = ['ETF', 'ETN'] if instrument.type in ['ETF', 'ETN'] else ['Stock']
            
            signals = self.etf_screener.screen_instruments(
                setup_filter=setup_name,
                min_confidence=0.0, # Get all signals for analysis
                max_signals=100,
                regime_filter=False, # Don't filter for single symbol analysis
                update_data=False,
                instrument_types=instrument_types
            )
            
            # Filter signals for this specific symbol
            symbol_signals = [s for s in signals if s.symbol == symbol]
            
            analysis_results = []
            for signal in symbol_signals:
                result = {
                    'setup_name': signal.setup_type.value,
                    'setup_display_name': signal.setup_type.value.replace('_', ' ').title(),
                    'confidence': signal.confidence,
                    'signal': signal.signal_strength.value,
                    'entry_price': signal.entry_price,
                    'stop_loss': signal.stop_loss,
                    'target_price': signal.target_price,
                    'notes': signal.notes or f'Signal from {signal.setup_type.value}'
                }
                analysis_results.append(result)
            
            return {
                'symbol': symbol,
                'instrument': {
                    'id': instrument.id,
                    'symbol': instrument.symbol,
                    'name': instrument.name,
                    'type': instrument.type,
                    'sector': instrument.sector,
                    'theme': instrument.theme
                },
                'analysis': {
                    'signals': analysis_results,
                    'total_signals': len(analysis_results),
                    'best_signal': analysis_results[0] if analysis_results else None
                },
                'analyzed_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'symbol': symbol,
                'error': f'Analysis failed: {str(e)}'
            }
    
    def _get_screening_universe(self, instrument_type: str = None) -> List[Dict]:
        """Get instruments for screening"""
        try:
            query = Instrument.query
            
            if instrument_type:
                if instrument_type.lower() == 'etf':
                    query = query.filter(Instrument.type.in_(['ETF', 'ETN']))
                elif instrument_type.lower() == 'stock':
                    query = query.filter(Instrument.type == 'Stock')
            
            instruments = query.order_by(Instrument.symbol).all()
            
            return [
                {
                    'id': inst.id,
                    'symbol': inst.symbol,
                    'name': inst.name,
                    'type': inst.type,
                    'sector': inst.sector,
                    'theme': inst.theme
                }
                for inst in instruments
            ]
            
        except Exception as e:
            return []
    
    def _analyze_single_setup(self, symbol: str, setup_name: str, 
                            min_confidence: float = 0.5) -> Optional[Dict]:
        """Analyze a symbol for a specific trade setup - DEPRECATED: Use CLI integration instead"""
        # This method is no longer used since we now use the CLI screener directly
        # Kept for backward compatibility
        return None
    
    def _analyze_all_setups(self, symbol: str, min_confidence: float = 0.5) -> Dict:
        """Analyze a symbol for all available setups"""
        try:
            setups = Setup.query.all()
            best_setup = None
            best_confidence = 0
            all_results = []
            
            for setup in setups:
                result = self._analyze_single_setup(symbol, setup.name, min_confidence)
                if result:
                    all_results.append(result)
                    if result['confidence'] > best_confidence:
                        best_confidence = result['confidence']
                        best_setup = result
            
            return {
                'best_setup': best_setup,
                'confidence': best_confidence,
                'all_setups': all_results,
                'total_setups_analyzed': len(setups)
            }
            
        except Exception as e:
            return {
                'error': f'Failed to analyze all setups: {str(e)}'
            }