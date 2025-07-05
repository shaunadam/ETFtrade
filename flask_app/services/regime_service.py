"""
Market Regime Service Layer

Integrates regime_detection.py with Flask application.
Provides web-friendly interface for market regime analysis.
"""

import sys
import os
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional

# Add parent directory to import CLI modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from regime_detection import RegimeDetector
from models import db, MarketRegime

class RegimeService:
    """Service for market regime operations in Flask app"""
    
    def __init__(self):
        # Use the correct database path - same as Flask config  
        import os
        db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'journal.db')
        self.regime_detector = RegimeDetector(db_path)
    
    def get_current_regime(self) -> Dict:
        """Get current market regime"""
        try:
            # Try to get from database first
            latest_regime = MarketRegime.query.order_by(MarketRegime.date.desc()).first()
            
            # If no recent data or data is stale (>2 days), detect fresh regime
            if not latest_regime or (datetime.now().date() - latest_regime.date).days > 2:
                regime_data = self.regime_detector.detect_current_regime()
                
                if regime_data:
                    # Convert RegimeData to dict format for web use
                    regime_dict = self._regime_data_to_dict(regime_data)
                    # Save to database
                    self._save_regime_to_db(regime_dict)
                    return regime_dict
            else:
                # Return cached regime data
                return {
                    'date': latest_regime.date.isoformat(),
                    'volatility_regime': latest_regime.volatility_regime,
                    'trend_regime': latest_regime.trend_regime,
                    'sector_rotation': latest_regime.sector_rotation,
                    'risk_on_off': latest_regime.risk_on_off,
                    'vix_level': latest_regime.vix_level,
                    'spy_vs_sma200': latest_regime.spy_vs_sma200,
                    'growth_value_ratio': latest_regime.growth_value_ratio,
                    'risk_on_off_ratio': latest_regime.risk_on_off_ratio,
                    'notes': latest_regime.notes,
                    'regime_summary': latest_regime.regime_summary
                }
            
            # Fallback to fresh detection
            regime_data = self.regime_detector.detect_current_regime()
            if regime_data:
                return self._regime_data_to_dict(regime_data)
            else:
                return {'error': 'Unable to detect current regime'}
            
        except Exception as e:
            return {'error': f'Failed to get current regime: {str(e)}'}
    
    def get_regime_history(self, days: int = 90) -> List[Dict]:
        """Get historical regime data"""
        try:
            start_date = datetime.now().date() - timedelta(days=days)
            
            regimes = MarketRegime.query.filter(
                MarketRegime.date >= start_date
            ).order_by(MarketRegime.date.desc()).all()
            
            return [
                {
                    'date': regime.date.isoformat(),
                    'volatility_regime': regime.volatility_regime,
                    'trend_regime': regime.trend_regime,
                    'sector_rotation': regime.sector_rotation,
                    'risk_on_off': regime.risk_on_off,
                    'vix_level': regime.vix_level,
                    'spy_vs_sma200': regime.spy_vs_sma200,
                    'growth_value_ratio': regime.growth_value_ratio,
                    'risk_on_off_ratio': regime.risk_on_off_ratio,
                    'regime_summary': regime.regime_summary
                }
                for regime in regimes
            ]
            
        except Exception as e:
            return [{'error': f'Failed to get regime history: {str(e)}'}]
    
    def update_regime_data(self) -> Dict:
        """Update regime data by running fresh detection"""
        try:
            regime_data = self.regime_detector.detect_current_regime()
            
            if regime_data:
                # Convert to dict format
                regime_dict = self._regime_data_to_dict(regime_data)
                # Save to database
                self._save_regime_to_db(regime_dict)
                
                return {
                    'success': True,
                    'message': 'Regime data updated successfully',
                    'data': regime_dict,
                    'updated_at': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to detect regime',
                    'data': None
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to update regime data: {str(e)}'
            }
    
    def get_regime_analysis(self) -> Dict:
        """Get comprehensive regime analysis"""
        try:
            # Get current regime
            current_regime = self.get_current_regime()
            
            # Get recent regime history (30 days)
            recent_history = self.get_regime_history(30)
            
            # Calculate regime stability (how often regime has changed)
            regime_changes = 0
            if len(recent_history) > 1:
                for i in range(1, len(recent_history)):
                    curr = recent_history[i-1]
                    prev = recent_history[i]
                    if (curr.get('volatility_regime') != prev.get('volatility_regime') or
                        curr.get('trend_regime') != prev.get('trend_regime') or
                        curr.get('risk_on_off') != prev.get('risk_on_off')):
                        regime_changes += 1
            
            stability_score = max(0, 1 - (regime_changes / max(1, len(recent_history))))
            
            # Get regime distribution
            regime_distribution = {}
            for regime in recent_history:
                key = f"{regime.get('volatility_regime', 'Unknown')}-{regime.get('trend_regime', 'Unknown')}"
                regime_distribution[key] = regime_distribution.get(key, 0) + 1
            
            return {
                'current_regime': current_regime,
                'recent_history': recent_history[:10],  # Last 10 days
                'stability_score': round(stability_score, 2),
                'regime_changes_30d': regime_changes,
                'regime_distribution': regime_distribution,
                'analysis_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'error': f'Failed to get regime analysis: {str(e)}'}
    
    def _save_regime_to_db(self, regime_data: Dict) -> bool:
        """Save regime data to database"""
        try:
            today = datetime.now().date()
            
            # Check if regime for today already exists
            existing = MarketRegime.query.filter(MarketRegime.date == today).first()
            
            if existing:
                # Update existing
                existing.volatility_regime = regime_data.get('volatility_regime', '')
                existing.trend_regime = regime_data.get('trend_regime', '')
                existing.sector_rotation = regime_data.get('sector_rotation', '')
                existing.risk_on_off = regime_data.get('risk_on_off', '')
                existing.vix_level = regime_data.get('vix_level')
                existing.spy_vs_sma200 = regime_data.get('spy_vs_sma200')
                existing.growth_value_ratio = regime_data.get('growth_value_ratio')
                existing.risk_on_off_ratio = regime_data.get('risk_on_off_ratio')
                existing.notes = regime_data.get('notes', '')
            else:
                # Create new
                new_regime = MarketRegime(
                    date=today,
                    volatility_regime=regime_data.get('volatility_regime', ''),
                    trend_regime=regime_data.get('trend_regime', ''),
                    sector_rotation=regime_data.get('sector_rotation', ''),
                    risk_on_off=regime_data.get('risk_on_off', ''),
                    vix_level=regime_data.get('vix_level'),
                    spy_vs_sma200=regime_data.get('spy_vs_sma200'),
                    growth_value_ratio=regime_data.get('growth_value_ratio'),
                    risk_on_off_ratio=regime_data.get('risk_on_off_ratio'),
                    notes=regime_data.get('notes', '')
                )
                db.session.add(new_regime)
            
            db.session.commit()
            return True
            
        except Exception as e:
            db.session.rollback()
            print(f"Failed to save regime to database: {e}")
            return False
    
    def _regime_data_to_dict(self, regime_data) -> Dict:
        """Convert RegimeData object to dictionary for web use"""
        return {
            'date': regime_data.date.isoformat(),
            'volatility_regime': regime_data.volatility_regime.value,
            'trend_regime': regime_data.trend_regime.value, 
            'sector_rotation': regime_data.sector_rotation.value,
            'risk_on_off': regime_data.risk_sentiment.value,
            'vix_level': regime_data.vix_level,
            'spy_vs_sma200': regime_data.spy_vs_sma200,
            'growth_value_ratio': regime_data.growth_value_ratio,
            'risk_on_off_ratio': regime_data.risk_on_off_ratio,
            'notes': '',
            'regime_summary': f"{regime_data.volatility_regime.value} Vol, {regime_data.trend_regime.value} Trend, {regime_data.risk_sentiment.value}"
        }