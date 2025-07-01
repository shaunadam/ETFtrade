#!/usr/bin/env python3
"""
Flask Web Application for ETF/Stock Trading System

Main application entry point that integrates all trading system modules
into a unified web interface with dark theme and interactive charts.
"""

import os
import sys
from flask import Flask, render_template, jsonify
from flask_sqlalchemy import SQLAlchemy

# Add parent directory to path to import existing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Initialize Flask extensions
db = SQLAlchemy()

def create_app():
    """Application factory pattern for Flask app creation"""
    app = Flask(__name__)
    
    # Load configuration
    from config import Config
    app.config.from_object(Config)
    
    # Initialize extensions
    db.init_app(app)
    
    # Register blueprints
    from blueprints.dashboard import dashboard_bp
    from blueprints.data import data_bp
    from blueprints.screener import screener_bp
    from blueprints.journal import journal_bp
    from blueprints.backtest import backtest_bp
    from blueprints.regime import regime_bp
    
    app.register_blueprint(dashboard_bp, url_prefix='/')
    app.register_blueprint(data_bp, url_prefix='/data')
    app.register_blueprint(screener_bp, url_prefix='/screener')
    app.register_blueprint(journal_bp, url_prefix='/journal')
    app.register_blueprint(backtest_bp, url_prefix='/backtest')
    app.register_blueprint(regime_bp, url_prefix='/regime')
    
    # Global error handlers
    @app.errorhandler(404)
    def not_found_error(error):
        return render_template('errors/404.html'), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        db.session.rollback()
        return render_template('errors/500.html'), 500
    
    # API health check
    @app.route('/api/health')
    def health_check():
        """Health check endpoint for monitoring"""
        return jsonify({
            'status': 'healthy',
            'app': 'ETF Trading System',
            'version': '1.0.0'
        })
    
    return app

# Create app instance
app = create_app()

if __name__ == '__main__':
    # Development server
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    port = int(os.getenv('FLASK_PORT', 5000))
    
    print("\n" + "="*60)
    print("🚀 ETF/Stock Trading System - Flask Web Interface")
    print("="*60)
    print(f"📊 Dashboard: http://localhost:{port}/")
    print(f"🔍 Screener:  http://localhost:{port}/screener/")
    print(f"📝 Journal:   http://localhost:{port}/journal/")
    print(f"⚡ Backtest:  http://localhost:{port}/backtest/")
    print(f"🌡️  Regime:    http://localhost:{port}/regime/")
    print(f"💾 Data:      http://localhost:{port}/data/")
    print("="*60)
    
    app.run(
        host='0.0.0.0',  # Allow external connections
        port=port,
        debug=debug_mode,
        use_reloader=debug_mode
    )