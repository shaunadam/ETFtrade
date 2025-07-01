"""
Flask Configuration Module

Environment-based configuration for the ETF/Stock Trading System Flask app.
Supports .env files for easy development and production deployment.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Base configuration class with common settings"""
    
    # Flask Core Settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-change-in-production-2024'
    FLASK_DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    FLASK_PORT = int(os.environ.get('FLASK_PORT', 5000))
    
    # Database Configuration
    # Default to existing journal.db in parent directory
    DATABASE_PATH = os.environ.get('DATABASE_PATH') or \
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'journal.db')
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or f'sqlite:///{DATABASE_PATH}'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_pre_ping': True,
        'pool_recycle': 300,
    }
    
    # Trading System Settings
    # Cache and data settings
    CACHE_ENABLED = os.environ.get('CACHE_ENABLED', 'True').lower() == 'true'
    DATA_UPDATE_TIMEOUT = int(os.environ.get('DATA_UPDATE_TIMEOUT', 300))  # 5 minutes
    
    # Market regime settings
    REGIME_UPDATE_INTERVAL = int(os.environ.get('REGIME_UPDATE_INTERVAL', 3600))  # 1 hour
    
    # Backtesting settings
    BACKTEST_TIMEOUT = int(os.environ.get('BACKTEST_TIMEOUT', 1800))  # 30 minutes
    BACKTEST_MAX_CONCURRENT = int(os.environ.get('BACKTEST_MAX_CONCURRENT', 1))
    
    # UI/UX Settings
    ITEMS_PER_PAGE = int(os.environ.get('ITEMS_PER_PAGE', 25))
    CHART_THEME = os.environ.get('CHART_THEME', 'plotly_dark')
    BOOTSTRAP_THEME = os.environ.get('BOOTSTRAP_THEME', 'dark')
    
    # Security Settings
    WTF_CSRF_ENABLED = os.environ.get('WTF_CSRF_ENABLED', 'True').lower() == 'true'
    WTF_CSRF_TIME_LIMIT = int(os.environ.get('WTF_CSRF_TIME_LIMIT', 3600))
    
    # Session Settings
    PERMANENT_SESSION_LIFETIME = int(os.environ.get('SESSION_LIFETIME', 86400))  # 24 hours
    SESSION_COOKIE_SECURE = os.environ.get('SESSION_COOKIE_SECURE', 'False').lower() == 'true'
    SESSION_COOKIE_HTTPONLY = True
    
    # Logging Configuration
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()
    LOG_FILE = os.environ.get('LOG_FILE', 'flask_app.log')
    
    @staticmethod
    def init_app(app):
        """Initialize app-specific configuration"""
        pass

class DevelopmentConfig(Config):
    """Development environment configuration"""
    DEBUG = True
    FLASK_DEBUG = True
    SQLALCHEMY_ECHO = os.environ.get('SQLALCHEMY_ECHO', 'False').lower() == 'true'
    WTF_CSRF_ENABLED = False  # Disable CSRF for easier development
    
    @classmethod
    def init_app(cls, app):
        Config.init_app(app)
        
        # Development-specific initialization
        import logging
        logging.basicConfig(level=logging.DEBUG)

class ProductionConfig(Config):
    """Production environment configuration"""
    DEBUG = False
    FLASK_DEBUG = False
    
    # Enhanced security for production
    SESSION_COOKIE_SECURE = True
    WTF_CSRF_ENABLED = True
    
    # Database connection pooling for production
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_pre_ping': True,
        'pool_recycle': 300,
        'pool_size': 10,
        'max_overflow': 20
    }
    
    @classmethod
    def init_app(cls, app):
        Config.init_app(app)
        
        # Production logging setup
        import logging
        from logging.handlers import RotatingFileHandler
        
        if not os.path.exists('logs'):
            os.mkdir('logs')
        
        file_handler = RotatingFileHandler(
            'logs/etf_trading_system.log',
            maxBytes=10240000,  # 10MB
            backupCount=10
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)
        
        app.logger.setLevel(logging.INFO)
        app.logger.info('ETF Trading System Flask App startup')

class TestingConfig(Config):
    """Testing environment configuration"""
    TESTING = True
    WTF_CSRF_ENABLED = False
    
    # Use in-memory database for testing
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    
    # Shorter timeouts for testing
    DATA_UPDATE_TIMEOUT = 30
    BACKTEST_TIMEOUT = 60

# Configuration dictionary for easy access
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

# Allow easy access to current config
Config = config[os.environ.get('FLASK_ENV', 'development')]