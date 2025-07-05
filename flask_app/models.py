"""
SQLAlchemy Models for ETF/Stock Trading System

Maps to existing journal.db schema with Flask-SQLAlchemy ORM.
Supports both ETFs and individual stocks with future-proof design.
"""

from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import CheckConstraint, UniqueConstraint

db = SQLAlchemy()

class Instrument(db.Model):
    """Instruments table - handles both ETFs and stocks"""
    __tablename__ = 'instruments'
    
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(20), unique=True, nullable=False, index=True)
    name = db.Column(db.String(200), nullable=False)
    type = db.Column(db.String(10), nullable=False, index=True)
    sector = db.Column(db.String(100))
    theme = db.Column(db.String(100))
    geography = db.Column(db.String(50))
    leverage = db.Column(db.String(10))
    volatility_profile = db.Column(db.String(20))
    avg_volume_req = db.Column(db.String(20))
    tags = db.Column(db.Text)
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    trades = db.relationship('Trade', backref='instrument', lazy='dynamic')
    correlations_as_instrument1 = db.relationship('Correlation', 
                                                  foreign_keys='Correlation.instrument1_id',
                                                  backref='instrument1', lazy='dynamic')
    correlations_as_instrument2 = db.relationship('Correlation',
                                                  foreign_keys='Correlation.instrument2_id', 
                                                  backref='instrument2', lazy='dynamic')
    
    __table_args__ = (
        CheckConstraint("type IN ('ETF', 'Stock', 'ETN')", name='check_instrument_type'),
    )
    
    def __repr__(self):
        return f'<Instrument {self.symbol}: {self.name}>'
    
    @property
    def tag_list(self):
        """Return tags as a list"""
        return self.tags.split(',') if self.tags else []
    
    def is_etf(self):
        """Check if instrument is an ETF"""
        return self.type in ('ETF', 'ETN')
    
    def is_stock(self):
        """Check if instrument is a stock"""
        return self.type == 'Stock'

class Setup(db.Model):
    """Trade setups table"""
    __tablename__ = 'setups'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    description = db.Column(db.Text)
    parameters = db.Column(db.Text)  # JSON string
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    trades = db.relationship('Trade', backref='setup', lazy='dynamic')
    
    def __repr__(self):
        return f'<Setup {self.name}>'

class Trade(db.Model):
    """Trades table"""
    __tablename__ = 'trades'
    
    id = db.Column(db.Integer, primary_key=True)
    instrument_id = db.Column(db.Integer, db.ForeignKey('instruments.id'), nullable=False, index=True)
    setup_id = db.Column(db.Integer, db.ForeignKey('setups.id'), index=True)
    entry_date = db.Column(db.Date, nullable=False, index=True)
    exit_date = db.Column(db.Date)
    size = db.Column(db.Float, nullable=False)
    entry_price = db.Column(db.Float, nullable=False)
    exit_price = db.Column(db.Float)
    r_planned = db.Column(db.Float)
    r_actual = db.Column(db.Float)
    notes = db.Column(db.Text)
    regime_at_entry = db.Column(db.String(200))
    status = db.Column(db.String(20), default='open', index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    snapshots = db.relationship('Snapshot', backref='trade', lazy='dynamic', cascade='all, delete-orphan')
    
    __table_args__ = (
        CheckConstraint("status IN ('open', 'closed', 'cancelled')", name='check_trade_status'),
    )
    
    def __repr__(self):
        return f'<Trade {self.id}: {self.instrument.symbol} {self.status}>'
    
    @property
    def is_open(self):
        """Check if trade is still open"""
        return self.status == 'open'
    
    @property
    def current_pnl(self):
        """Calculate current P&L (requires current price)"""
        if not self.is_open or not self.exit_price:
            return None
        return (self.exit_price - self.entry_price) * self.size
    
    @property
    def days_held(self):
        """Calculate days held in position"""
        end_date = self.exit_date or datetime.utcnow().date()
        return (end_date - self.entry_date).days

class Snapshot(db.Model):
    """Trade snapshots table"""
    __tablename__ = 'snapshots'
    
    id = db.Column(db.Integer, primary_key=True)
    trade_id = db.Column(db.Integer, db.ForeignKey('trades.id'), nullable=False, index=True)
    date = db.Column(db.Date, nullable=False)
    price = db.Column(db.Float, nullable=False)
    notes = db.Column(db.Text)
    chart_path = db.Column(db.String(500))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Snapshot {self.id}: Trade {self.trade_id} on {self.date}>'

class MarketRegime(db.Model):
    """Market regimes table"""
    __tablename__ = 'market_regimes'
    
    date = db.Column(db.Date, primary_key=True)
    volatility_regime = db.Column(db.String(20), nullable=False)
    trend_regime = db.Column(db.String(20), nullable=False)
    sector_rotation = db.Column(db.String(20), nullable=False)
    risk_on_off = db.Column(db.String(20), nullable=False)
    vix_level = db.Column(db.Float)
    spy_vs_sma200 = db.Column(db.Float)
    growth_value_ratio = db.Column(db.Float)
    risk_on_off_ratio = db.Column(db.Float)
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<MarketRegime {self.date}: {self.volatility_regime}/{self.trend_regime}>'
    
    @property
    def regime_summary(self):
        """Get a summary string of the regime"""
        return f"{self.volatility_regime} Vol, {self.trend_regime} Trend, {self.risk_on_off}"

class Correlation(db.Model):
    """Correlation tracking table"""
    __tablename__ = 'correlations'
    
    id = db.Column(db.Integer, primary_key=True)
    instrument1_id = db.Column(db.Integer, db.ForeignKey('instruments.id'), nullable=False)
    instrument2_id = db.Column(db.Integer, db.ForeignKey('instruments.id'), nullable=False)
    correlation_30d = db.Column(db.Float)
    correlation_90d = db.Column(db.Float)
    correlation_252d = db.Column(db.Float)
    date_calculated = db.Column(db.Date, nullable=False, index=True)
    
    __table_args__ = (
        UniqueConstraint('instrument1_id', 'instrument2_id', 'date_calculated', 
                        name='unique_correlation_per_date'),
    )
    
    def __repr__(self):
        return f'<Correlation {self.instrument1.symbol}-{self.instrument2.symbol} on {self.date_calculated}>'

class PriceData(db.Model):
    """Price data caching table"""
    __tablename__ = 'price_data'
    
    symbol = db.Column(db.String(20), primary_key=True)
    date = db.Column(db.Date, primary_key=True)
    open = db.Column(db.Float, nullable=False)
    high = db.Column(db.Float, nullable=False)
    low = db.Column(db.Float, nullable=False)
    close = db.Column(db.Float, nullable=False)
    volume = db.Column(db.Integer, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<PriceData {self.symbol} {self.date}: ${self.close:.2f}>'

class Indicator(db.Model):
    """Technical indicators caching table"""
    __tablename__ = 'indicators'  
    
    symbol = db.Column(db.String(20), primary_key=True)
    date = db.Column(db.Date, primary_key=True)
    indicator_name = db.Column(db.String(50), primary_key=True)
    value = db.Column(db.Float, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Indicator {self.symbol} {self.indicator_name} {self.date}: {self.value:.4f}>'

class RateMetadata(db.Model):
    """Risk-free rate metadata table"""
    __tablename__ = 'rate_metadata'
    
    symbol = db.Column(db.String(20), primary_key=True)
    description = db.Column(db.String(200), nullable=False)
    source = db.Column(db.String(100), nullable=False)
    data_type = db.Column(db.String(20), nullable=False)
    priority = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    rates = db.relationship('RiskFreeRate', backref='metadata', lazy='dynamic')
    
    __table_args__ = (
        CheckConstraint("data_type IN ('yield', 'etf_proxy')", name='check_rate_data_type'),
    )
    
    def __repr__(self):
        return f'<RateMetadata {self.symbol}: {self.description}>'

class RiskFreeRate(db.Model):
    """Risk-free rates data table"""
    __tablename__ = 'risk_free_rates'
    
    symbol = db.Column(db.String(20), db.ForeignKey('rate_metadata.symbol'), primary_key=True)
    date = db.Column(db.Date, primary_key=True)
    rate_type = db.Column(db.String(20), nullable=False)
    value = db.Column(db.Float, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        CheckConstraint("rate_type IN ('yield', 'etf_return')", name='check_rate_type'),
    )
    
    def __repr__(self):
        return f'<RiskFreeRate {self.symbol} {self.date}: {self.value:.4f}%>'

# Utility functions for common queries
def get_instruments_by_type(instrument_type=None):
    """Get instruments filtered by type"""
    query = Instrument.query
    if instrument_type:
        query = query.filter(Instrument.type == instrument_type)
    return query.order_by(Instrument.symbol).all()

def get_open_trades():
    """Get all open trades with instrument details"""
    return Trade.query.filter(Trade.status == 'open').join(Instrument).all()

def get_latest_regime():
    """Get the most recent market regime"""
    return MarketRegime.query.order_by(MarketRegime.date.desc()).first()

def get_price_data(symbol, start_date=None, end_date=None):
    """Get price data for a symbol within date range"""
    query = PriceData.query.filter(PriceData.symbol == symbol)
    if start_date:
        query = query.filter(PriceData.date >= start_date)
    if end_date:
        query = query.filter(PriceData.date <= end_date)
    return query.order_by(PriceData.date).all()

def get_indicator_data(symbol, indicator_name, start_date=None, end_date=None):
    """Get indicator data for a symbol within date range"""
    query = Indicator.query.filter(
        Indicator.symbol == symbol,
        Indicator.indicator_name == indicator_name
    )
    if start_date:
        query = query.filter(Indicator.date >= start_date)
    if end_date:
        query = query.filter(Indicator.date <= end_date)
    return query.order_by(Indicator.date).all()