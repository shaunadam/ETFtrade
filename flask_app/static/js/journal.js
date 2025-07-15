/**
 * Journal JavaScript Functions
 * 
 * Client-side functionality for the trade journal interface
 */

// Initialize journal functionality when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeJournal();
});

function initializeJournal() {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Auto-refresh positions every 5 minutes if on positions page
    if (window.location.pathname.includes('/journal/positions')) {
        setInterval(refreshPositionData, 300000); // 5 minutes
    }
}

// Trade management functions
function closeTradeModal(tradeId, symbol) {
    document.getElementById('closeTradeId').value = tradeId;
    document.getElementById('closeSymbol').value = symbol;
    document.getElementById('exitDate').value = new Date().toISOString().split('T')[0];
    
    const modal = new bootstrap.Modal(document.getElementById('closeTradeModal'));
    modal.show();
}

function submitCloseTrade() {
    const form = document.getElementById('closeTradeForm');
    const formData = new FormData(form);
    const tradeId = document.getElementById('closeTradeId').value;
    
    const data = {
        exit_price: parseFloat(formData.get('exit_price')),
        exit_date: formData.get('exit_date'),
        exit_reason: formData.get('exit_reason')
    };
    
    // Validate required fields
    if (!data.exit_price || data.exit_price <= 0) {
        showAlert('Exit price is required and must be greater than 0', 'danger');
        return;
    }
    
    // Show loading state
    const submitBtn = document.querySelector('#closeTradeModal .btn-danger');
    const originalText = submitBtn.innerHTML;
    submitBtn.innerHTML = '<i class="bi bi-hourglass-split"></i> Closing...';
    submitBtn.disabled = true;
    
    fetch(`/journal/api/trades/${tradeId}/close`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
        if (result.success) {
            showAlert(`Trade closed successfully. P&L: $${result.pnl_dollar?.toFixed(2) || '0.00'}`, 'success');
            setTimeout(() => location.reload(), 1500);
        } else {
            showAlert('Error closing trade: ' + result.error, 'danger');
            submitBtn.innerHTML = originalText;
            submitBtn.disabled = false;
        }
    })
    .catch(error => {
        showAlert('Error closing trade: ' + error.message, 'danger');
        submitBtn.innerHTML = originalText;
        submitBtn.disabled = false;
    });
}

// Trade details modal
function viewTradeDetails(tradeId) {
    // Show loading state
    const modal = document.getElementById('tradeDetailsModal');
    const content = document.getElementById('tradeDetailsContent');
    content.innerHTML = '<div class="text-center py-4"><i class="bi bi-hourglass-split"></i> Loading...</div>';
    
    const modalInstance = new bootstrap.Modal(modal);
    modalInstance.show();
    
    // Fetch trade details
    fetch(`/journal/api/trades?limit=1000`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                const trade = data.trades.find(t => t.id === tradeId);
                if (trade) {
                    displayTradeDetails(trade);
                } else {
                    content.innerHTML = '<div class="alert alert-warning">Trade not found</div>';
                }
            } else {
                content.innerHTML = '<div class="alert alert-danger">Error loading trade details</div>';
            }
        })
        .catch(error => {
            content.innerHTML = '<div class="alert alert-danger">Error: ' + error.message + '</div>';
        });
}

function displayTradeDetails(trade) {
    const formatCurrency = (value) => value ? '$' + value.toFixed(2) : 'N/A';
    const formatPercent = (value) => value ? value.toFixed(1) + '%' : 'N/A';
    const formatR = (value) => value ? value.toFixed(1) + 'R' : 'N/A';
    
    const pnlClass = trade.pnl_dollar > 0 ? 'text-success' : trade.pnl_dollar < 0 ? 'text-danger' : '';
    const statusClass = trade.status === 'open' ? 'bg-success' : trade.status === 'closed' ? 'bg-secondary' : 'bg-warning';
    
    const content = `
        <div class="row">
            <div class="col-md-6">
                <h6 class="text-info">Basic Information</h6>
                <table class="table table-sm table-dark">
                    <tr><td>Symbol:</td><td><strong>${trade.symbol}</strong></td></tr>
                    <tr><td>Instrument:</td><td>${trade.instrument_name}</td></tr>
                    <tr><td>Type:</td><td>${trade.instrument_type}</td></tr>
                    <tr><td>Sector:</td><td>${trade.sector || 'N/A'}</td></tr>
                    <tr><td>Setup:</td><td>${trade.setup_name || 'Manual'}</td></tr>
                    <tr><td>Status:</td><td><span class="badge ${statusClass}">${trade.status}</span></td></tr>
                </table>
            </div>
            <div class="col-md-6">
                <h6 class="text-info">Trade Details</h6>
                <table class="table table-sm table-dark">
                    <tr><td>Entry Date:</td><td>${trade.entry_date}</td></tr>
                    <tr><td>Exit Date:</td><td>${trade.exit_date || 'N/A'}</td></tr>
                    <tr><td>Days Held:</td><td>${trade.days_held}</td></tr>
                    <tr><td>Entry Price:</td><td>${formatCurrency(trade.entry_price)}</td></tr>
                    <tr><td>Exit Price:</td><td>${formatCurrency(trade.exit_price)}</td></tr>
                    <tr><td>Size:</td><td>${trade.size}</td></tr>
                </table>
            </div>
        </div>
        <div class="row">
            <div class="col-md-6">
                <h6 class="text-info">Risk Management</h6>
                <table class="table table-sm table-dark">
                    <tr><td>Stop Loss:</td><td>${formatCurrency(trade.stop_loss)}</td></tr>
                    <tr><td>Target Price:</td><td>${formatCurrency(trade.target_price)}</td></tr>
                    <tr><td>Planned R:</td><td>${formatR(trade.r_planned)}</td></tr>
                    <tr><td>Actual R:</td><td>${formatR(trade.r_actual)}</td></tr>
                </table>
            </div>
            <div class="col-md-6">
                <h6 class="text-info">Performance</h6>
                <table class="table table-sm table-dark">
                    <tr><td>P&L (Dollar):</td><td class="${pnlClass}">${formatCurrency(trade.pnl_dollar)}</td></tr>
                    <tr><td>P&L (%):</td><td class="${pnlClass}">${formatPercent(trade.pnl_percent)}</td></tr>
                    <tr><td>Commission:</td><td>${formatCurrency(trade.commission)}</td></tr>
                    <tr><td>Regime:</td><td>${trade.regime_at_entry || 'N/A'}</td></tr>
                </table>
            </div>
        </div>
        ${trade.entry_reason ? `
        <div class="row">
            <div class="col-12">
                <h6 class="text-info">Entry Reason</h6>
                <p class="text-light">${trade.entry_reason}</p>
            </div>
        </div>
        ` : ''}
        ${trade.exit_reason ? `
        <div class="row">
            <div class="col-12">
                <h6 class="text-info">Exit Reason</h6>
                <p class="text-light">${trade.exit_reason}</p>
            </div>
        </div>
        ` : ''}
        ${trade.notes ? `
        <div class="row">
            <div class="col-12">
                <h6 class="text-info">Notes</h6>
                <p class="text-light">${trade.notes}</p>
            </div>
        </div>
        ` : ''}
    `;
    
    document.getElementById('tradeDetailsContent').innerHTML = content;
}

// Risk calculator for add/edit forms
function updateRiskCalculator() {
    const entryPrice = parseFloat(document.getElementById('entry_price')?.value) || 0;
    const size = parseFloat(document.getElementById('size')?.value) || 0;
    const stopLoss = parseFloat(document.getElementById('stop_loss')?.value) || 0;
    
    const positionValue = entryPrice * size;
    const riskAmount = stopLoss > 0 ? Math.abs(entryPrice - stopLoss) * size : 0;
    const riskPercentage = positionValue > 0 ? (riskAmount / positionValue) * 100 : 0;
    
    // Update display elements if they exist
    const positionValueEl = document.getElementById('position-value');
    const riskAmountEl = document.getElementById('risk-amount');
    const riskPercentageEl = document.getElementById('risk-percentage');
    
    if (positionValueEl) positionValueEl.textContent = '$' + positionValue.toFixed(2);
    if (riskAmountEl) riskAmountEl.textContent = '$' + riskAmount.toFixed(2);
    if (riskPercentageEl) riskPercentageEl.textContent = riskPercentage.toFixed(1) + '%';
}

// Form validation
function validateTradeForm(form) {
    const requiredFields = ['symbol', 'entry_date', 'entry_price', 'size'];
    let isValid = true;
    
    requiredFields.forEach(fieldName => {
        const field = form.querySelector(`[name="${fieldName}"]`);
        if (field && !field.value.trim()) {
            showFieldError(field, `${fieldName.replace('_', ' ')} is required`);
            isValid = false;
        }
    });
    
    // Validate numeric fields
    const numericFields = ['entry_price', 'exit_price', 'size', 'stop_loss', 'target_price'];
    numericFields.forEach(fieldName => {
        const field = form.querySelector(`[name="${fieldName}"]`);
        if (field && field.value && parseFloat(field.value) <= 0) {
            showFieldError(field, `${fieldName.replace('_', ' ')} must be greater than 0`);
            isValid = false;
        }
    });
    
    return isValid;
}

function showFieldError(field, message) {
    // Remove existing error
    const existingError = field.parentNode.querySelector('.text-danger');
    if (existingError) existingError.remove();
    
    // Add error message
    const errorDiv = document.createElement('div');
    errorDiv.className = 'text-danger small mt-1';
    errorDiv.textContent = message;
    field.parentNode.appendChild(errorDiv);
    
    // Highlight field
    field.classList.add('is-invalid');
    
    // Remove error on input
    field.addEventListener('input', function() {
        field.classList.remove('is-invalid');
        if (errorDiv.parentNode) errorDiv.remove();
    }, { once: true });
}

// Utility functions
function showAlert(message, type = 'info') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Insert at top of main content
    const main = document.querySelector('main .container-fluid') || document.body;
    main.insertBefore(alertDiv, main.firstChild);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.remove();
        }
    }, 5000);
}

function formatCurrency(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    }).format(value);
}

function formatPercent(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'percent',
        minimumFractionDigits: 1,
        maximumFractionDigits: 1
    }).format(value / 100);
}

// Data refresh functions
function refreshPositionData() {
    fetch('/journal/api/positions')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                updatePositionCards(data.data);
            }
        })
        .catch(error => {
            console.error('Error refreshing position data:', error);
        });
}

function updatePositionCards(positionData) {
    // Update summary cards if they exist
    const totalPositionsEl = document.querySelector('[data-metric="total-positions"]');
    const totalValueEl = document.querySelector('[data-metric="total-value"]');
    const totalPnlEl = document.querySelector('[data-metric="total-pnl"]');
    
    if (totalPositionsEl && positionData.summary) {
        totalPositionsEl.textContent = positionData.summary.total_positions;
    }
    if (totalValueEl && positionData.summary) {
        totalValueEl.textContent = formatCurrency(positionData.summary.total_value);
    }
    if (totalPnlEl && positionData.summary) {
        totalPnlEl.textContent = formatCurrency(positionData.summary.total_pnl);
        totalPnlEl.className = positionData.summary.total_pnl >= 0 ? 'text-success' : 'text-danger';
    }
}

// Export functions for global use
window.JournalJS = {
    closeTradeModal,
    submitCloseTrade,
    viewTradeDetails,
    updateRiskCalculator,
    validateTradeForm,
    showAlert,
    formatCurrency,
    formatPercent,
    refreshPositionData
};