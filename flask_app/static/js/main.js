/**
 * Main JavaScript for ETF/Stock Trading System Flask App
 * Handles common functionality, charts, and UI interactions
 */

// Global application state
window.TradingApp = {
    charts: {},
    intervals: {},
    config: {
        refreshInterval: 60000, // 1 minute
        chartRefreshInterval: 300000, // 5 minutes
        apiTimeout: 30000 // 30 seconds
    }
};

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

/**
 * Initialize the application
 */
function initializeApp() {
    console.log('🚀 Initializing ETF/Stock Trading System...');
    
    // Initialize tooltips
    initializeTooltips();
    
    // Initialize DataTables
    initializeDataTables();
    
    // Set up navigation active states
    setActiveNavigation();
    
    // Start auto-refresh if on dashboard
    if (window.location.pathname === '/' || window.location.pathname === '/dashboard/') {
        startDashboardRefresh();
    }
    
    // Initialize chart refresh for chart pages
    if (document.querySelectorAll('.chart-container').length > 0) {
        startChartRefresh();
    }
    
    console.log('✅ Application initialized successfully');
}

/**
 * Initialize Bootstrap tooltips
 */
function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

/**
 * Initialize DataTables with dark theme
 */
function initializeDataTables() {
    // Check if DataTables is available
    if (typeof $.fn.DataTable === 'undefined') {
        console.log('DataTables not loaded, skipping initialization');
        return;
    }
    
    // Initialize any table with class 'datatable'
    $('.datatable').each(function() {
        if (!$.fn.DataTable.isDataTable(this)) {
            $(this).DataTable({
                pageLength: 25,
                order: [[0, "desc"]],
                responsive: true,
                language: {
                    search: "Search:",
                    lengthMenu: "Show _MENU_ entries",
                    info: "Showing _START_ to _END_ of _TOTAL_ entries",
                    paginate: {
                        first: "First",
                        last: "Last",
                        next: "Next",
                        previous: "Previous"
                    }
                },
                dom: '<"row"<"col-sm-12 col-md-6"l><"col-sm-12 col-md-6"f>>' +
                     '<"row"<"col-sm-12"tr>>' +
                     '<"row"<"col-sm-12 col-md-5"i><"col-sm-12 col-md-7"p>>',
            });
        }
    });
}

/**
 * Set active navigation state based on current URL
 */
function setActiveNavigation() {
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.navbar-nav .nav-link');
    
    navLinks.forEach(link => {
        link.classList.remove('active');
        
        // Check if current path matches link href
        const linkPath = new URL(link.href).pathname;
        if (currentPath.startsWith(linkPath) && linkPath !== '/') {
            link.classList.add('active');
        } else if (currentPath === '/' && linkPath === '/') {
            link.classList.add('active');
        }
    });
}

/**
 * Start dashboard auto-refresh
 */
function startDashboardRefresh() {
    console.log('📊 Starting dashboard auto-refresh...');
    
    TradingApp.intervals.dashboard = setInterval(() => {
        refreshDashboardData();
    }, TradingApp.config.refreshInterval);
}

/**
 * Start chart auto-refresh
 */
function startChartRefresh() {
    console.log('📈 Starting chart auto-refresh...');
    
    TradingApp.intervals.charts = setInterval(() => {
        refreshAllCharts();
    }, TradingApp.config.chartRefreshInterval);
}

/**
 * Refresh dashboard data
 */
async function refreshDashboardData() {
    try {
        const response = await fetch('/api/dashboard/status', {
            timeout: TradingApp.config.apiTimeout
        });
        
        if (response.ok) {
            const data = await response.json();
            updateDashboardElements(data);
            console.log('✅ Dashboard data refreshed');
        }
    } catch (error) {
        console.error('❌ Error refreshing dashboard:', error);
    }
}

/**
 * Update dashboard elements with new data
 */
function updateDashboardElements(data) {
    // Update status indicators
    if (data.status) {
        updateStatusIndicators(data.status);
    }
    
    // Update regime information
    if (data.regime) {
        updateRegimeDisplay(data.regime);
    }
    
    // Update last updated timestamp
    document.getElementById('lastUpdated').textContent = new Date().toLocaleString();
}

/**
 * Update status indicators
 */
function updateStatusIndicators(status) {
    const indicators = {
        'database': status.database_connected,
        'cache': status.cache_healthy,
        'data': status.data_fresh
    };
    
    Object.entries(indicators).forEach(([key, value]) => {
        const indicator = document.querySelector(`[data-status="${key}"]`);
        if (indicator) {
            indicator.className = `status-indicator ${value ? 'success' : 'danger'}`;
        }
    });
}

/**
 * Update regime display
 */
function updateRegimeDisplay(regime) {
    Object.entries(regime).forEach(([key, data]) => {
        const element = document.querySelector(`[data-regime="${key}"]`);
        if (element && data.label) {
            element.textContent = data.label;
            element.className = `badge regime-badge regime-${data.value}`;
        }
    });
}

/**
 * Refresh all charts on the page
 */
function refreshAllCharts() {
    Object.keys(TradingApp.charts).forEach(chartId => {
        const chart = TradingApp.charts[chartId];
        if (chart && chart.refresh) {
            chart.refresh();
        }
    });
}

/**
 * Create a Plotly chart with dark theme
 */
function createChart(containerId, data, layout = {}, config = {}) {
    // Merge with dark theme layout
    const darkLayout = {
        ...window.PLOTLY_LAYOUT_DARK,
        ...layout
    };
    
    // Merge with global config
    const chartConfig = {
        ...window.PLOTLY_CONFIG,
        ...config
    };
    
    // Create the chart
    Plotly.newPlot(containerId, data, darkLayout, chartConfig);
    
    // Store chart reference for refresh
    TradingApp.charts[containerId] = {
        element: document.getElementById(containerId),
        data: data,
        layout: darkLayout,
        config: chartConfig,
        refresh: function() {
            Plotly.redraw(containerId);
        }
    };
    
    return TradingApp.charts[containerId];
}

/**
 * Update an existing chart
 */
function updateChart(containerId, newData, newLayout = {}) {
    if (TradingApp.charts[containerId]) {
        const chart = TradingApp.charts[containerId];
        chart.data = newData;
        chart.layout = { ...chart.layout, ...newLayout };
        
        Plotly.react(containerId, chart.data, chart.layout, chart.config);
    }
}

/**
 * Show loading overlay
 */
function showLoading(message = 'Loading...') {
    const overlay = document.createElement('div');
    overlay.className = 'loading-overlay';
    overlay.id = 'loadingOverlay';
    overlay.innerHTML = `
        <div class="text-center">
            <div class="loading-spinner"></div>
            <div class="mt-3 text-light">${message}</div>
        </div>
    `;
    document.body.appendChild(overlay);
}

/**
 * Hide loading overlay
 */
function hideLoading() {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.remove();
    }
}

/**
 * Show toast notification
 */
function showToast(message, type = 'info', duration = 5000) {
    const toastContainer = getOrCreateToastContainer();
    
    const toast = document.createElement('div');
    toast.className = `toast align-items-center text-white bg-${type} border-0`;
    toast.setAttribute('role', 'alert');
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">${message}</div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
        </div>
    `;
    
    toastContainer.appendChild(toast);
    
    const bsToast = new bootstrap.Toast(toast, { autohide: true, delay: duration });
    bsToast.show();
    
    // Remove from DOM after hiding
    toast.addEventListener('hidden.bs.toast', () => {
        toast.remove();
    });
}

/**
 * Get or create toast container
 */
function getOrCreateToastContainer() {
    let container = document.getElementById('toastContainer');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toastContainer';
        container.className = 'toast-container position-fixed bottom-0 end-0 p-3';
        container.style.zIndex = '9999';
        document.body.appendChild(container);
    }
    return container;
}

/**
 * Format currency values
 */
function formatCurrency(value, currency = 'USD') {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: currency,
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    }).format(value);
}

/**
 * Format percentage values
 */
function formatPercentage(value, decimals = 2) {
    return (value * 100).toFixed(decimals) + '%';
}

/**
 * Format large numbers with appropriate suffixes
 */
function formatLargeNumber(value) {
    if (Math.abs(value) >= 1e9) {
        return (value / 1e9).toFixed(1) + 'B';
    } else if (Math.abs(value) >= 1e6) {
        return (value / 1e6).toFixed(1) + 'M';
    } else if (Math.abs(value) >= 1e3) {
        return (value / 1e3).toFixed(1) + 'K';
    }
    return value.toString();
}

/**
 * Make API calls with timeout and error handling
 */
async function apiCall(url, options = {}) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), TradingApp.config.apiTimeout);
    
    try {
        const response = await fetch(url, {
            ...options,
            signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        clearTimeout(timeoutId);
        console.error('API call failed:', error);
        throw error;
    }
}

/**
 * Cleanup function when leaving page
 */
window.addEventListener('beforeunload', function() {
    // Clear all intervals
    Object.values(TradingApp.intervals).forEach(interval => {
        clearInterval(interval);
    });
    
    console.log('🧹 Cleaned up intervals and resources');
});

/**
 * Progress tracking functionality
 */
class ProgressTracker {
    constructor() {
        this.currentOperation = null;
        this.steps = [];
        this.modal = null;
    }
    
    /**
     * Start a new operation with progress tracking
     */
    startOperation(operationName, steps) {
        this.currentOperation = operationName;
        this.steps = steps.map((step, index) => ({
            id: index,
            title: step.title,
            description: step.description,
            status: 'pending', // pending, active, completed, error
            startTime: null,
            endTime: null
        }));
        
        this.showProgressModal();
        this.updateDisplay();
    }
    
    /**
     * Update step status
     */
    updateStep(stepIndex, status, message = null) {
        if (stepIndex >= 0 && stepIndex < this.steps.length) {
            const step = this.steps[stepIndex];
            
            if (status === 'active' && step.status !== 'active') {
                step.startTime = new Date();
            }
            
            if ((status === 'completed' || status === 'error') && step.status === 'active') {
                step.endTime = new Date();
            }
            
            step.status = status;
            if (message) {
                step.description = message;
            }
            
            this.updateDisplay();
        }
    }
    
    /**
     * Complete the operation
     */
    completeOperation(success = true, message = null) {
        if (success) {
            // Mark all remaining steps as completed
            this.steps.forEach(step => {
                if (step.status === 'pending' || step.status === 'active') {
                    step.status = 'completed';
                    step.endTime = new Date();
                }
            });
        }
        
        this.updateDisplay();
        
        // Auto-close after delay if successful
        if (success) {
            setTimeout(() => {
                this.hideProgressModal();
            }, 2000);
        }
    }
    
    /**
     * Show progress modal
     */
    showProgressModal() {
        if (!this.modal) {
            this.createProgressModal();
        }
        
        this.modal.show();
    }
    
    /**
     * Hide progress modal
     */
    hideProgressModal() {
        if (this.modal) {
            this.modal.hide();
        }
    }
    
    /**
     * Create progress modal DOM
     */
    createProgressModal() {
        const modalHtml = `
            <div class="modal fade operation-progress-modal" id="progressModal" tabindex="-1" data-bs-backdrop="static">
                <div class="modal-dialog">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title text-light">
                                <i class="bi bi-gear me-2"></i>
                                <span id="operationTitle">Processing...</span>
                            </h5>
                        </div>
                        <div class="modal-body">
                            <div id="progressSteps"></div>
                            <div class="mt-3">
                                <div class="progress">
                                    <div class="progress-bar progress-bar-animated" id="overallProgress" style="width: 0%"></div>
                                </div>
                                <div class="progress-text" id="progressText">0%</div>
                            </div>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-outline-light" data-bs-dismiss="modal" style="display: none;" id="closeProgressBtn">
                                Close
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        document.body.insertAdjacentHTML('beforeend', modalHtml);
        this.modal = new bootstrap.Modal(document.getElementById('progressModal'));
    }
    
    /**
     * Update progress display
     */
    updateDisplay() {
        if (!this.modal) return;
        
        // Update operation title
        document.getElementById('operationTitle').textContent = this.currentOperation;
        
        // Update steps
        const stepsContainer = document.getElementById('progressSteps');
        stepsContainer.innerHTML = this.steps.map(step => {
            const icon = this.getStepIcon(step.status);
            const timing = this.getStepTiming(step);
            
            return `
                <div class="operation-step ${step.status}">
                    <div class="step-icon">
                        <i class="bi ${icon}"></i>
                    </div>
                    <div class="step-content">
                        <div class="step-title">${step.title}</div>
                        <div class="step-description">${step.description}</div>
                    </div>
                    <div class="step-timing">${timing}</div>
                </div>
            `;
        }).join('');
        
        // Update overall progress
        const completedSteps = this.steps.filter(s => s.status === 'completed').length;
        const totalSteps = this.steps.length;
        const progressPercent = Math.round((completedSteps / totalSteps) * 100);
        
        const progressBar = document.getElementById('overallProgress');
        const progressText = document.getElementById('progressText');
        
        if (progressBar && progressText) {
            progressBar.style.width = `${progressPercent}%`;
            progressText.textContent = `${progressPercent}%`;
        }
        
        // Show close button if all steps completed or any error
        const hasError = this.steps.some(s => s.status === 'error');
        const allCompleted = this.steps.every(s => s.status === 'completed');
        
        if (hasError || allCompleted) {
            const closeBtn = document.getElementById('closeProgressBtn');
            if (closeBtn) {
                closeBtn.style.display = 'block';
            }
        }
    }
    
    /**
     * Get icon for step status
     */
    getStepIcon(status) {
        switch (status) {
            case 'pending': return 'bi-circle';
            case 'active': return 'bi-arrow-clockwise';
            case 'completed': return 'bi-check-circle';
            case 'error': return 'bi-x-circle';
            default: return 'bi-circle';
        }
    }
    
    /**
     * Get timing display for step
     */
    getStepTiming(step) {
        if (step.startTime && step.endTime) {
            const duration = step.endTime - step.startTime;
            return `${Math.round(duration / 1000)}s`;
        } else if (step.startTime) {
            const duration = new Date() - step.startTime;
            return `${Math.round(duration / 1000)}s`;
        }
        return '';
    }
}

// Create global progress tracker instance
window.TradingApp.progressTracker = new ProgressTracker();

// Export functions for global use
window.TradingApp.utils = {
    createChart,
    updateChart,
    showLoading,
    hideLoading,
    showToast,
    formatCurrency,
    formatPercentage,
    formatLargeNumber,
    apiCall,
    ProgressTracker
};