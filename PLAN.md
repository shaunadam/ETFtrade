# PLAN.md: Consolidated Project Progress and Plan

This document consolidates the current in-progress elements and confirmed progress across various project planning and development files.


# Current Project:


## Briefing Document: Trade Journal Integration Outline

### 1. Task Architecture Overview

The "Trade Journal Integration" task aims to introduce a robust system for logging, tracking, and managing individual trade entries within the existing ETFtrade application. This involves extending the current Flask web application to provide a dedicated user interface for trade journaling, alongside the necessary backend logic and database schema modifications.

The integration will follow a standard Model-View-Controller (MVC) pattern (or similar, given Flask's structure) where:
*   **Model**: A new database model will be defined to represent individual trade journal entries, including fields for trade details, performance metrics, and correlation data. This model will interact with the SQLite database.
*   **Service Layer**: A dedicated service will encapsulate the business logic for all journal-related operations, such as creating, reading, updating, deleting (CRUD) entries, and performing calculations like correlation tracking. This promotes separation of concerns and reusability.
*   **Controller (Blueprint/Routes)**: Flask blueprints will define the API endpoints and routes necessary for the web interface to interact with the journal service. These routes will handle HTTP requests, call the appropriate service methods, and render HTML templates.
*   **View (Templates/Static Files)**: HTML templates will provide the user interface for adding, viewing, and managing trade journal entries. Client-side JavaScript will enhance interactivity.

### 2. Connection of Files

The integration will involve modifications and additions across several key areas of the existing codebase:

*   **Database Schema (`flask_app/models.py`, `init_database.py`)**: A new `TradeJournalEntry` model will be added to `flask_app/models.py` to define the structure of journal data. `init_database.py` will need to be updated to ensure this new table is created when the application's database is initialized.
*   **Backend Logic (`flask_app/services/journal_service.py`, `flask_app/services/trade_service.py`, `portfolio_risk.py`)**: A new `journal_service.py` will be created to handle all specific journal operations. Existing services like `trade_service.py` might be modified to automatically log trades into the journal upon execution or completion. `portfolio_risk.py` could potentially be extended to utilize journal data for more advanced correlation tracking or risk assessments.
*   **Web Application Routing (`flask_app/blueprints/journal/__init__.py`, `flask_app/__init__.py`)**: The `flask_app/blueprints/journal/__init__.py` file will house the Flask routes for the journal interface. This blueprint will then be registered in the main `flask_app/__init__.py` file to make the journal routes accessible within the application.
*   **User Interface (`flask_app/templates/journal/`, `flask_app/static/js/journal.js`, `flask_app/templates/base.html`)**: New HTML templates will be created within `flask_app/templates/journal/` for various journal views (e.g., adding, viewing, editing entries). `flask_app/static/js/journal.js` will contain any client-side JavaScript for interactive elements. The `flask_app/templates/base.html` will be updated to include navigation links to the new trade journal section.
*   **CLI Integration (`trade_manager.py`)**: While the primary focus is the web interface, `trade_manager.py` (which handles trade execution and management via CLI) might be reviewed or modified to ensure seamless integration with the new journal, potentially allowing for CLI-based logging or reporting.

### 3. Specific Goal of the Task

The specific goal is to provide a fully functional, user-friendly trade journaling system within the ETFtrade Flask application. This system must allow users to:
*   Record detailed information for each trade (e.g., entry/exit dates, prices, instrument, setup used, profit/loss).
*   Track and display correlations between trades or trade setups.
*   Manage (add, view, edit, delete) their trade entries through a dedicated web interface.

This integration will enhance the application's utility by providing a historical record of trading activity, enabling performance analysis, and supporting continuous improvement of trading strategies.

---

### Definitive List of Essential File Paths

**New Files to Create:**

*   `flask_app/services/journal_service.py`
*   `flask_app/templates/journal/add_entry.html`
*   `flask_app/templates/journal/view_entries.html`
*   `flask_app/templates/journal/edit_entry.html`
*   `flask_app/static/js/journal.js`

**Existing Files to Read/Modify:**

*   `flask_app/models.py`
*   `flask_app/blueprints/journal/__init__.py`
*   `flask_app/__init__.py`
*   `init_database.py`
*   `trade_manager.py`
*   `flask_app/templates/base.html`
*   `flask_app/app.py`
*   `flask_app/services/trade_service.py`
*   `portfolio_risk.py`





## Future Work Items (after this complete):

*   **Trade Journal Integration**: Implement trade journal functionality, correlation tracking, and web interface for trade management.

*   **Risk Management Calculator**:
    *   Position Sizing Calculator (risk-based, account size, entry/stop/target inputs).
    *   Portfolio Risk Assessment (exposure, correlation-adjusted risk, drawdown projections).
    *   Trade Validation (pre-trade assessment, size recommendations, guideline compliance).

*   **Reporting Tools**: Develop performance reporting vs. benchmarks, daily/weekly reporting, and regime-based performance analysis.
*   **Optimization & Expansion**: Focus on strategy refinement, parameter optimization tools, and dynamic production parameter optimization.
