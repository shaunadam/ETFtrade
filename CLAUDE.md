# Claude.md: Project Briefing Document

## 1. Project Architecture Overview

This project is a sophisticated Python-based ETF and stock swing trading system. It is designed for long-term capital growth with a focus on controlled drawdowns, aiming to outperform benchmarks like SPY. The system features a dual interface approach:

*   **Command-Line Interface (CLI)**: For core operations like screening, data caching, regime detection, trade setup execution, and backtesting. Key CLI tools include `screener.py`, `data_cache.py`, `regime_detection.py`, `trade_setups.py`, and `backtest.py`.
*   **Flask Web Application**: A modular web application providing a professional user interface with interactive dashboards, charting, and comprehensive trading workflow management. It integrates all CLI functionalities and offers features like real-time progress updates and configuration UIs.

The system relies on a robust technical stack including Python, Flask, Pandas, NumPy, Matplotlib/Plotly, and SQLite for data storage. It emphasizes intelligent data caching to minimize API calls and incorporates a sophisticated market regime detection system to inform trading strategies. Risk management is a core component, with defined rules for position sizing, capital risk per trade, and sector concentration.

The project is structured into distinct development phases, progressing from foundational strategy development to advanced features like trade journaling, reporting, and optimization.

## 2. Connection of Markdown Files

The markdown files in the root of this repository serve as critical documentation and planning artifacts for the project:

*   **`README.md`**: Acts as the primary entry point. It provides a high-level overview of the project's objectives, features, quick start guide, project structure, core components, usage examples for both CLI and Flask app, performance metrics, and development status. It outlines the initial focus on ETFs and the future-proofing for individual stocks.
*   **`CLAUDE.md`**: This document itself. It's intended as a briefing for AI assistants, detailing the project's architecture, key components, technical stack, trading strategy framework, data caching system, development phases, success metrics, and development preferences.
*  **`PLAN.md`**: The short term plan of next steps to work on. 

Collectively, these files provide a comprehensive view of the project's goals, architecture, current status, and future development roadmap.

## 3. Specific Task Goal

The current task is to act as a technical project manager to consolidate project planning information. This involves:
1.  Reading all markdown (`.md`) files in the root of the repository.
2.  Creating a detailed briefing document (`claude.md`) for another AI assistant, explaining the project architecture, file connections, and the overall goal of this task.
3.  Creating a consolidated plan (`PLAN.md`) that summarizes the current in-progress elements and confirmed progress from the relevant markdown files.
