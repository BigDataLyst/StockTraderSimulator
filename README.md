# 📈 Stock Trader Simulator

A sophisticated **machine learning-powered stock market analysis engine** that discovers structural lead-lag relationships across multiple assets, identifies regime-specific trading signals, and trains high-confidence predictive models for intraday trading opportunities.

## 🎯 Project Overview

This project combines **quantitative finance**, **statistical analysis**, and **machine learning** to:

1. **Fetch & Aggregate** historical 5-minute bar data from Yahoo Finance
2. **Detect Lead-Lag Patterns** across multiple stock tickers using correlation analysis
3. **Classify Market Regimes** (Standard, Panic, Manic) to understand asymmetric relationships
4. **Generate ML Features** for predictive modeling
5. **Train XGBoost Models** for high-confidence directional signals
6. **Backtest Signals** with walk-forward validation
7. **Deploy Live Scanner** to identify real-time opportunities in the market

## 🏗️ Architecture

### Core Components

| Module | Purpose |
|--------|---------|
| **get_data.py** | Downloads 60 days of 5-min OHLCV data, detects lead-lag relationships, and exports ML training matrix |
| **train_model.py** | Trains XGBoost classifier with time-series cross-validation and probability-based filtering |
| **live_scanner.py** | Monitors live market data and triggers high-confidence trading signals in real-time |
| **tickers.txt** | Configuration file listing tracked stock symbols |
| **requirements.txt** | Python dependencies |

## 🔧 Setup & Installation

### Prerequisites

- **Python 3.8+**
- **pip** package manager

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/BigDataLyst/StockTraderSimulator.git
cd StockTraderSimulator

# Install dependencies
pip install -r requirements.txt
