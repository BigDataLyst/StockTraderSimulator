
# The best approach for training models is to download free bulk historical data once, store
# it locally as compressed Parquet or CSV files, and use lightweight API free tiers for your
# model's target inputs. API providers heavily restrict historical deep-dives on free tiers,
# making local storage essential for machine learning development.
#
# Recommended Data Sources & Local Storage Stack
#
# Provider / Tool      | Data Depth (Free)             | Target Focus                   | Local Storage Best Practice
# -------------------- | ----------------------------- | ------------------------------ | -------------------------------------
# Yahoo Finance        | Last 60 days of 5-min bars    | Machine learning training      | Save as Parquet using pandas
# Alpha Vantage        | Up to 2 years (sliced)        | Historical backtesting         | Append daily to a Local SQLite DB
# Alpaca Markets       | Unlimited (recent limitations)| Live algorithmic execution     | Stream into a DuckDB instance
##########
import json
import warnings
from pathlib import Path
import yfinance as yf
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

# Suppress potential performance warnings from statsmodels optimization loops
warnings.filterwarnings('ignore')

# Development toggle: set to False to skip all remote yfinance downloads during local iteration.
ENABLE_YFINANCE_FETCHES = True

BASE_DIR = Path(__file__).resolve().parent
TICKERS_FILE = BASE_DIR / "tickers.txt"
PARQUET_FILE = BASE_DIR / "stock_history.parquet"
STATE_FILE = BASE_DIR / "ticker_state.json"
ML_EXPORT_FILE = BASE_DIR / "ml_training_features.parquet"

def load_tickers(path=TICKERS_FILE):
    """Read tickers from a simple text file, ignoring blank lines and comments."""
    tickers = []
    if not path.exists():
        raise FileNotFoundError(f"Ticker list not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            ticker = line.strip()
            if not ticker or ticker.startswith("#"):
                continue
            tickers.append(ticker)

    return tickers


def load_processed_tickers(path=STATE_FILE):
    """Load the previously processed ticker list from a metadata file."""
    if not path.exists():
        return set()

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return set(data.get("processed_tickers", []))


def save_processed_tickers(tickers, path=STATE_FILE):
    """Persist the processed ticker list to a lightweight metadata file."""
    payload = {"processed_tickers": sorted(set(tickers))}
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def fetch_ticker_data(ticker):
    print(f"Fetching historical 5-minute data for {ticker}...")

    df = yf.download(tickers=ticker, period="60d", interval="5m", auto_adjust=False, progress=False)

    if df.empty:
        print(f"No data retrieved for {ticker}. Check ticker symbol or interval limits.")
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df.rename(columns={"Datetime": "Timestamp"}, inplace=True, errors="ignore")
    df["Ticker"] = ticker

    expected_columns = ["Ticker", "Timestamp", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for column in expected_columns:
        if column not in df.columns:
            df[column] = pd.NA

    return df[["Ticker", "Timestamp", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]


def load_existing_parquet(path=PARQUET_FILE):
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def merge_new_data(existing_df, new_df):
    """Append new rows and keep the newest value for each (Ticker, Timestamp) pair."""
    if existing_df.empty:
        return new_df.copy()

    if new_df.empty:
        return existing_df.copy()

    combined = pd.concat([existing_df, new_df], ignore_index=True, sort=False)
    combined = combined.drop_duplicates(subset=["Ticker", "Timestamp"], keep="last")
    combined = combined.sort_values(["Ticker", "Timestamp"]).reset_index(drop=True)
    return combined


def update_parquet_with_tickers(tickers, path=PARQUET_FILE):
    existing_df = load_existing_parquet(path)
    combined_df = existing_df.copy() if not existing_df.empty else pd.DataFrame()
    processed_tickers = load_processed_tickers()

    if not ENABLE_YFINANCE_FETCHES:
        print("Development mode: skipping yfinance fetches. Using existing parquet data only.")
        if combined_df.empty:
            print("No parquet data exists yet.")
        else:
            print(f"Existing parquet file is still available at {path.resolve()}")
        return

    new_tickers = [ticker for ticker in tickers if ticker not in processed_tickers]

    if not new_tickers:
        print("No new tickers found in tickers.txt since the last run.")
        if combined_df.empty:
            print("No parquet data exists yet.")
        else:
            print(f"Existing parquet file is still available at {path.resolve()}")
        return

    for ticker in new_tickers:
        new_df = fetch_ticker_data(ticker)
        if new_df is None:
            continue

        combined_df = merge_new_data(combined_df, new_df)

    if combined_df.empty:
        print("No data available to save.")
        return

    combined_df = combined_df.sort_values(["Ticker", "Timestamp"]).reset_index(drop=True)
    combined_df.to_parquet(path, compression="snappy")

    processed_tickers.update(tickers)
    save_processed_tickers(processed_tickers)
    print(f"Success! Saved updated data to {path.resolve()}")


# Example: process tickers listed in the text file
update_parquet_with_tickers(load_tickers())

##########
# Loads millions of rows into a DataFrame in milliseconds
df = pd.read_parquet(PARQUET_FILE)
print(df.head(1))

##########

def calculate_lagged_correlations(df_parquet, max_lag_minutes=15, interval_minutes=5):
    """
    Processes long-format Parquet data, pivots it, handles cross-day leakage,
    and isolates predictive leads.
    """
    if df_parquet.empty:
        print("No data available to calculate correlations.")
        return pd.DataFrame()

    # 1. Pivot the flat Parquet data into wide format (Rows=Timestamp, Cols=Ticker)
    # Using 'Close' or 'Adj Close' price for correlation
    df_pivoted = df_parquet.pivot(index="Timestamp", columns="Ticker", values="Close")
    
    # 2. Ensure the index is a DatetimeIndex and drop incomplete rows
    df_pivoted.index = pd.to_datetime(df_pivoted.index)
    df_pivoted = df_pivoted.dropna(how="all")

    # 3. Calculate percentage returns
    returns = df_pivoted.pct_change()
    
    # Add a Date column strictly to prevent cross-day shifting leakage
    returns["_Date"] = returns.index.date
    
    max_row_shifts = max_lag_minutes // interval_minutes
    results = []
    tickers = [col for col in returns.columns if col != "_Date"]

    # 4. Compute correlations safely within daily boundaries
    for leader in tickers:
        for follower in tickers:
            if leader == follower:
                continue
                
            for shift in range(1, max_row_shifts + 1):
                # Shift leader inside daily groups so market close doesn't bleed into next market open
                leader_shifted = returns.groupby("_Date")[leader].shift(shift)
                
                # Drop NaNs created by shifting and percentage changes
                valid_idx = leader_shifted.notna() & returns[follower].notna()
                
                # Guard against zero-variance slices (flat lines)
                if leader_shifted[valid_idx].std() == 0 or returns.loc[valid_idx, follower].std() == 0:
                    continue
                    
                correlation = leader_shifted[valid_idx].corr(returns.loc[valid_idx, follower])
                
                # Check for statistical significance/strength 
                # Note: 0.40 on 5-minute bars is incredibly rare and high; consider 0.15+ as a strong lead
                if pd.notna(correlation) and abs(correlation) > 0.15:
                    results.append({
                        "Leader": leader,
                        "Follower": follower,
                        "Lag_Minutes": shift * interval_minutes,
                        "Correlation": round(correlation, 4)
                    })

    if not results:
        print("No significant lagged correlations found over the threshold.")
        return pd.DataFrame()
        
    return pd.DataFrame(results).sort_values(by="Correlation", ascending=False)

def calculate_advanced_signals(df_parquet, max_lag_minutes=15, interval_minutes=5, rolling_window_days=1):
    """
    Optimized high-frequency lead-lag engine. Adjusts threshold metrics 
    to separate structural alpha from 5-minute microstructure noise.
    """
    if df_parquet.empty:
        print("No data available to calculate advanced signals.")
        return pd.DataFrame()

    # Drop any null or unexpected row fragments in the Ticker column
    df_clean = df_parquet.dropna(subset=["Ticker"])
    df_clean = df_clean[df_clean["Ticker"].astype(str).str.strip() != ""]

    # 1. Pivot flat file data into parallel time-series arrays
    df_pivoted = df_clean.pivot(index="Timestamp", columns="Ticker", values="Close")
    df_pivoted.index = pd.to_datetime(df_pivoted.index)
    df_pivoted = df_pivoted.dropna(how="all")

    # 2. Extract percentage returns
    returns = df_pivoted.pct_change()
    returns["_Date"] = returns.index.date
    
    # Filter out fragmented or short trading sessions (e.g., market holidays)
    day_counts = returns["_Date"].value_counts()
    valid_days = day_counts[day_counts > 50].index
    returns = returns[returns["_Date"].isin(valid_days)]

    max_row_shifts = max_lag_minutes // interval_minutes
    tickers = [col for col in returns.columns if col != "_Date"]
    results = []

    # ~78 five-minute bars exist per standard US stock market session
    rolling_rows_window = int(rolling_window_days * 78)

    print(f"Scanning {len(tickers)} assets using lower noise thresholds...")

    for leader in tickers:
        for follower in tickers:
            if leader == follower:
                continue
                
            for shift in range(1, max_row_shifts + 1):
                # Shift within daily boundaries to isolate overnight gap distortions
                leader_shifted = returns.groupby("_Date")[leader].shift(shift)
                
                analysis_df = pd.DataFrame({
                    "Leader_Lagged": leader_shifted,
                    "Follower": returns[follower],
                    "_Date": returns["_Date"]
                }).dropna()

                if len(analysis_df) < (rolling_rows_window + 10):
                    continue

                if analysis_df["Leader_Lagged"].std() == 0 or analysis_df["Follower"].std() == 0:
                    continue

                # A. Noise-adjusted Correlation Filter
                base_corr = analysis_df["Leader_Lagged"].corr(analysis_df["Follower"])
                if pd.isna(base_corr) or abs(base_corr) < 0.04:  # Lowered from 0.12 to capture intraday leads
                    continue

                # B. Dynamic Rolling Module
                rolling_corr = analysis_df["Leader_Lagged"].rolling(window=rolling_rows_window).corr(analysis_df["Follower"])
                rolling_std = rolling_corr.std()

                # C. OLS Regression Engine
                X = sm.add_constant(analysis_df["Leader_Lagged"])
                y = analysis_df["Follower"]
                
                try:
                    model = sm.OLS(y, X).fit()
                    p_value = model.pvalues.iloc[1]  # Safely extract slope coefficient p-value
                    t_stat = model.tvalues.iloc[1]
                except Exception:
                    continue

                # D. Strict Statistical Significance Check
                # Keep p-value barrier strict (< 0.05) to ensure the signal isn't random
                if p_value > 0.05:
                    continue

                results.append({
                    "Leader": leader,
                    "Follower": follower,
                    "Lag_Min": shift * interval_minutes,
                    "Corr": round(base_corr, 4),
                    "Roll_Std": round(rolling_std, 4) if pd.notna(rolling_std) else 0.0,
                    "T_Stat": round(t_stat, 2),
                    "P_Value": f"{p_value:.4e}" if p_value < 0.0001 else round(p_value, 4)
                })

    if not results:
        print("\nNo pairs matched even the lower noise-adjusted criteria.")
        return pd.DataFrame()
        
    return pd.DataFrame(results).sort_values(by="Corr", ascending=False)

def analyze_and_export_ml_matrix(df_parquet, max_lag_minutes=15, interval_minutes=5, rolling_window_days=1):
    """
    Tri-Regime Analysis Engine: Evaluates Lead-Lag behavior across 
    Standard, Panic (Downward), and Manic (Upward Momentum) environments.
    """
    if df_parquet.empty:
        print("No historical data to process.")
        return pd.DataFrame()

    df_clean = df_parquet.dropna(subset=["Ticker"])
    df_clean = df_clean[df_clean["Ticker"].astype(str).str.strip() != ""]

    # 1. Pivot structural flat file data into parallel time-series arrays
    df_pivoted = df_clean.pivot(index="Timestamp", columns="Ticker", values="Close")
    df_pivoted.index = pd.to_datetime(df_pivoted.index)
    df_pivoted = df_pivoted.dropna(how="all")

    # 2. Extract percentage returns
    returns = df_pivoted.pct_change()
    returns["_Date"] = returns.index.date
    
    day_counts = returns["_Date"].value_counts()
    valid_days = day_counts[day_counts > 50].index
    returns = returns[returns["_Date"].isin(valid_days)]

    max_row_shifts = max_lag_minutes // interval_minutes
    tickers = [col for col in returns.columns if col != "_Date"]
    tri_regime_results = []
    
    rolling_rows_window = int(rolling_window_days * 78)
    print(f"Executing Tri-Regime Analysis (Standard vs Panic vs Mania) across {len(tickers)} assets...")

    # --- PART 1: TRI-REGIME DISCOVERY ---
    for leader in tickers:
        for follower in tickers:
            if leader == follower:
                continue
                
            for shift in range(1, max_row_shifts + 1):
                leader_shifted = returns.groupby("_Date")[leader].shift(shift)
                
                analysis_df = pd.DataFrame({
                    "Leader_Lagged": leader_shifted,
                    "Follower": returns[follower],
                    "_Date": returns["_Date"]
                }).dropna()

                if len(analysis_df) < (rolling_rows_window + 10):
                    continue

                base_corr = analysis_df["Leader_Lagged"].corr(analysis_df["Follower"])
                if pd.isna(base_corr) or abs(base_corr) < 0.04:
                    continue

                # Partition into Three Distinct Regimes based on Leader Behavior
                panic_slice = analysis_df[analysis_df["Leader_Lagged"] < -0.0005]   # Clear down-bars
                manic_slice = analysis_df[analysis_df["Leader_Lagged"] > 0.0005]    # Clear up-bars
                standard_slice = analysis_df[(analysis_df["Leader_Lagged"] >= -0.0005) & 
                                             (analysis_df["Leader_Lagged"] <= 0.0005)] # Noise/Flat

                if len(panic_slice) > 10 and len(manic_slice) > 10 and len(standard_slice) > 10:
                    panic_corr = panic_slice["Leader_Lagged"].corr(panic_slice["Follower"])
                    manic_corr = manic_slice["Leader_Lagged"].corr(manic_slice["Follower"])
                    standard_corr = standard_slice["Leader_Lagged"].corr(standard_slice["Follower"])
                else:
                    panic_corr, manic_corr, standard_corr = np.nan, np.nan, np.nan

                # Statistical Validity Check
                X = sm.add_constant(analysis_df["Leader_Lagged"])
                y = analysis_df["Follower"]
                try:
                    model = sm.OLS(y, X).fit()
                    p_value = model.pvalues.iloc[1]  # Slope coefficient p-value
                    t_stat = model.tvalues.iloc[1]  # Slope coefficient t-statistic
                except Exception:
                    continue

                if pd.isna(p_value) or p_value > 0.05:
                    continue

                # Tag Triggers based on which regime shows dominant correlation intensity
                asym_trigger = "PANIC" if (pd.notna(panic_corr) and abs(panic_corr) > abs(standard_corr) * 1.3 and abs(panic_corr) > abs(manic_corr)) else "NO"
                if pd.notna(manic_corr) and abs(manic_corr) > abs(standard_corr) * 1.3 and abs(manic_corr) > abs(panic_corr):
                    asym_trigger = "MANIA"

                tri_regime_results.append({
                    "Leader": leader,
                    "Follower": follower,
                    "Lag_Min": shift * interval_minutes,
                    "Base_Corr": round(base_corr, 4),
                    "Standard_Corr": round(standard_corr, 4) if pd.notna(standard_corr) else 0.0,
                    "Panic_Corr": round(panic_corr, 4) if pd.notna(panic_corr) else 0.0,
                    "Manic_Corr": round(manic_corr, 4) if pd.notna(manic_corr) else 0.0,
                    "Regime_Trigger": asym_trigger,
                    "T_Stat": round(t_stat, 2) if pd.notna(t_stat) else 0.0
                })

    df_report = pd.DataFrame(tri_regime_results)

    # =====================================================================
    # NEW ELEMENT: HISTORICAL REGIME DRIFT TRACKING LOG
    # =====================================================================
    if not df_report.empty:
        # Define paths for the historical tracker log
        HISTORICAL_LOG_FILE = BASE_DIR / "lead_lag_history_log.parquet"
        
        # 1. Tag the entire snapshot with the current operational execution time
        df_snapshot = df_report.copy()
        df_snapshot["Snapshot_Timestamp"] = pd.Timestamp.now()
        
        # 2. Safely read and append to the historical rolling frame
        if HISTORICAL_LOG_FILE.exists():
            try:
                df_existing_log = pd.read_parquet(HISTORICAL_LOG_FILE)
                df_combined_log = pd.concat([df_existing_log, df_snapshot], ignore_index=True)
                print(f"📈 Appended {len(df_snapshot)} regime rows to the rolling historical log database.")
            except Exception as e:
                print(f"⚠️ Error reading log parquet, re-initializing: {e}")
                df_combined_log = df_snapshot
        else:
            print("🚀 Creating brand new historical log tracking database...")
            df_combined_log = df_snapshot
            
        # 3. Persist log update to local disk storage
        df_combined_log.to_parquet(HISTORICAL_LOG_FILE, compression="snappy")
    # =====================================================================

    # --- PART 2: ML FEATURE EXPORT ---
    print("\nAssembling Machine Learning Matrix...")
    ml_rows = []
    if not df_report.empty:
        # Pull top performing pairs from any dynamic regime to extract structural alpha
        top_pairs = df_report.sort_values(by="Base_Corr", ascending=False).head(20)
        
        for _, row in top_pairs.iterrows():
            leader_tk = row["Leader"]
            follower_tk = row["Follower"]

            feature_build = pd.DataFrame({
                "Target_Follower_Return": returns[follower_tk],
                "Leader_Raw_Return": returns[leader_tk]
            }, index=returns.index)

            for l in range(1, max_row_shifts + 1):
                feature_build[f"{leader_tk}_Lag_{l*5}m"] = feature_build["Leader_Raw_Return"].shift(l)
                feature_build[f"{follower_tk}_Self_Lag_{l*5}m"] = feature_build["Target_Follower_Return"].shift(l)

            feature_build[f"{leader_tk}_RollVol_15m"] = feature_build["Leader_Raw_Return"].rolling(3).std()
            feature_build["Future_Follower_Return"] = feature_build["Target_Follower_Return"].shift(-1)
            feature_build["ML_Target_Class"] = np.where(feature_build["Future_Follower_Return"] > 0.0002, 1, 
                                               np.where(feature_build["Future_Follower_Return"] < -0.0002, -1, 0))
            
            feature_build["Asset_Pair_Context"] = f"{leader_tk}_predicts_{follower_tk}"
            ml_rows.append(feature_build.dropna())

        if ml_rows:
            ml_matrix_final = pd.concat(ml_rows, axis=0)
            ml_matrix_final.to_parquet(ML_EXPORT_FILE, compression="snappy")
            print(f"🤖 Machine learning matrix saved! ({len(ml_matrix_final)} rows online across multiple regimes.)")

    # Sort output primarily by Manic buying patterns to review the new findings
    return df_report.sort_values(by="Manic_Corr", ascending=False)

# =====================================================================
# MAIN EXECUTION PIPELINE
# =====================================================================
def main():
    # 1. Update Database
    try:
        update_parquet_with_tickers(load_tickers())
    except FileNotFoundError as e:
        print(f"Skipping database storage pull: {e}")

    # 2. Core Execution Engine
    if PARQUET_FILE.exists():
        df_history = pd.read_parquet(PARQUET_FILE)
        print(f"\nSuccessfully read {len(df_history)} multi-asset entries from Local Storage.")
        
        # 3. Compute Tri-Regime Metrics and Export ML Matrix
        print("Initiating Multi-Regime Structural Framework...")
        tri_regime_summary = analyze_and_export_ml_matrix(
            df_parquet=df_history, 
            max_lag_minutes=15, 
            interval_minutes=5,
            rolling_window_days=1
        )
        
        # 4. Display Formatted Structural Reports
        if not tri_regime_summary.empty:
            print("\n" + "="*114)
            print("                 STRUCTURAL TRI-REGIME LEAD-LAG ALPHA ENGINE ENGINE REPORT           ")
            print("="*114)
            
            # --- SECTION 1: MANIC BUYING MOMENTUM ---
            print("\n🔥 [REGIME A] TOP MANIC MOMENTUM LEADS (Sorted by Highest Manic_Corr)")
            print("-" * 114)
            manic_top = tri_regime_summary.sort_values(by="Manic_Corr", ascending=False).head(10)
            print(manic_top.to_string(index=False))
            print("-" * 114)
            
            # --- SECTION 2: PANIC SELLING HEDGES ---
            print("\n🚨 [REGIME B] TOP PANIC LIQUIDATION LEADS (Sorted by Highest Panic_Corr)")
            print("-" * 114)
            panic_top = tri_regime_summary.sort_values(by="Panic_Corr", ascending=False).head(10)
            print(panic_top.to_string(index=False))
            print("-" * 114)
            
            # --- SECTION 3: STANDARD MARKET COMPANIONS ---
            print("\n⚖️ [REGIME C] TOP STANDARD MARKET CO-MOVEMENTS (Sorted by Highest Standard_Corr)")
            print("-" * 114)
            standard_top = tri_regime_summary.sort_values(by="Standard_Corr", ascending=False).head(10)
            print(standard_top.to_string(index=False))
            print("-" * 114)
            
            # --- EXPLANATORY KEY FOOTER ---
            print("\n💡 TRI-REGIME STRATEGY DISCOVERY KEY:")
            print(" -> Regime_Trigger = MANIA : Relationship strengthens dramatically during upward surges (FOMO-Driven).")
            print(" -> Regime_Trigger = PANIC : Relationship strengthens dramatically during downward flushes (Liquidity-Driven).")
            print(" -> Standard_Corr vs Others: Identifies assets moving purely on macroeconomic or constant index weight trends.")
            print("="*114)
        else:
            print("\nZero systemic mathematical signals verified inside this window slice.")
            
    else:
        print(f"Target local database missing at: {PARQUET_FILE}")

if __name__ == "__main__":
    main()