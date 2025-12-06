import pandas as pd
import numpy as np
import os

# ==========================================
# 1. Core Calculation Function
# ==========================================

def calc_norm(series, window=20):
    """
    Calculates rolling normalization: (X - Min) / (Max - Min)
    """
    roll_min = series.rolling(window).min()
    roll_max = series.rolling(window).max()
    denom = roll_max - roll_min
    
    # Handle division by zero (ignore warnings, replace inf with nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        res = (series - roll_min) / denom
    
    res = res.replace([np.inf, -np.inf], np.nan)
    return res

def calculate_34_factors(df):
    """
    Input: OHLCV DataFrame
    Output: DataFrame with 34 calculated factors
    """
    # Preprocessing: Convert column names to lowercase
    df.columns = [c.lower() for c in df.columns]
    
    # Handle "N.A." strings and ensure numeric types
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Base variables
    o = df['open']
    h = df['high']
    l = df['low']
    c = df['close']
    v = df['volume']
    
    # Pre-calculate common lags
    c_lag_4 = c.shift(4)
    c_lag_19 = c.shift(19)
    
    factors = pd.DataFrame(index=df.index)

    # --- Open Series ---
    factors['OHLC-open_lag_1'] = calc_norm(o / c_lag_19, 20)
    factors['OHLC-open_lag_2'] = calc_norm((o.shift(1) / c_lag_4).abs(), 19)
    factors['OHLC-open_lag_3'] = calc_norm(o.shift(2) / c_lag_19, 20)
    factors['OHLC-open_lag_4'] = calc_norm(o.shift(3) / c_lag_19, 20)
    factors['OHLC-open_lag_5'] = calc_norm(o.shift(4) / c_lag_19, 20)

    # --- High Series ---
    factors['OHLC-high_lag_1'] = calc_norm(h / c_lag_19, 20)
    factors['OHLC-high_lag_2'] = calc_norm(h.shift(1) / c_lag_19, 20)
    factors['OHLC-high_lag_3'] = calc_norm(h.shift(2) / c_lag_19, 20)
    factors['OHLC-high_lag_4'] = calc_norm(h.shift(3) / c_lag_19, 20)
    factors['OHLC-high_lag_5'] = calc_norm(h.shift(4) / c_lag_19, 20)

    # --- Low Series ---
    factors['OHLC-low_lag_1'] = calc_norm(l / c_lag_19, 20)
    factors['OHLC-low_lag_2'] = calc_norm(l.shift(1) / c_lag_19, 20)
    factors['OHLC-low_lag_3'] = calc_norm((l.shift(2) / c_lag_4).abs(), 19)
    factors['OHLC-low_lag_4'] = calc_norm(l.shift(3) / c_lag_19, 20)
    factors['OHLC-low_lag_5'] = calc_norm((l.shift(4) / c_lag_4).abs(), 19)

    # --- Close Series ---
    factors['OHLC-close_lag_1'] = calc_norm(c / c_lag_19, 20)
    factors['OHLC-close_lag_2'] = calc_norm(c.shift(1) / c_lag_19, 20)
    factors['OHLC-close_lag_3'] = calc_norm((c.shift(2) / c_lag_4).abs(), 19)
    factors['OHLC-close_lag_4'] = calc_norm(c.shift(3) / c_lag_19, 20)
    # Fixed: Changed denominator to c_lag_19 to avoid division by zero
    factors['OHLC-close_lag_5'] = calc_norm((c.shift(4) / c_lag_19).abs(), 20)

    # --- MA Series ---
    ma_1_base = (c / c_lag_4).abs().rolling(5).mean()
    factors['OHLC-ma_lag_1'] = calc_norm(ma_1_base, 19)
    factors['OHLC-ma_lag_2'] = calc_norm((c.shift(1) / c_lag_19).rolling(20).mean(), 20)
    factors['OHLC-ma_lag_3'] = calc_norm((c.shift(2) / c_lag_19).rolling(20).mean(), 20)
    factors['OHLC-ma_lag_4'] = calc_norm((c.shift(3) / c_lag_19).rolling(20).mean(), 20)
    factors['OHLC-ma_lag_5'] = calc_norm((c.shift(4) / c_lag_19).rolling(20).mean(), 20)

    # --- High-Low Series ---
    factors['OHLC-high_low_lag_1'] = calc_norm((h / c_lag_19) - (l / c_lag_19), 20)
    factors['OHLC-high_low_lag_2'] = calc_norm((h.shift(1) / c_lag_19) - (l.shift(1) / c_lag_19), 20)
    hl_3_base = (h.shift(2) / c_lag_4) - (l.shift(2) / c_lag_4)
    factors['OHLC-high_low_lag_3'] = calc_norm(hl_3_base.abs(), 19)
    factors['OHLC-high_low_lag_4'] = calc_norm((h.shift(3) / c_lag_19) - (l.shift(3) / c_lag_19), 20)
    factors['OHLC-high_low_lag_5'] = calc_norm((h.shift(4) / c_lag_19) - (l.shift(4) / c_lag_19), 20)

    # --- Volume Series ---
    factors['OHLC-vol_lag_1'] = calc_norm(v.abs(), 19)
    factors['OHLC-vol_lag_2'] = calc_norm(v.shift(1).abs(), 19)
    factors['OHLC-vol_lag_3'] = calc_norm((v.shift(2) / c_lag_4).abs(), 19)
    factors['OHLC-vol_lag_4'] = calc_norm(v.shift(3).abs(), 19)
    factors['OHLC-vol_lag_5'] = calc_norm(v.shift(4).abs(), 19)

    return factors

# ==========================================
# 2. Main Program 
# ==========================================

def main():
    # 1. Define Paths
    # script_dir: Where this script is located (D:\...\RL-FA25-Final\OHLC)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    factor_output_dir = os.path.join(script_dir, 'factor_outputs')
    os.makedirs(factor_output_dir, exist_ok=True)
    
    # data_file: consolidated CSV (D:\...\RL-FA25-Final\data.csv)
    data_file = os.path.join(script_dir, '..', 'data.csv')
    if not os.path.exists(data_file):
        print(f"Data file not found: {data_file}")
        return

    print(f"Loading consolidated file: {data_file}")
    try:
        raw_df = pd.read_csv(data_file)
    except Exception as e:
        print(f"Failed to read {data_file}: {e}")
        return

    if 'DateTime' not in raw_df.columns:
        print("DateTime column missing in data file.")
        return

    raw_df['DateTime'] = pd.to_datetime(raw_df['DateTime'])

    # Detect tickers from column prefixes (e.g., VOO_Close -> VOO)
    tickers = sorted({col.split('_', 1)[0] for col in raw_df.columns if col != 'DateTime'})
    if not tickers:
        print("No ticker columns detected in data file.")
        return

    print(f"Detected tickers: {', '.join(tickers)}")

    all_dfs = []
    for ticker in tickers:
        required_cols = {
            'open': f'{ticker}_Open',
            'high': f'{ticker}_High',
            'low': f'{ticker}_Low',
            'close': f'{ticker}_Close',
            'volume': f'{ticker}_Volume'
        }

        missing_cols = [col for col in required_cols.values() if col not in raw_df.columns]
        if missing_cols:
            print(f"Skipping {ticker}: Missing columns {missing_cols}")
            continue

        print(f"Processing: {ticker}")

        df = raw_df[['DateTime'] + list(required_cols.values())].copy()
        df.rename(columns={
            'DateTime': 'datetime',
            required_cols['open']: 'open',
            required_cols['high']: 'high',
            required_cols['low']: 'low',
            required_cols['close']: 'close',
            required_cols['volume']: 'volume'
        }, inplace=True)

        df.set_index('datetime', inplace=True)

        try:
            df_factors = calculate_34_factors(df)
            df_factors['ticker'] = ticker
            
            cols = ['ticker'] + [c for c in df_factors.columns if c != 'ticker']
            df_factors = df_factors[cols]
            
            all_dfs.append(df_factors)
        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    if all_dfs:
        print("\nMerging panel data...")
        panel_df = pd.concat(all_dfs, axis=0)
        panel_df.index.name = 'DateTime'
        panel_df.sort_values(by=['DateTime', 'ticker'], inplace=True)
        factor_columns = [c for c in panel_df.columns if c != 'ticker']
        
        # ==========================================
        # SAVE RAW DATA -> to OHLC folder (script_dir)
        # ==========================================
        output_path = os.path.join(script_dir, 'OHLC_34.csv')
        panel_df.to_csv(output_path)
        print(f"\nSuccess! Raw file saved to: {output_path}")

        # Save each factor to its own raw CSV
        print("Saving per-factor raw files...")
        for factor in factor_columns:
            factor_df = panel_df[['ticker', factor]].reset_index()
            factor_df = factor_df[['DateTime', 'ticker', factor]]
            factor_path = os.path.join(factor_output_dir, f"{factor}_raw.csv")
            factor_df.to_csv(factor_path, index=False)
        print(f"Raw factor files saved under: {factor_output_dir}")

        # ==========================================
        # Quality Check & CLEAN SAVE
        # ==========================================
        print("\n" + "="*40)
        print("         Data Quality Report")
        print("="*40)

        failed_factors = panel_df.columns[panel_df.isnull().all()]
        
        if len(failed_factors) > 0:
            print(f"❌ CRITICAL WARNING! Found {len(failed_factors)} factor(s) entirely empty:")
            print(list(failed_factors))
        else:
            print("✅ Factor calculation passed: No entirely empty columns found.")
            
            rows_with_nan = panel_df.isnull().any(axis=1).sum()
            print(f"\n✅ Detected warm-up NaNs: {rows_with_nan} (Normal behavior)")

            print("-" * 40)
            print("Generating final cleaned version...")
            
            # Drop rows with NaN
            df_clean = panel_df.dropna()
            
            # ==========================================
            # SAVE CLEAN DATA -> to OHLC folder (script_dir)
            # ==========================================
            clean_path = os.path.join(script_dir, 'OHLC_34_clean.csv')
            df_clean.to_csv(clean_path)
            
            print(f"🎉 Cleaning complete! Valid data rows: {len(df_clean)}")
            print(f"Final ready-to-use file: {clean_path}")

            # Save each factor to its own clean CSV
            print("Saving per-factor clean files...")
            for factor in factor_columns:
                factor_df = df_clean[['ticker', factor]].reset_index()
                factor_df = factor_df[['DateTime', 'ticker', factor]]
                factor_path = os.path.join(factor_output_dir, f"{factor}_clean.csv")
                factor_df.to_csv(factor_path, index=False)
            print(f"Clean factor files saved under: {factor_output_dir}")

    else:
        print("No valid data generated.")

if __name__ == "__main__":
    main()
