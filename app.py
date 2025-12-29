import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor

# ==========================================
# 1. CONFIGURATION & STYLE
# ==========================================
st.set_page_config(page_title="Prath's Market Scanner", layout="wide", page_icon="🎯")
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #FAFAFA; }
    div[data-testid="stMetricValue"] { font-size: 20px; color: #00CC96; }
    .stRadio > div { flex-direction: row; } 
    .stDataFrame { border: 1px solid #333; }
    h3 { border-bottom: 2px solid #333; padding-bottom: 10px; }
    
    /* Style for the consolidated table */
    .consolidated-table th {
        text-align: center !important;
        background-color: #1e2130 !important;
    }
    .consolidated-table td {
        text-align: center !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SECURITY & AUTH
# ==========================================
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets.get("PASSWORD", "Sniper2025"):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Enter Access Key", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Enter Access Key", type="password", on_change=password_entered, key="password"
        )
        st.error("⛔ Access Denied")
        return False
    else:
        # Password correct.
        return True

# ==========================================
# 3. MATH WIZ (Advanced Logic)
# ==========================================
class MathWiz:
    @staticmethod
    def calculate_choppiness(high, low, close, length=14):
        try:
            tr1 = high - low
            tr2 = (high - close.shift(1)).abs()
            tr3 = (low - close.shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            atr_sum = tr.rolling(window=length).sum()
            high_max = high.rolling(window=length).max()
            low_min = low.rolling(window=length).min()
            
            range_diff = high_max - low_min
            range_diff.replace(0, np.nan, inplace=True)

            numerator = np.log10(atr_sum / range_diff)
            denominator = np.log10(length)
            return 100 * (numerator / denominator)
        except:
            return pd.Series(dtype='float64')

    @staticmethod
    def identify_strict_swings(df, neighbor_count=3):
        """
        Identifies peaks/valleys that are higher/lower than `neighbor_count` candles
        to the left AND right.
        """
        is_swing_high = pd.Series(True, index=df.index)
        is_swing_low = pd.Series(True, index=df.index)
        
        for i in range(1, neighbor_count + 1):
            is_swing_high &= (df['High'] > df['High'].shift(i))
            is_swing_low &= (df['Low'] < df['Low'].shift(i))
            is_swing_high &= (df['High'] > df['High'].shift(-i))
            is_swing_low &= (df['Low'] < df['Low'].shift(-i))
            
        return is_swing_high, is_swing_low

    @staticmethod
    def find_fvg(df):
        bull_fvg = (df['Low'] > df['High'].shift(2))
        bear_fvg = (df['High'] < df['Low'].shift(2))
        return bull_fvg, bear_fvg
    
    @staticmethod
    def check_ifvg_reversal(df):
        """
        Checks the specific 5-candle iFVG Reversal structure on the LATEST data.
        Returns: 'Bull', 'Bear', or None
        """
        # Take last 5 candles and drop NaNs to ensure clean data
        subset = df.iloc[-5:].copy().dropna()
        if len(subset) < 5: return None

        c1 = subset.iloc[0]
        c3 = subset.iloc[2]
        c5 = subset.iloc[4] # Current candle
        
        # Bullish iFVG Scan:
        # 1. Bearish FVG between C1 and C3 (Drop) -> C3 High < C1 Low
        # 2. Bullish FVG between C3 and C5 (Pop)  -> C5 Low > C3 High
        is_bear_fvg_first = c3['High'] < c1['Low']
        is_bull_fvg_second = c5['Low'] > c3['High']
        
        if is_bear_fvg_first and is_bull_fvg_second:
            return 'Bull'

        # Bearish iFVG Scan:
        # 1. Bullish FVG between C1 and C3 (Pop)  -> C3 Low > C1 High
        # 2. Bearish FVG between C3 and C5 (Drop) -> C5 High < C3 Low
        is_bull_fvg_first = c3['Low'] > c1['High']
        is_bear_fvg_second = c5['High'] < c3['Low']
        
        if is_bull_fvg_first and is_bear_fvg_second:
            return 'Bear'
            
        return None

    @staticmethod
    def find_unmitigated_fvg_zone(df, threshold_pct=0.05):
        if len(df) < 5: return False
        current_price = df['Close'].iloc[-1]
        lookback = min(len(df), 50)
        
        for i in range(len(df)-1, len(df)-lookback, -1):
            curr_low = df['Low'].iloc[i]
            prev_high = df['High'].iloc[i-2]
            
            if curr_low > prev_high: 
                gap_top = curr_low
                gap_bottom = prev_high
                subsequent_data = df.iloc[i+1:]
                if not subsequent_data.empty:
                    if (subsequent_data['Low'] < gap_bottom).any():
                        continue 
                upper_bound = gap_bottom * (1 + threshold_pct)
                lower_bound = gap_bottom * (1 - threshold_pct)
                if lower_bound <= current_price <= upper_bound:
                    return True 
        return False

    @staticmethod
    def check_consecutive_candles(df, num_candles):
        """
        Checks for consecutive red or green candles at the end of the dataframe.
        
        Args:
            df: DataFrame with OHLC data
            num_candles: Number of consecutive candles to check
            
        Returns:
            'Bull' if num_candles consecutive red candles (bullish reversal potential)
            'Bear' if num_candles consecutive green candles (bearish reversal potential)
            None if neither condition is met
        """
        if len(df) < num_candles:
            return None
        
        # Get the last num_candles
        recent = df.iloc[-num_candles:]
        
        # Check if all candles are red (Close < Open)
        all_red = all(recent['Close'] < recent['Open'])
        
        # Check if all candles are green (Close > Open)
        all_green = all(recent['Close'] > recent['Open'])
        
        if all_red:
            return 'Bull'  # Consecutive red = bullish reversal potential
        elif all_green:
            return 'Bear'  # Consecutive green = bearish reversal potential
        
        return None

# ==========================================
# 4. ROBUST DATA ENGINE
# ==========================================
@st.cache_data
def load_tickers(market_choice):
    if "Macro" in market_choice:
        filename = "macro_tickers.csv"
    elif "India" in market_choice:
        filename = "ind_tickers.csv"
    else:
        filename = "us_tickers.csv"
        
    if not os.path.exists(filename):
        st.error(f"❌ '{filename}' not found. Ensure CSV files are in the repository.")
        return []
    try:
        df = pd.read_csv(filename)
        
        # For macro tickers, return the Ticker column directly
        if "Macro" in market_choice:
            return df['Ticker'].tolist()
        
        raw = df.iloc[:, 0].tolist()
        clean = [str(t).strip().upper() for t in raw]
        if "India" in market_choice:
            clean = [t if t.endswith(".NS") else f"{t}.NS" for t in clean]
        return clean
    except: return []

@st.cache_data
def load_macro_metadata():
    """Load macro ticker metadata for display purposes"""
    filename = "macro_tickers.csv"
    if not os.path.exists(filename):
        return {}
    try:
        df = pd.read_csv(filename)
        return dict(zip(df['Ticker'], df['Name']))
    except:
        return {}

@st.cache_data(ttl=3600)
def fetch_bulk_data(tickers, interval="1d", period="2y"):
    if not tickers: return None
    data = yf.download(tickers, period=period, interval=interval, group_by='ticker', progress=True, threads=True)
    return data

def resample_custom(df, timeframe):
    if df.empty: return df
    agg_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
    try:
        if timeframe == "1D": return df.resample("1D").agg(agg_dict).dropna()
        if timeframe == "1W": return df.resample("W-FRI").agg(agg_dict).dropna()
        if timeframe == "1M": return df.resample("ME").agg(agg_dict).dropna()

        # Long Term Strict
        df_monthly = df.resample('MS').agg(agg_dict).dropna()
        if timeframe == "3M": return df_monthly.resample('QE').agg(agg_dict).dropna()
        if timeframe == "6M":
            df_monthly['Year'] = df_monthly.index.year
            df_monthly['Half'] = np.where(df_monthly.index.month <= 6, 1, 2)
            df_6m = df_monthly.groupby(['Year', 'Half']).agg(agg_dict)
            new_index = []
            for (year, half) in df_6m.index:
                month = 6 if half == 1 else 12
                new_index.append(pd.Timestamp(year=year, month=month, day=30))
            df_6m.index = pd.DatetimeIndex(new_index)
            return df_6m.sort_index()
        if timeframe == "12M": return df_monthly.resample('YE').agg(agg_dict).dropna()
    except: return df
    return df

# ==========================================
# 5. PARALLEL ANALYSIS ENGINE
# ==========================================

def analyze_ticker(ticker, df_daily_raw, df_monthly_raw):
    """
    Runs ALL scans for ONE ticker in a single pass.
    Returns a dictionary with ticker info and all scan results.
    """
    results = {
        'Ticker': ticker,
        'Price': 0,
        # FVG Breakouts
        'FVG_1D': False, 'FVG_1W': False, 'FVG_1M': False,
        # iFVG Reversals
        'iFVG_1D': False, 'iFVG_1W': False, 'iFVG_1M': False,
        # Order Flow
        'OB_1D': False, 'OB_1W': False, 'OB_1M': False, 'OB_6M': False,
        # Strong Support
        'Support': False,
        # Reversal Candidates
        'RevCand_1D': False, 'RevCand_1W': False, 'RevCand_1M': False,
        # Squeeze
        'Squeeze_1D': False, 'Squeeze_1W': False,
        # Track if any bullish signal found
        'has_signal': False
    }
    
    try:
        # 1D, 1W from Daily Data
        d_1d = resample_custom(df_daily_raw, "1D")
        d_1w = resample_custom(df_daily_raw, "1W")
        
        # 1M, 3M, 6M, 12M from Monthly Data
        d_1m = resample_custom(df_monthly_raw, "1M")
        d_3m = resample_custom(df_monthly_raw, "3M")
        d_6m = resample_custom(df_monthly_raw, "6M")
        d_12m = resample_custom(df_monthly_raw, "12M")
        
        data_map = {
            "1D": d_1d, "1W": d_1w, "1M": d_1m, 
            "3M": d_3m, "6M": d_6m, "12M": d_12m
        }
        
        # Set price from daily data
        if not d_1d.empty:
            results['Price'] = round(d_1d['Close'].iloc[-1], 2)
        
    except: 
        return results

    # 1. FVG & Order Flow & iFVG (Iterate Timeframes)
    for tf in ["1D", "1W", "1M", "6M"]:
        if tf not in data_map or len(data_map[tf]) < 5: continue
        
        df = data_map[tf].copy() 
        
        # Run Standard FVG/Swing Logic
        df['Bull_FVG'], df['Bear_FVG'] = MathWiz.find_fvg(df)
        df['Is_Swing_High'], df['Is_Swing_Low'] = MathWiz.identify_strict_swings(df)
        
        curr = df.iloc[-1]
        prev = df.iloc[-2]

        # --- A. SNIPER FVG ENTRIES ---
        if curr['Bull_FVG'] and tf in ["1D", "1W", "1M"]:
            past_swings = df[df['Is_Swing_High']]
            if not past_swings.empty:
                last_swing_high = past_swings['High'].iloc[-1]
                if curr['Close'] > last_swing_high and prev['Close'] <= last_swing_high:
                    results[f'FVG_{tf}'] = True
                    results['has_signal'] = True

        # --- B. Order Flow ---
        subset = df.iloc[-4:].copy()
        if len(subset) == 4:
            c_anc, c1, c2, c3 = subset.iloc[0], subset.iloc[1], subset.iloc[2], subset.iloc[3]
            
            # Bullish OB
            if (c_anc['Close'] < c_anc['Open'] and 
                c1['Close'] > c1['Open'] and c2['Close'] > c2['Open'] and c3['Close'] > c3['Open']):
                if c3['Low'] > c1['High'] and c3['Close'] > c_anc['High']:
                    results[f'OB_{tf}'] = True
                    results['has_signal'] = True
        
        # --- C. iFVG REVERSALS (1D, 1W, 1M only) ---
        if tf in ["1D", "1W", "1M"]:
            ifvg_status = MathWiz.check_ifvg_reversal(df)
            if ifvg_status == "Bull":
                results[f'iFVG_{tf}'] = True
                results['has_signal'] = True

    # 2. STRONG SUPPORT
    sup_tf = []
    if MathWiz.find_unmitigated_fvg_zone(d_3m): sup_tf.append("3M")
    if MathWiz.find_unmitigated_fvg_zone(d_6m): sup_tf.append("6M")
    if MathWiz.find_unmitigated_fvg_zone(d_12m): sup_tf.append("12M")
    
    if len(sup_tf) >= 2:
        results['Support'] = True
        results['has_signal'] = True

    # 3. SQUEEZE
    if not d_1d.empty:
        chop_series_d = MathWiz.calculate_choppiness(d_1d['High'], d_1d['Low'], d_1d['Close'])
        if not chop_series_d.empty and not pd.isna(chop_series_d.iloc[-1]):
            chop_d = chop_series_d.iloc[-1]
            if chop_d > 59:
                results['Squeeze_1D'] = True
                results['has_signal'] = True
                
    if not d_1w.empty:
        chop_series_w = MathWiz.calculate_choppiness(d_1w['High'], d_1w['Low'], d_1w['Close'])
        if not chop_series_w.empty and not pd.isna(chop_series_w.iloc[-1]):
            chop_w = chop_series_w.iloc[-1]
            if chop_w > 59:
                results['Squeeze_1W'] = True
                results['has_signal'] = True

    # 4. REVERSAL CANDIDATES (Consecutive Candle Scan)
    # 1D: 5+ consecutive red candles
    if not d_1d.empty and len(d_1d) >= 5:
        rev_1d = MathWiz.check_consecutive_candles(d_1d, 5)
        if rev_1d == 'Bull':
            results['RevCand_1D'] = True
            results['has_signal'] = True
    
    # 1W: 4+ consecutive red candles
    if not d_1w.empty and len(d_1w) >= 4:
        rev_1w = MathWiz.check_consecutive_candles(d_1w, 4)
        if rev_1w == 'Bull':
            results['RevCand_1W'] = True
            results['has_signal'] = True
    
    # 1M: 3+ consecutive red candles
    if not d_1m.empty and len(d_1m) >= 3:
        rev_1m = MathWiz.check_consecutive_candles(d_1m, 3)
        if rev_1m == 'Bull':
            results['RevCand_1M'] = True
            results['has_signal'] = True

    return results


def analyze_ticker_bearish(ticker, df_daily_raw, df_monthly_raw):
    """
    Runs ALL BEARISH scans for ONE ticker in a single pass.
    Returns a dictionary with ticker info and all scan results.
    """
    results = {
        'Ticker': ticker,
        'Price': 0,
        # FVG Breakdowns
        'FVG_1D': False, 'FVG_1W': False, 'FVG_1M': False,
        # iFVG Reversals
        'iFVG_1D': False, 'iFVG_1W': False, 'iFVG_1M': False,
        # Order Flow
        'OB_1D': False, 'OB_1W': False, 'OB_1M': False, 'OB_6M': False,
        # Trend Reversal (Exhaustion)
        'Exhaustion': False,
        # Reversal Candidates (Overbought)
        'RevCand_1D': False, 'RevCand_1W': False, 'RevCand_1M': False,
        # Track if any bearish signal found
        'has_signal': False
    }
    
    try:
        # 1D, 1W from Daily Data
        d_1d = resample_custom(df_daily_raw, "1D")
        d_1w = resample_custom(df_daily_raw, "1W")
        
        # 1M, 3M, 6M, 12M from Monthly Data
        d_1m = resample_custom(df_monthly_raw, "1M")
        d_3m = resample_custom(df_monthly_raw, "3M")
        d_6m = resample_custom(df_monthly_raw, "6M")
        d_12m = resample_custom(df_monthly_raw, "12M")
        
        data_map = {
            "1D": d_1d, "1W": d_1w, "1M": d_1m, 
            "3M": d_3m, "6M": d_6m, "12M": d_12m
        }
        
        # Set price from daily data
        if not d_1d.empty:
            results['Price'] = round(d_1d['Close'].iloc[-1], 2)
        
    except: 
        return results

    # 1. FVG & Order Flow & iFVG (Iterate Timeframes)
    for tf in ["1D", "1W", "1M", "6M"]:
        if tf not in data_map or len(data_map[tf]) < 5: continue
        
        df = data_map[tf].copy() 
        
        # Run Standard FVG/Swing Logic
        df['Bull_FVG'], df['Bear_FVG'] = MathWiz.find_fvg(df)
        df['Is_Swing_High'], df['Is_Swing_Low'] = MathWiz.identify_strict_swings(df)
        
        curr = df.iloc[-1]
        prev = df.iloc[-2]

        # --- A. SNIPER FVG BREAKDOWNS ---
        if curr['Bear_FVG'] and tf in ["1D", "1W", "1M"]:
            past_swings = df[df['Is_Swing_Low']]
            if not past_swings.empty:
                last_swing_low = past_swings['Low'].iloc[-1]
                if curr['Close'] < last_swing_low and prev['Close'] >= last_swing_low:
                    results[f'FVG_{tf}'] = True
                    results['has_signal'] = True

        # --- B. Order Flow (Bearish) ---
        subset = df.iloc[-4:].copy()
        if len(subset) == 4:
            c_anc, c1, c2, c3 = subset.iloc[0], subset.iloc[1], subset.iloc[2], subset.iloc[3]
            
            # Bearish OB
            if (c_anc['Close'] > c_anc['Open'] and 
                c1['Close'] < c1['Open'] and c2['Close'] < c2['Open'] and c3['Close'] < c3['Open']):
                if c3['High'] < c1['Low'] and c3['Close'] < c_anc['Low']:
                    results[f'OB_{tf}'] = True
                    results['has_signal'] = True
        
        # --- C. iFVG REVERSALS (1D, 1W, 1M only) ---
        if tf in ["1D", "1W", "1M"]:
            ifvg_status = MathWiz.check_ifvg_reversal(df)
            if ifvg_status == "Bear":
                results[f'iFVG_{tf}'] = True
                results['has_signal'] = True

    # 2. TREND REVERSAL (Exhaustion) - Weekly Chop < 25
    if not d_1w.empty:
        chop_series = MathWiz.calculate_choppiness(d_1w['High'], d_1w['Low'], d_1w['Close'])
        if not chop_series.empty and not pd.isna(chop_series.iloc[-1]):
            chop_w = chop_series.iloc[-1]
            if chop_w < 25:
                results['Exhaustion'] = True
                results['has_signal'] = True

    # 3. REVERSAL CANDIDATES (Consecutive Green Candles - Overbought)
    # 1D: 5+ consecutive green candles
    if not d_1d.empty and len(d_1d) >= 5:
        rev_1d = MathWiz.check_consecutive_candles(d_1d, 5)
        if rev_1d == 'Bear':
            results['RevCand_1D'] = True
            results['has_signal'] = True
    
    # 1W: 4+ consecutive green candles
    if not d_1w.empty and len(d_1w) >= 4:
        rev_1w = MathWiz.check_consecutive_candles(d_1w, 4)
        if rev_1w == 'Bear':
            results['RevCand_1W'] = True
            results['has_signal'] = True
    
    # 1M: 3+ consecutive green candles
    if not d_1m.empty and len(d_1m) >= 3:
        rev_1m = MathWiz.check_consecutive_candles(d_1m, 3)
        if rev_1m == 'Bear':
            results['RevCand_1M'] = True
            results['has_signal'] = True

    return results


def create_consolidated_table(results_list, is_macro=False, macro_names=None, scan_type='bullish'):
    """
    Creates a consolidated DataFrame with multi-level columns for display.
    """
    if not results_list:
        return None
    
    # Filter only stocks with signals
    filtered = [r for r in results_list if r.get('has_signal', False)]
    
    if not filtered:
        return None
    
    # Create base dataframe
    df = pd.DataFrame(filtered)
    
    # Add Name column for macro
    if is_macro and macro_names:
        df['Name'] = df['Ticker'].map(macro_names).fillna(df['Ticker'])
    
    # Convert boolean to checkmark
    check = "✅"
    empty = ""
    
    if scan_type == 'bullish':
        # Build the display dataframe
        display_data = []
        for _, row in df.iterrows():
            display_row = {
                ('Stock', 'Ticker'): row['Ticker'],
                ('Stock', 'Price'): row['Price'],
                ('FVG Breakouts', '1D'): check if row.get('FVG_1D') else empty,
                ('FVG Breakouts', '1W'): check if row.get('FVG_1W') else empty,
                ('FVG Breakouts', '1M'): check if row.get('FVG_1M') else empty,
                ('iFVG Reversals', '1D'): check if row.get('iFVG_1D') else empty,
                ('iFVG Reversals', '1W'): check if row.get('iFVG_1W') else empty,
                ('iFVG Reversals', '1M'): check if row.get('iFVG_1M') else empty,
                ('Order Flow', '1D'): check if row.get('OB_1D') else empty,
                ('Order Flow', '1W'): check if row.get('OB_1W') else empty,
                ('Order Flow', '1M'): check if row.get('OB_1M') else empty,
                ('Order Flow', '6M'): check if row.get('OB_6M') else empty,
                ('Strong Support', '3M/6M/12M'): check if row.get('Support') else empty,
                ('Reversal Candidates', '1D'): check if row.get('RevCand_1D') else empty,
                ('Reversal Candidates', '1W'): check if row.get('RevCand_1W') else empty,
                ('Reversal Candidates', '1M'): check if row.get('RevCand_1M') else empty,
                ('Squeeze', '1D'): check if row.get('Squeeze_1D') else empty,
                ('Squeeze', '1W'): check if row.get('Squeeze_1W') else empty,
            }
            if is_macro and macro_names:
                display_row[('Stock', 'Name')] = row.get('Name', row['Ticker'])
            display_data.append(display_row)
    else:  # bearish
        display_data = []
        for _, row in df.iterrows():
            display_row = {
                ('Stock', 'Ticker'): row['Ticker'],
                ('Stock', 'Price'): row['Price'],
                ('FVG Breakdowns', '1D'): check if row.get('FVG_1D') else empty,
                ('FVG Breakdowns', '1W'): check if row.get('FVG_1W') else empty,
                ('FVG Breakdowns', '1M'): check if row.get('FVG_1M') else empty,
                ('iFVG Reversals', '1D'): check if row.get('iFVG_1D') else empty,
                ('iFVG Reversals', '1W'): check if row.get('iFVG_1W') else empty,
                ('iFVG Reversals', '1M'): check if row.get('iFVG_1M') else empty,
                ('Order Flow', '1D'): check if row.get('OB_1D') else empty,
                ('Order Flow', '1W'): check if row.get('OB_1W') else empty,
                ('Order Flow', '1M'): check if row.get('OB_1M') else empty,
                ('Order Flow', '6M'): check if row.get('OB_6M') else empty,
                ('Trend Exhaustion', '1W'): check if row.get('Exhaustion') else empty,
                ('Reversal Candidates', '1D'): check if row.get('RevCand_1D') else empty,
                ('Reversal Candidates', '1W'): check if row.get('RevCand_1W') else empty,
                ('Reversal Candidates', '1M'): check if row.get('RevCand_1M') else empty,
            }
            if is_macro and macro_names:
                display_row[('Stock', 'Name')] = row.get('Name', row['Ticker'])
            display_data.append(display_row)
    
    display_df = pd.DataFrame(display_data)
    display_df.columns = pd.MultiIndex.from_tuples(display_df.columns)
    
    # Reorder columns to put Name after Ticker if macro
    if is_macro and macro_names:
        if scan_type == 'bullish':
            col_order = [
                ('Stock', 'Ticker'), ('Stock', 'Name'), ('Stock', 'Price'),
                ('FVG Breakouts', '1D'), ('FVG Breakouts', '1W'), ('FVG Breakouts', '1M'),
                ('iFVG Reversals', '1D'), ('iFVG Reversals', '1W'), ('iFVG Reversals', '1M'),
                ('Order Flow', '1D'), ('Order Flow', '1W'), ('Order Flow', '1M'), ('Order Flow', '6M'),
                ('Strong Support', '3M/6M/12M'),
                ('Reversal Candidates', '1D'), ('Reversal Candidates', '1W'), ('Reversal Candidates', '1M'),
                ('Squeeze', '1D'), ('Squeeze', '1W'),
            ]
        else:
            col_order = [
                ('Stock', 'Ticker'), ('Stock', 'Name'), ('Stock', 'Price'),
                ('FVG Breakdowns', '1D'), ('FVG Breakdowns', '1W'), ('FVG Breakdowns', '1M'),
                ('iFVG Reversals', '1D'), ('iFVG Reversals', '1W'), ('iFVG Reversals', '1M'),
                ('Order Flow', '1D'), ('Order Flow', '1W'), ('Order Flow', '1M'), ('Order Flow', '6M'),
                ('Trend Exhaustion', '1W'),
                ('Reversal Candidates', '1D'), ('Reversal Candidates', '1W'), ('Reversal Candidates', '1M'),
            ]
        display_df = display_df[col_order]
    
    return display_df


# ==========================================
# 6. MAIN DASHBOARD UI
# ==========================================
def main():
    if not check_password():
        st.stop()
        
    st.title("Prath's Market Scanner")

    if 'init_done' not in st.session_state:
        st.cache_data.clear()
        st.session_state.init_done = True
    
    # Initialize scan state
    if 'scan_triggered' not in st.session_state:
        st.session_state.scan_triggered = False
        
    market = st.sidebar.selectbox("Market Sector", [
        "US Markets (Nasdaq 100)", 
        "Indian Markets (Nifty 200)",
        "🌍 Macro Scanner (Global Indices & Commodities)"
    ])
    
    is_macro = "Macro" in market
    tickers = load_tickers(market)
    macro_names = load_macro_metadata() if is_macro else {}
    
    # Sidebar buttons
    col_scan, col_refresh = st.sidebar.columns(2)
    with col_scan:
        if st.button("🎯 Run Scan", type="primary", use_container_width=True):
            st.session_state.scan_triggered = True
    with col_refresh:
        if st.button("🔄 Refresh", type="secondary", use_container_width=True):
            st.cache_data.clear()
            st.session_state.scan_triggered = False
            st.rerun()

    # Show welcome message if scan not triggered
    if not st.session_state.scan_triggered:
        st.info("👋 Welcome! Select a market and click **🎯 Run Scan** to analyze stocks.")
        
        # Show market info
        if is_macro:
            market_name = "Macro Scanner"
            st.markdown(f"""
            ### Ready to Scan: {market_name}
            
            **Macro Assets Covered:**
            - 🇺🇸 **US Sectors:** XLC, XLY, XLP, XLE, XLF, XLV, XLI, XLK, XLB, XLRE, XLU
            - 🇮🇳 **India Sectors:** Nifty Auto, Bank, IT, Pharma, Metal, Realty, FMCG, Energy, PSU Bank, Media
            - 🌏 **Global Indices:** Nikkei 225, Dow Jones, Nifty 50, Midcap, Smallcap
            - 🥇 **Commodities:** Gold, Silver, Platinum, Copper, Natural Gas, Crude Oil
            - 💱 **Currencies:** USD/INR, CAD/INR, USD/CAD, EUR/USD, DXY
            - ₿ **Crypto:** Bitcoin, Ethereum, XRP
            - 📊 **Bonds:** US 10Y, US 30Y Treasury Yields
            
            **Timeframes:** 1D, 1W, 1M, 3M, 6M, 12M
            """)
        else:
            market_name = "Nasdaq 100" if "US" in market else "Nifty 200"
            st.markdown(f"""
            ### Ready to Scan: {market_name}
            
            **Available Scans:**
            - 🐂 **Bullish:** FVG Breakouts, iFVG Reversals, Order Flow, Strong Support, Reversal Candidates, Squeeze
            - 🐻 **Bearish:** FVG Breakdowns, iFVG Reversals, Order Flow, Trend Exhaustion, Reversal Candidates
            
            **Timeframes:** 1D, 1W, 1M, 3M, 6M, 12M
            """)
        
        if tickers:
            st.caption(f"📊 {len(tickers)} {'assets' if is_macro else 'stocks'} loaded and ready to scan")
        return

    # Run scan only when triggered
    if tickers and st.session_state.scan_triggered:
        with st.status("🚀 Running Parallel Scans...", expanded=True) as status:
            
            st.write("Fetching Market Data...")
            data_d = fetch_bulk_data(tickers, interval="1d", period="2y")
            data_m = fetch_bulk_data(tickers, interval="1mo", period="max")
            
            st.write("Processing Algorithms...")
            
            bullish_results = []
            bearish_results = []
            is_multi = len(tickers) > 1
            
            with ThreadPoolExecutor() as executor:
                bull_futures = []
                bear_futures = []
                
                for ticker in tickers:
                    try:
                        df_d = data_d[ticker] if is_multi else data_d
                        df_m = data_m[ticker] if is_multi else data_m
                        
                        if not df_d.empty and not df_m.empty:
                            bull_futures.append(executor.submit(analyze_ticker, ticker, df_d, df_m))
                            bear_futures.append(executor.submit(analyze_ticker_bearish, ticker, df_d, df_m))
                    except: continue
                
                for f in bull_futures:
                    try:
                        res = f.result()
                        if res: bullish_results.append(res)
                    except: continue
                
                for f in bear_futures:
                    try:
                        res = f.result()
                        if res: bearish_results.append(res)
                    except: continue
            
            status.update(label="✅ Analysis Complete", state="complete", expanded=False)

        # Create consolidated tables
        bull_table = create_consolidated_table(bullish_results, is_macro, macro_names, 'bullish')
        bear_table = create_consolidated_table(bearish_results, is_macro, macro_names, 'bearish')
        
        # --- DISPLAY ---
        st.header("🐂 Bullish Scans")
        st.caption("Stocks showing bullish setups across multiple timeframes")
        
        if bull_table is not None and not bull_table.empty:
            st.dataframe(
                bull_table,
                hide_index=True,
                use_container_width=True,
                height=min(len(bull_table) * 35 + 60, 600)
            )
        else:
            st.info("No bullish setups found.")
        
        st.divider()
        
        st.header("🐻 Bearish Scans")
        st.caption("Stocks showing bearish setups across multiple timeframes")
        
        if bear_table is not None and not bear_table.empty:
            st.dataframe(
                bear_table,
                hide_index=True,
                use_container_width=True,
                height=min(len(bear_table) * 35 + 60, 600)
            )
        else:
            st.info("No bearish setups found.")
        
        # Summary stats
        st.divider()
        st.subheader("📊 Scan Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            bull_count = len([r for r in bullish_results if r.get('has_signal')])
            st.metric("Bullish Setups", bull_count)
        
        with col2:
            bear_count = len([r for r in bearish_results if r.get('has_signal')])
            st.metric("Bearish Setups", bear_count)
        
        with col3:
            st.metric("Total Scanned", len(tickers))

if __name__ == "__main__":

    main()
