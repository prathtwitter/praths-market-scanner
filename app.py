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
    
    /* Compact table styling */
    .stDataFrame [data-testid="stDataFrameResizable"] {
        max-width: 100%;
    }
    div[data-testid="stDataFrame"] > div {
        overflow-x: auto;
    }
    .stDataFrame th {
        white-space: nowrap !important;
        font-size: 12px !important;
        padding: 4px 8px !important;
    }
    .stDataFrame td {
        text-align: center !important;
        padding: 4px 8px !important;
        font-size: 13px !important;
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
        subset = df.iloc[-5:].copy().dropna()
        if len(subset) < 5: return None

        c1 = subset.iloc[0]
        c3 = subset.iloc[2]
        c5 = subset.iloc[4]
        
        is_bear_fvg_first = c3['High'] < c1['Low']
        is_bull_fvg_second = c5['Low'] > c3['High']
        
        if is_bear_fvg_first and is_bull_fvg_second:
            return 'Bull'

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
        if len(df) < num_candles:
            return None
        
        recent = df.iloc[-num_candles:]
        all_red = all(recent['Close'] < recent['Open'])
        all_green = all(recent['Close'] > recent['Open'])
        
        if all_red:
            return 'Bull'
        elif all_green:
            return 'Bear'
        
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

def count_active_scanners(results, scan_type='bullish'):
    """
    Count how many DIFFERENT scanners have at least one True signal.
    Excludes Squeeze from the count.
    """
    scanners_with_signals = set()
    
    if scan_type == 'bullish':
        # Order Flow scanner
        if results.get('OB_1D') or results.get('OB_1W') or results.get('OB_1M'):
            scanners_with_signals.add('OB')
        # FVG scanner
        if results.get('FVG_1D') or results.get('FVG_1W') or results.get('FVG_1M'):
            scanners_with_signals.add('FVG')
        # Reversal Candidates scanner
        if results.get('RevCand_1D') or results.get('RevCand_1W') or results.get('RevCand_1M'):
            scanners_with_signals.add('RevCand')
        # iFVG scanner
        if results.get('iFVG_1D') or results.get('iFVG_1W') or results.get('iFVG_1M'):
            scanners_with_signals.add('iFVG')
        # Strong Support scanner
        if results.get('Support'):
            scanners_with_signals.add('Support')
        # Squeeze excluded from count
    else:  # bearish
        # Order Flow scanner
        if results.get('OB_1D') or results.get('OB_1W') or results.get('OB_1M'):
            scanners_with_signals.add('OB')
        # FVG scanner
        if results.get('FVG_1D') or results.get('FVG_1W') or results.get('FVG_1M'):
            scanners_with_signals.add('FVG')
        # Reversal Candidates scanner
        if results.get('RevCand_1D') or results.get('RevCand_1W') or results.get('RevCand_1M'):
            scanners_with_signals.add('RevCand')
        # iFVG scanner
        if results.get('iFVG_1D') or results.get('iFVG_1W') or results.get('iFVG_1M'):
            scanners_with_signals.add('iFVG')
        # Exhaustion scanner
        if results.get('Exhaustion'):
            scanners_with_signals.add('Exhaustion')
    
    return len(scanners_with_signals)


def analyze_ticker(ticker, df_daily_raw, df_monthly_raw):
    """
    Runs ALL scans for ONE ticker in a single pass.
    Returns a dictionary with ticker info and all scan results.
    """
    results = {
        'Ticker': ticker,
        'Price': 0,
        # Order Flow
        'OB_1D': False, 'OB_1W': False, 'OB_1M': False,
        # FVG Breakouts
        'FVG_1D': False, 'FVG_1W': False, 'FVG_1M': False,
        # Reversal Candidates
        'RevCand_1D': False, 'RevCand_1W': False, 'RevCand_1M': False,
        # iFVG Reversals
        'iFVG_1D': False, 'iFVG_1W': False, 'iFVG_1M': False,
        # Strong Support
        'Support': False,
        # Squeeze
        'Squeeze_1D': False, 'Squeeze_1W': False,
        # Track signals
        'has_signal': False,
        'scanner_count': 0
    }
    
    try:
        d_1d = resample_custom(df_daily_raw, "1D")
        d_1w = resample_custom(df_daily_raw, "1W")
        d_1m = resample_custom(df_monthly_raw, "1M")
        d_3m = resample_custom(df_monthly_raw, "3M")
        d_6m = resample_custom(df_monthly_raw, "6M")
        d_12m = resample_custom(df_monthly_raw, "12M")
        
        data_map = {
            "1D": d_1d, "1W": d_1w, "1M": d_1m, 
            "3M": d_3m, "6M": d_6m, "12M": d_12m
        }
        
        if not d_1d.empty:
            results['Price'] = round(d_1d['Close'].iloc[-1], 2)
        
    except: 
        return results

    for tf in ["1D", "1W", "1M"]:
        if tf not in data_map or len(data_map[tf]) < 5: continue
        
        df = data_map[tf].copy() 
        df['Bull_FVG'], df['Bear_FVG'] = MathWiz.find_fvg(df)
        df['Is_Swing_High'], df['Is_Swing_Low'] = MathWiz.identify_strict_swings(df)
        
        curr = df.iloc[-1]
        prev = df.iloc[-2]

        # FVG ENTRIES
        if curr['Bull_FVG']:
            past_swings = df[df['Is_Swing_High']]
            if not past_swings.empty:
                last_swing_high = past_swings['High'].iloc[-1]
                if curr['Close'] > last_swing_high and prev['Close'] <= last_swing_high:
                    results[f'FVG_{tf}'] = True
                    results['has_signal'] = True

        # Order Flow
        subset = df.iloc[-4:].copy()
        if len(subset) == 4:
            c_anc, c1, c2, c3 = subset.iloc[0], subset.iloc[1], subset.iloc[2], subset.iloc[3]
            if (c_anc['Close'] < c_anc['Open'] and 
                c1['Close'] > c1['Open'] and c2['Close'] > c2['Open'] and c3['Close'] > c3['Open']):
                if c3['Low'] > c1['High'] and c3['Close'] > c_anc['High']:
                    results[f'OB_{tf}'] = True
                    results['has_signal'] = True
        
        # iFVG REVERSALS
        ifvg_status = MathWiz.check_ifvg_reversal(df)
        if ifvg_status == "Bull":
            results[f'iFVG_{tf}'] = True
            results['has_signal'] = True

    # STRONG SUPPORT
    sup_tf = []
    if MathWiz.find_unmitigated_fvg_zone(d_3m): sup_tf.append("3M")
    if MathWiz.find_unmitigated_fvg_zone(d_6m): sup_tf.append("6M")
    if MathWiz.find_unmitigated_fvg_zone(d_12m): sup_tf.append("12M")
    
    if len(sup_tf) >= 2:
        results['Support'] = True
        results['has_signal'] = True

    # SQUEEZE
    if not d_1d.empty:
        chop_series_d = MathWiz.calculate_choppiness(d_1d['High'], d_1d['Low'], d_1d['Close'])
        if not chop_series_d.empty and not pd.isna(chop_series_d.iloc[-1]):
            if chop_series_d.iloc[-1] > 59:
                results['Squeeze_1D'] = True
                results['has_signal'] = True
                
    if not d_1w.empty:
        chop_series_w = MathWiz.calculate_choppiness(d_1w['High'], d_1w['Low'], d_1w['Close'])
        if not chop_series_w.empty and not pd.isna(chop_series_w.iloc[-1]):
            if chop_series_w.iloc[-1] > 59:
                results['Squeeze_1W'] = True
                results['has_signal'] = True

    # REVERSAL CANDIDATES
    if not d_1d.empty and len(d_1d) >= 5:
        if MathWiz.check_consecutive_candles(d_1d, 5) == 'Bull':
            results['RevCand_1D'] = True
            results['has_signal'] = True
    
    if not d_1w.empty and len(d_1w) >= 4:
        if MathWiz.check_consecutive_candles(d_1w, 4) == 'Bull':
            results['RevCand_1W'] = True
            results['has_signal'] = True
    
    if not d_1m.empty and len(d_1m) >= 3:
        if MathWiz.check_consecutive_candles(d_1m, 3) == 'Bull':
            results['RevCand_1M'] = True
            results['has_signal'] = True

    # Count active scanners (excluding Squeeze)
    results['scanner_count'] = count_active_scanners(results, 'bullish')

    return results


def analyze_ticker_bearish(ticker, df_daily_raw, df_monthly_raw):
    """
    Runs ALL BEARISH scans for ONE ticker in a single pass.
    """
    results = {
        'Ticker': ticker,
        'Price': 0,
        'OB_1D': False, 'OB_1W': False, 'OB_1M': False,
        'FVG_1D': False, 'FVG_1W': False, 'FVG_1M': False,
        'RevCand_1D': False, 'RevCand_1W': False, 'RevCand_1M': False,
        'iFVG_1D': False, 'iFVG_1W': False, 'iFVG_1M': False,
        'Exhaustion': False,
        'has_signal': False,
        'scanner_count': 0
    }
    
    try:
        d_1d = resample_custom(df_daily_raw, "1D")
        d_1w = resample_custom(df_daily_raw, "1W")
        d_1m = resample_custom(df_monthly_raw, "1M")
        d_3m = resample_custom(df_monthly_raw, "3M")
        d_6m = resample_custom(df_monthly_raw, "6M")
        d_12m = resample_custom(df_monthly_raw, "12M")
        
        data_map = {"1D": d_1d, "1W": d_1w, "1M": d_1m, "3M": d_3m, "6M": d_6m, "12M": d_12m}
        
        if not d_1d.empty:
            results['Price'] = round(d_1d['Close'].iloc[-1], 2)
        
    except: 
        return results

    for tf in ["1D", "1W", "1M"]:
        if tf not in data_map or len(data_map[tf]) < 5: continue
        
        df = data_map[tf].copy() 
        df['Bull_FVG'], df['Bear_FVG'] = MathWiz.find_fvg(df)
        df['Is_Swing_High'], df['Is_Swing_Low'] = MathWiz.identify_strict_swings(df)
        
        curr = df.iloc[-1]
        prev = df.iloc[-2]

        # FVG BREAKDOWNS
        if curr['Bear_FVG']:
            past_swings = df[df['Is_Swing_Low']]
            if not past_swings.empty:
                last_swing_low = past_swings['Low'].iloc[-1]
                if curr['Close'] < last_swing_low and prev['Close'] >= last_swing_low:
                    results[f'FVG_{tf}'] = True
                    results['has_signal'] = True

        # Order Flow (Bearish)
        subset = df.iloc[-4:].copy()
        if len(subset) == 4:
            c_anc, c1, c2, c3 = subset.iloc[0], subset.iloc[1], subset.iloc[2], subset.iloc[3]
            if (c_anc['Close'] > c_anc['Open'] and 
                c1['Close'] < c1['Open'] and c2['Close'] < c2['Open'] and c3['Close'] < c3['Open']):
                if c3['High'] < c1['Low'] and c3['Close'] < c_anc['Low']:
                    results[f'OB_{tf}'] = True
                    results['has_signal'] = True
        
        # iFVG REVERSALS
        ifvg_status = MathWiz.check_ifvg_reversal(df)
        if ifvg_status == "Bear":
            results[f'iFVG_{tf}'] = True
            results['has_signal'] = True

    # EXHAUSTION
    if not d_1w.empty:
        chop_series = MathWiz.calculate_choppiness(d_1w['High'], d_1w['Low'], d_1w['Close'])
        if not chop_series.empty and not pd.isna(chop_series.iloc[-1]):
            if chop_series.iloc[-1] < 25:
                results['Exhaustion'] = True
                results['has_signal'] = True

    # REVERSAL CANDIDATES (Overbought)
    if not d_1d.empty and len(d_1d) >= 5:
        if MathWiz.check_consecutive_candles(d_1d, 5) == 'Bear':
            results['RevCand_1D'] = True
            results['has_signal'] = True
    
    if not d_1w.empty and len(d_1w) >= 4:
        if MathWiz.check_consecutive_candles(d_1w, 4) == 'Bear':
            results['RevCand_1W'] = True
            results['has_signal'] = True
    
    if not d_1m.empty and len(d_1m) >= 3:
        if MathWiz.check_consecutive_candles(d_1m, 3) == 'Bear':
            results['RevCand_1M'] = True
            results['has_signal'] = True

    results['scanner_count'] = count_active_scanners(results, 'bearish')

    return results


def create_consolidated_table(results_list, is_macro=False, macro_names=None, scan_type='bullish'):
    """
    Creates a consolidated DataFrame with timeframes as main columns and scanners as sub-columns.
    Only includes tickers with signals from 2+ DIFFERENT scanners (excluding Squeeze).
    """
    if not results_list:
        return None
    
    # Filter: must have signals from 2+ different scanners
    filtered = [r for r in results_list if r.get('scanner_count', 0) >= 2]
    
    if not filtered:
        return None
    
    filtered = sorted(filtered, key=lambda x: x.get('scanner_count', 0), reverse=True)
    df = pd.DataFrame(filtered)
    
    if is_macro and macro_names:
        df['Name'] = df['Ticker'].map(macro_names).fillna(df['Ticker'])
    
    check = "✅"
    empty = ""
    
    if scan_type == 'bullish':
        display_data = []
        for _, row in df.iterrows():
            display_row = {
                ('Info', 'Ticker'): row['Ticker'],
                ('Info', 'Price'): row['Price'],
                # 1D timeframe
                ('1D', 'OB'): check if row.get('OB_1D') else empty,
                ('1D', 'FVG'): check if row.get('FVG_1D') else empty,
                ('1D', 'Rev'): check if row.get('RevCand_1D') else empty,
                ('1D', 'iFVG'): check if row.get('iFVG_1D') else empty,
                ('1D', 'Sqz'): check if row.get('Squeeze_1D') else empty,
                # 1W timeframe
                ('1W', 'OB'): check if row.get('OB_1W') else empty,
                ('1W', 'FVG'): check if row.get('FVG_1W') else empty,
                ('1W', 'Rev'): check if row.get('RevCand_1W') else empty,
                ('1W', 'iFVG'): check if row.get('iFVG_1W') else empty,
                ('1W', 'Sqz'): check if row.get('Squeeze_1W') else empty,
                # 1M timeframe
                ('1M', 'OB'): check if row.get('OB_1M') else empty,
                ('1M', 'FVG'): check if row.get('FVG_1M') else empty,
                ('1M', 'Rev'): check if row.get('RevCand_1M') else empty,
                ('1M', 'iFVG'): check if row.get('iFVG_1M') else empty,
                # Long-term
                ('LT', 'Sup'): check if row.get('Support') else empty,
            }
            if is_macro and macro_names:
                display_row[('Info', 'Name')] = row.get('Name', row['Ticker'])
            display_data.append(display_row)
            
        display_df = pd.DataFrame(display_data)
        display_df.columns = pd.MultiIndex.from_tuples(display_df.columns)
        
        if is_macro and macro_names:
            col_order = [
                ('Info', 'Ticker'), ('Info', 'Name'), ('Info', 'Price'),
                ('1D', 'OB'), ('1D', 'FVG'), ('1D', 'Rev'), ('1D', 'iFVG'), ('1D', 'Sqz'),
                ('1W', 'OB'), ('1W', 'FVG'), ('1W', 'Rev'), ('1W', 'iFVG'), ('1W', 'Sqz'),
                ('1M', 'OB'), ('1M', 'FVG'), ('1M', 'Rev'), ('1M', 'iFVG'),
                ('LT', 'Sup'),
            ]
            display_df = display_df[col_order]
            
    else:  # bearish
        display_data = []
        for _, row in df.iterrows():
            display_row = {
                ('Info', 'Ticker'): row['Ticker'],
                ('Info', 'Price'): row['Price'],
                # 1D timeframe
                ('1D', 'OB'): check if row.get('OB_1D') else empty,
                ('1D', 'FVG'): check if row.get('FVG_1D') else empty,
                ('1D', 'Rev'): check if row.get('RevCand_1D') else empty,
                ('1D', 'iFVG'): check if row.get('iFVG_1D') else empty,
                # 1W timeframe
                ('1W', 'OB'): check if row.get('OB_1W') else empty,
                ('1W', 'FVG'): check if row.get('FVG_1W') else empty,
                ('1W', 'Rev'): check if row.get('RevCand_1W') else empty,
                ('1W', 'iFVG'): check if row.get('iFVG_1W') else empty,
                ('1W', 'Exh'): check if row.get('Exhaustion') else empty,
                # 1M timeframe
                ('1M', 'OB'): check if row.get('OB_1M') else empty,
                ('1M', 'FVG'): check if row.get('FVG_1M') else empty,
                ('1M', 'Rev'): check if row.get('RevCand_1M') else empty,
                ('1M', 'iFVG'): check if row.get('iFVG_1M') else empty,
            }
            if is_macro and macro_names:
                display_row[('Info', 'Name')] = row.get('Name', row['Ticker'])
            display_data.append(display_row)
            
        display_df = pd.DataFrame(display_data)
        display_df.columns = pd.MultiIndex.from_tuples(display_df.columns)
        
        if is_macro and macro_names:
            col_order = [
                ('Info', 'Ticker'), ('Info', 'Name'), ('Info', 'Price'),
                ('1D', 'OB'), ('1D', 'FVG'), ('1D', 'Rev'), ('1D', 'iFVG'),
                ('1W', 'OB'), ('1W', 'FVG'), ('1W', 'Rev'), ('1W', 'iFVG'), ('1W', 'Exh'),
                ('1M', 'OB'), ('1M', 'FVG'), ('1M', 'Rev'), ('1M', 'iFVG'),
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
    
    col_scan, col_refresh = st.sidebar.columns(2)
    with col_scan:
        if st.button("🎯 Run Scan", type="primary", use_container_width=True):
            st.session_state.scan_triggered = True
    with col_refresh:
        if st.button("🔄 Refresh", type="secondary", use_container_width=True):
            st.cache_data.clear()
            st.session_state.scan_triggered = False
            st.rerun()

    # Legend in sidebar
    st.sidebar.divider()
    st.sidebar.subheader("📖 Legend")
    st.sidebar.markdown("""
    **OB** = Order Flow  
    **FVG** = Fair Value Gap  
    **Rev** = Reversal Candidate  
    **iFVG** = Inverse FVG  
    **Sqz** = Squeeze  
    **Sup** = Strong Support  
    **Exh** = Trend Exhaustion
    """)

    if not st.session_state.scan_triggered:
        st.info("👋 Welcome! Select a market and click **🎯 Run Scan** to analyze stocks.")
        
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
            """)
        else:
            market_name = "Nasdaq 100" if "US" in market else "Nifty 200"
            st.markdown(f"""
            ### Ready to Scan: {market_name}
            
            **Scan Types:** Order Flow, FVG, Reversal Candidates, iFVG, Strong Support, Squeeze
            
            **Timeframes:** 1D, 1W, 1M (Long-term for Support)
            
            **Filter:** Only shows stocks with signals from 2+ different scan types
            """)
        
        if tickers:
            st.caption(f"📊 {len(tickers)} {'assets' if is_macro else 'stocks'} loaded and ready to scan")
        return

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

        bull_table = create_consolidated_table(bullish_results, is_macro, macro_names, 'bullish')
        bear_table = create_consolidated_table(bearish_results, is_macro, macro_names, 'bearish')
        
        st.header("🐂 Bullish Scans")
        st.caption("Stocks with signals from 2+ different scan types (cross-scanner confluence)")
        
        if bull_table is not None and not bull_table.empty:
            st.dataframe(
                bull_table,
                hide_index=True,
                use_container_width=True,
                height=min(len(bull_table) * 35 + 60, 500)
            )
        else:
            st.info("No stocks found with cross-scanner confluence.")
        
        st.divider()
        
        st.header("🐻 Bearish Scans")
        st.caption("Stocks with signals from 2+ different scan types (cross-scanner confluence)")
        
        if bear_table is not None and not bear_table.empty:
            st.dataframe(
                bear_table,
                hide_index=True,
                use_container_width=True,
                height=min(len(bear_table) * 35 + 60, 500)
            )
        else:
            st.info("No stocks found with cross-scanner confluence.")
        
        st.divider()
        st.subheader("📊 Scan Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            bull_count = len([r for r in bullish_results if r.get('scanner_count', 0) >= 2])
            st.metric("Bullish Confluence", bull_count)
        
        with col2:
            bear_count = len([r for r in bearish_results if r.get('scanner_count', 0) >= 2])
            st.metric("Bearish Confluence", bear_count)
        
        with col3:
            total_signals = len([r for r in bullish_results if r.get('has_signal')]) + len([r for r in bearish_results if r.get('has_signal')])
            st.metric("Total Signals", total_signals)
        
        with col4:
            st.metric("Scanned", len(tickers))

if __name__ == "__main__":
    main()