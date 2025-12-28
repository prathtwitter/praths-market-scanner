import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import os
import datetime
from concurrent.futures import ThreadPoolExecutor

# --- AI LIBRARIES ---
try:
    import google.generativeai as genai
    from duckduckgo_search import DDGS
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

# ==========================================
# 1. CONFIGURATION & STYLE
# ==========================================
st.set_page_config(page_title="Prath's Sniper v6.2", layout="wide", page_icon="🎯")
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #FAFAFA; }
    div[data-testid="stMetricValue"] { font-size: 18px; color: #00CC96; }
    
    /* Button Styling */
    .stButton>button { 
        width: 100%; 
        border-radius: 8px; 
        font-weight: bold;
        height: 45px;
        background-color: #1f2937;
        color: white;
        border: 1px solid #374151;
        margin-top: 5px;
        margin-bottom: 15px;
    }
    .stButton>button:hover {
        border-color: #00CC96;
        color: #00CC96;
    }
    
    /* AI Box Styling */
    .ai-box {
        border: 2px solid #00CC96;
        background-color: #162b26;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    
    /* Headers & Text */
    h4 { padding-top: 10px; margin-bottom: 0px; color: #e5e7eb; }
    div[data-testid="stCaptionContainer"] {
        color: #9ca3af;
        font-size: 13px;
        margin-bottom: 10px;
        line-height: 1.4;
    }
    
    /* Image Container */
    .stImage img { border-radius: 5px; border: 1px solid #333; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SECURITY & AUTH
# ==========================================
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False
    if st.session_state.password_correct:
        return True
    def password_entered():
        correct_password = st.secrets.get("PASSWORD", "Sniper2025")
        if st.session_state["password"] == correct_password:
            st.session_state.password_correct = True
            del st.session_state["password"]
        else:
            st.session_state.password_correct = False
    st.text_input("Enter Access Key", type="password", on_change=password_entered, key="password")
    if not st.session_state.password_correct:
        st.error("⛔ Access Denied")
        return False
    return True

# ==========================================
# 3. AI ANALYST ENGINE
# ==========================================
class AI_Analyst:
    @staticmethod
    def get_earnings_date_bulk(tickers):
        results = {}
        def fetch_single(ticker):
            try:
                stock = yf.Ticker(ticker)
                dates_df = stock.get_earnings_dates(limit=8)
                if dates_df is not None and not dates_df.empty:
                    now = pd.Timestamp.now().tz_localize(dates_df.index.dtype.tz) if dates_df.index.tz else pd.Timestamp.now()
                    future = dates_df[dates_df.index > now].sort_index()
                    if not future.empty:
                        nxt = future.index[0]
                        delta = (nxt - now).days
                        return ticker, f"{nxt.strftime('%b %d')} ({delta}d)"
                cal = stock.calendar
                if cal is not None and not cal.empty:
                    val = cal.iloc[0, 0] if isinstance(cal, pd.DataFrame) else cal.get('Earnings Date', [None])[0]
                    if val and isinstance(val, (datetime.date, datetime.datetime)):
                        delta = (val - datetime.date.today()).days
                        return ticker, f"{val.strftime('%b %d')} ({delta}d)"
                return ticker, "-"
            except:
                return ticker, "-"

        with ThreadPoolExecutor(max_workers=8) as ex:
            future_to_ticker = {ex.submit(fetch_single, t): t for t in tickers}
            for future in future_to_ticker:
                results[future.result()[0]] = future.result()[1]
        return results

    @staticmethod
    def generate_top_pick(df, scan_name):
        if not AI_AVAILABLE: return "⚠️ AI Library Missing"
        if "GEMINI_API_KEY" not in st.secrets: return "⚠️ API Key Missing"
        
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        
        data_str = df[['Ticker', 'Price', 'Chop', 'TF', 'Info']].to_string(index=False)
        
        prompt = f"""
        You are an elite Sniper Trader.
        I just ran a scan for **{scan_name}**.
        Here are the results:
        
        {data_str}
        
        **Your Mission:**
        1.  Analyze setups based on Price, Chop (Choppiness Index), and Timeframe.
        2.  **Filter:** Reject stocks with earnings in < 3 days.
        3.  **Logic:** - FVG Breakouts: Look for strong displacement relative to the opposing gap.
            - Reversals/Squeezes: Prefer Chop > 60.
        4.  **Select ONE Top Pick.**
        
        **Output Format (Strict):**
        **🏆 Top Pick: [TICKER]**
        **Rationale:** [2 sentences technical reason.]
        **Trade Plan:** [Stop Loss idea].
        """
        
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            res = model.generate_content(prompt)
            return res.text
        except Exception as e:
            return f"AI Error: {str(e)}"

# ==========================================
# 4. MATH WIZ & LOGIC
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
    def find_fvg(df):
        """
        Fair Value Gap Detection:
        - Bullish FVG: Low of candle 3 > High of candle 1 (gap up, imbalance)
        - Bearish FVG: High of candle 3 < Low of candle 1 (gap down, imbalance)
        
        Returns boolean series where True at index i means candle i is the 3rd candle of an FVG pattern
        """
        # Bullish: Low[i] > High[i-2] (current candle's low is above candle 1's high)
        bull_fvg = (df['Low'] > df['High'].shift(2))
        # Bearish: High[i] < Low[i-2] (current candle's high is below candle 1's low)
        bear_fvg = (df['High'] < df['Low'].shift(2))
        return bull_fvg, bear_fvg
    
    @staticmethod
    def check_ifvg_reversal(df):
        subset = df.iloc[-5:].copy().dropna()
        if len(subset) < 5: return None
        c1, c3, c5 = subset.iloc[0], subset.iloc[2], subset.iloc[4]
        if (c3['High'] < c1['Low']) and (c5['Low'] > c3['High']): return 'Bull'
        if (c3['Low'] > c1['High']) and (c5['High'] < c3['Low']): return 'Bear'
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
                gap_top, gap_bottom = curr_low, prev_high
                if not df.iloc[i+1:].empty and (df.iloc[i+1:]['Low'] < gap_bottom).any(): continue 
                upper, lower = gap_bottom * (1 + threshold_pct), gap_bottom * (1 - threshold_pct)
                if lower <= current_price <= upper: return True 
        return False

# ==========================================
# 5. DATA ENGINE
# ==========================================
@st.cache_data
def load_tickers(market_choice):
    if "India" in market_choice:
        filename = "ind_tickers.csv"
    else:
        filename = "us_tickers.csv"
    if not os.path.exists(filename): return []
    try:
        df = pd.read_csv(filename)
        clean = [str(t).strip().upper() for t in df.iloc[:, 0].tolist()]
        if "India" in market_choice:
            clean = [t if t.endswith(".NS") else f"{t}.NS" for t in clean]
            clean = clean[:200] 
        return clean
    except: return []

@st.cache_data(ttl=3600)
def fetch_bulk_data(tickers):
    if not tickers: return None
    # Auto adjust ensures we get proper prices
    return yf.download(tickers, period="2y", interval="1d", group_by='ticker', progress=False, threads=True, auto_adjust=True)

@st.cache_data(ttl=3600)
def fetch_monthly_data(tickers):
    if not tickers: return None
    return yf.download(tickers, period="max", interval="1mo", group_by='ticker', progress=False, threads=True, auto_adjust=True)

def resample_custom(df, timeframe):
    if df.empty: return df
    agg = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
    try:
        if timeframe == "1D": return df.resample("1D").agg(agg).dropna()
        if timeframe == "1W": return df.resample("W-FRI").agg(agg).dropna()
        if timeframe == "1M": return df.resample("ME").agg(agg).dropna()
        
        # For quarterly and above, first resample to monthly
        df_m = df.resample('MS').agg(agg).dropna()
        
        if timeframe == "3M": 
            return df_m.resample('QS').agg(agg).dropna()
        
        if timeframe == "6M":
            # Manual 6-month grouping: H1 = Jan-Jun, H2 = Jul-Dec
            df_m = df_m.copy()
            df_m['year'] = df_m.index.year
            df_m['half'] = np.where(df_m.index.month <= 6, 1, 2)
            
            # Group by year and half
            grouped = df_m.groupby(['year', 'half']).agg({
                'Open': 'first',
                'High': 'max', 
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            })
            
            # Create proper datetime index for the result
            new_index = []
            for (year, half) in grouped.index:
                month = 1 if half == 1 else 7
                new_index.append(pd.Timestamp(year=year, month=month, day=1))
            
            grouped.index = pd.DatetimeIndex(new_index)
            return grouped.sort_index()
        
        if timeframe == "12M": 
            return df_m.resample('YS').agg(agg).dropna()
    except Exception as e:
        return df
    return df

# ==========================================
# 6. SCAN LOGIC
# ==========================================
def scan_logic(ticker, df_d, df_m, scan_type):
    try:
        data_map = {
            "1D": resample_custom(df_d, "1D"),
            "1W": resample_custom(df_d, "1W"),
            "1M": resample_custom(df_m, "1M"),
            "3M": resample_custom(df_m, "3M"),
            "6M": resample_custom(df_m, "6M"),
            "12M": resample_custom(df_m, "12M")
        }
    except: return []

    results = []
    
    for tf in ["1D", "1W", "1M", "6M"]:
        if tf not in data_map or len(data_map[tf]) < 5: continue
        df = data_map[tf].copy()
        
        # Calculate Indicators
        chop = 0
        if "Chop" not in df.columns:
            chop_s = MathWiz.calculate_choppiness(df['High'], df['Low'], df['Close'])
            chop = round(chop_s.iloc[-1], 2) if not chop_s.empty else 0
        
        curr = df.iloc[-1]
        price = round(curr['Close'], 2)

        # ----------------------------------------
        # FVG SCAN (OPPOSING FVG BREAKOUT LOGIC)
        # ----------------------------------------
        if "FVG" in scan_type:
            # Need at least 3 candles for FVG detection
            if len(df) < 3:
                continue
            
            # Get the last 3 candles for FVG check
            # Candle indices: -3 (oldest), -2 (middle), -1 (current/latest)
            candle_1_high = df['High'].iloc[-3]  # High of candle 1 (oldest of the 3)
            candle_1_low = df['Low'].iloc[-3]    # Low of candle 1
            candle_3_high = df['High'].iloc[-1]  # High of candle 3 (current)
            candle_3_low = df['Low'].iloc[-1]    # Low of candle 3 (current)
            
            # BULLISH FVG: Low of candle 3 > High of candle 1 (gap UP between them)
            # This means there's unfilled space - price jumped up leaving a gap
            latest_has_bull_fvg = candle_3_low > candle_1_high
            
            # BEARISH FVG: High of candle 3 < Low of candle 1 (gap DOWN between them)
            # This means there's unfilled space - price dropped leaving a gap
            latest_has_bear_fvg = candle_3_high < candle_1_low
            
            # Pre-calculate historical FVGs
            df['Bull_FVG'], df['Bear_FVG'] = MathWiz.find_fvg(df)
            
            # --- BULLISH FVG BREAKOUT ---
            # Condition: Latest candle creates Bullish FVG AND breaks swing high from most recent Bearish FVG
            if "Bullish" in scan_type and latest_has_bull_fvg:
                # Find most recent Bearish FVG BEFORE the current candle
                bear_fvg_indices = np.where(df['Bear_FVG'].values)[0]
                # Exclude current candle (index -1 = len(df)-1)
                bear_fvg_indices = bear_fvg_indices[bear_fvg_indices < len(df) - 1]
                
                if len(bear_fvg_indices) > 0:
                    # Get the most recent bearish FVG (this is candle 3 of that pattern)
                    last_bear_fvg_candle3_idx = bear_fvg_indices[-1]
                    # Candle 1 of that bearish FVG pattern is 2 positions before its candle 3
                    bear_fvg_candle1_idx = last_bear_fvg_candle3_idx - 2
                    
                    if bear_fvg_candle1_idx >= 0:
                        # Swing High = High of Candle 1 of the Bearish FVG
                        swing_high = df['High'].iloc[bear_fvg_candle1_idx]
                        
                        # Trigger: Current candle's Close breaks above this Swing High
                        if curr['Close'] > swing_high:
                            info_txt = f"BullFVG✓ SwingHi:{round(swing_high, 2)}"
                            results.append({"Ticker": ticker, "Price": price, "Chop": chop, "TF": tf, "Info": info_txt})

            # --- BEARISH FVG BREAKDOWN ---
            # Condition: Latest candle creates Bearish FVG AND breaks swing low from most recent Bullish FVG
            if "Bearish" in scan_type and latest_has_bear_fvg:
                # Find most recent Bullish FVG BEFORE the current candle
                bull_fvg_indices = np.where(df['Bull_FVG'].values)[0]
                # Exclude current candle
                bull_fvg_indices = bull_fvg_indices[bull_fvg_indices < len(df) - 1]
                
                if len(bull_fvg_indices) > 0:
                    # Get the most recent bullish FVG (this is candle 3 of that pattern)
                    last_bull_fvg_candle3_idx = bull_fvg_indices[-1]
                    # Candle 1 of that bullish FVG pattern is 2 positions before its candle 3
                    bull_fvg_candle1_idx = last_bull_fvg_candle3_idx - 2
                    
                    if bull_fvg_candle1_idx >= 0:
                        # Swing Low = Low of Candle 1 of the Bullish FVG
                        swing_low = df['Low'].iloc[bull_fvg_candle1_idx]
                        
                        # Trigger: Current candle's Close breaks below this Swing Low
                        if curr['Close'] < swing_low:
                            info_txt = f"BearFVG✓ SwingLo:{round(swing_low, 2)}"
                            results.append({"Ticker": ticker, "Price": price, "Chop": chop, "TF": tf, "Info": info_txt})

        # ----------------------------------------
        # ORDER BLOCKS
        # ----------------------------------------
        if "Order Block" in scan_type:
            sub = df.iloc[-4:].copy()
            if len(sub) == 4:
                c0, c1, c2, c3 = sub.iloc[0], sub.iloc[1], sub.iloc[2], sub.iloc[3]
                if "Bullish" in scan_type:
                    if (c0['Close']<c0['Open'] and c1['Close']>c1['Open'] and c2['Close']>c2['Open'] and c3['Close']>c3['Open']):
                        if c3['Low']>c1['High'] and c3['Close']>c0['High']:
                             results.append({"Ticker": ticker, "Price": price, "Chop": chop, "TF": tf, "Info": ""})
                if "Bearish" in scan_type:
                    if (c0['Close']>c0['Open'] and c1['Close']<c1['Open'] and c2['Close']<c2['Open'] and c3['Close']<c3['Open']):
                        if c3['High']<c1['Low'] and c3['Close']<c0['Low']:
                             results.append({"Ticker": ticker, "Price": price, "Chop": chop, "TF": tf, "Info": ""})

        # ----------------------------------------
        # iFVG REVERSAL
        # ----------------------------------------
        if "iFVG" in scan_type and tf in ["1D", "1W", "1M"]:
            res = MathWiz.check_ifvg_reversal(df)
            if "Bullish" in scan_type and res == "Bull":
                results.append({"Ticker": ticker, "Price": price, "Chop": chop, "TF": tf, "Info": ""})
            if "Bearish" in scan_type and res == "Bear":
                results.append({"Ticker": ticker, "Price": price, "Chop": chop, "TF": tf, "Info": ""})

    if "Support" in scan_type:
        sup_tf = []
        for t in ["3M", "6M", "12M"]:
            if MathWiz.find_unmitigated_fvg_zone(data_map[t]): sup_tf.append(t)
        if len(sup_tf) >= 2:
            c3m = MathWiz.calculate_choppiness(data_map["3M"]['High'], data_map["3M"]['Low'], data_map["3M"]['Close'])
            c_val = round(c3m.iloc[-1], 2) if not c3m.empty else 0
            results.append({"Ticker": ticker, "Price": round(data_map["3M"]['Close'].iloc[-1], 2), "Chop": c_val, "TF": "Multi", "Info": ",".join(sup_tf)})

    if "Squeeze" in scan_type:
         if not data_map["1D"].empty:
            c_d = MathWiz.calculate_choppiness(data_map["1D"]['High'], data_map["1D"]['Low'], data_map["1D"]['Close']).iloc[-1]
            if c_d > 59: results.append({"Ticker": ticker, "Price": round(data_map["1D"]['Close'].iloc[-1], 2), "Chop": round(c_d, 2), "TF": "1D", "Info": ""})
         if not data_map["1W"].empty:
            c_w = MathWiz.calculate_choppiness(data_map["1W"]['High'], data_map["1W"]['Low'], data_map["1W"]['Close']).iloc[-1]
            if c_w > 59: results.append({"Ticker": ticker, "Price": round(data_map["1W"]['Close'].iloc[-1], 2), "Chop": round(c_w, 2), "TF": "1W", "Info": ""})

    return results

# ==========================================
# 7. MAIN UI
# ==========================================
def main():
    if not check_password(): st.stop()
    
    st.sidebar.title("🌍 Global Pulse")
    global_tickers = {
        "NIFTY 50": "^NSEI", "BANKNIFTY": "^NSEBANK", "DJI": "^DJI", "SPX": "^GSPC",
        "USD/INR": "USDINR=X", "CAD/INR": "CADINR=X", "GOLD": "GC=F", "SILVER": "SI=F",
        "BTC/USD": "BTC-USD"
    }
    
    g_data = yf.download(list(global_tickers.values()), period="5d", interval="1d", progress=False)
    for name, sym in global_tickers.items():
        try:
            if isinstance(g_data.columns, pd.MultiIndex): hist = g_data['Close'][sym].dropna()
            else: hist = g_data['Close'].dropna()
            if not hist.empty:
                st.sidebar.metric(name, f"{hist.iloc[-1]:,.2f}", f"{(hist.iloc[-1]-hist.iloc[-2]):+.2f}")
        except: pass
    
    st.sidebar.divider()

    st.title("Prath's Sniper v6.2")
    
    col_mkt, col_status = st.columns([1, 2])
    with col_mkt:
        market = st.selectbox("Select Market", ["US Markets (Nasdaq 100)", "Indian Markets (Nifty 200)"])
    
    tickers = load_tickers(market)
    if not tickers:
        st.error("No tickers found. Check CSV files.")
        st.stop()
        
    st.write("### 🎯 Select Strategy")
    
    repo_base = "https://raw.githubusercontent.com/prathtwitter/praths-market-scanner/main/"
    
    c1, c2, c3 = st.columns(3)
    scan_request = None
    
    # --- BULLISH COLUMN ---
    with c1:
        st.write("#### 🐂 Bullish")
        
        if st.button("Bullish FVG", help="Latest candle created a Bullish FVG AND broke the High of the most recent Bearish FVG."): scan_request = "Bullish FVG"
        if st.button("Bullish OB", help="Last down-candle before an impulsive upward move that broke structure."): scan_request = "Bullish Order Block"
        if st.button("Bullish iFVG", help="A failed Bearish Gap that price has reclaimed and flipped into support."): scan_request = "Bullish iFVG"

    # --- BEARISH COLUMN ---
    with c2:
        st.write("#### 🐻 Bearish")
        
        if st.button("Bearish FVG", help="Latest candle created a Bearish FVG AND broke the Low of the most recent Bullish FVG."): scan_request = "Bearish FVG"
        if st.button("Bearish OB", help="Last up-candle before an impulsive downward move that broke structure."): scan_request = "Bearish Order Block"
        if st.button("Bearish iFVG", help="A failed Bullish Gap that price has broken below and flipped into resistance."): scan_request = "Bearish iFVG"

    # --- VOLATILITY COLUMN ---
    with c3:
        st.write("#### ⚡ Volatility")
        
        if st.button("Strong Support", help="Price reacting off a high-timeframe (3M/6M) unmitigated Gap."): scan_request = "Strong Support"
        if st.button("Squeeze", help="Price coiling in an extremely tight range (Chop > 60)."): scan_request = "Squeeze"

    # --- EXECUTION ---
    if scan_request:
        st.divider()
        st.subheader(f"🚀 Running Scan: {scan_request}...")
        
        with st.spinner("Fetching Market Data..."):
            data_d = fetch_bulk_data(tickers)
            data_m = fetch_monthly_data(tickers)
        
        if data_d is None or data_d.empty:
            st.error("Failed to fetch market data. Please try again later.")
            st.stop()

        all_res = []
        is_multi = len(tickers) > 1
        progress_bar = st.progress(0)
        
        with ThreadPoolExecutor() as ex:
            futures = []
            for t in tickers:
                try:
                    df_d = data_d[t] if is_multi else data_d
                    df_m = data_m[t] if is_multi else data_m
                    if not df_d.empty:
                        futures.append(ex.submit(scan_logic, t, df_d, df_m, scan_request))
                except: continue
            
            for i, f in enumerate(futures):
                try:
                    res = f.result()
                    if res: all_res.extend(res)
                except: pass
                progress_bar.progress((i + 1) / len(tickers))
        
        progress_bar.empty()
        df_final = pd.DataFrame(all_res)
        
        if df_final.empty:
            st.warning("No high-probability setups found right now. (Strict Filters Applied)")
        else:
            with st.spinner("Fetching Earnings Dates..."):
                u_tickers = df_final['Ticker'].unique().tolist()
                e_map = AI_Analyst.get_earnings_date_bulk(u_tickers)
                # Ensure we don't overwrite specific info if it exists, otherwise use Earnings
                df_final['Earnings'] = df_final['Ticker'].map(e_map)
                
                # If 'Info' is empty/generic, append Earnings, otherwise combine
                df_final['Info'] = df_final.apply(
                    lambda x: f"{x['Info']} | 📅 {x['Earnings']}" if x['Info'] and x['Earnings'] != '-' else (x['Earnings'] if x['Earnings'] != '-' else x['Info']), 
                    axis=1
                )
            
            with st.spinner(f"🧠 AI is selecting Top Pick from {len(df_final)} candidates..."):
                top_pick_text = AI_Analyst.generate_top_pick(df_final, scan_request)
            
            st.markdown(f"""
            <div class="ai-box">
                <h2>🤖 AI Strategic Verdict</h2>
                {top_pick_text}
            </div>
            """, unsafe_allow_html=True)
            
            st.write("### 📋 Full Scan Results (By Timeframe)")
            
            # --- MODIFIED DISPLAY LOGIC START ---
            
            # We iterate through the specific timeframes to create separate tables
            # 'Multi' covers the Support scan which aggregates multiple TFs
            target_timeframes = ["1D", "1W", "1M", "3M", "6M", "12M", "Multi"]
            
            found_any = False
            
            for tf in target_timeframes:
                # Filter the dataframe for the specific Timeframe
                subset = df_final[df_final['TF'] == tf]
                
                if not subset.empty:
                    found_any = True
                    st.markdown(f"#### ⏱️ **{tf} Timeframe** ({len(subset)} Setups)")
                    
                    st.dataframe(
                        subset,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                            "Price": st.column_config.NumberColumn("Price", format="%.2f"),
                            "Chop": st.column_config.NumberColumn("Chop Index", format="%.2f"),
                            "TF": st.column_config.TextColumn("Timeframe", width="small"),
                            "Info": st.column_config.TextColumn("Signal Info / Earnings", width="large")
                        }
                    )
                    st.divider() # Adds a line separator between tables
            
            if not found_any:
                st.info("Results processed but no specific timeframe buckets matched.")
            
            # --- MODIFIED DISPLAY LOGIC END ---

    # --- STRATEGY REFERENCE (COLLAPSIBLE) ---
    st.divider()
    with st.expander("📚 Strategy Reference Guide (Hover over buttons for quick definitions)"):
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            st.image(f"{repo_base}FVG%20Bullish%20Breakout.jpg", caption="Bullish FVG", use_container_width=True)
            st.image(f"{repo_base}Bullish%20Order%20Block.jpg", caption="Bullish OB", use_container_width=True)
            st.image(f"{repo_base}Bullish%20iFVG.jpg", caption="Bullish iFVG", use_container_width=True)
        with rc2:
            st.image(f"{repo_base}FVG%20Bearish%20Breakdown.jpg", caption="Bearish FVG", use_container_width=True)
            st.image(f"{repo_base}Bearish%20Order%20Block.jpg", caption="Bearish OB", use_container_width=True)
            st.image(f"{repo_base}Bearish%20iFVG.jpg", caption="Bearish iFVG", use_container_width=True)
        with rc3:
            st.image(f"{repo_base}Strong%20Support.jpg", caption="Strong Support", use_container_width=True)
            st.image(f"{repo_base}Volatility%20Squeeze.jpg", caption="Squeeze", use_container_width=True)

if __name__ == "__main__":
    main()