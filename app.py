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
st.set_page_config(page_title="Prath's Sniper v6.0", layout="wide", page_icon="🎯")
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
    
    /* Headers */
    h4 { padding-top: 10px; margin-bottom: 0px; color: #e5e7eb; }
    
    /* Definitions */
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
    
    if "password_correct" in st.session_state and not st.session_state.password_correct:
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
        
        extra_context = ""
        if "Fair Value Area" in scan_name:
            extra_context = "For FVAs, the 'Info' column shows the actionable Zone (Retracement Range) and confirms an FVG triggered the break."

        data_str = df[['Ticker', 'Price', 'Chop', 'TF', 'Info']].to_string(index=False)
        
        prompt = f"""
        You are an elite Sniper Trader.
        I just ran a scan for **{scan_name}**.
        {extra_context}
        Here are the results:
        
        {data_str}
        
        **Your Mission:**
        1.  Analyze setups based on Price, Chop (Choppiness Index), and Timeframe.
        2.  **Filter:** Reject stocks with earnings in < 3 days.
        3.  **Logic:** - Breakouts (FVG/OB): Prefer Chop < 40.
            - Reversals/Squeezes: Prefer Chop > 60.
            - **Fair Value Areas (FVA):** Look for the most solid 1-2-3 structure with a confirmed FVG.
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

    @staticmethod
    def check_fair_value_area(df, direction):
        # NEW LOGIC: Market Mastery Framework (1-2-3 Swing)
        # Swing Definition: neighbor_count=1 (Immediate left/right)
        
        if len(df) < 15: return None
        
        # 1. Identify Swings (The Anchors)
        df['Is_H'], df['Is_L'] = MathWiz.identify_strict_swings(df, neighbor_count=1)
        
        # Get indices of swings
        swing_highs = df.index[df['Is_H']].tolist()
        swing_lows = df.index[df['Is_L']].tolist()
        
        if not swing_highs or not swing_lows: return None
        
        curr_price = df['Close'].iloc[-1]
        
        # ----------------------------------------
        # BULLISH FVA: Buy(A) -> Sell(B) -> Buy(Break A)
        # ----------------------------------------
        if direction == "Bullish":
            # Check the last few candles for a Break of Structure
            # Ideally, the break happened recently (last 3-5 bars)
            recent_candles = df.iloc[-5:]
            
            # Find the most recent Swing High (Point A) that is *before* the current breakout
            # We iterate backwards through Swing Highs
            for h_idx in reversed(swing_highs):
                if h_idx in recent_candles.index: continue # Skip if the high is the current candle itself
                
                point_a_high = df.loc[h_idx]['High']
                
                # Step C (Break): Has price recently Closed ABOVE Point A?
                # Check if any of the recent candles closed above A
                if (recent_candles['Close'] > point_a_high).any():
                    
                    # Step B (Retracement): Was there a Swing Low AFTER Point A but BEFORE the Break?
                    # Find lows between h_idx and now
                    valid_lows = [l for l in swing_lows if l > h_idx]
                    
                    if valid_lows:
                        point_b_idx = valid_lows[0] # The first low after high A
                        point_b_low = df.loc[point_b_idx]['Low']
                        
                        # Step 4 (Validation): Did the move from B to Break create a FVG?
                        # Analyze leg from Point B to Current
                        breakout_leg = df.loc[point_b_idx:]
                        has_fvg = False
                        
                        # Calculate FVG for this leg specifically
                        leg_bull, _ = MathWiz.find_fvg(breakout_leg)
                        if leg_bull.any(): has_fvg = True
                        
                        if has_fvg:
                            # Valid FVA Found! 
                            # Zone is range of Swing 2 (Retracement): High A to Low B
                            return f"Zone: {round(point_b_low, 2)} - {round(point_a_high, 2)} (FVG Confirmed)"
            
        # ----------------------------------------
        # BEARISH FVA: Sell(A) -> Buy(B) -> Sell(Break A)
        # ----------------------------------------
        if direction == "Bearish":
            recent_candles = df.iloc[-5:]
            
            # Find Swing Low (Point A)
            for l_idx in reversed(swing_lows):
                if l_idx in recent_candles.index: continue
                
                point_a_low = df.loc[l_idx]['Low']
                
                # Step C (Break): Closed BELOW Point A?
                if (recent_candles['Close'] < point_a_low).any():
                    
                    # Step B (Retracement): Swing High AFTER A?
                    valid_highs = [h for h in swing_highs if h > l_idx]
                    
                    if valid_highs:
                        point_b_idx = valid_highs[0]
                        point_b_high = df.loc[point_b_idx]['High']
                        
                        # Validation: FVG in breakdown leg?
                        breakdown_leg = df.loc[point_b_idx:]
                        has_fvg = False
                        
                        _, leg_bear = MathWiz.find_fvg(breakdown_leg)
                        if leg_bear.any(): has_fvg = True
                        
                        if has_fvg:
                            # Valid FVA Found
                            return f"Zone: {round(point_a_low, 2)} - {round(point_b_high, 2)} (FVG Confirmed)"

        return None

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
        df_m = df.resample('MS').agg(agg).dropna()
        if timeframe == "3M": return df_m.resample('QE').agg(agg).dropna()
        if timeframe == "6M":
            df_m['Y'], df_m['H'] = df_m.index.year, np.where(df_m.index.month <= 6, 1, 2)
            df_6m = df_m.groupby(['Y', 'H']).agg(agg)
            return df_6m
        if timeframe == "12M": return df_m.resample('YE').agg(agg).dropna()
    except: return df
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
        chop = 0
        if "Chop" not in df.columns:
            chop_s = MathWiz.calculate_choppiness(df['High'], df['Low'], df['Close'])
            chop = round(chop_s.iloc[-1], 2) if not chop_s.empty else 0
        curr = df.iloc[-1]; price = round(curr['Close'], 2)

        # --- STANDARD SCANS ---
        if "FVG" in scan_type and "Fair Value Area" not in scan_type:
            df['Bull'], df['Bear'] = MathWiz.find_fvg(df)
            df['Is_H'], df['Is_L'] = MathWiz.identify_strict_swings(df)
            if "Bullish" in scan_type and curr['Bull']:
                past_swings = df[df['Is_H']]
                if not past_swings.empty:
                    if (df['Close'].iloc[-3:] > past_swings['High'].iloc[-1]).any():
                        results.append({"Ticker": ticker, "Price": price, "Chop": chop, "TF": tf, "Info": ""})
            if "Bearish" in scan_type and curr['Bear']:
                past_swings = df[df['Is_L']]
                if not past_swings.empty:
                    if (df['Close'].iloc[-3:] < past_swings['Low'].iloc[-1]).any():
                        results.append({"Ticker": ticker, "Price": price, "Chop": chop, "TF": tf, "Info": ""})

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

        if "iFVG" in scan_type and tf in ["1D", "1W", "1M"]:
            res = MathWiz.check_ifvg_reversal(df)
            if "Bullish" in scan_type and res == "Bull":
                results.append({"Ticker": ticker, "Price": price, "Chop": chop, "TF": tf, "Info": ""})
            if "Bearish" in scan_type and res == "Bear":
                results.append({"Ticker": ticker, "Price": price, "Chop": chop, "TF": tf, "Info": ""})

        # --- FAIR VALUE AREA SCAN (1-2-3 SEQUENCE) ---
        if "Fair Value Area" in scan_type:
            direction = "Bullish" if "Bullish" in scan_type else "Bearish"
            fva_status = MathWiz.check_fair_value_area(df, direction)
            if fva_status:
                results.append({"Ticker": ticker, "Price": price, "Chop": chop, "TF": tf, "Info": fva_status})

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

    st.title("Prath's Sniper v6.0")
    
    col_mkt, col_status = st.columns([1, 2])
    with col_mkt:
        market = st.selectbox("Select Market", ["US Markets (Nasdaq 100)", "Indian Markets (Nifty 200)"])
    
    tickers = load_tickers(market)
    if not tickers:
        st.error("No tickers found. Check CSV files.")
        st.stop()
        
    st.write("### 🎯 Select Strategy to Scan")
    
    repo_base = "https://raw.githubusercontent.com/prathtwitter/praths-market-scanner/main/"
    
    c1, c2, c3 = st.columns(3)
    scan_request = None
    
    # --- BULLISH COLUMN ---
    with c1:
        st.header("🐂 Bullish")
        
        st.markdown("#### FVG Breakouts")
        st.caption("A strong bullish candle creating a Fair Value Gap while simultaneously breaking a recent swing high structure.")
        if st.button("Run: Bullish FVG"): scan_request = "Bullish FVG"
        st.image(f"{repo_base}FVG%20Bullish%20Breakout.jpg", use_container_width=True)
        
        st.markdown("#### Order Blocks")
        st.caption("The last down-candle before an impulsive upward move that broke market structure, acting as a buy zone.")
        if st.button("Run: Bullish OB"): scan_request = "Bullish Order Block"
        st.image(f"{repo_base}Bullish%20Order%20Block.jpg", use_container_width=True)
        
        st.markdown("#### iFVG Reversal")
        st.caption("A failed bearish Fair Value Gap that price has reclaimed and closed above, flipping it into support.")
        if st.button("Run: Bullish iFVG"): scan_request = "Bullish iFVG"
        st.image(f"{repo_base}Bullish%20iFVG.jpg", use_container_width=True)

    # --- BEARISH COLUMN ---
    with c2:
        st.header("🐻 Bearish")
        
        st.markdown("#### FVG Breakdowns")
        st.caption("A strong bearish candle creating a Fair Value Gap while simultaneously breaking a recent swing low structure.")
        if st.button("Run: Bearish FVG"): scan_request = "Bearish FVG"
        st.image(f"{repo_base}FVG%20Bearish%20Breakdown.jpg", use_container_width=True)
        
        st.markdown("#### Order Blocks")
        st.caption("The last up-candle before an impulsive downward move that broke market structure, acting as a sell zone.")
        if st.button("Run: Bearish OB"): scan_request = "Bearish Order Block"
        st.image(f"{repo_base}Bearish%20Order%20Block.jpg", use_container_width=True)
        
        st.markdown("#### iFVG Reversal")
        st.caption("A failed bullish Fair Value Gap that price has broken and closed below, flipping it into resistance.")
        if st.button("Run: Bearish iFVG"): scan_request = "Bearish iFVG"
        st.image(f"{repo_base}Bearish%20iFVG.jpg", use_container_width=True)

    # --- VOLATILITY & FVA COLUMN ---
    with c3:
        st.header("⚡ Volatility & FVA")
        
        st.markdown("#### Fair Value Areas (FVA)")
        st.caption("Valid 1-2-3 Sequence: Impulse -> Retracement -> Break. Scanner validates if the breakout leg contained an FVG.")
        if st.button("Run: Bullish FVA"): scan_request = "Bullish Fair Value Area"
        if st.button("Run: Bearish FVA"): scan_request = "Bearish Fair Value Area"
        # Placeholder for FVA until you upload one
        st.image("https://placehold.co/600x400/162b26/00CC96?text=FVA+Scanner%0A(1-2-3+Move+%2B+FVG)", use_container_width=True)

        st.markdown("#### Strong Support Zones")
        st.caption("Price reacting off a high-timeframe (3M/6M) unmitigated Fair Value Gap, signaling major institutional interest.")
        if st.button("Run: Strong Support"): scan_request = "Strong Support"
        st.image(f"{repo_base}Strong%20Support.jpg", use_container_width=True)
        
        st.markdown("#### Volatility Squeezes")
        st.caption("Price consolidating in an extremely tight range (Choppiness Index > 60), signaling an imminent explosive move.")
        if st.button("Run: Volatility Squeeze"): scan_request = "Squeeze"
        st.image(f"{repo_base}Volatility%20Squeeze.jpg", use_container_width=True)

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
            st.warning("No setups found matching this exact logic right now.")
        else:
            with st.spinner("Fetching Earnings Dates..."):
                u_tickers = df_final['Ticker'].unique().tolist()
                e_map = AI_Analyst.get_earnings_date_bulk(u_tickers)
                df_final['Info'] = df_final['Ticker'].map(e_map).fillna(df_final.get('Info', "-"))
            
            with st.spinner(f"🧠 AI is selecting Top Pick from {len(df_final)} candidates..."):
                top_pick_text = AI_Analyst.generate_top_pick(df_final, scan_request)
            
            st.markdown(f"""
            <div class="ai-box">
                <h2>🤖 AI Strategic Verdict</h2>
                {top_pick_text}
            </div>
            """, unsafe_allow_html=True)
            
            st.write("### 📋 Full Scan Results")
            st.dataframe(df_final, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()