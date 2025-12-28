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
st.set_page_config(page_title="Prath's Market Scanner v3.0", layout="wide", page_icon="🎯")
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #FAFAFA; }
    div[data-testid="stMetricValue"] { font-size: 18px; color: #00CC96; }
    div[data-testid="stMetricDelta"] { font-size: 14px; }
    .stDataFrame { border: 1px solid #333; }
    h3 { border-bottom: 2px solid #333; padding-bottom: 10px; }
    /* Highlight selected rows */
    .stDataFrame div[data-testid="stTable"] tr:hover {
        background-color: #2b2b2b;
    }
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
        if st.session_state["password"] == st.secrets.get("PASSWORD", "Sniper2025"):
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
# 3. AI ANALYST ENGINE (Updated)
# ==========================================
class AI_Analyst:
    @staticmethod
    def get_market_news(ticker):
        if not AI_AVAILABLE: return "AI Libraries not installed."
        try:
            with DDGS() as ddgs:
                query = f"{ticker} stock news financial results analysis"
                results = list(ddgs.news(query, max_results=2))
                news_summary = ""
                for r in results:
                    news_summary += f"- [{r['title']}]({r['url']}) (Source: {r['source']})\n"
                return news_summary if news_summary else "No specific news found."
        except Exception as e:
            return f"Could not fetch news: {str(e)}"

    @staticmethod
    def get_earnings_date(ticker):
        try:
            stock = yf.Ticker(ticker)
            # Try getting calendar
            cal = stock.calendar
            if cal is not None and not cal.empty:
                # Calendar format varies by yfinance version, usually 'Earnings Date' or 0
                earnings_date = cal.iloc[0, 0] if isinstance(cal, pd.DataFrame) else cal.get('Earnings Date', [None])[0]
                if earnings_date:
                    days_left = (earnings_date - datetime.date.today()).days
                    return f"{earnings_date} ({days_left} days left)"
            return "Date not available"
        except:
            return "Data unavailable"

    @staticmethod
    def generate_deep_dive(ticker, signal_type, price, timeframe, chop_val):
        if not AI_AVAILABLE: return "⚠️ AI libraries missing."
        if "GEMINI_API_KEY" not in st.secrets: return "⚠️ GEMINI_API_KEY missing in secrets."

        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        
        # 1. Fetch Data
        news = AI_Analyst.get_market_news(ticker)
        earnings = AI_Analyst.get_earnings_date(ticker)
        
        # 2. Prompt
        prompt = f"""
        You are an elite, unbiased financial strategist.
        The stock {ticker} was filtered by a quantitative scanner for a **{signal_type}** on the {timeframe} timeframe.
        Current Price: {price}
        Choppiness Index: {chop_val} (Note: <38 is trending, >61 is squeeze/chop).
        Next Earnings: {earnings}

        Top News Context:
        {news}

        Provide a strategic "Deep Dive" analysis (max 150 words):
        1. **Contextualize the Filter:** Why is this setup appearing now? Connect the technical signal ({signal_type}) with the news/earnings timeline.
        2. **Strategic Verdict:** Is this a high-probability "Sniper" entry or a potential trap? Be critical. 
        3. **Key Risk:** What is the one thing that invalidates this trade?

        Format clearly with bold headers.
        """
        
        try:
            model = genai.GenerativeModel('gemini-3-flash-preview')  # or the stable version once available
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"⚠️ Analysis failed: {str(e)}"

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

# ==========================================
# 5. DATA ENGINE
# ==========================================
@st.cache_data
def load_tickers(market_choice):
    filename = "ind_tickers.csv" if "India" in market_choice else "us_tickers.csv"
    if not os.path.exists(filename): return []
    try:
        df = pd.read_csv(filename)
        clean = [str(t).strip().upper() for t in df.iloc[:, 0].tolist()]
        if "India" in market_choice: clean = [t if t.endswith(".NS") else f"{t}.NS" for t in clean]
        return clean
    except: return []

@st.cache_data(ttl=3600)
def fetch_bulk_data(tickers, interval="1d", period="2y"):
    if not tickers: return None
    return yf.download(tickers, period=period, interval=interval, group_by='ticker', progress=False, threads=True)

def resample_custom(df, timeframe):
    if df.empty: return df
    agg = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
    try:
        if timeframe == "1D": return df.resample("1D").agg(agg).dropna()
        if timeframe == "1W": return df.resample("W-FRI").agg(agg).dropna()
        if timeframe == "1M": return df.resample("ME").agg(agg).dropna()
        
        # Chop calculation helper needs full series, but resampling reduces rows.
        # We calculate Chop on resampled data inside analyze_ticker.
        df_m = df.resample('MS').agg(agg).dropna()
        if timeframe == "3M": return df_m.resample('QE').agg(agg).dropna()
        if timeframe == "6M":
            df_m['Y'], df_m['H'] = df_m.index.year, np.where(df_m.index.month <= 6, 1, 2)
            df_6m = df_m.groupby(['Y', 'H']).agg(agg)
            # Reconstruct index roughly
            return df_6m
        if timeframe == "12M": return df_m.resample('YE').agg(agg).dropna()
    except: return df
    return df

# ==========================================
# 6. ANALYSIS ENGINE
# ==========================================
def analyze_ticker(ticker, df_d_raw, df_m_raw):
    results = []
    try:
        data_map = {
            "1D": resample_custom(df_d_raw, "1D"),
            "1W": resample_custom(df_d_raw, "1W"),
            "1M": resample_custom(df_m_raw, "1M"),
            "3M": resample_custom(df_m_raw, "3M"),
            "6M": resample_custom(df_m_raw, "6M"),
            "12M": resample_custom(df_m_raw, "12M")
        }
    except: return []

    for tf in ["1D", "1W", "1M", "6M"]:
        if tf not in data_map or len(data_map[tf]) < 5: continue
        df = data_map[tf].copy()
        
        # Calc Chop for this timeframe
        chop_series = MathWiz.calculate_choppiness(df['High'], df['Low'], df['Close'])
        current_chop = round(chop_series.iloc[-1], 2) if not chop_series.empty else 0

        df['Bull_FVG'], df['Bear_FVG'] = MathWiz.find_fvg(df)
        df['Is_High'], df['Is_Low'] = MathWiz.identify_strict_swings(df)
        
        curr, prev = df.iloc[-1], df.iloc[-2]
        price = round(curr['Close'], 2)

        # FVG SNIPER
        if curr['Bull_FVG']:
            past = df[df['Is_High']]
            if not past.empty and curr['Close'] > past['High'].iloc[-1] and prev['Close'] <= past['High'].iloc[-1]:
                results.append({"Ticker": ticker, "Price": price, "Type": "Bull_FVG", "TF": tf, "Chop": current_chop})
        
        if curr['Bear_FVG']:
            past = df[df['Is_Low']]
            if not past.empty and curr['Close'] < past['Low'].iloc[-1] and prev['Close'] >= past['Low'].iloc[-1]:
                results.append({"Ticker": ticker, "Price": price, "Type": "Bear_FVG", "TF": tf, "Chop": current_chop})

        # ORDER BLOCKS
        sub = df.iloc[-4:].copy()
        if len(sub) == 4:
            c0, c1, c2, c3 = sub.iloc[0], sub.iloc[1], sub.iloc[2], sub.iloc[3]
            if (c0['Close']<c0['Open'] and c1['Close']>c1['Open'] and c2['Close']>c2['Open'] and c3['Close']>c3['Open']):
                if c3['Low']>c1['High'] and c3['Close']>c0['High']:
                    results.append({"Ticker": ticker, "Price": price, "Type": "Bull_OB", "TF": tf, "Chop": current_chop})
            if (c0['Close']>c0['Open'] and c1['Close']<c1['Open'] and c2['Close']<c2['Open'] and c3['Close']<c3['Open']):
                if c3['High']<c1['Low'] and c3['Close']<c0['Low']:
                    results.append({"Ticker": ticker, "Price": price, "Type": "Bear_OB", "TF": tf, "Chop": current_chop})
        
        # iFVG
        if tf in ["1D", "1W", "1M"]:
            ifvg = MathWiz.check_ifvg_reversal(df)
            if ifvg: results.append({"Ticker": ticker, "Price": price, "Type": f"{ifvg}_iFVG", "TF": tf, "Chop": current_chop})

    # SUPPORT
    sup_tf = []
    for t in ["3M", "6M", "12M"]:
        if MathWiz.find_unmitigated_fvg_zone(data_map[t]): sup_tf.append(t)
    if len(sup_tf) >= 2:
        # For support, use 3M chop as proxy
        c3m = MathWiz.calculate_choppiness(data_map["3M"]['High'], data_map["3M"]['Low'], data_map["3M"]['Close'])
        c_val = round(c3m.iloc[-1], 2) if not c3m.empty else 0
        results.append({"Ticker": ticker, "Price": round(data_map["3M"]['Close'].iloc[-1], 2), "Type": "Strong_Support", "TF": "Multiple", "Chop": c_val, "Info": ",".join(sup_tf)})

    # SQUEEZE/REVERSAL (Existing logic)
    if not data_map["1W"].empty:
        c_w = MathWiz.calculate_choppiness(data_map["1W"]['High'], data_map["1W"]['Low'], data_map["1W"]['Close']).iloc[-1]
        if c_w < 25: results.append({"Ticker": ticker, "Price": round(data_map["1W"]['Close'].iloc[-1], 2), "Type": "Reversal", "TF": "1W", "Chop": round(c_w, 2)})
        if c_w > 59: results.append({"Ticker": ticker, "Price": round(data_map["1W"]['Close'].iloc[-1], 2), "Type": "Squeeze", "TF": "1W", "Chop": round(c_w, 2)})
    if not data_map["1D"].empty:
        c_d = MathWiz.calculate_choppiness(data_map["1D"]['High'], data_map["1D"]['Low'], data_map["1D"]['Close']).iloc[-1]
        if c_d > 59: results.append({"Ticker": ticker, "Price": round(data_map["1D"]['Close'].iloc[-1], 2), "Type": "Squeeze", "TF": "1D", "Chop": round(c_d, 2)})

    return results

# ==========================================
# 7. MAIN UI
# ==========================================
def display_interactive_table(df, key_prefix):
    """
    Displays a dataframe with selection enabled. Returns the selected row.
    """
    if df.empty:
        st.caption("No setups found.")
        return None
        
    # Add a 'Deep Dive' column for UI clarity (acts as the checkbox)
    df_display = df.copy()
    
    # Configure columns
    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        on_select="rerun", # Triggers reload when row selected
        selection_mode="single-row",
        key=f"table_{key_prefix}"
    )
    
    # Check session state for selection
    selection_state = st.session_state.get(f"table_{key_prefix}", {})
    if selection_state and "rows" in selection_state and selection_state["rows"]:
        selected_idx = selection_state["rows"][0]
        return df_display.iloc[selected_idx]
    return None

def main():
    if not check_password(): st.stop()
    
    # --- GLOBAL SIDEBAR TRACKER ---
    st.sidebar.title("🌍 Global Pulse")
    global_tickers = {
        "NIFTY 50": "^NSEI", "BANKNIFTY": "^NSEBANK", "DJI": "^DJI", "SPX": "^GSPC",
        "USD/INR": "USDINR=X", "CAD/INR": "CADINR=X", "GOLD": "GC=F", "SILVER": "SI=F",
        "US OIL": "CL=F", "BTC/USD": "BTC-USD", "Total Mkt (Proxy)": "^GSPC" # Proxy for TradingView ticker
    }
    
    # Fetch live data for sidebar (cached short term)
    g_data = yf.download(list(global_tickers.values()), period="5d", interval="1d", progress=False)
    
    for name, sym in global_tickers.items():
        try:
            # Handle multi-index columns from yfinance
            if isinstance(g_data.columns, pd.MultiIndex):
                hist = g_data['Close'][sym].dropna()
            else:
                hist = g_data['Close'].dropna()
            
            if not hist.empty:
                last_price = hist.iloc[-1]
                prev_price = hist.iloc[-2]
                delta = last_price - prev_price
                st.sidebar.metric(name, f"{last_price:,.2f}", f"{delta:+.2f}")
        except:
            st.sidebar.metric(name, "Err", "0.00")

    st.sidebar.divider()
    
    # --- MAIN APP ---
    st.title("Prath's Market Scanner v3.0")
    
    market = st.sidebar.selectbox("Select Market", ["US Markets (Nasdaq 100)", "Indian Markets (Nifty 500)"])
    tickers = load_tickers(market)
    
    if st.sidebar.button("🔄 Run Analysis", type="primary"):
        st.cache_data.clear()
        st.rerun()

    if tickers:
        data_d = fetch_bulk_data(tickers, "1d", "2y")
        data_m = fetch_bulk_data(tickers, "1mo", "max")
        
        all_res = []
        is_multi = len(tickers) > 1
        
        with ThreadPoolExecutor() as ex:
            futures = []
            for t in tickers:
                try:
                    df_d = data_d[t] if is_multi else data_d
                    df_m = data_m[t] if is_multi else data_m
                    if not df_d.empty: futures.append(ex.submit(analyze_ticker, t, df_d, df_m))
                except: continue
            
            for f in futures:
                try:
                    if f.result(): all_res.extend(f.result())
                except: continue
        
        df_all = pd.DataFrame(all_res)
        
        # --- UI LAYOUT ---
        st.info("💡 **Interactive Mode:** Select any row in the tables below to generate an AI Deep Dive Analysis.")
        
        # Helper to filter
        def get_df(type_n, tf=None):
            if df_all.empty: return pd.DataFrame()
            mask = (df_all['Type'] == type_n)
            if tf: mask &= (df_all['TF'] == tf)
            return df_all[mask][['Ticker', 'Price', 'Chop', 'TF']]

        c1, c2 = st.columns(2)
        
        selected_stock_data = None
        
        with c1:
            st.header("🐂 Bullish Scans")
            st.subheader("FVG Breakouts")
            sel = display_interactive_table(get_df("Bull_FVG"), "bf")
            if sel is not None: selected_stock_data = (sel, "Bull_FVG")

            st.subheader("Order Blocks")
            sel = display_interactive_table(get_df("Bull_OB"), "bob")
            if sel is not None: selected_stock_data = (sel, "Bull_OB")
            
            st.subheader("iFVG Reversals")
            sel = display_interactive_table(get_df("Bull_iFVG_iFVG"), "bif") # Note: Fixed type naming in analyze
            if sel is not None: selected_stock_data = (sel, "Bull_iFVG")

        with c2:
            st.header("🐻 Bearish Scans")
            st.subheader("FVG Breakdowns")
            sel = display_interactive_table(get_df("Bear_FVG"), "brf")
            if sel is not None: selected_stock_data = (sel, "Bear_FVG")

            st.subheader("Order Blocks")
            sel = display_interactive_table(get_df("Bear_OB"), "brob")
            if sel is not None: selected_stock_data = (sel, "Bear_OB")

            st.subheader("iFVG Reversals")
            sel = display_interactive_table(get_df("Bear_iFVG_iFVG"), "brif")
            if sel is not None: selected_stock_data = (sel, "Bear_iFVG")

        st.divider()
        st.subheader("Strong Support (3M/6M/12M)")
        if not df_all.empty:
            sup_df = df_all[df_all['Type'] == "Strong_Support"][['Ticker', 'Price', 'Chop', 'Info']]
            sel = display_interactive_table(sup_df, "sup")
            if sel is not None: selected_stock_data = (sel, "Strong_Support")
        
        # --- DEEP DIVE MODAL ---
        if selected_stock_data:
            row, s_type = selected_stock_data
            ticker = row['Ticker']
            
            @st.dialog(f"🧠 Deep Dive: {ticker}")
            def show_analysis():
                st.write(f"**Signal:** {s_type} | **Price:** {row['Price']} | **Chop:** {row['Chop']}")
                
                with st.spinner("Analyzing Market Context & News..."):
                    # Generate Analysis
                    analysis = AI_Analyst.generate_deep_dive(
                        ticker, s_type, row['Price'], 
                        row.get('TF', 'N/A'), row['Chop']
                    )
                    st.markdown(analysis)
                    
                    st.divider()
                    st.caption("Disclaimer: AI analysis is for informational purposes only.")
            
            show_analysis()

if __name__ == "__main__":
    main()