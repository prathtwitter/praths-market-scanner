import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor

# --- NEW IMPORTS FOR AI ---
try:
    import google.generativeai as genai
    from duckduckgo_search import DDGS
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

# ==========================================
# 1. CONFIGURATION & STYLE
# ==========================================
st.set_page_config(page_title="Prath's Market Scanner v2.1 (AI)", layout="wide", page_icon="🎯")
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #FAFAFA; }
    div[data-testid="stMetricValue"] { font-size: 20px; color: #00CC96; }
    .stRadio > div { flex-direction: row; } 
    .stDataFrame { border: 1px solid #333; }
    h3 { border-bottom: 2px solid #333; padding-bottom: 10px; }
    .stButton>button { width: 100%; border-radius: 5px; }
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
        st.text_input(
            "Enter Access Key", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        st.text_input(
            "Enter Access Key", type="password", on_change=password_entered, key="password"
        )
        st.error("⛔ Access Denied")
        return False
    else:
        return True

# ==========================================
# 3. AI ANALYST ENGINE (Phase 4)
# ==========================================
class AI_Analyst:
    @staticmethod
    def get_market_news(ticker):
        """Fetches top 3 recent news headlines for the ticker."""
        if not AI_AVAILABLE: return "AI Libraries not installed."
        
        try:
            with DDGS() as ddgs:
                # Search for specific financial news
                query = f"{ticker} stock news financial results or catalyst"
                results = list(ddgs.news(query, max_results=3))
                
                news_summary = ""
                for r in results:
                    news_summary += f"- {r['title']} (Source: {r['source']})\n"
                return news_summary if news_summary else "No recent specific news found."
        except Exception as e:
            return f"Could not fetch live news: {str(e)}"

    @staticmethod
    def generate_briefing(ticker, signal_type, price, timeframe):
        """Sends data + news to Gemini to generate a strategic briefing."""
        if not AI_AVAILABLE: return "⚠️ AI libraries missing. Please add google-generativeai to requirements.txt"
        if "GEMINI_API_KEY" not in st.secrets: return "⚠️ GEMINI_API_KEY missing in secrets."

        # Configure Gemini
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        
        # 1. Get Live Context
        news_context = AI_Analyst.get_market_news(ticker)
        
        # 2. Construct the Prompt
        prompt = f"""
        You are a senior hedge fund strategist. 
        The stock {ticker} has just triggered a **{signal_type}** signal on the {timeframe} timeframe at a price of ${price}.
        
        Here is the recent news context found:
        {news_context}
        
        Please provide a concise, 3-sentence briefing:
        1. **The Catalyst:** Based on the news, what is driving this move? (Earnings, macro, product launch, etc.)
        2. **The Technical Context:** Why is this {signal_type} significant right now?
        3. **Strategic Verdict:** Is this likely a trap or a valid continuation? (Be skeptical/realistic).
        
        Keep it professional, direct, and under 100 words.
        """
        
        # 3. Call Gemini
        try:
            model = genai.GenerativeModel('gemini-3-flash-preview')  # or the stable version once available
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"⚠️ AI Analysis failed: {str(e)}"

# ==========================================
# 4. MATH WIZ (Advanced Logic)
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
        # Take last 5 candles and drop NaNs
        subset = df.iloc[-5:].copy().dropna()
        if len(subset) < 5: return None

        c1 = subset.iloc[0]
        c3 = subset.iloc[2]
        c5 = subset.iloc[4] # Current candle
        
        # Bullish iFVG Scan: Bear FVG (Drop) -> Bull FVG (Pop)
        is_bear_fvg_first = c3['High'] < c1['Low']
        is_bull_fvg_second = c5['Low'] > c3['High']
        if is_bear_fvg_first and is_bull_fvg_second:
            return 'Bull'

        # Bearish iFVG Scan: Bull FVG (Pop) -> Bear FVG (Drop)
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

# ==========================================
# 5. ROBUST DATA ENGINE
# ==========================================
@st.cache_data
def load_tickers(market_choice):
    filename = "ind_tickers.csv" if "India" in market_choice else "us_tickers.csv"
    if not os.path.exists(filename):
        st.error(f"❌ '{filename}' not found. Ensure CSV files are in the repository.")
        return []
    try:
        df = pd.read_csv(filename)
        raw = df.iloc[:, 0].tolist()
        clean = [str(t).strip().upper() for t in raw]
        if "India" in market_choice:
            clean = [t if t.endswith(".NS") else f"{t}.NS" for t in clean]
        return clean
    except: return []

@st.cache_data(ttl=3600)
def fetch_bulk_data(tickers, interval="1d", period="2y"):
    if not tickers: return None
    data = yf.download(tickers, period=period, interval=interval, group_by='ticker', progress=True, threads=True)
    return data

def filter_top_5_by_cap(df_results):
    if df_results.empty: return df_results
    tickers = df_results['Ticker'].unique().tolist()
    caps = []
    
    for t in tickers:
        try:
            val = yf.Ticker(t).fast_info['market_cap']
            caps.append(val)
        except:
            try:
                val = yf.Ticker(t).info.get('marketCap', 0)
                caps.append(val)
            except:
                caps.append(0)
    
    cap_df = pd.DataFrame({'Ticker': tickers, 'MarketCap': caps})
    merged = df_results.merge(cap_df, on='Ticker', how='left')
    merged = merged.sort_values(by='MarketCap', ascending=False, na_position='last').head(5)
    return merged.drop(columns=['MarketCap'])

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
# 6. PARALLEL ANALYSIS ENGINE
# ==========================================
def analyze_ticker(ticker, df_daily_raw, df_monthly_raw):
    results = []
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
    except: return []

    # 1. FVG & ORDER BLOCKS & iFVG
    for tf in ["1D", "1W", "1M", "6M"]:
        if tf not in data_map or len(data_map[tf]) < 5: continue
        
        df = data_map[tf].copy() 
        df['Bull_FVG'], df['Bear_FVG'] = MathWiz.find_fvg(df)
        df['Is_Swing_High'], df['Is_Swing_Low'] = MathWiz.identify_strict_swings(df)
        
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        price = round(curr['Close'], 2)

        # --- A. SNIPER FVG ENTRIES ---
        if curr['Bull_FVG']:
            past_swings = df[df['Is_Swing_High']]
            if not past_swings.empty:
                last_swing_high = past_swings['High'].iloc[-1]
                if curr['Close'] > last_swing_high and prev['Close'] <= last_swing_high:
                    results.append({"Ticker": ticker, "Price": price, "Type": "Bull_FVG", "TF": tf})
        
        if curr['Bear_FVG']:
            past_swings = df[df['Is_Swing_Low']]
            if not past_swings.empty:
                last_swing_low = past_swings['Low'].iloc[-1]
                if curr['Close'] < last_swing_low and prev['Close'] >= last_swing_low:
                    results.append({"Ticker": ticker, "Price": price, "Type": "Bear_FVG", "TF": tf})

        # --- B. ORDER BLOCKS ---
        subset = df.iloc[-4:].copy()
        if len(subset) == 4:
            c_anc, c1, c2, c3 = subset.iloc[0], subset.iloc[1], subset.iloc[2], subset.iloc[3]
            
            if (c_anc['Close'] < c_anc['Open'] and 
                c1['Close'] > c1['Open'] and c2['Close'] > c2['Open'] and c3['Close'] > c3['Open']):
                if c3['Low'] > c1['High'] and c3['Close'] > c_anc['High']:
                    results.append({"Ticker": ticker, "Price": price, "Type": "Bull_OB", "TF": tf})

            if (c_anc['Close'] > c_anc['Open'] and 
                c1['Close'] < c1['Open'] and c2['Close'] < c2['Open'] and c3['Close'] < c3['Open']):
                if c3['High'] < c1['Low'] and c3['Close'] < c_anc['Low']:
                    results.append({"Ticker": ticker, "Price": price, "Type": "Bear_OB", "TF": tf})
        
        # --- C. iFVG REVERSALS ---
        if tf in ["1D", "1W", "1M"]:
            ifvg_status = MathWiz.check_ifvg_reversal(df)
            if ifvg_status == "Bull":
                results.append({"Ticker": ticker, "Price": price, "Type": "Bull_iFVG", "TF": tf})
            elif ifvg_status == "Bear":
                results.append({"Ticker": ticker, "Price": price, "Type": "Bear_iFVG", "TF": tf})

    # 2. STRONG SUPPORT
    sup_tf = []
    if MathWiz.find_unmitigated_fvg_zone(d_3m): sup_tf.append("3M")
    if MathWiz.find_unmitigated_fvg_zone(d_6m): sup_tf.append("6M")
    if MathWiz.find_unmitigated_fvg_zone(d_12m): sup_tf.append("12M")
    
    if len(sup_tf) >= 2:
         results.append({
             "Ticker": ticker, 
             "Price": round(d_3m['Close'].iloc[-1], 2) if not d_3m.empty else 0, 
             "Type": "Strong_Support", 
             "Info": ", ".join(sup_tf)
         })

    # 3. REVERSALS & SQUEEZES
    if not d_1w.empty:
        chop_series = MathWiz.calculate_choppiness(d_1w['High'], d_1w['Low'], d_1w['Close'])
        if not chop_series.empty and not pd.isna(chop_series.iloc[-1]):
            chop_w = chop_series.iloc[-1]
            if chop_w < 25:
                results.append({"Ticker": ticker, "Price": round(d_1w['Close'].iloc[-1], 2), "Type": "Reversal", "Info": round(chop_w, 2)})

    if not d_1d.empty:
        chop_series_d = MathWiz.calculate_choppiness(d_1d['High'], d_1d['Low'], d_1d['Close'])
        if not chop_series_d.empty and not pd.isna(chop_series_d.iloc[-1]):
            chop_d = chop_series_d.iloc[-1]
            if chop_d > 59:
                results.append({"Ticker": ticker, "Price": round(d_1d['Close'].iloc[-1], 2), "Type": "Squeeze", "TF": "1D", "Info": round(chop_d, 2)})
                
    if not d_1w.empty:
        chop_series_w2 = MathWiz.calculate_choppiness(d_1w['High'], d_1w['Low'], d_1w['Close'])
        if not chop_series_w2.empty and not pd.isna(chop_series_w2.iloc[-1]):
            chop_w2 = chop_series_w2.iloc[-1]
            if chop_w2 > 59:
                 results.append({"Ticker": ticker, "Price": round(d_1w['Close'].iloc[-1], 2), "Type": "Squeeze", "TF": "1W", "Info": round(chop_w2, 2)})

    return results

# ==========================================
# 7. MAIN DASHBOARD UI
# ==========================================
def main():
    if not check_password():
        st.stop()
        
    st.title("Prath's Market Scanner")

    if 'init_done' not in st.session_state:
        st.cache_data.clear()
        st.session_state.init_done = True
        
    market = st.sidebar.selectbox("Market Sector", ["US Markets (Nasdaq 100)", "Indian Markets (Nifty 500)"])
    tickers = load_tickers(market)
    
    if st.sidebar.button("🔄 Refresh Dashboard", type="primary"):
        st.cache_data.clear()
        st.rerun()

    if tickers:
        with st.status("🚀 Running Parallel Scans...", expanded=True) as status:
            st.write("Fetching Market Data...")
            data_d = fetch_bulk_data(tickers, interval="1d", period="2y")
            data_m = fetch_bulk_data(tickers, interval="1mo", period="max")
            
            st.write("Processing Algorithms...")
            all_results = []
            is_multi = len(tickers) > 1
            
            with ThreadPoolExecutor() as executor:
                futures = []
                for ticker in tickers:
                    try:
                        df_d = data_d[ticker] if is_multi else data_d
                        df_m = data_m[ticker] if is_multi else data_m
                        if not df_d.empty and not df_m.empty:
                            futures.append(executor.submit(analyze_ticker, ticker, df_d, df_m))
                    except: continue
                
                for f in futures:
                    try:
                        res = f.result()
                        if res: all_results.extend(res)
                    except: continue
            
            df_all = pd.DataFrame(all_results)
            if df_all.empty:
                st.warning("No setups found.")
                status.update(label="Analysis Complete (No matches)", state="complete")
                return

            def get_res(type_name, tf=None):
                if tf: return df_all[(df_all['Type'] == type_name) & (df_all['TF'] == tf)]
                return df_all[df_all['Type'] == type_name]

            # Get Result Subsets
            bf_1d = get_res("Bull_FVG", "1D")
            bf_1w = get_res("Bull_FVG", "1W")
            bf_1m = get_res("Bull_FVG", "1M")
            brf_1d = get_res("Bear_FVG", "1D")
            brf_1w = get_res("Bear_FVG", "1W")
            brf_1m = get_res("Bear_FVG", "1M")
            bif_1d = get_res("Bull_iFVG", "1D")
            bif_1w = get_res("Bull_iFVG", "1W")
            bif_1m = get_res("Bull_iFVG", "1M")
            brif_1d = get_res("Bear_iFVG", "1D")
            brif_1w = get_res("Bear_iFVG", "1W")
            brif_1m = get_res("Bear_iFVG", "1M")
            bob_1d = get_res("Bull_OB", "1D")
            bob_1w = get_res("Bull_OB", "1W")
            bob_1m = get_res("Bull_OB", "1M")
            bob_6m = get_res("Bull_OB", "6M") 
            brob_1d = get_res("Bear_OB", "1D")
            brob_1w = get_res("Bear_OB", "1W")
            brob_1m = get_res("Bear_OB", "1M")
            brob_6m = get_res("Bear_OB", "6M") 
            sup_res = get_res("Strong_Support")
            rev_res = get_res("Reversal")
            sq_1d = get_res("Squeeze", "1D")
            sq_1w = get_res("Squeeze", "1W")

            status.update(label="✅ Analysis Complete", state="complete", expanded=False)

        # --- DISPLAY GRID ---
        col_bull, col_bear = st.columns(2)
        
        with col_bull:
            st.header("🐂 Bullish Scans")
            st.subheader("FVG Breakouts"); st.caption("Fresh break of structure on FVG candle.")
            st.write("**1D**"); st.dataframe(filter_top_5_by_cap(bf_1d)[['Ticker', 'Price']], hide_index=True, use_container_width=True)
            st.write("**1W**"); st.dataframe(filter_top_5_by_cap(bf_1w)[['Ticker', 'Price']], hide_index=True, use_container_width=True)
            st.write("**1M**"); st.dataframe(filter_top_5_by_cap(bf_1m)[['Ticker', 'Price']], hide_index=True, use_container_width=True)

            st.subheader("🔄 iFVG Reversals"); st.caption("5-Candle V-Shape: Drop → Pop")
            if bif_1d.empty and bif_1w.empty and bif_1m.empty: st.info("No iFVG setups found.")
            else:
                st.write("**1D**"); st.dataframe(filter_top_5_by_cap(bif_1d)[['Ticker', 'Price']], hide_index=True, use_container_width=True)
                st.write("**1W**"); st.dataframe(filter_top_5_by_cap(bif_1w)[['Ticker', 'Price']], hide_index=True, use_container_width=True)
                st.write("**1M**"); st.dataframe(filter_top_5_by_cap(bif_1m)[['Ticker', 'Price']], hide_index=True, use_container_width=True)
            
            st.subheader("Order Block Breakouts")
            st.write("**1D**"); st.dataframe(filter_top_5_by_cap(bob_1d)[['Ticker', 'Price']], hide_index=True, use_container_width=True)
            st.write("**1W**"); st.dataframe(filter_top_5_by_cap(bob_1w)[['Ticker', 'Price']], hide_index=True, use_container_width=True)
            st.write("**1M**"); st.dataframe(filter_top_5_by_cap(bob_1m)[['Ticker', 'Price']], hide_index=True, use_container_width=True)
            st.write("**6M**"); st.dataframe(filter_top_5_by_cap(bob_6m)[['Ticker', 'Price']], hide_index=True, use_container_width=True)
            
            st.subheader("🛡️ Strong FVG Support"); st.dataframe(filter_top_5_by_cap(sup_res), hide_index=True, use_container_width=True)

        with col_bear:
            st.header("🐻 Bearish Scans")
            st.subheader("FVG Breakdowns"); st.caption("Fresh break of structure on FVG candle.")
            st.write("**1D**"); st.dataframe(filter_top_5_by_cap(brf_1d)[['Ticker', 'Price']], hide_index=True, use_container_width=True)
            st.write("**1W**"); st.dataframe(filter_top_5_by_cap(brf_1w)[['Ticker', 'Price']], hide_index=True, use_container_width=True)
            st.write("**1M**"); st.dataframe(filter_top_5_by_cap(brf_1m)[['Ticker', 'Price']], hide_index=True, use_container_width=True)
            
            st.subheader("🔄 iFVG Reversals"); st.caption("5-Candle Inverted V: Pop → Drop")
            if brif_1d.empty and brif_1w.empty and brif_1m.empty: st.info("No iFVG setups found.")
            else:
                st.write("**1D**"); st.dataframe(filter_top_5_by_cap(brif_1d)[['Ticker', 'Price']], hide_index=True, use_container_width=True)
                st.write("**1W**"); st.dataframe(filter_top_5_by_cap(brif_1w)[['Ticker', 'Price']], hide_index=True, use_container_width=True)
                st.write("**1M**"); st.dataframe(filter_top_5_by_cap(brif_1m)[['Ticker', 'Price']], hide_index=True, use_container_width=True)

            st.subheader("Order Block Breakdowns")
            st.write("**1D**"); st.dataframe(filter_top_5_by_cap(brob_1d)[['Ticker', 'Price']], hide_index=True, use_container_width=True)
            st.write("**1W**"); st.dataframe(filter_top_5_by_cap(brob_1w)[['Ticker', 'Price']], hide_index=True, use_container_width=True)
            st.write("**1M**"); st.dataframe(filter_top_5_by_cap(brob_1m)[['Ticker', 'Price']], hide_index=True, use_container_width=True)
            st.write("**6M**"); st.dataframe(filter_top_5_by_cap(brob_6m)[['Ticker', 'Price']], hide_index=True, use_container_width=True)
            
            st.subheader("🔄 Trend Reversals (Exhaustion)"); st.write("**1W (Chop < 25)**")
            st.dataframe(filter_top_5_by_cap(rev_res), hide_index=True, use_container_width=True)

        st.divider()
        st.header("⚡ Volatility Squeeze Watchlist")
        c1, c2 = st.columns(2)
        with c1: st.write("**Daily Squeeze**"); st.dataframe(filter_top_5_by_cap(sq_1d), hide_index=True, use_container_width=True)
        with c2: st.write("**Weekly Squeeze**"); st.dataframe(filter_top_5_by_cap(sq_1w), hide_index=True, use_container_width=True)

        # === AI STRATEGIC BRIEFING SECTION ===
        st.divider()
        st.header("🤖 AI Strategic Analyst")
        
        found_tickers = df_all['Ticker'].unique().tolist()
        
        if found_tickers:
            c_sel, c_btn = st.columns([3, 1])
            with c_sel:
                selected_ticker = st.selectbox("Select a Ticker to Analyze:", found_tickers)
            with c_btn:
                st.write("") # Spacer
                run_ai = st.button("Generate Briefing ✨", type="primary")
            
            if run_ai and selected_ticker:
                ticker_data = df_all[df_all['Ticker'] == selected_ticker].iloc[0]
                signal_type = ticker_data['Type']
                price = ticker_data['Price']
                tf = ticker_data.get('TF', 'N/A')
                
                with st.spinner(f"🔍 Analyzing {selected_ticker} market context..."):
                    briefing = AI_Analyst.generate_briefing(selected_ticker, signal_type, price, tf)
                
                st.success(f"**Strategic Briefing: {selected_ticker}**")
                st.markdown(briefing)
                with st.expander("View Source Headlines"):
                    st.write(AI_Analyst.get_market_news(selected_ticker))
        else:
            st.info("No tickers available to analyze.")

if __name__ == "__main__":
    main()