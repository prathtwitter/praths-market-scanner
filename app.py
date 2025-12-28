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