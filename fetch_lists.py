import pandas as pd
import requests
import io

def fetch_nasdaq100():
    print("🇺🇸 Fetching Nasdaq 100 (NDX) tickers...")
    try:
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        
        # FIX: Send headers so Wikipedia thinks we are a real browser
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        # 1. Get raw HTML with headers
        response = requests.get(url, headers=headers)
        
        # 2. Parse the HTML text
        tables = pd.read_html(io.StringIO(response.text))
        
        # 3. Find the table with 'Ticker' or 'Symbol'
        ndx_df = None
        for t in tables:
            if "Ticker" in t.columns:
                ndx_df = t
                break
            elif "Symbol" in t.columns: # Fallback
                ndx_df = t
                ndx_df.rename(columns={"Symbol": "Ticker"}, inplace=True)
                break
        
        if ndx_df is not None:
            tickers = ndx_df["Ticker"].tolist()
            pd.DataFrame(tickers, columns=["Ticker"]).to_csv("us_tickers.csv", index=False)
            print(f"✅ Success! Saved {len(tickers)} NDX tickers to 'us_tickers.csv'")
        else:
            print("❌ Could not locate NDX table on Wikipedia.")
            
    except Exception as e:
        print(f"❌ Error fetching NDX: {e}")

def fetch_nifty500():
    print("🇮🇳 Fetching Nifty 500 tickers from NSE...")
    try:
        url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            csv_data = io.StringIO(response.content.decode('utf-8'))
            df = pd.read_csv(csv_data)
            
            if 'Symbol' in df.columns:
                tickers = df['Symbol'].apply(lambda x: f"{x}.NS").tolist()
                pd.DataFrame(tickers, columns=["Ticker"]).to_csv("ind_tickers.csv", index=False)
                print(f"✅ Success! Saved {len(tickers)} Nifty 500 tickers to 'ind_tickers.csv'")
            else:
                print("❌ CSV format changed. Could not find 'Symbol' column.")
        else:
            print(f"❌ Failed to download from NSE. Status Code: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error fetching Nifty 500: {e}")

if __name__ == "__main__":
    fetch_nasdaq100()
    print("---")
    fetch_nifty500()
    print("\n🚀 Ready. You can now run 'streamlit run app.py'")