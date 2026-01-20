from yahooquery import Ticker
import pandas as pd
import os
import time

def update_stock_data(ticker_symbol, filename, start_year=2020):
    """
    Update stock data dari Yahoo Finance
    """
    try:
        filepath = f'data/{filename}'
        
        # Load existing data
        if os.path.exists(filepath):
            existing_data = pd.read_csv(filepath)
            existing_data['Date'] = pd.to_datetime(existing_data['Date'])
            last_date = existing_data['Date'].max()
        else:
            existing_data = pd.DataFrame()
            last_date = pd.to_datetime(f'{start_year}-01-01')
        
        # Download from Yahoo
        ticker = Ticker(ticker_symbol)
        data = ticker.history(start=f'{start_year}-01-01', end='2026-12-31')
        
        if isinstance(data.index, pd.MultiIndex):
            data = data.reset_index(level=0, drop=True)
        
        # Fix timezone
        data.index = pd.to_datetime(data.index.astype(str).str[:10])
        data = data.reset_index()
        
        # Transform
        new_data = pd.DataFrame({
            'Date': data['date'],
            'Open': data['open'].astype(float),
            'High': data['high'].astype(float),
            'Low': data['low'].astype(float),
            'Close': data['close'].astype(float),
            'Volume': data['volume'].astype(int)
        })
        
        # Filter new data only
        if not existing_data.empty:
            new_data = new_data[new_data['Date'] > last_date]
            
            if len(new_data) == 0:
                return True  # No new data
            
            combined = pd.concat([existing_data, new_data], ignore_index=True)
        else:
            combined = new_data
        
        # Remove duplicates
        combined = combined.drop_duplicates(subset='Date', keep='last')
        
        # Sort descending
        combined = combined.sort_values('Date', ascending=False).reset_index(drop=True)
        
        # Save
        combined.to_csv(filepath, index=False)
        
        return True
        
    except Exception as e:
        print(f"Error updating {ticker_symbol}: {e}")
        return False


if __name__ == "__main__":
    # Manual run via terminal
    stocks = {
        'BBCA.JK': 'Bank Central Asia Stock Price History.csv',
        'BBRI.JK': 'Bank Rakyat Indonesia Stock Price History.csv',
        'BMRI.JK': 'Bank Mandiri Stock Price History.csv',
        'BBNI.JK': 'Bank Negara Indonesia Stock Price History.csv',
        'BBTN.JK': 'Bank Tabungan Negara Stock Price History.csv'
    }
    
    print("Updating stock data...")
    for ticker, filename in stocks.items():
        print(f"  {ticker}...", end=" ")
        if update_stock_data(ticker, filename):
            print("✓")
        else:
            print("✗")
        time.sleep(2)
    print("Done!")