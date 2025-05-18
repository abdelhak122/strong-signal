import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
from datetime import datetime
import json
import os

# Set page configuration
st.set_page_config(
    page_title="تقرير إشارات السكالبينج",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Available trading pairs
SYMBOLS = {
    'EUR/USD': 'EURUSD=X',
    'XAU/USD': 'GC=F',
    'BTC/USD': 'BTC-USD',
    'NAS100': '^NDX',
    'ETH/USD': 'ETH-USD',
    'USD/JPY': 'JPY=X',
    'GBP/USD': 'GBPUSD=X',
    'S&P500': '^GSPC'
}

# Default settings for technical indicators
SETTINGS = {
    'sma_fast': 10,
    'sma_slow': 30,
    'ema_fast': 8,
    'ema_medium': 21,
    'ema_slow': 50,
    'rsi_period': 14,
    'rsi_overbought': 70,
    'rsi_oversold': 30,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'bollinger_period': 20,
    'bollinger_std': 2,
    'atr_period': 14,
    'stoch_k': 14,
    'stoch_d': 3,
    'stoch_overbought': 80,
    'stoch_oversold': 20,
    'volume_sma': 20,
    'risk_reward_ratio': 1.5,
    'confidence_threshold': 70
}

# Fetch data from Yahoo Finance
def fetch_data(symbol, period='5d', interval='5m'):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            return pd.DataFrame()
        return df
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return pd.DataFrame()

# Calculate Simple Moving Average
def sma(data, window):
    return data['Close'].rolling(window=window).mean()

# Calculate Exponential Moving Average
def ema(data, window):
    return data['Close'].ewm(span=window, adjust=False).mean()

# Calculate Relative Strength Index
def rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Calculate MACD
def macd(data, fast=12, slow=26, signal=9):
    ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

# Calculate Bollinger Bands
def bollinger_bands(data, window=20, std=2):
    sma_line = data['Close'].rolling(window=window).mean()
    std_dev = data['Close'].rolling(window=window).std()
    upper_band = sma_line + (std_dev * std)
    lower_band = sma_line - (std_dev * std)
    return upper_band, sma_line, lower_band

# Calculate Stochastic Oscillator
def stochastic(data, k_period=14, d_period=3):
    lowest_low = data['Low'].rolling(window=k_period).min()
    highest_high = data['High'].rolling(window=k_period).max()
    k = 100 * ((data['Close'] - lowest_low) / (highest_high - lowest_low))
    d = k.rolling(window=d_period).mean()
    return k, d

# Calculate Average True Range
def atr(data, period=14):
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(window=period).mean()

# Analyze price action
def analyze_price_action(df, settings=SETTINGS):
    # Check if we have enough data
    if df.empty or len(df) < 50:
        return {
            'signal': "no_signal",
            'tp': None,
            'sl': None,
            'last_close': None
        }

    try:
        # Calculate technical indicators
        df['SMA_fast'] = sma(df, settings['sma_fast'])
        df['SMA_slow'] = sma(df, settings['sma_slow'])
        df['RSI'] = rsi(df, settings['rsi_period'])

        macd_line, signal_line, histogram = macd(
            df,
            settings['macd_fast'],
            settings['macd_slow'],
            settings['macd_signal']
        )
        df['MACD'] = macd_line
        df['MACD_signal'] = signal_line
        df['MACD_histogram'] = histogram

        k, d = stochastic(df, settings['stoch_k'], settings['stoch_d'])
        df['Stoch_K'] = k
        df['Stoch_D'] = d

        df['ATR'] = atr(df, settings['atr_period'])

        # Get latest indicator values
        last_close = float(df['Close'].iloc[-1]) if not df.empty and not pd.isna(df['Close'].iloc[-1]) else None
        last_rsi = float(df['RSI'].iloc[-1]) if not df.empty and not pd.isna(df['RSI'].iloc[-1]) else None
        last_macd = float(df['MACD'].iloc[-1]) if not df.empty and not pd.isna(df['MACD'].iloc[-1]) else None
        last_macd_signal = float(df['MACD_signal'].iloc[-1]) if not df.empty and not pd.isna(df['MACD_signal'].iloc[-1]) else None
        last_stoch_k = float(df['Stoch_K'].iloc[-1]) if not df.empty and not pd.isna(df['Stoch_K'].iloc[-1]) else None
        last_stoch_d = float(df['Stoch_D'].iloc[-1]) if not df.empty and not pd.isna(df['Stoch_D'].iloc[-1]) else None

        # Analyze signals
        buy_signals = 0
        sell_signals = 0

        # SMA signals
        if not pd.isna(df['SMA_fast'].iloc[-1]) and not pd.isna(df['SMA_slow'].iloc[-1]):
            # SMA fast crosses above SMA slow (buy signal)
            if df['SMA_fast'].iloc[-2] < df['SMA_slow'].iloc[-2] and df['SMA_fast'].iloc[-1] > df['SMA_slow'].iloc[-1]:
                buy_signals += 1

            # SMA fast crosses below SMA slow (sell signal)
            if df['SMA_fast'].iloc[-2] > df['SMA_slow'].iloc[-2] and df['SMA_fast'].iloc[-1] < df['SMA_slow'].iloc[-1]:
                sell_signals += 1

        # RSI signals
        if last_rsi is not None:
            if last_rsi < settings['rsi_oversold']:
                buy_signals += 1

            if last_rsi > settings['rsi_overbought']:
                sell_signals += 1

        # MACD signals
        if last_macd is not None and last_macd_signal is not None:
            # MACD line crosses above signal line (buy signal)
            if df['MACD'].iloc[-2] < df['MACD_signal'].iloc[-2] and df['MACD'].iloc[-1] > df['MACD_signal'].iloc[-1]:
                buy_signals += 1

            # MACD line crosses below signal line (sell signal)
            if df['MACD'].iloc[-2] > df['MACD_signal'].iloc[-2] and df['MACD'].iloc[-1] < df['MACD_signal'].iloc[-1]:
                sell_signals += 1

        # Stochastic signals
        if last_stoch_k is not None and last_stoch_d is not None:
            # %K crosses above %D in oversold territory (buy signal)
            if (df['Stoch_K'].iloc[-2] < df['Stoch_D'].iloc[-2] and df['Stoch_K'].iloc[-1] > df['Stoch_D'].iloc[-1] and
                    df['Stoch_K'].iloc[-1] < settings['stoch_oversold']):
                buy_signals += 1

            # %K crosses below %D in overbought territory (sell signal)
            if (df['Stoch_K'].iloc[-2] > df['Stoch_D'].iloc[-2] and df['Stoch_K'].iloc[-1] < df['Stoch_D'].iloc[-1] and
                    df['Stoch_K'].iloc[-1] > settings['stoch_overbought']):
                sell_signals += 1

        # Determine final signal
        signal = "no_signal"
        if buy_signals >= 2:
            signal = "buy"
        elif sell_signals >= 2:
            signal = "sell"

        # Calculate take profit and stop loss levels
        tp = None
        sl = None

        if signal == "buy" and last_close is not None and not pd.isna(df['ATR'].iloc[-1]):
            # Use ATR for stop loss calculation
            atr_value = float(df['ATR'].iloc[-1])
            sl = last_close - (atr_value * 1.5)
            tp = last_close + (atr_value * 1.5 * settings['risk_reward_ratio'])

        elif signal == "sell" and last_close is not None and not pd.isna(df['ATR'].iloc[-1]):
            # Use ATR for stop loss calculation
            atr_value = float(df['ATR'].iloc[-1])
            sl = last_close + (atr_value * 1.5)
            tp = last_close - (atr_value * 1.5 * settings['risk_reward_ratio'])

        # Round values
        tp = round(float(tp), 2) if tp is not None else None
        sl = round(float(sl), 2) if sl is not None else None
        last_close = round(float(last_close), 2) if last_close is not None else None

        return {
            'signal': signal,
            'tp': tp,
            'sl': sl,
            'last_close': last_close
        }
    except Exception as e:
        st.error(f"Error in technical analysis: {str(e)}")
        return {
            'signal': "error",
            'tp': None,
            'sl': None,
            'last_close': None
        }

# Apply custom CSS for Arabic RTL layout
def apply_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700&display=swap');
    
    html, body, [class*="css"] {
        direction: rtl;
        font-family: 'Tajawal', sans-serif !important;
    }
    
    .main {
        background-color: #0e1525;
        color: #ffffff;
    }
    
    .stApp {
        background-color: #0e1525;
    }
    
    h1, h2, h3 {
        color: #ffc107 !important;
        text-align: center;
        font-weight: bold;
    }
    
    .header-text {
        text-align: center;
        color: #a0aec0;
        margin-bottom: 20px;
    }
    
    .report-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 20px;
        direction: rtl;
    }
    
    .report-table th {
        background-color: #ffc107;
        color: #0e1525;
        text-align: center;
        padding: 10px;
        font-weight: bold;
    }
    
    .report-table td {
        padding: 10px;
        text-align: center;
        border: 1px solid #1e293b;
    }
    
    .report-table tr:nth-child(odd) {
        background-color: #1e293b;
    }
    
    .report-table tr:nth-child(even) {
        background-color: #0f172a;
    }
    
    .buy-signal {
        color: #4ade80;
        font-weight: bold;
    }
    
    .sell-signal {
        color: #f87171;
        font-weight: bold;
    }
    
    .no-signal {
        color: #94a3b8;
        font-style: italic;
    }
    
    .footer {
        text-align: center;
        margin-top: 30px;
        padding-top: 20px;
        border-top: 1px solid #334155;
        color: #94a3b8;
        font-size: 0.9em;
    }
    
    .footer a {
        color: #ffc107;
        text-decoration: none;
    }
    
    .stButton>button {
        background-color: #ffc107;
        color: #0e1525;
        font-weight: bold;
        border: none;
    }
    
    .stButton>button:hover {
        background-color: #f59e0b;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# Format signal text in Arabic
def format_signal(signal):
    if signal == "buy":
        return '<span class="buy-signal">شراء</span>'
    elif signal == "sell":
        return '<span class="sell-signal">بيع</span>'
    else:
        return '<span class="no-signal">لا توجد إشارة</span>'

# Main function
def main():
    # Apply custom CSS
    apply_custom_css()
    
    # Title and description
    st.markdown('<h1>تقرير إشارات السكالبينج</h1>', unsafe_allow_html=True)
    st.markdown('<p class="header-text">هذا التقرير يعرض إشارات الشراء والبيع بناءً على تحليل السعر باستخدام مؤشرات SMA و RSI.<br>يتم التنبيه في حال ظهور إشارة مهمة لتنبيهك.</p>', unsafe_allow_html=True)
    
    # Sidebar for settings
    with st.sidebar:
        st.header("الإعدادات")
        
        # Timeframe selection
        timeframe = st.selectbox(
            "الإطار الزمني",
            ["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
            index=1
        )
        
        # Determine data period based on timeframe
        if timeframe in ["1m", "5m", "15m", "30m"]:
            period = "1d"
        elif timeframe in ["1h", "4h"]:
            period = "5d"
        else:
            period = "1mo"
        
        # Update button
        if st.button("تحديث الآن"):
            st.success("✅ تم التحديث بنجاح!")
    
    # Create table header
    table_html = """
    <table class="report-table">
        <tr>
            <th>الرمز</th>
            <th>السعر الحالي</th>
            <th>الإشارة</th>
            <th>هدف الربح (TP)</th>
            <th>وقف الخسارة (SL)</th>
        </tr>
    """
    
    # Process all symbols
    for symbol_name, symbol in SYMBOLS.items():
        try:
            df = fetch_data(symbol, period=period, interval=timeframe)
            
            if not df.empty:
                # Analyze data
                analysis = analyze_price_action(df, SETTINGS)
                
                # Format values
                current_price = f"{analysis['last_close']:.2f}" if analysis['last_close'] is not None else "-"
                tp_value = f"{analysis['tp']:.2f}" if analysis['tp'] is not None else "-"
                sl_value = f"{analysis['sl']:.2f}" if analysis['sl'] is not None else "-"
                signal_text = format_signal(analysis['signal'])
                
                # Add row to table
                table_html += f"""
                <tr>
                    <td>{symbol_name}</td>
                    <td>{current_price}</td>
                    <td>{signal_text}</td>
                    <td>{tp_value}</td>
                    <td>{sl_value}</td>
                </tr>
                """
            else:
                # Add empty row if data fetch failed
                table_html += f"""
                <tr>
                    <td>{symbol_name}</td>
                    <td>-</td>
                    <td><span class="no-signal">لا توجد بيانات</span></td>
                    <td>-</td>
                    <td>-</td>
                </tr>
                """
        except Exception as e:
            st.error(f"Error processing {symbol_name}: {str(e)}")
            # Add error row
            table_html += f"""
            <tr>
                <td>{symbol_name}</td>
                <td>-</td>
                <td><span class="no-signal">خطأ</span></td>
                <td>-</td>
                <td>-</td>
            </tr>
            """
    
    # Close table
    table_html += "</table>"
    
    # Display table
    st.markdown(table_html, unsafe_allow_html=True)
    
    # Footer with contact info
    st.markdown("""
    <div class="footer">
        © 2025 عبد الحق - هاتف: 0664959709 - إيميل: <a href="mailto:abdelhak122@gmail.com">abdelhak122@gmail.com</a> - واتساب: 0664959709
    </div>
    """, unsafe_allow_html=True)
    
    # Add last update info
    st.markdown(f"""
    <div style='text-align:center;color:#64748b;font-size:0.8em;margin-top:10px;'>
        آخر تحديث: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
    """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
