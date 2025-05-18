import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
import json
import random
import os
import io
import base64

# Set page configuration
st.set_page_config(
    page_title="Trading Signals Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme colors
THEME = {
    'bg_color': '#1e1e2f',
    'text_color': '#ffffff',
    'primary_color': '#4CAF50',
    'secondary_color': '#2196F3',
    'accent_color': '#FF9800',
    'panel_bg': '#2a2a45',
    'buy_color': '#4CAF50',
    'sell_color': '#F44336',
    'neutral_color': '#9E9E9E',
    'chart_bg': '#2a2a45',
    'grid_color': '#3a3a5a'
}

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
    'confidence_threshold': 70,
    'enable_alerts': True,
    'alert_volume': 0.7,
    'strategy': 'scalping'
}

# Initialize session state for alerts
if 'enable_alerts' not in st.session_state:
    st.session_state.enable_alerts = True

if 'alert_volume' not in st.session_state:
    st.session_state.alert_volume = 0.7

# Store previous signals to avoid duplicate alerts
previous_signals = {}

# Signal history
signal_history = []

# History file path
HISTORY_FILE = 'signal_history.json'

# Load signal history from file
def load_signal_history():
    global signal_history
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r') as f:
                signal_history = json.load(f)
    except Exception as e:
        st.error(f"Error loading signal history: {str(e)}")
        signal_history = []

# Save signal history to file
def save_signal_history():
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(signal_history, f)
    except Exception as e:
        st.error(f"Error saving signal history: {str(e)}")

# Add new signal to history
def add_to_history(symbol, signal_type, price, confidence, timestamp=None):
    global signal_history

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Add new signal at the beginning of the list
    signal_history.insert(0, {
        'symbol': symbol,
        'signal': signal_type,
        'price': price,
        'confidence': confidence,
        'timestamp': timestamp
    })

    # Ensure history doesn't exceed max size
    if len(signal_history) > 50:
        signal_history = signal_history[:50]

    # Save history
    save_signal_history()

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

# Analyze candlestick patterns
def analyze_candlestick_patterns(df):
    patterns = {}

    # Calculate properties for current and previous candles
    current_idx = -1
    prev_idx = -2

    if len(df) < 2:
        return {
            'bullish_engulfing': False,
            'bearish_engulfing': False,
            'hammer': False,
            'shooting_star': False,
            'doji': False
        }

    # Calculate body size and direction
    current_body_size = abs(df['Close'].iloc[current_idx] - df['Open'].iloc[current_idx])
    prev_body_size = abs(df['Close'].iloc[prev_idx] - df['Open'].iloc[prev_idx])

    current_body_direction = 1 if df['Close'].iloc[current_idx] > df['Open'].iloc[current_idx] else -1
    prev_body_direction = 1 if df['Close'].iloc[prev_idx] > df['Open'].iloc[prev_idx] else -1

    # Calculate shadows
    current_upper_shadow = df['High'].iloc[current_idx] - max(df['Open'].iloc[current_idx], df['Close'].iloc[current_idx])
    current_lower_shadow = min(df['Open'].iloc[current_idx], df['Close'].iloc[current_idx]) - df['Low'].iloc[current_idx]

    # Calculate body to range ratio
    current_range = df['High'].iloc[current_idx] - df['Low'].iloc[current_idx]
    if current_range > 0:
        body_to_range_ratio = current_body_size / current_range
    else:
        body_to_range_ratio = 0

    # Bullish Engulfing pattern
    patterns['bullish_engulfing'] = (
        (current_body_direction == 1) and
        (prev_body_direction == -1) and
        (df['Open'].iloc[current_idx] < df['Close'].iloc[prev_idx]) and
        (df['Close'].iloc[current_idx] > df['Open'].iloc[prev_idx])
    )

    # Bearish Engulfing pattern
    patterns['bearish_engulfing'] = (
        (current_body_direction == -1) and
        (prev_body_direction == 1) and
        (df['Open'].iloc[current_idx] > df['Close'].iloc[prev_idx]) and
        (df['Close'].iloc[current_idx] < df['Open'].iloc[prev_idx])
    )

    # Hammer pattern
    patterns['hammer'] = (
        (body_to_range_ratio < 0.3) and
        (current_lower_shadow > 2 * current_body_size) and
        (current_upper_shadow < 0.5 * current_body_size)
    )

    # Shooting Star pattern
    patterns['shooting_star'] = (
        (body_to_range_ratio < 0.3) and
        (current_upper_shadow > 2 * current_body_size) and
        (current_lower_shadow < 0.5 * current_body_size)
    )

    # Doji pattern
    patterns['doji'] = current_body_size < 0.1 * current_range if current_range > 0 else False

    return patterns

# Find support and resistance levels
def find_support_resistance(df, window=10):
    pivots = {'support': [], 'resistance': []}

    if len(df) < window * 2:
        return pivots

    # Identify local peaks and troughs
    for i in range(window, len(df) - window):
        # Check if this is a local peak
        if all(df['High'].iloc[i] > df['High'].iloc[i-j] for j in range(1, window)) and \
           all(df['High'].iloc[i] > df['High'].iloc[i+j] for j in range(1, window)):
            pivots['resistance'].append(float(df['High'].iloc[i]))

        # Check if this is a local trough
        if all(df['Low'].iloc[i] < df['Low'].iloc[i-j] for j in range(1, window)) and \
           all(df['Low'].iloc[i] < df['Low'].iloc[i+j] for j in range(1, window)):
            pivots['support'].append(float(df['Low'].iloc[i]))

    # Filter out closely spaced levels
    if pivots['resistance']:
        filtered_resistance = [pivots['resistance'][0]]
        for level in pivots['resistance'][1:]:
            if all(abs(level - r) / r > 0.005 for r in filtered_resistance):
                filtered_resistance.append(level)
        pivots['resistance'] = sorted(filtered_resistance)[-2:] if len(filtered_resistance) >= 2 else filtered_resistance

    if pivots['support']:
        filtered_support = [pivots['support'][0]]
        for level in pivots['support'][1:]:
            if all(abs(level - s) / s > 0.005 for s in filtered_support):
                filtered_support.append(level)
        pivots['support'] = sorted(filtered_support)[:2] if len(filtered_support) >= 2 else filtered_support

    return pivots

# Analyze volume
def analyze_volume(df, volume_sma_period=20):
    if 'Volume' not in df.columns or df['Volume'].isnull().all():
        return {
            'high_volume': False,
            'low_volume': False,
            'volume_confirms_trend': False
        }

    if len(df) < volume_sma_period + 1:
        return {
            'high_volume': False,
            'low_volume': False,
            'volume_confirms_trend': False
        }

    try:
        volume_sma = df['Volume'].rolling(window=volume_sma_period).mean()
        if pd.isna(volume_sma.iloc[-1]) or pd.isna(df['Volume'].iloc[-1]):
            return {
                'high_volume': False,
                'low_volume': False,
                'volume_confirms_trend': False
            }

        volume_ratio = df['Volume'].iloc[-1] / volume_sma.iloc[-1]

        # Determine if volume is high or low
        high_volume = volume_ratio > 1.5
        low_volume = volume_ratio < 0.5

        # Determine if volume confirms trend
        volume_confirms_trend = (
            (df['Close'].iloc[-1] > df['Close'].iloc[-2] and df['Volume'].iloc[-1] > df['Volume'].iloc[-2]) or
            (df['Close'].iloc[-1] < df['Close'].iloc[-2] and df['Volume'].iloc[-1] > df['Volume'].iloc[-2])
        )

        return {
            'high_volume': high_volume,
            'low_volume': low_volume,
            'volume_confirms_trend': volume_confirms_trend
        }
    except Exception as e:
        st.error(f"Error analyzing volume: {str(e)}")
        return {
            'high_volume': False,
            'low_volume': False,
            'volume_confirms_trend': False
        }

# Analyze trend
def analyze_trend(df, ema_fast=8, ema_medium=21, ema_slow=50):
    if len(df) < max(ema_fast, ema_medium, ema_slow) + 1:
        return "unknown"

    try:
        df['ema_fast'] = ema(df, ema_fast)
        df['ema_medium'] = ema(df, ema_medium)
        df['ema_slow'] = ema(df, ema_slow)

        # Determine trend based on EMA alignment
        if pd.isna(df['ema_fast'].iloc[-1]) or pd.isna(df['ema_medium'].iloc[-1]) or pd.isna(df['ema_slow'].iloc[-1]):
            return "unknown"

        if df['ema_fast'].iloc[-1] > df['ema_medium'].iloc[-1] > df['ema_slow'].iloc[-1]:
            trend = "strong_uptrend"
        elif df['ema_fast'].iloc[-1] > df['ema_medium'].iloc[-1] and df['ema_medium'].iloc[-1] < df['ema_slow'].iloc[-1]:
            trend = "uptrend_resistance"
        elif df['ema_fast'].iloc[-1] < df['ema_medium'].iloc[-1] and df['ema_medium'].iloc[-1] > df['ema_slow'].iloc[-1]:
            trend = "downtrend_support"
        elif df['ema_fast'].iloc[-1] < df['ema_medium'].iloc[-1] < df['ema_slow'].iloc[-1]:
            trend = "strong_downtrend"
        else:
            trend = "ranging"

        return trend
    except Exception as e:
        st.error(f"Error analyzing trend: {str(e)}")
        return "unknown"

# Comprehensive price action analysis
def analyze_price_action(df, settings=SETTINGS, strategy='scalping'):
    # Check if we have enough data
    if df.empty or len(df) < 50:
        return {
            'signal': "no_signal",
            'confidence': 0,
            'tp': None,
            'sl': None,
            'trend': "unknown",
            'signals_details': [],
            'support_resistance': {'support': [], 'resistance': []},
            'last_close': None,
            'rsi': None,
            'macd': None,
            'macd_signal': None,
            'stoch_k': None,
            'stoch_d': None
        }

    try:
        # Calculate technical indicators
        df['SMA_fast'] = sma(df, settings['sma_fast'])
        df['SMA_slow'] = sma(df, settings['sma_slow'])
        df['EMA_fast'] = ema(df, settings['ema_fast'])
        df['EMA_medium'] = ema(df, settings['ema_medium'])
        df['EMA_slow'] = ema(df, settings['ema_slow'])
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

        upper_band, middle_band, lower_band = bollinger_bands(
            df,
            settings['bollinger_period'],
            settings['bollinger_std']
        )
        df['BB_upper'] = upper_band
        df['BB_middle'] = middle_band
        df['BB_lower'] = lower_band

        k, d = stochastic(df, settings['stoch_k'], settings['stoch_d'])
        df['Stoch_K'] = k
        df['Stoch_D'] = d

        df['ATR'] = atr(df, settings['atr_period'])

        # Analyze candlestick patterns
        candlestick_patterns = analyze_candlestick_patterns(df)

        # Analyze support and resistance levels
        support_resistance = find_support_resistance(df)

        # Analyze volume
        volume_analysis = analyze_volume(df, settings['volume_sma'])

        # Analyze trend
        trend = analyze_trend(df, settings['ema_fast'], settings['ema_medium'], settings['ema_slow'])

        # Get latest indicator values
        last_close = float(df['Close'].iloc[-1])
        last_rsi = float(df['RSI'].iloc[-1]) if not pd.isna(df['RSI'].iloc[-1]) else None
        last_macd = float(df['MACD'].iloc[-1]) if not pd.isna(df['MACD'].iloc[-1]) else None
        last_macd_signal = float(df['MACD_signal'].iloc[-1]) if not pd.isna(df['MACD_signal'].iloc[-1]) else None
        last_stoch_k = float(df['Stoch_K'].iloc[-1]) if not pd.isna(df['Stoch_K'].iloc[-1]) else None
        last_stoch_d = float(df['Stoch_D'].iloc[-1]) if not pd.isna(df['Stoch_D'].iloc[-1]) else None

        # Analyze signals
        buy_signals = []
        sell_signals = []

        # Strategy-specific signal weights
        signal_weights = {
            'scalping': {
                'moving_averages': 1.5,
                'oscillators': 2.0,
                'patterns': 1.0,
                'volume': 0.5
            },
            'day_trading': {
                'moving_averages': 1.0,
                'oscillators': 1.5,
                'patterns': 1.5,
                'volume': 1.0
            },
            'swing': {
                'moving_averages': 2.0,
                'oscillators': 1.0,
                'patterns': 1.5,
                'volume': 1.5
            },
            'position': {
                'moving_averages': 2.5,
                'oscillators': 0.5,
                'patterns': 1.0,
                'volume': 2.0
            }
        }

        # Moving Average signals
        if not pd.isna(df['SMA_fast'].iloc[-1]) and not pd.isna(df['SMA_slow'].iloc[-1]):
            # SMA fast crosses above SMA slow (buy signal)
            if df['SMA_fast'].iloc[-2] < df['SMA_slow'].iloc[-2] and df['SMA_fast'].iloc[-1] > df['SMA_slow'].iloc[-1]:
                buy_signals.append(f"SMA {settings['sma_fast']} crossed above SMA {settings['sma_slow']}")

            # SMA fast crosses below SMA slow (sell signal)
            if df['SMA_fast'].iloc[-2] > df['SMA_slow'].iloc[-2] and df['SMA_fast'].iloc[-1] < df['SMA_slow'].iloc[-1]:
                sell_signals.append(f"SMA {settings['sma_fast']} crossed below SMA {settings['sma_slow']}")

        if not pd.isna(df['EMA_fast'].iloc[-1]) and not pd.isna(df['EMA_medium'].iloc[-1]):
            # EMA fast crosses above EMA medium (buy signal)
            if df['EMA_fast'].iloc[-2] < df['EMA_medium'].iloc[-2] and df['EMA_fast'].iloc[-1] > df['EMA_medium'].iloc[-1]:
                buy_signals.append(f"EMA {settings['ema_fast']} crossed above EMA {settings['ema_medium']}")

            # EMA fast crosses below EMA medium (sell signal)
            if df['EMA_fast'].iloc[-2] > df['EMA_medium'].iloc[-2] and df['EMA_fast'].iloc[-1] < df['EMA_medium'].iloc[-1]:
                sell_signals.append(f"EMA {settings['ema_fast']} crossed below EMA {settings['ema_medium']}")

        # RSI signals
        if last_rsi is not None:
            if last_rsi < settings['rsi_oversold']:
                buy_signals.append(f"RSI in oversold territory ({last_rsi:.1f})")

            if last_rsi > settings['rsi_overbought']:
                sell_signals.append(f"RSI in overbought territory ({last_rsi:.1f})")

        # MACD signals
        if last_macd is not None and last_macd_signal is not None:
            # MACD line crosses above signal line (buy signal)
            if df['MACD'].iloc[-2] < df['MACD_signal'].iloc[-2] and df['MACD'].iloc[-1] > df['MACD_signal'].iloc[-1]:
                buy_signals.append("MACD crossed above signal line")

            # MACD line crosses below signal line (sell signal)
            if df['MACD'].iloc[-2] > df['MACD_signal'].iloc[-2] and df['MACD'].iloc[-1] < df['MACD_signal'].iloc[-1]:
                sell_signals.append("MACD crossed below signal line")

        # Stochastic signals
        if last_stoch_k is not None and last_stoch_d is not None:
            # %K crosses above %D in oversold territory (buy signal)
            if (df['Stoch_K'].iloc[-2] < df['Stoch_D'].iloc[-2] and df['Stoch_K'].iloc[-1] > df['Stoch_D'].iloc[-1] and
                    df['Stoch_K'].iloc[-1] < settings['stoch_oversold']):
                buy_signals.append("Stochastic %K crossed above %D in oversold territory")

            # %K crosses below %D in overbought territory (sell signal)
            if (df['Stoch_K'].iloc[-2] > df['Stoch_D'].iloc[-2] and df['Stoch_K'].iloc[-1] < df['Stoch_D'].iloc[-1] and
                    df['Stoch_K'].iloc[-1] > settings['stoch_overbought']):
                sell_signals.append("Stochastic %K crossed below %D in overbought territory")

            # Stochastic in oversold territory
            if last_stoch_k < settings['stoch_oversold'] and last_stoch_d < settings['stoch_oversold']:
                buy_signals.append(f"Stochastic in oversold territory (%K: {last_stoch_k:.1f}, %D: {last_stoch_d:.1f})")

            # Stochastic in overbought territory
            if last_stoch_k > settings['stoch_overbought'] and last_stoch_d > settings['stoch_overbought']:
                sell_signals.append(f"Stochastic in overbought territory (%K: {last_stoch_k:.1f}, %D: {last_stoch_d:.1f})")

        # Bollinger Bands signals
        if 'BB_upper' in df.columns and 'BB_lower' in df.columns:
            if not pd.isna(df['BB_lower'].iloc[-1]) and not pd.isna(df['BB_upper'].iloc[-1]):
                if df['Close'].iloc[-1] < df['BB_lower'].iloc[-1]:
                    buy_signals.append("Price below lower Bollinger Band")

                if df['Close'].iloc[-1] > df['BB_upper'].iloc[-1]:
                    sell_signals.append("Price above upper Bollinger Band")

        # Candlestick pattern signals
        if candlestick_patterns['bullish_engulfing']:
            buy_signals.append("Bullish Engulfing pattern")

        if candlestick_patterns['bearish_engulfing']:
            sell_signals.append("Bearish Engulfing pattern")

        if candlestick_patterns['hammer']:
            buy_signals.append("Hammer pattern")

        if candlestick_patterns['shooting_star']:
            sell_signals.append("Shooting Star pattern")

        # Volume signals
        if 'Volume' in df.columns and not df['Volume'].isnull().all():
            if volume_analysis.get('high_volume') and volume_analysis.get('volume_confirms_trend'):
                if len(df) > 1 and not pd.isna(df['Close'].iloc[-1]) and not pd.isna(df['Close'].iloc[-2]):
                    if df['Close'].iloc[-1] > df['Close'].iloc[-2]:
                        buy_signals.append("High volume confirms uptrend")
                    else:
                        sell_signals.append("High volume confirms downtrend")

        # Calculate signal confidence with strategy weights
        weights = signal_weights.get(strategy, signal_weights['scalping'])
        
        # Count signals by category
        ma_buy = sum(1 for s in buy_signals if "SMA" in s or "EMA" in s)
        ma_sell = sum(1 for s in sell_signals if "SMA" in s or "EMA" in s)
        
        osc_buy = sum(1 for s in buy_signals if "RSI" in s or "Stochastic" in s or "MACD" in s)
        osc_sell = sum(1 for s in sell_signals if "RSI" in s or "Stochastic" in s or "MACD" in s)
        
        pattern_buy = sum(1 for s in buy_signals if "pattern" in s or "Bollinger" in s)
        pattern_sell = sum(1 for s in sell_signals if "pattern" in s or "Bollinger" in s)
        
        vol_buy = sum(1 for s in buy_signals if "volume" in s)
        vol_sell = sum(1 for s in sell_signals if "volume" in s)
        
        # Calculate weighted confidence
        buy_confidence = (
            (ma_buy * weights['moving_averages']) +
            (osc_buy * weights['oscillators']) +
            (pattern_buy * weights['patterns']) +
            (vol_buy * weights['volume'])
        ) * 10  # Scale to percentage
        
        sell_confidence = (
            (ma_sell * weights['moving_averages']) +
            (osc_sell * weights['oscillators']) +
            (pattern_sell * weights['patterns']) +
            (vol_sell * weights['volume'])
        ) * 10  # Scale to percentage

        # Determine final signal based on confidence threshold
        signal = "no_signal"
        confidence = 0
        signals_details = []

        if buy_confidence >= settings['confidence_threshold'] and buy_confidence > sell_confidence:
            signal = "buy"
            confidence = buy_confidence
            signals_details = buy_signals
        elif sell_confidence >= settings['confidence_threshold'] and sell_confidence > buy_confidence:
            signal = "sell"
            confidence = sell_confidence
            signals_details = sell_signals

        # Calculate take profit and stop loss levels
        tp = None
        sl = None
        risk_reward_ratio = settings['risk_reward_ratio']

        if signal == "buy" and not pd.isna(df['ATR'].iloc[-1]):
            # Use ATR for stop loss calculation
            atr_value = float(df['ATR'].iloc[-1])
            sl = last_close - (atr_value * 1.5)
            tp = last_close + (atr_value * 1.5 * risk_reward_ratio)

            # Adjust stop loss based on support levels if available
            if support_resistance['support']:
                nearest_support = max([s for s in support_resistance['support'] if s < last_close], default=sl)
                sl = max(sl, nearest_support)

            # Adjust take profit based on resistance levels if available
            if support_resistance['resistance']:
                nearest_resistance = min([r for r in support_resistance['resistance'] if r > last_close], default=tp)
                tp = min(tp, nearest_resistance)

        elif signal == "sell" and not pd.isna(df['ATR'].iloc[-1]):
            # Use ATR for stop loss calculation
            atr_value = float(df['ATR'].iloc[-1])
            sl = last_close + (atr_value * 1.5)
            tp = last_close - (atr_value * 1.5 * risk_reward_ratio)

            # Adjust stop loss based on resistance levels if available
            if support_resistance['resistance']:
                nearest_resistance = min([r for r in support_resistance['resistance'] if r > last_close], default=sl)
                sl = min(sl, nearest_resistance)

            # Adjust take profit based on support levels if available
            if support_resistance['support']:
                nearest_support = max([s for s in support_resistance['support'] if s < last_close], default=tp)
                tp = max(tp, nearest_support)

        # Round values
        tp = round(float(tp), 5) if tp is not None else None
        sl = round(float(sl), 5) if sl is not None else None

        return {
            'signal': signal,
            'confidence': confidence,
            'tp': tp,
            'sl': sl,
            'trend': trend,
            'signals_details': signals_details,
            'support_resistance': support_resistance,
            'last_close': last_close,
            'rsi': last_rsi,
            'macd': last_macd,
            'macd_signal': last_macd_signal,
            'stoch_k': last_stoch_k,
            'stoch_d': last_stoch_d
        }
    except Exception as e:
        st.error(f"Error in technical analysis: {str(e)}")
        return {
            'signal': "error",
            'confidence': 0,
            'tp': None,
            'sl': None,
            'trend': "unknown",
            'signals_details': [],
            'support_resistance': {'support': [], 'resistance': []},
            'last_close': None,
            'rsi': None,
            'macd': None,
            'macd_signal': None,
            'stoch_k': None,
            'stoch_d': None
        }

# Create alert sound
def create_alert_sound():
    audio_html = """
    <script>
    function playAlertSound(volume) {
        var audioContext = new (window.AudioContext || window.webkitAudioContext)();
        var oscillator = audioContext.createOscillator();
        var gainNode = audioContext.createGain();
        
        oscillator.type = 'sine';
        oscillator.frequency.setValueAtTime(880, audioContext.currentTime);
        oscillator.frequency.setValueAtTime(440, audioContext.currentTime + 0.25);
        
        gainNode.gain.setValueAtTime(volume, audioContext.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.001, audioContext.currentTime + 0.5);
        
        oscillator.connect(gainNode);
        gainNode.connect(audioContext.destination);
        
        oscillator.start();
        oscillator.stop(audioContext.currentTime + 0.5);
    }
    </script>
    """
    return audio_html

# Trigger alert sound
def trigger_alert(volume=0.7):
    return f"""
    <script>
    playAlertSound({volume});
    </script>
    """

# Format signal with colors
def format_signal(signal):
    if signal == "buy":
        return f"<span style='color:#4CAF50;font-weight:bold;font-size:1.5em;'>BUY</span>"
    elif signal == "sell":
        return f"<span style='color:#F44336;font-weight:bold;font-size:1.5em;'>SELL</span>"
    else:
        return f"<span style='color:#9E9E9E;font-size:1.5em;'>NO SIGNAL</span>"

# Format confidence with colors
def format_confidence(confidence):
    if confidence >= 80:
        return f"<span style='color:#4CAF50;font-weight:bold;font-size:1.5em;'>{confidence:.1f}%</span>"
    elif confidence >= 60:
        return f"<span style='color:#FFC107;font-weight:bold;font-size:1.5em;'>{confidence:.1f}%</span>"
    else:
        return f"<span style='color:#F44336;font-weight:bold;font-size:1.5em;'>{confidence:.1f}%</span>"

# Format trend with colors
def format_trend(trend):
    if trend == "strong_uptrend":
        return f"<span style='color:#4CAF50;font-weight:bold;font-size:1.5em;'>Strong Uptrend</span>"
    elif trend == "uptrend_resistance":
        return f"<span style='color:#8BC34A;font-weight:bold;font-size:1.5em;'>Uptrend with Resistance</span>"
    elif trend == "downtrend_support":
        return f"<span style='color:#FF9800;font-weight:bold;font-size:1.5em;'>Downtrend with Support</span>"
    elif trend == "strong_downtrend":
        return f"<span style='color:#F44336;font-weight:bold;font-size:1.5em;'>Strong Downtrend</span>"
    elif trend == "ranging":
        return f"<span style='color:#9E9E9E;font-weight:bold;font-size:1.5em;'>Ranging</span>"
    else:
        return f"<span style='color:#9E9E9E;font-size:1.5em;'>Unknown</span>"

# Apply custom CSS
def apply_custom_css():
    colors = THEME
    
    st.markdown(f"""
    <style>
    .main {{
        background-color: {colors['bg_color']};
        color: {colors['text_color']};
    }}
    .stApp {{
        background-color: {colors['bg_color']};
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: {colors['primary_color']} !important;
    }}
    .stButton>button {{
        background-color: {colors['primary_color']};
        color: {colors['bg_color']};
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
    }}
    .stButton>button:hover {{
        background-color: {colors['secondary_color']};
    }}
    .signal-card {{
        background-color: {colors['panel_bg']};
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
        border-left: 4px solid {colors['primary_color']};
    }}
    .signal-title {{
        font-size: 1.2em;
        font-weight: bold;
        margin-bottom: 10px;
        color: {colors['text_color']};
    }}
    .signal-value {{
        font-size: 1.8em;
        font-weight: bold;
        margin-bottom: 5px;
    }}
    .signal-details {{
        font-size: 0.9em;
        color: {colors['text_color']};
        opacity: 0.8;
        margin-top: 10px;
    }}
    .price-info {{
        display: flex;
        justify-content: space-between;
        margin-top: 10px;
    }}
    .price-item {{
        text-align: center;
    }}
    .price-label {{
        font-size: 0.9em;
        color: {colors['text_color']};
        opacity: 0.8;
    }}
    .price-value {{
        font-size: 1.1em;
        font-weight: bold;
    }}
    .signal-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
        gap: 15px;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {colors['panel_bg']};
        color: {colors['text_color']};
        border-radius: 4px 4px 0 0;
    }}
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {{
        font-size: 1rem;
    }}
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        background-color: {colors['primary_color']};
        color: {colors['bg_color']};
    }}
    .stDataFrame {{
        color: {colors['text_color']};
    }}
    .stDataFrame [data-testid="stTable"] {{
        color: {colors['text_color']};
    }}
    .stDataFrame th {{
        background-color: {colors['panel_bg']};
        color: {colors['text_color']};
    }}
    .stDataFrame td {{
        color: {colors['text_color']};
    }}
    @media (max-width: 768px) {{
        .signal-grid {{
            grid-template-columns: 1fr;
        }}
    }}
    </style>
    """, unsafe_allow_html=True)

# Display signal card
def display_signal_card(symbol_name, analysis):
    signal_html = f"""
    <div class="signal-card">
        <div class="signal-title">{symbol_name}</div>
        <div class="signal-value">{format_signal(analysis['signal'])}</div>
        <div class="signal-details">
            <strong>Confidence:</strong> {format_confidence(analysis['confidence'])}
            <br>
            <strong>Trend:</strong> {format_trend(analysis['trend'])}
        </div>
        <div class="price-info">
            <div class="price-item">
                <div class="price-label">Current Price</div>
                <div class="price-value">{analysis['last_close']:.5f}</div>
            </div>
    """
    
    if analysis['tp'] is not None:
        signal_html += f"""
            <div class="price-item">
                <div class="price-label">Take Profit</div>
                <div class="price-value" style="color:{THEME['buy_color']};">{analysis['tp']:.5f}</div>
            </div>
        """
    
    if analysis['sl'] is not None:
        signal_html += f"""
            <div class="price-item">
                <div class="price-label">Stop Loss</div>
                <div class="price-value" style="color:{THEME['sell_color']};">{analysis['sl']:.5f}</div>
            </div>
        """
    
    signal_html += """
        </div>
    </div>
    """
    
    return signal_html

# Main function
def main():
    # Apply custom CSS
    apply_custom_css()
    
    # Add alert sound JavaScript
    st.markdown(create_alert_sound(), unsafe_allow_html=True)
    
    # App title
    st.title("Trading Signals Dashboard")
    st.markdown("### Real-time signals for all trading pairs")

    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Timeframe selection
        timeframe = st.selectbox(
            "Timeframe",
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
        
        # Alert settings
        enable_alerts = st.checkbox(
            "Enable Sound Alerts",
            value=st.session_state.enable_alerts
        )
        
        if enable_alerts != st.session_state.enable_alerts:
            st.session_state.enable_alerts = enable_alerts
        
        if enable_alerts:
            alert_volume = st.slider(
                "Alert Volume",
                0.1, 1.0, st.session_state.alert_volume, 0.1
            )
            
            if alert_volume != st.session_state.alert_volume:
                st.session_state.alert_volume = alert_volume
        
        # Update button
        if st.button("Update Now"):
            st.success("✅ Updated successfully!")
    
    # Create tabs
    tabs = st.tabs([
        "📊 All Signals", 
        "📋 Signal History"
    ])
    
    # All Signals tab
    with tabs[0]:
        st.markdown("<div class='signal-grid'>", unsafe_allow_html=True)
        
        # Process all symbols
        for symbol_name, symbol in SYMBOLS.items():
            with st.spinner(f"Analyzing {symbol_name}..."):
                df = fetch_data(symbol, period=period, interval=timeframe)
                
                if not df.empty:
                    # Analyze data
                    analysis = analyze_price_action(df, SETTINGS)
                    
                    # Display signal card
                    st.markdown(display_signal_card(symbol_name, analysis), unsafe_allow_html=True)
                    
                    # Check if we need to trigger an alert
                    if st.session_state.enable_alerts and analysis['signal'] != "no_signal" and analysis['signal'] != "error":
                        # Check if this is a new signal
                        symbol_key = f"{symbol_name}_{timeframe}"
                        if symbol_key not in previous_signals or previous_signals[symbol_key] != analysis['signal']:
                            previous_signals[symbol_key] = analysis['signal']
                            st.markdown(trigger_alert(st.session_state.alert_volume), unsafe_allow_html=True)
                            
                            # Add to history
                            add_to_history(
                                symbol_name,
                                analysis['signal'],
                                analysis['last_close'],
                                analysis['confidence']
                            )
                else:
                    st.error(f"Error fetching data for {symbol_name}")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # History tab
    with tabs[1]:
        # Load signal history
        load_signal_history()
        
        if signal_history:
            # Create DataFrame from signal history
            history_df = pd.DataFrame(signal_history)
            
            # Format display
            history_df.columns = ["Symbol", "Signal", "Price", "Confidence", "Timestamp"]
            
            # Display table
            st.dataframe(history_df, use_container_width=True)
            
            # Clear history button
            if st.button("Clear History"):
                signal_history.clear()
                save_signal_history()
                st.success("Signal history cleared successfully.")
        else:
            st.info("Signal history is empty.")
    
    # Add last update info
    st.markdown("---")
    st.markdown(f"<div style='text-align:center;color:#aaa;font-size:0.8em;'>Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>", unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
