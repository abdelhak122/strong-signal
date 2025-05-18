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
from matplotlib import font_manager

# Set page configuration
st.set_page_config(
    page_title="Trading Signals Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for language
if 'language' not in st.session_state:
    st.session_state.language = 'en'

# Language dictionary
lang = {
    'en': {
        'title': "Professional Trading Signals Platform",
        'subtitle': "Advanced Technical Analysis for Scalping",
        'settings': "Settings",
        'select_pair': "Select Trading Pair:",
        'timeframe': "Timeframe:",
        'advanced_settings': "Advanced Settings",
        'moving_averages': "Moving Averages",
        'oscillators': "Oscillators",
        'macd_settings': "MACD Settings",
        'bollinger_settings': "Bollinger Bands",
        'stochastic_settings': "Stochastic",
        'other_settings': "Other Settings",
        'confidence_threshold': "Signal Confidence Threshold (%)",
        'risk_reward': "Risk/Reward Ratio",
        'display_options': "Display Options",
        'show_charts': "Show Charts",
        'save_history': "Save Signal History",
        'show_tips': "Show Trading Tips",
        'update_button': "Update Now",
        'trading_tip': "Trading Tip",
        'dashboard': "Dashboard",
        'charts': "Charts",
        'history': "Signal History",
        'instructions': "Instructions",
        'signal': "Signal",
        'confidence': "Confidence",
        'trend': "Trend",
        'current_price': "Current Price",
        'take_profit': "Take Profit",
        'stop_loss': "Stop Loss",
        'signal_details': "Signal Details",
        'support_resistance': "Support & Resistance",
        'resistance': "Resistance",
        'support': "Support",
        'technical_indicators': "Technical Indicators",
        'no_signals': "No sufficient signals for a decision.",
        'no_levels': "No specific levels identified.",
        'not_available': "Not available",
        'rsi': "RSI",
        'overbought': "Overbought",
        'oversold': "Oversold",
        'neutral': "Neutral",
        'signal_line': "Signal Line",
        'difference': "Difference",
        'status': "Status",
        'positive': "Positive",
        'negative': "Negative",
        'stochastic': "Stochastic",
        'charts_disabled': "Charts are disabled. You can enable them in Advanced Settings.",
        'history_empty': "Signal history is empty or disabled. You can enable history saving in Advanced Settings.",
        'clear_history': "Clear History",
        'history_cleared': "Signal history cleared successfully.",
        'how_to_use': "How to Use",
        'important_notice': "Important Notice",
        'last_update': "Last update",
        'buy': "BUY",
        'sell': "SELL",
        'no_signal': "NO SIGNAL",
        'error': "ERROR",
        'strong_uptrend': "Strong Uptrend",
        'uptrend_resistance': "Uptrend with Resistance",
        'downtrend_support': "Downtrend with Support",
        'strong_downtrend': "Strong Downtrend",
        'ranging': "Ranging",
        'unknown': "Unknown",
        'language_option': "Language",
        'theme_option': "Theme",
        'light': "Light",
        'dark': "Dark",
        'blue': "Blue",
        'green': "Green",
        'strategy': "Strategy",
        'scalping': "Scalping",
        'day_trading': "Day Trading",
        'swing': "Swing",
        'position': "Position",
        'signal_strength': "Signal Strength",
        'weak': "Weak",
        'moderate': "Moderate",
        'strong': "Strong",
        'very_strong': "Very Strong",
        'enable_alerts': "Enable Sound Alerts",
        'alert_volume': "Alert Volume",
        'data_loading': "Loading data...",
        'data_analyzing': "Analyzing data...",
        'no_data': "No data available for the selected pair. Check your internet connection or try another pair.",
        'instructions_steps': [
            "Select a trading pair from the dropdown menu in the sidebar.",
            "Choose the appropriate timeframe for your analysis.",
            "Customize technical indicators in the Advanced Settings section if needed.",
            "Monitor the dashboard for current signals, confidence level, and trend.",
            "Use the suggested take profit and stop loss levels to manage risk in your trades.",
            "Check signal details to understand the reasons behind the current recommendation.",
            "Review the chart for visual analysis of price and technical indicators.",
            "Click the 'Update Now' button in the sidebar to get the latest data and analysis.",
            "Enable sound alerts to get notified when strong signals appear."
        ],
        'important_notices': [
            "This application is for educational and informational purposes only and not a trading recommendation.",
            "Always conduct your own analysis and consult a financial advisor before making investment decisions.",
            "Trading involves risk, and you can lose more than your initial investment.",
            "Past performance is not indicative of future results."
        ],
        'trading_tips': [
            "Always set a stop loss to protect your capital.",
            "Don't risk more than 1-2% of your capital on a single trade.",
            "Look for signals that align with the overall market trend.",
            "Avoid trading during major economic news releases to prevent high volatility.",
            "Use a risk/reward ratio of at least 1:1.5 for your trades.",
            "Avoid overtrading; quality of trades is more important than quantity.",
            "Check multiple timeframes before making a trading decision.",
            "Patience is one of the most important qualities of a successful trader.",
            "Keep a trading journal to analyze and improve your performance.",
            "Don't chase losses with additional trades; stick to your trading plan.",
            "Signals with confidence above 80% are the most reliable.",
            "Avoid trading when you're tired or under psychological pressure.",
            "Remember that scalping requires high focus and continuous monitoring.",
            "Diversify the financial instruments you trade to reduce risk.",
            "Learn from your mistakes and continuously improve your strategy."
        ]
    },
    'ar': {
        'title': "منصة إشارات التداول الاحترافية",
        'subtitle': "تحليل فني متقدم للسكالبينج",
        'settings': "الإعدادات",
        'select_pair': "اختر زوج التداول:",
        'timeframe': "الإطار الزمني:",
        'advanced_settings': "إعدادات متقدمة",
        'moving_averages': "المتوسطات المتحركة",
        'oscillators': "المذبذبات",
        'macd_settings': "إعدادات MACD",
        'bollinger_settings': "بولينجر باند",
        'stochastic_settings': "ستوكاستيك",
        'other_settings': "إعدادات أخرى",
        'confidence_threshold': "حد الثقة للإشارة (%)",
        'risk_reward': "نسبة المخاطرة/المكافأة",
        'display_options': "خيارات العرض",
        'show_charts': "عرض الرسوم البيانية",
        'save_history': "حفظ سجل الإشارات",
        'show_tips': "عرض نصائح التداول",
        'update_button': "تحديث الآن",
        'trading_tip': "نصيحة تداول",
        'dashboard': "لوحة المعلومات",
        'charts': "الرسوم البيانية",
        'history': "سجل الإشارات",
        'instructions': "التعليمات",
        'signal': "الإشارة",
        'confidence': "الثقة",
        'trend': "الاتجاه",
        'current_price': "السعر الحالي",
        'take_profit': "هدف الربح",
        'stop_loss': "وقف الخسارة",
        'signal_details': "تفاصيل الإشارات",
        'support_resistance': "الدعم والمقاومة",
        'resistance': "المقاومة",
        'support': "الدعم",
        'technical_indicators': "المؤشرات الفنية",
        'no_signals': "لا توجد إشارات كافية لاتخاذ قرار.",
        'no_levels': "لا توجد مستويات محددة.",
        'not_available': "غير متوفر",
        'rsi': "RSI",
        'overbought': "ذروة شراء",
        'oversold': "ذروة بيع",
        'neutral': "محايد",
        'signal_line': "خط الإشارة",
        'difference': "الفرق",
        'status': "الحالة",
        'positive': "إيجابي",
        'negative': "سلبي",
        'stochastic': "ستوكاستيك",
        'charts_disabled': "الرسوم البيانية معطلة. يمكنك تفعيلها من الإعدادات المتقدمة.",
        'history_empty': "سجل الإشارات فارغ أو معطل. يمكنك تفعيل حفظ السجل من الإعدادات المتقدمة.",
        'clear_history': "مسح السجل",
        'history_cleared': "تم مسح سجل الإشارات بنجاح.",
        'how_to_use': "كيفية الاستخدام",
        'important_notice': "تنبيه مهم",
        'last_update': "آخر تحديث",
        'buy': "شراء",
        'sell': "بيع",
        'no_signal': "لا توجد إشارة",
        'error': "حدث خطأ",
        'strong_uptrend': "صاعد قوي",
        'uptrend_resistance': "صاعد مع مقاومة",
        'downtrend_support': "هابط مع دعم",
        'strong_downtrend': "هابط قوي",
        'ranging': "متذبذب",
        'unknown': "غير معروف",
        'language_option': "اللغة",
        'theme_option': "المظهر",
        'light': "فاتح",
        'dark': "داكن",
        'blue': "أزرق",
        'green': "أخضر",
        'strategy': "الاستراتيجية",
        'scalping': "سكالبينج",
        'day_trading': "تداول يومي",
        'swing': "سوينج",
        'position': "مركز",
        'signal_strength': "قوة الإشارة",
        'weak': "ضعيفة",
        'moderate': "متوسطة",
        'strong': "قوية",
        'very_strong': "قوية جداً",
        'enable_alerts': "تفعيل التنبيهات الصوتية",
        'alert_volume': "مستوى صوت التنبيه",
        'data_loading': "جاري تحميل البيانات...",
        'data_analyzing': "جاري تحليل البيانات...",
        'no_data': "لا يمكن جلب بيانات الزوج المحدد. تأكد من اتصالك بالإنترنت أو جرب زوجاً آخر.",
        'instructions_steps': [
            "اختر زوج التداول من القائمة المنسدلة في الشريط الجانبي.",
            "حدد الإطار الزمني المناسب لتحليلك.",
            "خصص المؤشرات الفنية في قسم الإعدادات المتقدمة إذا لزم الأمر.",
            "راقب لوحة المعلومات للحصول على الإشارات الحالية ومستوى الثقة والاتجاه.",
            "استخدم مستويات هدف الربح ووقف الخسارة المقترحة لإدارة المخاطر في صفقاتك.",
            "تحقق من تفاصيل الإشارة لفهم أسباب التوصية الحالية.",
            "راجع الرسم البياني للحصول على تحليل بصري للسعر والمؤشرات الفنية.",
            "اضغط على زر 'تحديث الآن' في الشريط الجانبي للحصول على أحدث البيانات والتحليلات.",
            "فعّل التنبيهات الصوتية للحصول على إشعار عند ظهور إشارات قوية."
        ],
        'important_notices': [
            "هذا التطبيق مخصص لأغراض تعليمية وإرشادية فقط وليس توصية للتداول.",
            "يجب عليك دائماً إجراء تحليلك الخاص واستشارة مستشار مالي قبل اتخاذ أي قرارات استثمارية.",
            "التداول ينطوي على مخاطر، ويمكن أن تخسر أكثر من استثمارك الأولي.",
            "الأداء السابق ليس مؤشراً على النتائج المستقبلية."
        ],
        'trading_tips': [
            "تأكد دائماً من وضع وقف الخسارة لحماية رأس المال.",
            "لا تخاطر بأكثر من 1-2% من رأس المال في الصفقة الواحدة.",
            "ابحث عن الإشارات التي تتوافق مع الاتجاه العام للسوق.",
            "تجنب التداول خلال الأخبار الاقتصادية المهمة لتجنب التقلبات الشديدة.",
            "استخدم نسبة مخاطرة/مكافأة لا تقل عن 1:1.5 للصفقات.",
            "تجنب المبالغة في التداول، جودة الصفقات أهم من كميتها.",
            "تحقق من عدة أطر زمنية قبل اتخاذ قرار التداول.",
            "الصبر من أهم صفات المتداول الناجح، انتظر الإشارات القوية.",
            "حافظ على سجل تداولاتك لتحليل أدائك وتحسينه باستمرار.",
            "لا تطارد الخسائر بصفقات إضافية، التزم بخطة التداول.",
            "الإشارات ذات نسبة الثقة فوق 80% هي الأكثر موثوقية.",
            "تجنب التداول عندما تكون متعباً أو تحت ضغط نفسي.",
            "تذكر أن السكالبينج يتطلب تركيزاً عالياً ومتابعة مستمرة.",
            "قم بتنويع الأدوات المالية التي تتداول عليها لتقليل المخاطر.",
            "تعلم من أخطائك وطور استراتيجيتك باستمرار."
        ]
    }
}

# Available trading pairs
symbols = {
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
DEFAULT_SETTINGS = {
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
    'show_charts': True,
    'save_history': True,
    'show_tips': True,
    'max_history': 50,
    'enable_alerts': True,
    'alert_volume': 0.7,
    'strategy': 'scalping'
}

# Theme colors
THEMES = {
    'light': {
        'bg_color': '#ffffff',
        'text_color': '#333333',
        'primary_color': '#1E88E5',
        'secondary_color': '#26A69A',
        'accent_color': '#FF5722',
        'panel_bg': '#f5f5f5',
        'buy_color': '#4CAF50',
        'sell_color': '#F44336',
        'neutral_color': '#9E9E9E',
        'chart_bg': '#f9f9f9',
        'grid_color': '#dddddd'
    },
    'dark': {
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
    },
    'blue': {
        'bg_color': '#0a192f',
        'text_color': '#e6f1ff',
        'primary_color': '#64ffda',
        'secondary_color': '#8892b0',
        'accent_color': '#ff5555',
        'panel_bg': '#112240',
        'buy_color': '#64ffda',
        'sell_color': '#ff5555',
        'neutral_color': '#8892b0',
        'chart_bg': '#112240',
        'grid_color': '#1d3b66'
    },
    'green': {
        'bg_color': '#0f2027',
        'text_color': '#e0e0e0',
        'primary_color': '#00b894',
        'secondary_color': '#55efc4',
        'accent_color': '#ff7675',
        'panel_bg': '#203a43',
        'buy_color': '#00b894',
        'sell_color': '#ff7675',
        'neutral_color': '#a0a0a0',
        'chart_bg': '#203a43',
        'grid_color': '#2c5364'
    }
}

# Initialize session state for theme
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

# Initialize session state for strategy
if 'strategy' not in st.session_state:
    st.session_state.strategy = 'scalping'

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
    if len(signal_history) > DEFAULT_SETTINGS['max_history']:
        signal_history = signal_history[:DEFAULT_SETTINGS['max_history']]

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
def analyze_price_action(df, settings=DEFAULT_SETTINGS, strategy='scalping'):
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

# Create technical chart
def create_technical_chart(df, symbol_name, theme):
    try:
        # Set chart style based on theme
        plt.style.use('dark_background' if theme != 'light' else 'default')
        
        # Create figure and subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Set background colors
        fig.patch.set_facecolor(THEMES[theme]['chart_bg'])
        ax1.set_facecolor(THEMES[theme]['chart_bg'])
        ax2.set_facecolor(THEMES[theme]['chart_bg'])
        ax3.set_facecolor(THEMES[theme]['chart_bg'])
        
        # Plot price and moving averages
        ax1.plot(df.index, df['Close'], label='Price', linewidth=2, color='white')

        if 'EMA_fast' in df.columns and not df['EMA_fast'].isnull().all():
            ax1.plot(df.index, df['EMA_fast'], label=f'EMA {DEFAULT_SETTINGS["ema_fast"]}', alpha=0.7, color=THEMES[theme]['primary_color'])

        if 'EMA_medium' in df.columns and not df['EMA_medium'].isnull().all():
            ax1.plot(df.index, df['EMA_medium'], label=f'EMA {DEFAULT_SETTINGS["ema_medium"]}', alpha=0.7, color=THEMES[theme]['secondary_color'])

        if 'EMA_slow' in df.columns and not df['EMA_slow'].isnull().all():
            ax1.plot(df.index, df['EMA_slow'], label=f'EMA {DEFAULT_SETTINGS["ema_slow"]}', alpha=0.7, color=THEMES[theme]['accent_color'])

        if 'BB_upper' in df.columns and not df['BB_upper'].isnull().all():
            ax1.plot(df.index, df['BB_upper'], '--', label='Upper BB', alpha=0.5, color=THEMES[theme]['sell_color'])

        if 'BB_middle' in df.columns and not df['BB_middle'].isnull().all():
            ax1.plot(df.index, df['BB_middle'], '--', label='Middle BB', alpha=0.5, color=THEMES[theme]['neutral_color'])

        if 'BB_lower' in df.columns and not df['BB_lower'].isnull().all():
            ax1.plot(df.index, df['BB_lower'], '--', label='Lower BB', alpha=0.5, color=THEMES[theme]['buy_color'])

        ax1.set_title(f'Technical Analysis for {symbol_name}', color=THEMES[theme]['text_color'])
        ax1.set_ylabel('Price', color=THEMES[theme]['text_color'])
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3, color=THEMES[theme]['grid_color'])
        ax1.tick_params(colors=THEMES[theme]['text_color'])

        # Plot MACD
        if 'MACD' in df.columns and not df['MACD'].isnull().all():
            ax2.plot(df.index, df['MACD'], label='MACD', color=THEMES[theme]['primary_color'])

        if 'MACD_signal' in df.columns and not df['MACD_signal'].isnull().all():
            ax2.plot(df.index, df['MACD_signal'], label='Signal Line', color=THEMES[theme]['secondary_color'])

        if 'MACD_histogram' in df.columns and not df['MACD_histogram'].isnull().all():
            # Color histogram bars based on value
            for i in range(len(df)):
                if i < len(df) - 1:  # Ensure we don't go out of bounds
                    if df['MACD_histogram'].iloc[i] >= 0:
                        ax2.bar(df.index[i], df['MACD_histogram'].iloc[i], color=THEMES[theme]['buy_color'], alpha=0.5, width=0.7)
                    else:
                        ax2.bar(df.index[i], df['MACD_histogram'].iloc[i], color=THEMES[theme]['sell_color'], alpha=0.5, width=0.7)

        ax2.axhline(y=0, color=THEMES[theme]['text_color'], linestyle='-', alpha=0.3)
        ax2.set_ylabel('MACD', color=THEMES[theme]['text_color'])
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3, color=THEMES[theme]['grid_color'])
        ax2.tick_params(colors=THEMES[theme]['text_color'])

        # Plot RSI and Stochastic
        if 'RSI' in df.columns and not df['RSI'].isnull().all():
            ax3.plot(df.index, df['RSI'], label='RSI', color=THEMES[theme]['primary_color'])

        if 'Stoch_K' in df.columns and not df['Stoch_K'].isnull().all():
            ax3.plot(df.index, df['Stoch_K'], label='Stochastic %K', color=THEMES[theme]['secondary_color'], alpha=0.7)

        if 'Stoch_D' in df.columns and not df['Stoch_D'].isnull().all():
            ax3.plot(df.index, df['Stoch_D'], label='Stochastic %D', color=THEMES[theme]['accent_color'], alpha=0.7)

        ax3.axhline(y=DEFAULT_SETTINGS['rsi_overbought'], color=THEMES[theme]['sell_color'], linestyle='--', alpha=0.5)
        ax3.axhline(y=DEFAULT_SETTINGS['rsi_oversold'], color=THEMES[theme]['buy_color'], linestyle='--', alpha=0.5)
        ax3.axhline(y=50, color=THEMES[theme]['text_color'], linestyle='-', alpha=0.3)
        ax3.set_ylim([0, 100])
        ax3.set_ylabel('Oscillators', color=THEMES[theme]['text_color'])
        ax3.set_xlabel('Time', color=THEMES[theme]['text_color'])
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3, color=THEMES[theme]['grid_color'])
        ax3.tick_params(colors=THEMES[theme]['text_color'])

        plt.tight_layout()
        
        return fig
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        return None

# Get random trading tip
def get_random_trading_tip(language='en'):
    return random.choice(lang[language]['trading_tips'])

# Format signal with colors
def format_signal(signal, language='en'):
    if signal == "buy" or signal == "شراء":
        return f"<span style='color:#4CAF50;font-weight:bold;font-size:1.5em;'>{lang[language]['buy']}</span>"
    elif signal == "sell" or signal == "بيع":
        return f"<span style='color:#F44336;font-weight:bold;font-size:1.5em;'>{lang[language]['sell']}</span>"
    else:
        return f"<span style='color:#9E9E9E;font-size:1.5em;'>{lang[language]['no_signal']}</span>"

# Format confidence with colors
def format_confidence(confidence):
    if confidence >= 80:
        return f"<span style='color:#4CAF50;font-weight:bold;font-size:1.5em;'>{confidence:.1f}%</span>"
    elif confidence >= 60:
        return f"<span style='color:#FFC107;font-weight:bold;font-size:1.5em;'>{confidence:.1f}%</span>"
    else:
        return f"<span style='color:#F44336;font-weight:bold;font-size:1.5em;'>{confidence:.1f}%</span>"

# Format trend with colors
def format_trend(trend, language='en'):
    if trend == "strong_uptrend":
        return f"<span style='color:#4CAF50;font-weight:bold;font-size:1.5em;'>{lang[language]['strong_uptrend']}</span>"
    elif trend == "uptrend_resistance":
        return f"<span style='color:#8BC34A;font-weight:bold;font-size:1.5em;'>{lang[language]['uptrend_resistance']}</span>"
    elif trend == "downtrend_support":
        return f"<span style='color:#FF9800;font-weight:bold;font-size:1.5em;'>{lang[language]['downtrend_support']}</span>"
    elif trend == "strong_downtrend":
        return f"<span style='color:#F44336;font-weight:bold;font-size:1.5em;'>{lang[language]['strong_downtrend']}</span>"
    elif trend == "ranging":
        return f"<span style='color:#9E9E9E;font-weight:bold;font-size:1.5em;'>{lang[language]['ranging']}</span>"
    else:
        return f"<span style='color:#9E9E9E;font-size:1.5em;'>{lang[language]['unknown']}</span>"

# Apply custom CSS based on theme
def apply_custom_css(theme):
    colors = THEMES[theme]
    
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
    .dashboard-container {{
        background-color: {colors['panel_bg']};
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }}
    .metric-card {{
        background-color: {colors['bg_color']};
        padding: 15px;
        border-radius: 5px;
        text-align: center;
        height: 100%;
        border: 1px solid {colors['grid_color']};
    }}
    .metric-title {{
        font-size: 1.1em;
        margin-bottom: 10px;
        color: {colors['text_color']};
        opacity: 0.8;
    }}
    .metric-value {{
        font-size: 1.8em;
        font-weight: bold;
    }}
    .indicator-container {{
        display: flex;
        flex-direction: column;
        background-color: {colors['bg_color']};
        padding: 15px;
        border-radius: 5px;
        height: 100%;
        border: 1px solid {colors['grid_color']};
    }}
    .indicator-title {{
        font-size: 1.1em;
        margin-bottom: 10px;
        color: {colors['text_color']};
        opacity: 0.8;
    }}
    .indicator-value {{
        font-size: 1.4em;
        font-weight: bold;
        margin-bottom: 5px;
    }}
    .indicator-details {{
        font-size: 0.9em;
        color: {colors['text_color']};
        opacity: 0.8;
    }}
    .chart-container {{
        background-color: {colors['panel_bg']};
        padding: 15px;
        border-radius: 5px;
        margin-top: 20px;
    }}
    .history-container {{
        background-color: {colors['panel_bg']};
        padding: 15px;
        border-radius: 5px;
        margin-top: 20px;
    }}
    .instructions-container {{
        background-color: {colors['panel_bg']};
        padding: 20px;
        border-radius: 5px;
        margin-top: 20px;
        border-left: 4px solid {colors['primary_color']};
    }}
    .instructions-title {{
        font-size: 1.3em;
        font-weight: bold;
        margin-bottom: 15px;
        color: {colors['primary_color']};
    }}
    .instructions-step {{
        margin-bottom: 10px;
        padding-left: 15px;
        position: relative;
    }}
    .instructions-step:before {{
        content: "•";
        color: {colors['primary_color']};
        font-weight: bold;
        position: absolute;
        left: 0;
    }}
    .tip-container {{
        background-color: {colors['panel_bg']};
        padding: 15px;
        border-radius: 5px;
        margin: 20px 0;
        border-left: 4px solid {colors['primary_color']};
    }}
    .tip-title {{
        font-weight: bold;
        color: {colors['primary_color']};
        margin-bottom: 10px;
    }}
    .tip-content {{
        line-height: 1.6;
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
    div[data-testid="stVerticalBlock"] div[style*="flex-direction: column;"] div[data-testid="stVerticalBlock"] {{
        background-color: {colors['panel_bg']};
        padding: 10px;
        border-radius: 5px;
    }}
    div[role="radiogroup"] label {{
        background-color: {colors['bg_color']};
        border: 1px solid {colors['grid_color']};
    }}
    div[role="radiogroup"] label[data-baseweb="radio"] [data-testid="stMarkdownContainer"] p {{
        color: {colors['text_color']};
    }}
    .stAlert {{
        background-color: {colors['panel_bg']};
        color: {colors['text_color']};
    }}
    .stAlert [data-testid="stMarkdownContainer"] {{
        color: {colors['text_color']};
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
    .stSlider [data-baseweb="slider"] {{
        background-color: {colors['grid_color']};
    }}
    .stSlider [data-baseweb="slider"] [data-testid="stThumbValue"] {{
        background-color: {colors['primary_color']};
        color: {colors['bg_color']};
    }}
    .stSlider [data-baseweb="slider"] [data-testid="stThumbValue"] [data-testid="stMarkdownContainer"] p {{
        color: {colors['bg_color']};
    }}
    .stCheckbox label p {{
        color: {colors['text_color']};
    }}
    .stSelectbox [data-baseweb="select"] {{
        background-color: {colors['bg_color']};
        color: {colors['text_color']};
        border-color: {colors['grid_color']};
    }}
    .stSelectbox [data-baseweb="select"] [data-testid="stMarkdownContainer"] p {{
        color: {colors['text_color']};
    }}
    @media (max-width: 768px) {{
        .metric-card {{
            margin-bottom: 10px;
        }}
        .indicator-container {{
            margin-bottom: 10px;
        }}
    }}
    </style>
    """, unsafe_allow_html=True)

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

# Display dashboard
def display_dashboard(analysis, selected_symbol_name, language='en'):
    st.markdown("<div class='dashboard-container'>", unsafe_allow_html=True)
    
    # Main indicators row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>{lang[language]['signal']}</div>
            <div class='metric-value'>{format_signal(analysis['signal'], language)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>{lang[language]['confidence']}</div>
            <div class='metric-value'>{format_confidence(analysis['confidence'])}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>{lang[language]['trend']}</div>
            <div class='metric-value'>{format_trend(analysis['trend'], language)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Price, TP, SL row
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        price_value = f"{analysis['last_close']:.5f}" if analysis['last_close'] is not None else lang[language]['not_available']
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>{lang[language]['current_price']}</div>
            <div class='metric-value'>{price_value}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        tp_value = f"<span style='color:#4CAF50;'>{analysis['tp']:.5f}</span>" if analysis['tp'] is not None else lang[language]['not_available']
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>{lang[language]['take_profit']}</div>
            <div class='metric-value'>{tp_value}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        sl_value = f"<span style='color:#F44336;'>{analysis['sl']:.5f}</span>" if analysis['sl'] is not None else lang[language]['not_available']
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>{lang[language]['stop_loss']}</div>
            <div class='metric-value'>{sl_value}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Display technical indicators
def display_technical_indicators(analysis, language='en'):
    st.markdown(f"<h3>{lang[language]['technical_indicators']}</h3>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div class='indicator-container'>", unsafe_allow_html=True)
        st.markdown(f"<div class='indicator-title'>{lang[language]['rsi']}</div>", unsafe_allow_html=True)
        
        if analysis['rsi'] is not None:
            if analysis['rsi'] > DEFAULT_SETTINGS['rsi_overbought']:
                rsi_color = "#F44336"
                rsi_status = lang[language]['overbought']
            elif analysis['rsi'] < DEFAULT_SETTINGS['rsi_oversold']:
                rsi_color = "#4CAF50"
                rsi_status = lang[language]['oversold']
            else:
                rsi_color = "#FFC107"
                rsi_status = lang[language]['neutral']
            
            st.markdown(f"""
            <div class='indicator-value' style='color:{rsi_color};'>{analysis['rsi']:.2f}</div>
            <div class='indicator-details'>{rsi_status}</div>
            <div class='indicator-details'>{lang[language]['overbought']}: {DEFAULT_SETTINGS['rsi_overbought']}</div>
            <div class='indicator-details'>{lang[language]['oversold']}: {DEFAULT_SETTINGS['rsi_oversold']}</div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='indicator-value'>{lang[language]['not_available']}</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='indicator-container'>", unsafe_allow_html=True)
        st.markdown("<div class='indicator-title'>MACD</div>", unsafe_allow_html=True)
        
        if analysis['macd'] is not None and analysis['macd_signal'] is not None:
            macd_diff = analysis['macd'] - analysis['macd_signal']
            macd_color = "#4CAF50" if macd_diff > 0 else "#F44336"
            macd_status = lang[language]['positive'] if macd_diff > 0 else lang[language]['negative']
            
            st.markdown(f"""
            <div class='indicator-value' style='color:{macd_color};'>{analysis['macd']:.5f}</div>
            <div class='indicator-details'>{lang[language]['signal_line']}: {analysis['macd_signal']:.5f}</div>
            <div class='indicator-details'>{lang[language]['difference']}: {macd_diff:.5f}</div>
            <div class='indicator-details'>{lang[language]['status']}: {macd_status}</div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='indicator-value'>{lang[language]['not_available']}</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='indicator-container'>", unsafe_allow_html=True)
        st.markdown(f"<div class='indicator-title'>{lang[language]['stochastic']}</div>", unsafe_allow_html=True)
        
        if analysis['stoch_k'] is not None and analysis['stoch_d'] is not None:
            if analysis['stoch_k'] > DEFAULT_SETTINGS['stoch_overbought'] and analysis['stoch_d'] > DEFAULT_SETTINGS['stoch_overbought']:
                stoch_color = "#F44336"
                stoch_status = lang[language]['overbought']
            elif analysis['stoch_k'] < DEFAULT_SETTINGS['stoch_oversold'] and analysis['stoch_d'] < DEFAULT_SETTINGS['stoch_oversold']:
                stoch_color = "#4CAF50"
                stoch_status = lang[language]['oversold']
            else:
                stoch_color = "#FFC107"
                stoch_status = lang[language]['neutral']
            
            st.markdown(f"""
            <div class='indicator-value' style='color:{stoch_color};'>%K: {analysis['stoch_k']:.2f}, %D: {analysis['stoch_d']:.2f}</div>
            <div class='indicator-details'>{stoch_status}</div>
            <div class='indicator-details'>{lang[language]['overbought']}: {DEFAULT_SETTINGS['stoch_overbought']}</div>
            <div class='indicator-details'>{lang[language]['oversold']}: {DEFAULT_SETTINGS['stoch_oversold']}</div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='indicator-value'>{lang[language]['not_available']}</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

# Display signal details and support/resistance levels
def display_signal_details(analysis, language='en'):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"<h3>{lang[language]['signal_details']}</h3>", unsafe_allow_html=True)
        
        if analysis['signals_details']:
            for detail in analysis['signals_details']:
                st.markdown(f"• {detail}")
        else:
            st.markdown(lang[language]['no_signals'])
    
    with col2:
        st.markdown(f"<h3>{lang[language]['support_resistance']}</h3>", unsafe_allow_html=True)
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown(f"<h4>{lang[language]['resistance']}</h4>", unsafe_allow_html=True)
            if analysis['support_resistance']['resistance']:
                for level in sorted(analysis['support_resistance']['resistance'], reverse=True):
                    st.markdown(f"• {level:.5f}")
            else:
                st.markdown(lang[language]['no_levels'])
        
        with col_b:
            st.markdown(f"<h4>{lang[language]['support']}</h4>", unsafe_allow_html=True)
            if analysis['support_resistance']['support']:
                for level in sorted(analysis['support_resistance']['support'], reverse=True):
                    st.markdown(f"• {level:.5f}")
            else:
                st.markdown(lang[language]['no_levels'])

# Display instructions
def display_instructions(language='en'):
    st.markdown("<div class='instructions-container'>", unsafe_allow_html=True)
    st.markdown(f"<div class='instructions-title'>📋 {lang[language]['how_to_use']}</div>", unsafe_allow_html=True)
    
    for step in lang[language]['instructions_steps']:
        st.markdown(f"<div class='instructions-step'>{step}</div>", unsafe_allow_html=True)
    
    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='instructions-title'>⚠️ {lang[language]['important_notice']}</div>", unsafe_allow_html=True)
    
    for notice in lang[language]['important_notices']:
        st.markdown(f"<div class='instructions-step'>{notice}</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Main function
def main():
    # Get current language and theme
    current_language = st.session_state.language
    current_theme = st.session_state.theme
    
    # Apply custom CSS based on theme
    apply_custom_css(current_theme)
    
    # Add alert sound JavaScript
    st.markdown(create_alert_sound(), unsafe_allow_html=True)
    
    # App title
    st.title(lang[current_language]['title'])
    st.markdown(f"### {lang[current_language]['subtitle']}")

    # Sidebar settings
    with st.sidebar:
        st.header(f"⚙️ {lang[current_language]['settings']}")
        
        # Language and theme selection
        col1, col2 = st.columns(2)
        with col1:
            selected_language = st.selectbox(
                lang[current_language]['language_option'],
                options=['en', 'ar'],
                format_func=lambda x: "English" if x == "en" else "العربية",
                index=0 if current_language == 'en' else 1
            )
            
            if selected_language != current_language:
                st.session_state.language = selected_language
                st.experimental_rerun()
        
        with col2:
            selected_theme = st.selectbox(
                lang[current_language]['theme_option'],
                options=['dark', 'light', 'blue', 'green'],
                format_func=lambda x: lang[current_language][x],
                index=list(THEMES.keys()).index(current_theme)
            )
            
            if selected_theme != current_theme:
                st.session_state.theme = selected_theme
                st.experimental_rerun()
        
        # Trading pair selection
        selected_symbol_name = st.selectbox(
            lang[current_language]['select_pair'],
            list(symbols.keys())
        )
        selected_symbol = symbols[selected_symbol_name]
        
        # Timeframe selection
        timeframe = st.selectbox(
            lang[current_language]['timeframe'],
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
        
        # Strategy selection
        selected_strategy = st.selectbox(
            lang[current_language]['strategy'],
            ["scalping", "day_trading", "swing", "position"],
            format_func=lambda x: lang[current_language][x],
            index=["scalping", "day_trading", "swing", "position"].index(st.session_state.strategy)
        )
        
        if selected_strategy != st.session_state.strategy:
            st.session_state.strategy = selected_strategy
        
        # Alert settings
        enable_alerts = st.checkbox(
            lang[current_language]['enable_alerts'],
            value=st.session_state.enable_alerts
        )
        
        if enable_alerts != st.session_state.enable_alerts:
            st.session_state.enable_alerts = enable_alerts
        
        if enable_alerts:
            alert_volume = st.slider(
                lang[current_language]['alert_volume'],
                0.1, 1.0, st.session_state.alert_volume, 0.1
            )
            
            if alert_volume != st.session_state.alert_volume:
                st.session_state.alert_volume = alert_volume
        
        # Advanced settings
        with st.expander(f"🔧 {lang[current_language]['advanced_settings']}"):
            # Moving Averages settings
            st.subheader(lang[current_language]['moving_averages'])
            sma_fast = st.slider("SMA Fast", 5, 50, DEFAULT_SETTINGS['sma_fast'])
            sma_slow = st.slider("SMA Slow", 10, 200, DEFAULT_SETTINGS['sma_slow'])
            ema_fast = st.slider("EMA Fast", 5, 50, DEFAULT_SETTINGS['ema_fast'])
            ema_medium = st.slider("EMA Medium", 10, 100, DEFAULT_SETTINGS['ema_medium'])
            ema_slow = st.slider("EMA Slow", 20, 200, DEFAULT_SETTINGS['ema_slow'])
            
            # Oscillators settings
            st.subheader(lang[current_language]['oscillators'])
            rsi_period = st.slider("RSI Period", 5, 30, DEFAULT_SETTINGS['rsi_period'])
            rsi_overbought = st.slider("RSI Overbought", 60, 90, DEFAULT_SETTINGS['rsi_overbought'])
            rsi_oversold = st.slider("RSI Oversold", 10, 40, DEFAULT_SETTINGS['rsi_oversold'])
            
            # MACD settings
            st.subheader(lang[current_language]['macd_settings'])
            macd_fast = st.slider("MACD Fast", 5, 20, DEFAULT_SETTINGS['macd_fast'])
            macd_slow = st.slider("MACD Slow", 15, 40, DEFAULT_SETTINGS['macd_slow'])
            macd_signal = st.slider("MACD Signal", 5, 15, DEFAULT_SETTINGS['macd_signal'])
            
            # Bollinger Bands settings
            st.subheader(lang[current_language]['bollinger_settings'])
            bollinger_period = st.slider("Bollinger Period", 10, 50, DEFAULT_SETTINGS['bollinger_period'])
            bollinger_std = st.slider("Standard Deviation", 1.0, 3.0, float(DEFAULT_SETTINGS['bollinger_std']), 0.1)
            
            # Stochastic settings
            st.subheader(lang[current_language]['stochastic_settings'])
            stoch_k = st.slider("Stochastic %K", 5, 30, DEFAULT_SETTINGS['stoch_k'])
            stoch_d = st.slider("Stochastic %D", 1, 10, DEFAULT_SETTINGS['stoch_d'])
            stoch_overbought = st.slider("Stochastic Overbought", 60, 90, DEFAULT_SETTINGS['stoch_overbought'])
            stoch_oversold = st.slider("Stochastic Oversold", 10, 40, DEFAULT_SETTINGS['stoch_oversold'])
            
            # Other settings
            st.subheader(lang[current_language]['other_settings'])
            confidence_threshold = st.slider(lang[current_language]['confidence_threshold'], 50, 90, DEFAULT_SETTINGS['confidence_threshold'])
            risk_reward_ratio = st.slider(lang[current_language]['risk_reward'], 1.0, 3.0, float(DEFAULT_SETTINGS['risk_reward_ratio']), 0.1)
            
            # Display options
            st.subheader(lang[current_language]['display_options'])
            show_charts = st.checkbox(lang[current_language]['show_charts'], DEFAULT_SETTINGS['show_charts'])
            save_history = st.checkbox(lang[current_language]['save_history'], DEFAULT_SETTINGS['save_history'])
            show_tips = st.checkbox(lang[current_language]['show_tips'], DEFAULT_SETTINGS['show_tips'])
        
        # Update settings
        settings = {
            'sma_fast': sma_fast,
            'sma_slow': sma_slow,
            'ema_fast': ema_fast,
            'ema_medium': ema_medium,
            'ema_slow': ema_slow,
            'rsi_period': rsi_period,
            'rsi_overbought': rsi_overbought,
            'rsi_oversold': rsi_oversold,
            'macd_fast': macd_fast,
            'macd_slow': macd_slow,
            'macd_signal': macd_signal,
            'bollinger_period': bollinger_period,
            'bollinger_std': bollinger_std,
            'atr_period': DEFAULT_SETTINGS['atr_period'],
            'stoch_k': stoch_k,
            'stoch_d': stoch_d,
            'stoch_overbought': stoch_overbought,
            'stoch_oversold': stoch_oversold,
            'volume_sma': DEFAULT_SETTINGS['volume_sma'],
            'risk_reward_ratio': risk_reward_ratio,
            'confidence_threshold': confidence_threshold,
            'show_charts': show_charts,
            'save_history': save_history,
            'show_tips': show_tips,
            'max_history': DEFAULT_SETTINGS['max_history']
        }
        
        # Update button
        if st.button(lang[current_language]['update_button']):
            st.success("✅ Updated successfully!")
    
    # Show trading tip
    if settings['show_tips']:
        tip = get_random_trading_tip(current_language)
        st.markdown(f"""
        <div class="tip-container">
            <div class="tip-title">💡 {lang[current_language]['trading_tip']}</div>
            <div class="tip-content">{tip}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Create tabs
    tabs = st.tabs([
        f"📊 {lang[current_language]['dashboard']}", 
        f"📈 {lang[current_language]['charts']}", 
        f"📋 {lang[current_language]['history']}", 
        f"ℹ️ {lang[current_language]['instructions']}"
    ])
    
    # Load signal history
    if settings['save_history']:
        load_signal_history()
    
    # Fetch and analyze data
    with st.spinner(lang[current_language]['data_loading']):
        df = fetch_data(selected_symbol, period=period, interval=timeframe)
        
        if not df.empty:
            # Analyze data
            with st.spinner(lang[current_language]['data_analyzing']):
                analysis = analyze_price_action(df, settings, st.session_state.strategy)
                
                # Dashboard tab
                with tabs[0]:
                    # Display dashboard
                    display_dashboard(analysis, selected_symbol_name, current_language)
                    
                    # Display technical indicators
                    display_technical_indicators(analysis, current_language)
                    
                    # Display signal details and support/resistance levels
                    display_signal_details(analysis, current_language)
                    
                    # Check if we need to trigger an alert
                    if st.session_state.enable_alerts and analysis['signal'] != "no_signal" and analysis['signal'] != "error":
                        # Check if this is a new signal
                        symbol_key = f"{selected_symbol_name}_{timeframe}"
                        if symbol_key not in previous_signals or previous_signals[symbol_key] != analysis['signal']:
                            previous_signals[symbol_key] = analysis['signal']
                            st.markdown(trigger_alert(st.session_state.alert_volume), unsafe_allow_html=True)
                            
                            # Add to history if enabled
                            if settings['save_history']:
                                add_to_history(
                                    selected_symbol_name,
                                    analysis['signal'],
                                    analysis['last_close'],
                                    analysis['confidence']
                                )
                
                # Charts tab
                with tabs[1]:
                    if settings['show_charts']:
                        st.markdown(f"<h3>{lang[current_language]['charts']}</h3>", unsafe_allow_html=True)
                        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                        fig = create_technical_chart(df, selected_symbol_name, current_theme)
                        if fig:
                            st.pyplot(fig)
                        st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.info(lang[current_language]['charts_disabled'])
                
                # History tab
                with tabs[2]:
                    st.markdown(f"<h3>{lang[current_language]['history']}</h3>", unsafe_allow_html=True)
                    st.markdown("<div class='history-container'>", unsafe_allow_html=True)
                    
                    if settings['save_history'] and signal_history:
                        # Create DataFrame from signal history
                        history_df = pd.DataFrame(signal_history)
                        
                        # Format display
                        history_df.columns = ["Symbol", "Signal", "Price", "Confidence", "Timestamp"]
                        
                        # Display table
                        st.dataframe(history_df, use_container_width=True)
                        
                        # Clear history button
                        if st.button(lang[current_language]['clear_history']):
                            signal_history.clear()
                            save_signal_history()
                            st.success(lang[current_language]['history_cleared'])
                    else:
                        st.info(lang[current_language]['history_empty'])
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Instructions tab
                with tabs[3]:
                    display_instructions(current_language)
        else:
            st.error(lang[current_language]['no_data'])
    
    # Add last update info
    st.markdown("---")
    st.markdown(f"<div style='text-align:center;color:#aaa;font-size:0.8em;'>{lang[current_language]['last_update']}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>", unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
