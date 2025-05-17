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

# تعيين الخطوط لدعم اللغة العربية في matplotlib
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# إعداد الصفحة
st.set_page_config(
    page_title="مؤشرات التداول - سكالبينج",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# قائمة الأزواج والرموز المتاحة للتداول
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

# إعدادات افتراضية للمؤشرات الفنية
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
    'confidence_threshold': 70,  # نسبة الثقة المطلوبة للإشارة (%)
    'update_interval': 15,       # فترة التحديث بالثواني
    'show_charts': True,         # عرض الرسوم البيانية
    'save_history': True,        # حفظ سجل الإشارات
    'show_tips': True,           # عرض نصائح التداول
    'max_history': 50            # الحد الأقصى لعدد الإشارات المحفوظة في السجل
}

# تخزين الإشارات السابقة لتجنب تكرار التنبيهات
previous_signals = {}

# سجل الإشارات
signal_history = []

# مسار ملف حفظ سجل الإشارات
HISTORY_FILE = 'signal_history.json'

# نصائح التداول
trading_tips = [
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

# تحميل سجل الإشارات من الملف
def load_signal_history():
    global signal_history
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r') as f:
                signal_history = json.load(f)
    except Exception as e:
        st.error(f"خطأ في تحميل سجل الإشارات: {str(e)}")
        signal_history = []

# حفظ سجل الإشارات إلى الملف
def save_signal_history():
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(signal_history, f)
    except Exception as e:
        st.error(f"خطأ في حفظ سجل الإشارات: {str(e)}")

# إضافة إشارة جديدة إلى السجل
def add_to_history(symbol, signal_type, price, confidence, timestamp=None):
    global signal_history

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # إضافة الإشارة الجديدة في بداية القائمة
    signal_history.insert(0, {
        'symbol': symbol,
        'signal': signal_type,
        'price': price,
        'confidence': confidence,
        'timestamp': timestamp
    })

    # التأكد من عدم تجاوز الحد الأقصى للسجل
    if len(signal_history) > DEFAULT_SETTINGS['max_history']:
        signal_history = signal_history[:DEFAULT_SETTINGS['max_history']]

    # حفظ السجل
    save_signal_history()

# جلب البيانات من Yahoo Finance
def fetch_data(symbol, period='5d', interval='5m'):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            st.error(f"لا توجد بيانات للرمز {symbol}")
            return pd.DataFrame()
        return df
    except Exception as e:
        st.error(f"خطأ في جلب بيانات {symbol}: {str(e)}")
        return pd.DataFrame()  # إرجاع إطار بيانات فارغ في حالة الخطأ

# حساب المتوسط المتحرك البسيط
def sma(data, window):
    return data['Close'].rolling(window=window).mean()

# حساب المتوسط المتحرك الأسي
def ema(data, window):
    return data['Close'].ewm(span=window, adjust=False).mean()

# حساب مؤشر القوة النسبية
def rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# حساب مؤشر MACD
def macd(data, fast=12, slow=26, signal=9):
    ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

# حساب مؤشر بولينجر باند
def bollinger_bands(data, window=20, std=2):
    sma_line = data['Close'].rolling(window=window).mean()
    std_dev = data['Close'].rolling(window=window).std()
    upper_band = sma_line + (std_dev * std)
    lower_band = sma_line - (std_dev * std)
    return upper_band, sma_line, lower_band

# حساب مؤشر ستوكاستيك
def stochastic(data, k_period=14, d_period=3):
    lowest_low = data['Low'].rolling(window=k_period).min()
    highest_high = data['High'].rolling(window=k_period).max()
    k = 100 * ((data['Close'] - lowest_low) / (highest_high - lowest_low))
    d = k.rolling(window=d_period).mean()
    return k, d

# حساب مؤشر متوسط المدى الحقيقي (ATR)
def atr(data, period=14):
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(window=period).mean()

# تحليل نماذج الشموع اليابانية
def analyze_candlestick_patterns(df):
    # تحليل آخر شمعة فقط بدلاً من إنشاء أعمدة جديدة
    patterns = {}

    # حساب خصائص الشموع للشمعة الحالية والسابقة
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

    # حساب أحجام الجسم والاتجاه
    current_body_size = abs(df['Close'].iloc[current_idx] - df['Open'].iloc[current_idx])
    prev_body_size = abs(df['Close'].iloc[prev_idx] - df['Open'].iloc[prev_idx])

    current_body_direction = 1 if df['Close'].iloc[current_idx] > df['Open'].iloc[current_idx] else -1
    prev_body_direction = 1 if df['Close'].iloc[prev_idx] > df['Open'].iloc[prev_idx] else -1

    # حساب الظلال العلوية والسفلية
    current_upper_shadow = df['High'].iloc[current_idx] - max(df['Open'].iloc[current_idx], df['Close'].iloc[current_idx])
    current_lower_shadow = min(df['Open'].iloc[current_idx], df['Close'].iloc[current_idx]) - df['Low'].iloc[current_idx]

    # حساب نسبة الجسم إلى المدى الكلي
    current_range = df['High'].iloc[current_idx] - df['Low'].iloc[current_idx]
    if current_range > 0:
        body_to_range_ratio = current_body_size / current_range
    else:
        body_to_range_ratio = 0

    # نموذج البلع الصاعد (Bullish Engulfing)
    patterns['bullish_engulfing'] = (
        (current_body_direction == 1) and  # الشمعة الحالية صاعدة
        (prev_body_direction == -1) and  # الشمعة السابقة هابطة
        (df['Open'].iloc[current_idx] < df['Close'].iloc[prev_idx]) and  # فتح الشمعة الحالية أقل من إغلاق الشمعة السابقة
        (df['Close'].iloc[current_idx] > df['Open'].iloc[prev_idx])  # إغلاق الشمعة الحالية أعلى من فتح الشمعة السابقة
    )

    # نموذج البلع الهابط (Bearish Engulfing)
    patterns['bearish_engulfing'] = (
        (current_body_direction == -1) and  # الشمعة الحالية هابطة
        (prev_body_direction == 1) and  # الشمعة السابقة صاعدة
        (df['Open'].iloc[current_idx] > df['Close'].iloc[prev_idx]) and  # فتح الشمعة الحالية أعلى من إغلاق الشمعة السابقة
        (df['Close'].iloc[current_idx] < df['Open'].iloc[prev_idx])  # إغلاق الشمعة الحالية أقل من فتح الشمعة السابقة
    )

    # نموذج المطرقة (Hammer) - جسم صغير في الأعلى وذيل سفلي طويل
    patterns['hammer'] = (
        (body_to_range_ratio < 0.3) and  # جسم صغير نسبياً
        (current_lower_shadow > 2 * current_body_size) and  # ذيل سفلي طويل
        (current_upper_shadow < 0.5 * current_body_size)  # ذيل علوي قصير
    )

    # نموذج النجمة الساقطة (Shooting Star) - جسم صغير في الأسفل وذيل علوي طويل
    patterns['shooting_star'] = (
        (body_to_range_ratio < 0.3) and  # جسم صغير نسبياً
        (current_upper_shadow > 2 * current_body_size) and  # ذيل علوي طويل
        (current_lower_shadow < 0.5 * current_body_size)  # ذيل سفلي قصير
    )

    # نموذج الدوجي (Doji) - جسم صغير جداً
    patterns['doji'] = current_body_size < 0.1 * current_range if current_range > 0 else False

    return patterns

# تحليل مستويات الدعم والمقاومة
def find_support_resistance(df, window=10):
    pivots = {'support': [], 'resistance': []}

    if len(df) < window * 2:
        return pivots

    # تحديد القمم والقيعان المحلية بطريقة أبسط
    for i in range(window, len(df) - window):
        # تحقق مما إذا كانت هذه قمة محلية
        if all(df['High'].iloc[i] > df['High'].iloc[i-j] for j in range(1, window)) and \
           all(df['High'].iloc[i] > df['High'].iloc[i+j] for j in range(1, window)):
            pivots['resistance'].append(float(df['High'].iloc[i]))

        # تحقق مما إذا كان هذا قاع محلي
        if all(df['Low'].iloc[i] < df['Low'].iloc[i-j] for j in range(1, window)) and \
           all(df['Low'].iloc[i] < df['Low'].iloc[i+j] for j in range(1, window)):
            pivots['support'].append(float(df['Low'].iloc[i]))

    # تصفية المستويات المتقاربة
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

# تحليل حجم التداول
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

        # تحديد ما إذا كان الحجم مرتفعاً أو منخفضاً
        high_volume = volume_ratio > 1.5
        low_volume = volume_ratio < 0.5

        # تحديد ما إذا كان الحجم يؤكد الاتجاه
        volume_confirms_trend = (
            (df['Close'].iloc[-1] > df['Close'].iloc[-2] and df['Volume'].iloc[-1] > df['Volume'].iloc[-2]) or  # ارتفاع السعر مع ارتفاع الحجم
            (df['Close'].iloc[-1] < df['Close'].iloc[-2] and df['Volume'].iloc[-1] > df['Volume'].iloc[-2])  # انخفاض السعر مع ارتفاع الحجم
        )

        return {
            'high_volume': high_volume,
            'low_volume': low_volume,
            'volume_confirms_trend': volume_confirms_trend
        }
    except Exception as e:
        st.error(f"خطأ في تحليل الحجم: {str(e)}")
        return {
            'high_volume': False,
            'low_volume': False,
            'volume_confirms_trend': False
        }

# تحليل الاتجاه العام
def analyze_trend(df, ema_fast=8, ema_medium=21, ema_slow=50):
    if len(df) < max(ema_fast, ema_medium, ema_slow) + 1:
        return "غير معروف"

    try:
        df['ema_fast'] = ema(df, ema_fast)
        df['ema_medium'] = ema(df, ema_medium)
        df['ema_slow'] = ema(df, ema_slow)

        # تحديد الاتجاه بناءً على ترتيب المتوسطات المتحركة
        if pd.isna(df['ema_fast'].iloc[-1]) or pd.isna(df['ema_medium'].iloc[-1]) or pd.isna(df['ema_slow'].iloc[-1]):
            return "غير معروف"

        if df['ema_fast'].iloc[-1] > df['ema_medium'].iloc[-1] > df['ema_slow'].iloc[-1]:
            trend = "صاعد قوي"
        elif df['ema_fast'].iloc[-1] > df['ema_medium'].iloc[-1] and df['ema_medium'].iloc[-1] < df['ema_slow'].iloc[-1]:
            trend = "صاعد مع مقاومة"
        elif df['ema_fast'].iloc[-1] < df['ema_medium'].iloc[-1] and df['ema_medium'].iloc[-1] > df['ema_slow'].iloc[-1]:
            trend = "هابط مع دعم"
        elif df['ema_fast'].iloc[-1] < df['ema_medium'].iloc[-1] < df['ema_slow'].iloc[-1]:
            trend = "هابط قوي"
        else:
            trend = "متذبذب"

        return trend
    except Exception as e:
        st.error(f"خطأ في تحليل الاتجاه: {str(e)}")
        return "غير معروف"

# التحليل الشامل للسعر والمؤشرات الفنية
def analyze_price_action(df, settings=DEFAULT_SETTINGS):
    # التحقق من وجود بيانات كافية
    if df.empty or len(df) < 50:
        return {
            'signal': "لا توجد بيانات كافية",
            'confidence': 0,
            'tp': None,
            'sl': None,
            'trend': "غير معروف",
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
        # حساب المؤشرات الفنية
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

        # تحليل نماذج الشموع
        candlestick_patterns = analyze_candlestick_patterns(df)

        # تحليل مستويات الدعم والمقاومة
        support_resistance = find_support_resistance(df)

        # تحليل الحجم
        volume_analysis = analyze_volume(df, settings['volume_sma'])

        # تحليل الاتجاه العام
        trend = analyze_trend(df, settings['ema_fast'], settings['ema_medium'], settings['ema_slow'])

        # الحصول على آخر قيم للمؤشرات
        last_close = float(df['Close'].iloc[-1])
        last_rsi = float(df['RSI'].iloc[-1]) if not pd.isna(df['RSI'].iloc[-1]) else None
        last_macd = float(df['MACD'].iloc[-1]) if not pd.isna(df['MACD'].iloc[-1]) else None
        last_macd_signal = float(df['MACD_signal'].iloc[-1]) if not pd.isna(df['MACD_signal'].iloc[-1]) else None
        last_stoch_k = float(df['Stoch_K'].iloc[-1]) if not pd.isna(df['Stoch_K'].iloc[-1]) else None
        last_stoch_d = float(df['Stoch_D'].iloc[-1]) if not pd.isna(df['Stoch_D'].iloc[-1]) else None

        # تحليل الإشارات
        buy_signals = []
        sell_signals = []

        # إشارات المتوسطات المتحركة
        if not pd.isna(df['SMA_fast'].iloc[-1]) and not pd.isna(df['SMA_slow'].iloc[-1]):
            # تقاطع المتوسط السريع للمتوسط البطيء من الأسفل (إشارة شراء)
            if df['SMA_fast'].iloc[-2] < df['SMA_slow'].iloc[-2] and df['SMA_fast'].iloc[-1] > df['SMA_slow'].iloc[-1]:
                buy_signals.append(f"تقاطع SMA {settings['sma_fast']} مع SMA {settings['sma_slow']} للأعلى")

            # تقاطع المتوسط السريع للمتوسط البطيء من الأعلى (إشارة بيع)
            if df['SMA_fast'].iloc[-2] > df['SMA_slow'].iloc[-2] and df['SMA_fast'].iloc[-1] < df['SMA_slow'].iloc[-1]:
                sell_signals.append(f"تقاطع SMA {settings['sma_fast']} مع SMA {settings['sma_slow']} للأسفل")

        if not pd.isna(df['EMA_fast'].iloc[-1]) and not pd.isna(df['EMA_medium'].iloc[-1]):
            # تقاطع المتوسط السريع للمتوسط المتوسط من الأسفل (إشارة شراء)
            if df['EMA_fast'].iloc[-2] < df['EMA_medium'].iloc[-2] and df['EMA_fast'].iloc[-1] > df['EMA_medium'].iloc[-1]:
                buy_signals.append(f"تقاطع EMA {settings['ema_fast']} مع EMA {settings['ema_medium']} للأعلى")

            # تقاطع المتوسط السريع للمتوسط المتوسط من الأعلى (إشارة بيع)
            if df['EMA_fast'].iloc[-2] > df['EMA_medium'].iloc[-2] and df['EMA_fast'].iloc[-1] < df['EMA_medium'].iloc[-1]:
                sell_signals.append(f"تقاطع EMA {settings['ema_fast']} مع EMA {settings['ema_medium']} للأسفل")

        # إشارات RSI
        if last_rsi is not None:
            if last_rsi < settings['rsi_oversold']:
                buy_signals.append(f"RSI في منطقة ذروة البيع ({last_rsi:.1f})")

            if last_rsi > settings['rsi_overbought']:
                sell_signals.append(f"RSI في منطقة ذروة الشراء ({last_rsi:.1f})")

        # إشارات MACD
        if last_macd is not None and last_macd_signal is not None:
            # تقاطع خط MACD لخط الإشارة من الأسفل (إشارة شراء)
            if df['MACD'].iloc[-2] < df['MACD_signal'].iloc[-2] and df['MACD'].iloc[-1] > df['MACD_signal'].iloc[-1]:
                buy_signals.append("تقاطع MACD مع خط الإشارة للأعلى")

            # تقاطع خط MACD لخط الإشارة من الأعلى (إشارة بيع)
            if df['MACD'].iloc[-2] > df['MACD_signal'].iloc[-2] and df['MACD'].iloc[-1] < df['MACD_signal'].iloc[-1]:
                sell_signals.append("تقاطع MACD مع خط الإشارة للأسفل")

        # إشارات ستوكاستيك
        if last_stoch_k is not None and last_stoch_d is not None:
            # تقاطع %K لـ %D من الأسفل في منطقة ذروة البيع (إشارة شراء)
            if (df['Stoch_K'].iloc[-2] < df['Stoch_D'].iloc[-2] and df['Stoch_K'].iloc[-1] > df['Stoch_D'].iloc[-1] and
                    df['Stoch_K'].iloc[-1] < settings['stoch_oversold']):
                buy_signals.append("تقاطع ستوكاستيك %K مع %D للأعلى في منطقة ذروة البيع")

            # تقاطع %K لـ %D من الأعلى في منطقة ذروة الشراء (إشارة بيع)
            if (df['Stoch_K'].iloc[-2] > df['Stoch_D'].iloc[-2] and df['Stoch_K'].iloc[-1] < df['Stoch_D'].iloc[-1] and
                    df['Stoch_K'].iloc[-1] > settings['stoch_overbought']):
                sell_signals.append("تقاطع ستوكاستيك %K مع %D للأسفل في منطقة ذروة الشراء")

            # ستوكاستيك في منطقة ذروة البيع
            if last_stoch_k < settings['stoch_oversold'] and last_stoch_d < settings['stoch_oversold']:
                buy_signals.append(f"ستوكاستيك في منطقة ذروة البيع (%K: {last_stoch_k:.1f}, %D: {last_stoch_d:.1f})")

            # ستوكاستيك في منطقة ذروة الشراء
            if last_stoch_k > settings['stoch_overbought'] and last_stoch_d > settings['stoch_overbought']:
                sell_signals.append(f"ستوكاستيك في منطقة ذروة الشراء (%K: {last_stoch_k:.1f}, %D: {last_stoch_d:.1f})")

        # إشارات بولينجر باند
        if 'BB_upper' in df.columns and 'BB_lower' in df.columns:
            if not pd.isna(df['BB_lower'].iloc[-1]) and not pd.isna(df['BB_upper'].iloc[-1]):
                if df['Close'].iloc[-1] < df['BB_lower'].iloc[-1]:
                    buy_signals.append("السعر تحت الحد السفلي لبولينجر باند")

                if df['Close'].iloc[-1] > df['BB_upper'].iloc[-1]:
                    sell_signals.append("السعر فوق الحد العلوي لبولينجر باند")

        # إشارات نماذج الشموع
        if candlestick_patterns['bullish_engulfing']:
            buy_signals.append("نموذج البلع الصاعد")

        if candlestick_patterns['bearish_engulfing']:
            sell_signals.append("نموذج البلع الهابط")

        if candlestick_patterns['hammer']:
            buy_signals.append("نموذج المطرقة")

        if candlestick_patterns['shooting_star']:
            sell_signals.append("نموذج النجمة الساقطة")

        # إشارات الحجم
        if 'Volume' in df.columns and not df['Volume'].isnull().all():
            if volume_analysis.get('high_volume') and volume_analysis.get('volume_confirms_trend'):
                if len(df) > 1 and not pd.isna(df['Close'].iloc[-1]) and not pd.isna(df['Close'].iloc[-2]):
                    if df['Close'].iloc[-1] > df['Close'].iloc[-2]:
                        buy_signals.append("ارتفاع الحجم يؤكد الاتجاه الصاعد")
                    else:
                        sell_signals.append("ارتفاع الحجم يؤكد الاتجاه الهابط")

        # حساب نسبة الثقة في الإشارة
        buy_confidence = (len(buy_signals) / 10) * 100  # 10 هو العدد الإجمالي للإشارات المحتملة
        sell_confidence = (len(sell_signals) / 10) * 100

        # تحديد الإشارة النهائية بناءً على نسبة الثقة
        signal = "لا توجد إشارة"
        confidence = 0
        signals_details = []

        if buy_confidence >= settings['confidence_threshold'] and buy_confidence > sell_confidence:
            signal = "شراء"
            confidence = buy_confidence
            signals_details = buy_signals
        elif sell_confidence >= settings['confidence_threshold'] and sell_confidence > buy_confidence:
            signal = "بيع"
            confidence = sell_confidence
            signals_details = sell_signals

        # حساب أهداف الربح ووقف الخسارة
        tp = None
        sl = None
        risk_reward_ratio = settings['risk_reward_ratio']

        if signal == "شراء" and not pd.isna(df['ATR'].iloc[-1]):
            # استخدام ATR لتحديد وقف الخسارة
            atr_value = float(df['ATR'].iloc[-1])
            sl = last_close - (atr_value * 1.5)
            tp = last_close + (atr_value * 1.5 * risk_reward_ratio)

            # تعديل وقف الخسارة بناءً على مستويات الدعم إن وجدت
            if support_resistance['support']:
                nearest_support = max([s for s in support_resistance['support'] if s < last_close], default=sl)
                sl = max(sl, nearest_support)

            # تعديل هدف الربح بناءً على مستويات المقاومة إن وجدت
            if support_resistance['resistance']:
                nearest_resistance = min([r for r in support_resistance['resistance'] if r > last_close], default=tp)
                tp = min(tp, nearest_resistance)

        elif signal == "بيع" and not pd.isna(df['ATR'].iloc[-1]):
            # استخدام ATR لتحديد وقف الخسارة
            atr_value = float(df['ATR'].iloc[-1])
            sl = last_close + (atr_value * 1.5)
            tp = last_close - (atr_value * 1.5 * risk_reward_ratio)

            # تعديل وقف الخسارة بناءً على مستويات المقاومة إن وجدت
            if support_resistance['resistance']:
                nearest_resistance = min([r for r in support_resistance['resistance'] if r > last_close], default=sl)
                sl = min(sl, nearest_resistance)

            # تعديل هدف الربح بناءً على مستويات الدعم إن وجدت
            if support_resistance['support']:
                nearest_support = max([s for s in support_resistance['support'] if s < last_close], default=tp)
                tp = max(tp, nearest_support)

        # تقريب القيم
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
        st.error(f"خطأ في التحليل الفني: {str(e)}")
        return {
            'signal': "حدث خطأ",
            'confidence': 0,
            'tp': None,
            'sl': None,
            'trend': "غير معروف",
            'signals_details': [],
            'support_resistance': {'support': [], 'resistance': []},
            'last_close': None,
            'rsi': None,
            'macd': None,
            'macd_signal': None,
            'stoch_k': None,
            'stoch_d': None
        }

# إنشاء رسم بياني للتحليل الفني
def create_technical_chart(df, symbol_name):
    try:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1, 1]})

        # رسم السعر والمتوسطات المتحركة
        ax1.plot(df.index, df['Close'], label='السعر', linewidth=2)

        if 'EMA_fast' in df.columns and not df['EMA_fast'].isnull().all():
            ax1.plot(df.index, df['EMA_fast'], label=f'EMA {DEFAULT_SETTINGS["ema_fast"]}', alpha=0.7)

        if 'EMA_medium' in df.columns and not df['EMA_medium'].isnull().all():
            ax1.plot(df.index, df['EMA_medium'], label=f'EMA {DEFAULT_SETTINGS["ema_medium"]}', alpha=0.7)

        if 'EMA_slow' in df.columns and not df['EMA_slow'].isnull().all():
            ax1.plot(df.index, df['EMA_slow'], label=f'EMA {DEFAULT_SETTINGS["ema_slow"]}', alpha=0.7)

        if 'BB_upper' in df.columns and not df['BB_upper'].isnull().all():
            ax1.plot(df.index, df['BB_upper'], 'r--', label='بولينجر العلوي', alpha=0.5)

        if 'BB_middle' in df.columns and not df['BB_middle'].isnull().all():
            ax1.plot(df.index, df['BB_middle'], 'g--', label='بولينجر الوسط', alpha=0.5)

        if 'BB_lower' in df.columns and not df['BB_lower'].isnull().all():
            ax1.plot(df.index, df['BB_lower'], 'r--', label='بولينجر السفلي', alpha=0.5)

        ax1.set_title(f'التحليل الفني لـ {symbol_name}')
        ax1.set_ylabel('السعر')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # رسم MACD
        if 'MACD' in df.columns and not df['MACD'].isnull().all():
            ax2.plot(df.index, df['MACD'], label='MACD', color='blue')

        if 'MACD_signal' in df.columns and not df['MACD_signal'].isnull().all():
            ax2.plot(df.index, df['MACD_signal'], label='إشارة MACD', color='red')

        if 'MACD_histogram' in df.columns and not df['MACD_histogram'].isnull().all():
            ax2.bar(df.index, df['MACD_histogram'], label='هيستوجرام MACD', color='green', alpha=0.5)

        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_ylabel('MACD')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)

        # رسم RSI و Stochastic
        if 'RSI' in df.columns and not df['RSI'].isnull().all():
            ax3.plot(df.index, df['RSI'], label='RSI', color='purple')

        if 'Stoch_K' in df.columns and not df['Stoch_K'].isnull().all():
            ax3.plot(df.index, df['Stoch_K'], label='Stochastic %K', color='blue', alpha=0.7)

        if 'Stoch_D' in df.columns and not df['Stoch_D'].isnull().all():
            ax3.plot(df.index, df['Stoch_D'], label='Stochastic %D', color='red', alpha=0.7)

        ax3.axhline(y=DEFAULT_SETTINGS['rsi_overbought'], color='red', linestyle='--', alpha=0.5)
        ax3.axhline(y=DEFAULT_SETTINGS['rsi_oversold'], color='green', linestyle='--', alpha=0.5)
        ax3.axhline(y=50, color='black', linestyle='-', alpha=0.3)
        ax3.set_ylim([0, 100])
        ax3.set_ylabel('المذبذبات')
        ax3.set_xlabel('الوقت')
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        
        return fig
    except Exception as e:
        st.error(f"خطأ في إنشاء الرسم البياني: {str(e)}")
        return None

# الحصول على نصيحة تداول عشوائية
def get_random_trading_tip():
    return random.choice(trading_tips)

# تنسيق الإشارة بالألوان
def format_signal(signal):
    if signal == "شراء":
        return f"<span style='color:#4CAF50;font-weight:bold;'>{signal}</span>"
    elif signal == "بيع":
        return f"<span style='color:#f44336;font-weight:bold;'>{signal}</span>"
    else:
        return f"<span style='color:#aaa;'>{signal}</span>"

# تنسيق نسبة الثقة بالألوان
def format_confidence(confidence):
    if confidence >= 80:
        return f"<span style='color:#4CAF50;font-weight:bold;'>{confidence:.1f}%</span>"
    elif confidence >= 60:
        return f"<span style='color:#FFC107;font-weight:bold;'>{confidence:.1f}%</span>"
    else:
        return f"<span style='color:#f44336;font-weight:bold;'>{confidence:.1f}%</span>"

# تنسيق الاتجاه بالألوان
def format_trend(trend):
    if "صاعد" in trend:
        return f"<span style='color:#4CAF50;font-weight:bold;'>{trend}</span>"
    elif "هابط" in trend:
        return f"<span style='color:#f44336;font-weight:bold;'>{trend}</span>"
    else:
        return f"<span style='color:#FFC107;font-weight:bold;'>{trend}</span>"

# تطبيق CSS للتطبيق
def apply_custom_css():
    st.markdown("""
    <style>
    .main {
        background-color: #1e1e2f;
        color: white;
        direction: rtl;
    }
    .stApp {
        direction: rtl;
    }
    .stTabs [data-baseweb="tab-list"] {
        direction: rtl;
    }
    .stTabs [data-baseweb="tab"] {
        direction: rtl;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #4CAF50 !important;
        direction: rtl;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #3e8e41;
    }
    .stSelectbox>div>div {
        direction: rtl;
    }
    .stNumberInput>div>div {
        direction: rtl;
    }
    .stSlider>div>div {
        direction: rtl;
    }
    .stCheckbox>div {
        direction: rtl;
    }
    .stRadio>div {
        direction: rtl;
    }
    .stTextInput>div>div {
        direction: rtl;
    }
    .stTextArea>div>div {
        direction: rtl;
    }
    .stDataFrame {
        direction: rtl;
    }
    .stTable {
        direction: rtl;
    }
    .stMarkdown {
        direction: rtl;
    }
    .stText {
        direction: rtl;
    }
    .stInfo {
        direction: rtl;
    }
    .stWarning {
        direction: rtl;
    }
    .stError {
        direction: rtl;
    }
    .stSuccess {
        direction: rtl;
    }
    .stExpander>div>div {
        direction: rtl;
    }
    .signal-buy {
        color: #4CAF50;
        font-weight: bold;
    }
    .signal-sell {
        color: #f44336;
        font-weight: bold;
    }
    .signal-none {
        color: #aaa;
    }
    .confidence-high {
        color: #4CAF50;
        font-weight: bold;
    }
    .confidence-medium {
        color: #FFC107;
        font-weight: bold;
    }
    .confidence-low {
        color: #f44336;
        font-weight: bold;
    }
    .trend-up {
        color: #4CAF50;
        font-weight: bold;
    }
    .trend-down {
        color: #f44336;
        font-weight: bold;
    }
    .trend-neutral {
        color: #FFC107;
        font-weight: bold;
    }
    .tip-container {
        background-color: #2a2a45;
        padding: 15px;
        border-radius: 5px;
        margin: 20px 0;
        border-left: 4px solid #4CAF50;
    }
    .tip-title {
        font-weight: bold;
        color: #4CAF50;
        margin-bottom: 10px;
    }
    .tip-content {
        line-height: 1.6;
    }
    .dashboard-container {
        background-color: #2a2a45;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #3a3a5a;
        padding: 15px;
        border-radius: 5px;
        text-align: center;
        height: 100%;
    }
    .metric-title {
        font-size: 1.1em;
        margin-bottom: 10px;
        color: #aaa;
    }
    .metric-value {
        font-size: 1.8em;
        font-weight: bold;
    }
    .indicator-container {
        display: flex;
        flex-direction: column;
        background-color: #3a3a5a;
        padding: 15px;
        border-radius: 5px;
        height: 100%;
    }
    .indicator-title {
        font-size: 1.1em;
        margin-bottom: 10px;
        color: #aaa;
    }
    .indicator-value {
        font-size: 1.4em;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .indicator-details {
        font-size: 0.9em;
        color: #ddd;
    }
    .chart-container {
        background-color: #3a3a5a;
        padding: 15px;
        border-radius: 5px;
        margin-top: 20px;
    }
    .history-container {
        background-color: #3a3a5a;
        padding: 15px;
        border-radius: 5px;
        margin-top: 20px;
    }
    .instructions-container {
        background-color: #3a3a5a;
        padding: 20px;
        border-radius: 5px;
        margin-top: 20px;
        border-left: 4px solid #4CAF50;
    }
    .instructions-title {
        font-size: 1.3em;
        font-weight: bold;
        margin-bottom: 15px;
        color: #4CAF50;
    }
    .instructions-step {
        margin-bottom: 10px;
        padding-right: 15px;
        position: relative;
    }
    .instructions-step:before {
        content: "•";
        color: #4CAF50;
        font-weight: bold;
        position: absolute;
        right: 0;
    }
    </style>
    """, unsafe_allow_html=True)

# عرض لوحة المعلومات الرئيسية
def display_dashboard(analysis, selected_symbol_name):
    st.markdown("<div class='dashboard-container'>", unsafe_allow_html=True)
    
    # صف المؤشرات الرئيسية
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-title'>الإشارة</div>
            <div class='metric-value'>{}</div>
        </div>
        """.format(format_signal(analysis['signal'])), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-title'>نسبة الثقة</div>
            <div class='metric-value'>{}</div>
        </div>
        """.format(format_confidence(analysis['confidence'])), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-title'>الاتجاه العام</div>
            <div class='metric-value'>{}</div>
        </div>
        """.format(format_trend(analysis['trend'])), unsafe_allow_html=True)
    
    # صف السعر وأهداف الربح ووقف الخسارة
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        price_value = f"{analysis['last_close']:.5f}" if analysis['last_close'] is not None else "غير متوفر"
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>السعر الحالي</div>
            <div class='metric-value'>{price_value}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        tp_value = f"<span style='color:#4CAF50;'>{analysis['tp']:.5f}</span>" if analysis['tp'] is not None else "غير متوفر"
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>هدف الربح</div>
            <div class='metric-value'>{tp_value}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        sl_value = f"<span style='color:#f44336;'>{analysis['sl']:.5f}</span>" if analysis['sl'] is not None else "غير متوفر"
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>وقف الخسارة</div>
            <div class='metric-value'>{sl_value}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# عرض المؤشرات الفنية
def display_technical_indicators(analysis):
    st.markdown("<h3>المؤشرات الفنية</h3>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div class='indicator-container'>", unsafe_allow_html=True)
        st.markdown("<div class='indicator-title'>RSI</div>", unsafe_allow_html=True)
        
        if analysis['rsi'] is not None:
            if analysis['rsi'] > DEFAULT_SETTINGS['rsi_overbought']:
                rsi_color = "#f44336"
                rsi_status = "ذروة شراء"
            elif analysis['rsi'] < DEFAULT_SETTINGS['rsi_oversold']:
                rsi_color = "#4CAF50"
                rsi_status = "ذروة بيع"
            else:
                rsi_color = "#FFC107"
                rsi_status = "محايد"
            
            st.markdown(f"""
            <div class='indicator-value' style='color:{rsi_color};'>{analysis['rsi']:.2f}</div>
            <div class='indicator-details'>{rsi_status}</div>
            <div class='indicator-details'>ذروة الشراء: {DEFAULT_SETTINGS['rsi_overbought']}</div>
            <div class='indicator-details'>ذروة البيع: {DEFAULT_SETTINGS['rsi_oversold']}</div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("<div class='indicator-value'>غير متوفر</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='indicator-container'>", unsafe_allow_html=True)
        st.markdown("<div class='indicator-title'>MACD</div>", unsafe_allow_html=True)
        
        if analysis['macd'] is not None and analysis['macd_signal'] is not None:
            macd_diff = analysis['macd'] - analysis['macd_signal']
            macd_color = "#4CAF50" if macd_diff > 0 else "#f44336"
            macd_status = "إيجابي" if macd_diff > 0 else "سلبي"
            
            st.markdown(f"""
            <div class='indicator-value' style='color:{macd_color};'>{analysis['macd']:.5f}</div>
            <div class='indicator-details'>الإشارة: {analysis['macd_signal']:.5f}</div>
            <div class='indicator-details'>الفرق: {macd_diff:.5f}</div>
            <div class='indicator-details'>الحالة: {macd_status}</div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("<div class='indicator-value'>غير متوفر</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='indicator-container'>", unsafe_allow_html=True)
        st.markdown("<div class='indicator-title'>ستوكاستيك</div>", unsafe_allow_html=True)
        
        if analysis['stoch_k'] is not None and analysis['stoch_d'] is not None:
            if analysis['stoch_k'] > DEFAULT_SETTINGS['stoch_overbought'] and analysis['stoch_d'] > DEFAULT_SETTINGS['stoch_overbought']:
                stoch_color = "#f44336"
                stoch_status = "ذروة شراء"
            elif analysis['stoch_k'] < DEFAULT_SETTINGS['stoch_oversold'] and analysis['stoch_d'] < DEFAULT_SETTINGS['stoch_oversold']:
                stoch_color = "#4CAF50"
                stoch_status = "ذروة بيع"
            else:
                stoch_color = "#FFC107"
                stoch_status = "محايد"
            
            st.markdown(f"""
            <div class='indicator-value' style='color:{stoch_color};'>%K: {analysis['stoch_k']:.2f}, %D: {analysis['stoch_d']:.2f}</div>
            <div class='indicator-details'>{stoch_status}</div>
            <div class='indicator-details'>ذروة الشراء: {DEFAULT_SETTINGS['stoch_overbought']}</div>
            <div class='indicator-details'>ذروة البيع: {DEFAULT_SETTINGS['stoch_oversold']}</div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("<div class='indicator-value'>غير متوفر</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

# عرض تفاصيل الإشارات ومستويات الدعم والمقاومة
def display_signal_details(analysis):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h3>تفاصيل الإشارات</h3>", unsafe_allow_html=True)
        
        if analysis['signals_details']:
            for detail in analysis['signals_details']:
                st.markdown(f"• {detail}")
        else:
            st.markdown("لا توجد إشارات كافية لاتخاذ قرار.")
    
    with col2:
        st.markdown("<h3>مستويات الدعم والمقاومة</h3>", unsafe_allow_html=True)
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("<h4>المقاومة</h4>", unsafe_allow_html=True)
            if analysis['support_resistance']['resistance']:
                for level in sorted(analysis['support_resistance']['resistance'], reverse=True):
                    st.markdown(f"• {level:.5f}")
            else:
                st.markdown("لا توجد مستويات مقاومة محددة.")
        
        with col_b:
            st.markdown("<h4>الدعم</h4>", unsafe_allow_html=True)
            if analysis['support_resistance']['support']:
                for level in sorted(analysis['support_resistance']['support'], reverse=True):
                    st.markdown(f"• {level:.5f}")
            else:
                st.markdown("لا توجد مستويات دعم محددة.")

# عرض تعليمات الاستخدام
def display_instructions():
    st.markdown("<div class='instructions-container'>", unsafe_allow_html=True)
    st.markdown("<div class='instructions-title'>📋 كيفية استخدام التطبيق</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='instructions-step'>اختر الزوج أو الأداة المالية من القائمة المنسدلة في الشريط الجانبي.</div>
    <div class='instructions-step'>حدد الإطار الزمني المناسب لتحليلك (1 دقيقة، 5 دقائق، 15 دقيقة، إلخ).</div>
    <div class='instructions-step'>يمكنك تخصيص إعدادات المؤشرات الفنية من خلال قسم "إعدادات متقدمة" في الشريط الجانبي.</div>
    <div class='instructions-step'>راقب لوحة المعلومات للحصول على الإشارات الحالية ونسبة الثقة والاتجاه العام.</div>
    <div class='instructions-step'>استخدم هدف الربح ووقف الخسارة المقترحين لإدارة المخاطر في صفقاتك.</div>
    <div class='instructions-step'>تحقق من تفاصيل الإشارات لفهم أسباب التوصية الحالية.</div>
    <div class='instructions-step'>راجع الرسم البياني للحصول على تحليل بصري للسعر والمؤشرات الفنية.</div>
    <div class='instructions-step'>اضغط على زر "تحديث الآن" في الشريط الجانبي للحصول على أحدث البيانات والتحليلات.</div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='instructions-title'>⚠️ تنبيه مهم</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='instructions-step'>هذا التطبيق مخصص لأغراض تعليمية وإرشادية فقط وليس توصية للتداول.</div>
    <div class='instructions-step'>يجب عليك دائماً إجراء تحليلك الخاص واستشارة مستشار مالي قبل اتخاذ أي قرارات استثمارية.</div>
    <div class='instructions-step'>تذكر أن التداول ينطوي على مخاطر، ويمكن أن تخسر أكثر من استثمارك الأولي.</div>
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# الدالة الرئيسية للتطبيق
def main():
    # تطبيق CSS المخصص
    apply_custom_css()

    # عنوان التطبيق
    st.title("🚀 مؤشرات التداول - سكالبينج")
    st.markdown("### نظام تحليل فني متقدم للتداول قصير المدى")

    # تحميل سجل الإشارات
    load_signal_history()

    # الشريط الجانبي للإعدادات
    with st.sidebar:
        st.header("⚙️ الإعدادات")
        
        # اختيار الزوج
        selected_symbol_name = st.selectbox(
            "اختر الزوج أو الأداة المالية:",
            list(symbols.keys())
        )
        selected_symbol = symbols[selected_symbol_name]
        
        # اختيار الفترة الزمنية
        timeframe = st.selectbox(
            "الإطار الزمني:",
            ["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
            index=1
        )
        
        # تحديد فترة البيانات
        if timeframe in ["1m", "5m", "15m", "30m"]:
            period = "1d"
        elif timeframe in ["1h", "4h"]:
            period = "5d"
        else:
            period = "1mo"
        
        # إعدادات متقدمة
        with st.expander("🔧 إعدادات متقدمة"):
            # إعدادات المتوسطات المتحركة
            st.subheader("المتوسطات المتحركة")
            sma_fast = st.slider("SMA سريع", 5, 50, DEFAULT_SETTINGS['sma_fast'])
            sma_slow = st.slider("SMA بطيء", 10, 200, DEFAULT_SETTINGS['sma_slow'])
            ema_fast = st.slider("EMA سريع", 5, 50, DEFAULT_SETTINGS['ema_fast'])
            ema_medium = st.slider("EMA متوسط", 10, 100, DEFAULT_SETTINGS['ema_medium'])
            ema_slow = st.slider("EMA بطيء", 20, 200, DEFAULT_SETTINGS['ema_slow'])
            
            # إعدادات المذبذبات
            st.subheader("المذبذبات")
            rsi_period = st.slider("فترة RSI", 5, 30, DEFAULT_SETTINGS['rsi_period'])
            rsi_overbought = st.slider("RSI ذروة الشراء", 60, 90, DEFAULT_SETTINGS['rsi_overbought'])
            rsi_oversold = st.slider("RSI ذروة البيع", 10, 40, DEFAULT_SETTINGS['rsi_oversold'])
            
            # إعدادات MACD
            st.subheader("MACD")
            macd_fast = st.slider("MACD سريع", 5, 20, DEFAULT_SETTINGS['macd_fast'])
            macd_slow = st.slider("MACD بطيء", 15, 40, DEFAULT_SETTINGS['macd_slow'])
            macd_signal = st.slider("MACD إشارة", 5, 15, DEFAULT_SETTINGS['macd_signal'])
            
            # إعدادات بولينجر باند
            st.subheader("بولينجر باند")
            bollinger_period = st.slider("فترة بولينجر", 10, 50, DEFAULT_SETTINGS['bollinger_period'])
            bollinger_std = st.slider("انحراف معياري", 1.0, 3.0, float(DEFAULT_SETTINGS['bollinger_std']), 0.1)
            
            # إعدادات ستوكاستيك
            st.subheader("ستوكاستيك")
            stoch_k = st.slider("فترة %K", 5, 30, DEFAULT_SETTINGS['stoch_k'])
            stoch_d = st.slider("فترة %D", 1, 10, DEFAULT_SETTINGS['stoch_d'])
            stoch_overbought = st.slider("ستوكاستيك ذروة الشراء", 60, 90, DEFAULT_SETTINGS['stoch_overbought'])
            stoch_oversold = st.slider("ستوكاستيك ذروة البيع", 10, 40, DEFAULT_SETTINGS['stoch_oversold'])
            
            # إعدادات أخرى
            st.subheader("إعدادات أخرى")
            confidence_threshold = st.slider("حد الثقة للإشارة (%)", 50, 90, DEFAULT_SETTINGS['confidence_threshold'])
            risk_reward_ratio = st.slider("نسبة المخاطرة/المكافأة", 1.0, 3.0, float(DEFAULT_SETTINGS['risk_reward_ratio']), 0.1)
            
            # خيارات العرض
            st.subheader("خيارات العرض")
            show_charts = st.checkbox("عرض الرسوم البيانية", DEFAULT_SETTINGS['show_charts'])
            save_history = st.checkbox("حفظ سجل الإشارات", DEFAULT_SETTINGS['save_history'])
            show_tips = st.checkbox("عرض نصائح التداول", DEFAULT_SETTINGS['show_tips'])
        
        # تحديث الإعدادات
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
            'update_interval': DEFAULT_SETTINGS['update_interval'],
            'show_charts': show_charts,
            'save_history': save_history,
            'show_tips': show_tips,
            'max_history': DEFAULT_SETTINGS['max_history']
        }
        
        # زر التحديث اليدوي
        if st.button("تحديث الآن"):
            st.success("تم تحديث البيانات بنجاح!")
    
    # عرض نصيحة تداول عشوائية
    if settings['show_tips']:
        tip = get_random_trading_tip()
        st.markdown(f"""
        <div class="tip-container">
            <div class="tip-title">💡 نصيحة تداول</div>
            <div class="tip-content">{tip}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # إنشاء علامات تبويب للتنقل بين أقسام التطبيق
    tabs = st.tabs(["📊 لوحة المعلومات", "📈 الرسوم البيانية", "📋 سجل الإشارات", "ℹ️ تعليمات الاستخدام"])
    
    # جلب البيانات وتحليلها
    with st.spinner("جاري تحليل البيانات..."):
        df = fetch_data(selected_symbol, period=period, interval=timeframe)
        
        if not df.empty:
            # تحليل البيانات
            analysis = analyze_price_action(df, settings)
            
            # علامة تبويب لوحة المعلومات
            with tabs[0]:
                # عرض لوحة المعلومات الرئيسية
                display_dashboard(analysis, selected_symbol_name)
                
                # عرض المؤشرات الفنية
                display_technical_indicators(analysis)
                
                # عرض تفاصيل الإشارات ومستويات الدعم والمقاومة
                display_signal_details(analysis)
                
                if analysis['signal'] != "لا توجد إشارة" and analysis['signal'] != "حدث خطأ":
                    # إضافة الإشارة إلى السجل إذا كانت جديدة
                    symbol_key = f"{selected_symbol_name}_{timeframe}"
                    if symbol_key not in previous_signals or previous_signals[symbol_key] != analysis['signal']:
                        previous_signals[symbol_key] = analysis['signal']
                        if settings['save_history']:
                            add_to_history(
                                selected_symbol_name,
                                analysis['signal'],
                                analysis['last_close'],
                                analysis['confidence']
                            )
            
            # علامة تبويب الرسوم البيانية
            with tabs[1]:
                if settings['show_charts']:
                    st.markdown("<h3>الرسم البياني</h3>", unsafe_allow_html=True)
                    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                    fig = create_technical_chart(df, selected_symbol_name)
                    if fig:
                        st.pyplot(fig)
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.info("الرسوم البيانية معطلة. يمكنك تفعيلها من الإعدادات المتقدمة في الشريط الجانبي.")
            
            # علامة تبويب سجل الإشارات
            with tabs[2]:
                st.markdown("<h3>سجل الإشارات</h3>", unsafe_allow_html=True)
                st.markdown("<div class='history-container'>", unsafe_allow_html=True)
                
                if settings['save_history'] and signal_history:
                    # إنشاء DataFrame من سجل الإشارات
                    history_df = pd.DataFrame(signal_history)
                    
                    # تنسيق العرض
                    history_df.columns = ["الزوج", "الإشارة", "السعر", "الثقة", "التاريخ"]
                    
                    # عرض الجدول
                    st.dataframe(history_df, use_container_width=True)
                    
                    # زر لمسح السجل
                    if st.button("مسح السجل"):
                        signal_history.clear()
                        save_signal_history()
                        st.success("تم مسح سجل الإشارات بنجاح.")
                else:
                    st.info("سجل الإشارات فارغ أو معطل. يمكنك تفعيل حفظ السجل من الإعدادات المتقدمة في الشريط الجانبي.")
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # علامة تبويب تعليمات الاستخدام
            with tabs[3]:
                display_instructions()
        else:
            st.error(f"لا يمكن جلب بيانات {selected_symbol_name}. تأكد من اتصالك بالإنترنت أو جرب زوجاً آخر.")
    
    # إضافة معلومات التحديث
    st.markdown("---")
    st.markdown(f"<div style='text-align:center;color:#aaa;font-size:0.8em;'>آخر تحديث: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>", unsafe_allow_html=True)

# تشغيل التطبيق
if __name__ == "__main__":
    main()
