import time
import asyncio
import requests
import numpy as np
import pandas as pd
import talib as ta
import nest_asyncio
from datetime import datetime, timezone
from binance.client import Client # Added Binance Client import

nest_asyncio.apply()

from telegram import Bot

# ===============================

# 2. GLOBAL CONFIG

# ===============================
SCORE_CONFIG = {
    # Bullish conditions points
    'BULL_EMA_CROSS': 10,
    'BULL_MACD_CROSS': 7,
    'BULL_RSI_MOMENTUM': 5,
    'BULL_STRONG_CANDLE': 12,

    # Bearish conditions points
    'BEAR_EMA_CROSS': 10,
    'BEAR_MACD_CROSS': 7,
    'BEAR_RSI_MOMENTUM': 5,
    'BEAR_STRONG_CANDLE': 12,

    # Penalties/adjustments
    'PENALTY_HIGH_ATR_REGIME': -20, # Penalize signals during high volatility
    'PENALTY_LOW_DIST_FROM_EMA': -3, # Minor penalty if too close to EMA (can be adjusted)

    # Thresholds
    'MIN_BULL_SCORE_THRESHOLD': 15, # Minimum score required for a bullish signal
    'MIN_BEAR_SCORE_THRESHOLD': 15  # Minimum score required for a bearish signal
}

print(f"Defined SCORE_CONFIG: {SCORE_CONFIG}")
TIMEFRAMES = ["15min", "1hour", "4hour", "5min", "1min"]
BASE_TIMEFRAME = "15min"
TIMEFRAME = BASE_TIMEFRAME
HISTORY_LIMIT = 500

# Binance Client Initialization
client = Client('', '', tld='us') # Initialized Binance client with 'us' TLD

# Map internal timeframe strings to Binance KLINE_INTERVAL constants
BINANCE_INTERVAL_MAP = {
    "1min": Client.KLINE_INTERVAL_1MINUTE,
    "5min": Client.KLINE_INTERVAL_5MINUTE,
    "15min": Client.KLINE_INTERVAL_15MINUTE,
    "1hour": Client.KLINE_INTERVAL_1HOUR,
    "4hour": Client.KLINE_INTERVAL_4HOUR,
}

# Account / Risk

ACCOUNT_BALANCE = 10000.0  # in quote currency (e.g., USDT). User should set appropriately
PAPER_TRADING = True       # only paper in this script - integration with exchange must be added separately

PORTFOLIO_CONFIG = { "max_open_trades": 40, "max_risk_per_trade": 0.01,    # fraction of account
"max_total_risk": 0.4, "max_per_symbol": 1, "correlation_threshold": 0.75 }

TELEGRAM_TOKEN = "6247722895:AAEN6xEpnih_OdzSYX8D7Ni2q9mc7vfpBGo"
TELEGRAM_CHAT_ID = "1378561635"

# Spread / Risk controls

CONSIDER_SPREAD = False
SPREAD_PERCENTAGE = 0.0005
STOP_LOSS_MAX_PERCENTAGE = 0.02  # 2% max default
TAKE_PROFIT_MIN_PERCENTAGE = 0.005

# ATR factors - tuned conservatively

ATR_SL_FACTOR = 1.5
ATR_TP_FACTOR = 2.5

bot = Bot(token=TELEGRAM_TOKEN)

# -------------------------------

# Utility: map timeframe string to seconds

# -------------------------------

TF_SECONDS = { "1min": 60, "5min": 5 * 60, "15min": 15 * 60, "1hour": 60 * 60, "4hour": 4 * 60 * 60 }

# ===============================

# 3. DATA FETCH (changed to Binance API)

# ===============================
exchange_info = client.get_exchange_info()
print(f"Successfully fetched Binance exchange information.")
def fetch_binance_ohlcv(symbol, timeframe_str, limit=HISTORY_LIMIT):
    interval = BINANCE_INTERVAL_MAP.get(timeframe_str)
    if not interval:
        print(f"❌ Invalid timeframe: {timeframe_str} for Binance")
        return pd.DataFrame()
    try:
        klines = client.get_historical_klines(symbol, interval, limit=limit)
        if klines:
            print(f"   ✅ Fetched {len(klines)} klines for {symbol} on {timeframe_str}")
    except Exception as e:
        print(f"❌ Error fetching Binance klines for {symbol} {timeframe_str}: {e}")
        return pd.DataFrame()

    if klines:
        # Convert to DataFrame, mapping indices to column names
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df = df.set_index('open_time')
        # Convert numeric columns
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume']) # Drop rows with NaN in essential columns
        return df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    else:
        print(f"⚠️ No Binance klines data for {symbol} on {timeframe_str}")
        return pd.DataFrame()

def fetch_binance_bid_ask(symbol):
    try:
        # Using get_orderbook_tickers for best bid/ask from Binance
        ticker = client.get_orderbook_tickers(symbol=symbol)
        bid = pd.to_numeric(ticker.get('bidPrice'), errors='coerce')
        ask = pd.to_numeric(ticker.get('askPrice'), errors='coerce')
        if bid is not None and ask is not None:
            print(f"   ✅ Fetched bid/ask for {symbol}: Bid={bid}, Ask={ask}")
        return bid, ask
    except Exception as e:
        print(f"❌ Error fetching Binance bid/ask for {symbol}: {e}")
        return None, None

def fetch_all_timeframe_data(symbol):
    all_tf_data = {}
    print(f"   Fetching data for {symbol} across timeframes...")
    for tf in TIMEFRAMES:
        df = fetch_binance_ohlcv(symbol, tf, limit=HISTORY_LIMIT)
        if not df.empty:
            all_tf_data[tf] = df
        else:
            print(f"   ⚠️ No data for {symbol} on {tf}")
    print(f"   ✅ Completed data fetching for {symbol}.")
    return all_tf_data

# ===============================

# 4. FEATURE ENGINEERING (kept original indicators but add EMA_100 and warm-up checks)

# ===============================

def add_candles(df):
    # Keep existing patterns and ensure robust NaN handling
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]

    df = df.copy()
    try:
        df["CDLENGULFING"]     = ta.CDLENGULFING(o, h, l, c)
        df["CDLHAMMER"]        = ta.CDLHAMMER(o, h, l, c)
        df["CDLSHOOTINGSTAR"]  = ta.CDLSHOOTINGSTAR(o, h, l, c)
        df["CDLMORNINGSTAR"]   = ta.CDLMORNINGSTAR(o, h, l, c)
        df["CDLEVENINGSTAR"]   = ta.CDLEVENINGSTAR(o, h, l, c)
    except Exception:
        # Fallback: fill zeros if TA-Lib fails on small series
        for col in ["CDLENGULFING", "CDLHAMMER", "CDLSHOOTINGSTAR", "CDLMORNINGSTAR", "CDLEVENINGSTAR"]:
            df[col] = 0

    # Existing Tweezer and Three Methods implementation (unchanged logic)
    df['CDLTWEEZERTOP'] = 0
    df['CDLTWEEZERBOTTOM'] = 0
    if len(df) > 1:
        for i in range(1, len(df)):
            if df.iloc[i].isnull().any() or df.iloc[i-1].isnull().any():
                continue
            if abs(df['high'].iloc[i] - df['high'].iloc[i-1]) / df['high'].iloc[i] < 0.001:
                if df['close'].iloc[i-1] > df['open'].iloc[i-1] and df['close'].iloc[i] < df['open'].iloc[i]:
                    df.loc[df.index[i], 'CDLTWEEZERTOP'] = -100
            if abs(df['low'].iloc[i] - df['low'].iloc[i-1]) / df['low'].iloc[i] < 0.001:
                if df['close'].iloc[i-1] < df['open'].iloc[i-1] and df['close'].iloc[i] > df['open'].iloc[i]:
                    df.loc[df.index[i], 'CDLTWEEZERBOTTOM'] = 100

    df['CDLRISINGTHREEMETHODS'] = 0
    df['CDLFALLINGTHREEMETHODS'] = 0
    if len(df) > 4:
        for i in range(4, len(df)):
            if df.iloc[i-4:i+1].isnull().any().any():
                continue
            # bullish
            c1_bullish = df['close'].iloc[i-4] > df['open'].iloc[i-4] and (df['close'].iloc[i-4] - df['open'].iloc[i-4]) / df['open'].iloc[i-4] > 0.01
            c2_small = abs(df['close'].iloc[i-3] - df['open'].iloc[i-3]) / df['open'].iloc[i-3] < 0.005
            c3_small = abs(df['close'].iloc[i-2] - df['open'].iloc[i-2]) / df['open'].iloc[i-2] < 0.005
            c4_small = abs(df['close'].iloc[i-1] - df['open'].iloc[i-1]) / df['open'].iloc[i-1] < 0.005
            c234_within_c1 = (df['low'].iloc[i-3] > df['low'].iloc[i-4] and df['high'].iloc[i-3] < df['high'].iloc[i-4] and
                              df['low'].iloc[i-2] > df['low'].iloc[i-4] and df['high'].iloc[i-2] < df['high'].iloc[i-4] and
                              df['low'].iloc[i-1] > df['low'].iloc[i-4] and df['high'].iloc[i-1] < df['high'].iloc[i-4])
            c5_bullish = df['close'].iloc[i] > df['open'].iloc[i] and (df['close'].iloc[i] - df['open'].iloc[i]) / df['open'].iloc[i] > 0.01
            c5_above = df['close'].iloc[i] > df['close'].iloc[i-4]
            if c1_bullish and c2_small and c3_small and c4_small and c234_within_c1 and c5_bullish and c5_above:
                df.loc[df.index[i], 'CDLRISINGTHREEMETHODS'] = 100
            # bearish
            c1_bear = df['close'].iloc[i-4] < df['open'].iloc[i-4] and (df['open'].iloc[i-4] - df['close'].iloc[i-4]) / df['open'].iloc[i-4] > 0.01
            c2_small_f = abs(df['close'].iloc[i-3] - df['open'].iloc[i-3]) / df['open'].iloc[i-3] < 0.005
            c3_small_f = abs(df['close'].iloc[i-2] - df['open'].iloc[i-2]) / df['open'].iloc[i-2] < 0.005
            c4_small_f = abs(df['close'].iloc[i-1] - df['open'].iloc[i-1]) / df['open'].iloc[i-1] < 0.005
            c234_within_c1_f = (df['low'].iloc[i-3] > df['low'].iloc[i-4] and df['high'].iloc[i-3] < df['high'].iloc[i-4] and
                                df['low'].iloc[i-2] > df['low'].iloc[i-4] and df['high'].iloc[i-2] < df['high'].iloc[i-4] and
                                df['low'].iloc[i-1] > df['low'].iloc[i-4] and df['high'].iloc[i-1] < df['high'].iloc[i-4])
            c5_bear = df['close'].iloc[i] < df['open'].iloc[i] and (df['open'].iloc[i] - df['close'].iloc[i]) / df['open'].iloc[i] > 0.01
            c5_below = df['close'].iloc[i] < df['close'].iloc[i-4]
            if c1_bear and c2_small_f and c3_small_f and c4_small_f and c234_within_c1_f and c5_bear and c5_below:
                df.loc[df.index[i], 'CDLFALLINGTHREEMETHODS'] = -100

    return df

def add_indicators(df):
    df = df.copy()
    # protect small frames
    if len(df) < 50:
        # compute what you can and return - calling functions will check NaNs properly
        df["EMA_50"] = ta.EMA(df["close"], 50)
        df["SMA_20"] = ta.SMA(df["close"], 20)
        df["RSI_14"] = ta.RSI(df["close"], 14)
        df["ATR_14"] = ta.ATR(df["high"], df["low"], df["close"], 14)
        print(f"   ⚙️ Indicators computed for small frame (len={len(df)}) - partial set")
        return df

    df["EMA_50"] = ta.EMA(df["close"], 50)
    df["EMA_100"] = ta.EMA(df["close"], 100)
    df["EMA_50_slope"] = df["EMA_50"].diff()
    df["SMA_20"] = ta.SMA(df["close"], 20)
    df["RSI_14"] = ta.RSI(df["close"], 14)
    df["ATR_14"] = ta.ATR(df["high"], df["low"], df["close"], 14)
    df["dist_from_ema"] = (df["close"] - df["EMA_50"]) / df["ATR_14"]
    df["atr_ratio"] = df["ATR_14"] / df["ATR_14"].rolling(50).mean()

    macd, macdsignal, macdhist = ta.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['MACDSIGNAL'] = macdsignal

    stoch_k, stoch_d = ta.STOCH(df['high'], df['low'], df['close'], fastk_period=14, slowk_period=3, slowd_period=3)
    df['STOCH_K'] = stoch_k
    df['STOCH_D'] = stoch_d
    print(f"   ⚙️ All indicators computed for full frame (len={len(df)})")
    return df

# ===============================

# 5. ENHANCED RULE-BASED SIGNAL ENGINE

# ===============================

def generate_signals(symbol, all_tf_data):
    """Returns: signal (1 buy, -1 sell, 0 none), entry, sl, tp, reason"""
    signal = 0
    entry_price = None
    sl = None
    tp = None
    signal_reason = ""
    bullish_score = 0
    bearish_score = 0
    factors = []

    print(f"      Calculating signals for {symbol}...")
    # --- higher timeframe trend filter: require EMA_100 direction on both 1h and 4h if available ---
    df_1h = all_tf_data.get('1hour')
    df_4h = all_tf_data.get('4hour')
    global_trend = 'neutral'

    # Require both TFs to exist and have EMA_100 available
    if df_1h is None or df_4h is None or df_1h.empty or df_4h.empty:
        signal_reason += 'Missing higher timeframe data; '
        print(f"      ❌ Skipping signal generation for {symbol}: {signal_reason}")
        return 0, None, None, None, signal_reason

    # Ensure indicators present
    for tf_df in (df_1h, df_4h):
        if 'EMA_100' not in tf_df.columns or tf_df['EMA_100'].isnull().all():
            signal_reason += 'EMA_100 not ready on higher TFs; '
            print(f"      ❌ Skipping signal generation for {symbol}: {signal_reason}")
            return 0, None, None, None, signal_reason

    last_1h = df_1h.iloc[-1]
    last_4h = df_4h.iloc[-1]

    bullish_higher = last_1h['close'] > last_1h['EMA_100'] and last_4h['close'] > last_4h['EMA_100']
    bearish_higher = last_1h['close'] < last_1h['EMA_100'] and last_4h['close'] < last_4h['EMA_100']

    if bullish_higher:
        global_trend = 'bullish'
        signal_reason += 'Global Trend: Bullish; '
    elif bearish_higher:
        global_trend = 'bearish'
        signal_reason += 'Global Trend: Bearish; '
    else:
        signal_reason += 'Global Trend: Neutral; '
        print(f"      ❌ Skipping signal generation for {symbol}: {signal_reason}")
        return 0, None, None, None, signal_reason

    # --- Base timeframe primary decision (require trend + at least one strong trigger) ---
    df_base = all_tf_data.get(BASE_TIMEFRAME)
    if df_base is None or df_base.empty:
        signal_reason += f'Missing {BASE_TIMEFRAME}; '
        print(f"      ❌ Skipping signal generation for {symbol}: {signal_reason}")
        return 0, None, None, None, signal_reason

    valid_df = df_base.dropna()

    if len(valid_df) < 2:
        signal_reason += "Not enough valid bars; "
        print(f"      ❌ Skipping signal generation for {symbol}: {signal_reason}")
        return 0, None, None, None, signal_reason

    last = valid_df.iloc[-1]
    prev = valid_df.iloc[-2]

    required_cols = ['EMA_50', 'EMA_50_slope', 'RSI_14', 'MACD', 'MACDSIGNAL', 'ATR_14', 'dist_from_ema', 'atr_ratio']
    for col in required_cols:
        if col not in df_base.columns or df_base[col].isnull().all() or pd.isna(last.get(col)):
            signal_reason += f'{col} not ready on {BASE_TIMEFRAME}; '
            print(f"      ❌ Skipping signal generation for {symbol}: {signal_reason}")
            return 0, None, None, None, signal_reason

    # Scoring Logic
    if global_trend == 'bullish':
        if prev['close'] < prev['EMA_50'] and last['close'] > last['EMA_50']:
            bullish_score += SCORE_CONFIG['BULL_EMA_CROSS']
            factors.append(f"BULL_EMA_CROSS ({SCORE_CONFIG['BULL_EMA_CROSS']})")

        if prev['MACD'] < prev['MACDSIGNAL'] and last['MACD'] > last['MACDSIGNAL']:
            bullish_score += SCORE_CONFIG['BULL_MACD_CROSS']
            factors.append(f"BULL_MACD_CROSS ({SCORE_CONFIG['BULL_MACD_CROSS']})")

        if last['RSI_14'] > 50 and last['RSI_14'] > prev['RSI_14']:
            bullish_score += SCORE_CONFIG['BULL_RSI_MOMENTUM']
            factors.append(f"BULL_RSI_MOMENTUM ({SCORE_CONFIG['BULL_RSI_MOMENTUM']})")

        if any([last.get('CDLENGULFING',0) > 0, last.get('CDLHAMMER',0) > 0, last.get('CDLRISINGTHREEMETHODS',0) > 0]):
            bullish_score += SCORE_CONFIG['BULL_STRONG_CANDLE']
            factors.append(f"BULL_STRONG_CANDLE ({SCORE_CONFIG['BULL_STRONG_CANDLE']})")

        # Penalty for high ATR regime
        if last['atr_ratio'] > 2.0: # Using 2.0 as threshold for high ATR based on common practice
            bullish_score += SCORE_CONFIG['PENALTY_HIGH_ATR_REGIME']
            factors.append(f"PENALTY_HIGH_ATR_REGIME ({SCORE_CONFIG['PENALTY_HIGH_ATR_REGIME']})")

        # Penalty if too close to EMA (or some other proximity logic, here it is based on dist_from_ema being small which might not be a penalty, but for testing, let's just make it configurable)
        if abs(last['dist_from_ema']) < 0.5: # If very close to EMA, might imply lack of clear trend continuation
            bullish_score += SCORE_CONFIG['PENALTY_LOW_DIST_FROM_EMA']
            factors.append(f"PENALTY_LOW_DIST_FROM_EMA ({SCORE_CONFIG['PENALTY_LOW_DIST_FROM_EMA']})")

        if bullish_score >= SCORE_CONFIG['MIN_BULL_SCORE_THRESHOLD']:
            signal = 1
            signal_reason += f'Bullish Signal (Score: {bullish_score}, Factors: {', '.join(factors)}); '

    elif global_trend == 'bearish':
        if prev['close'] > prev['EMA_50'] and last['close'] < last['EMA_50']:
            bearish_score += SCORE_CONFIG['BEAR_EMA_CROSS']
            factors.append(f"BEAR_EMA_CROSS ({SCORE_CONFIG['BEAR_EMA_CROSS']})")

        if prev['MACD'] > prev['MACDSIGNAL'] and last['MACD'] < last['MACDSIGNAL']:
            bearish_score += SCORE_CONFIG['BEAR_MACD_CROSS']
            factors.append(f"BEAR_MACD_CROSS ({SCORE_CONFIG['BEAR_MACD_CROSS']})")

        if last['RSI_14'] < 50 and last['RSI_14'] < prev['RSI_14']:
            bearish_score += SCORE_CONFIG['BEAR_RSI_MOMENTUM']
            factors.append(f"BEAR_RSI_MOMENTUM ({SCORE_CONFIG['BEAR_RSI_MOMENTUM']})")

        if any([last.get('CDLSHOOTINGSTAR',0) < 0, last.get('CDLEVENINGSTAR',0) < 0, last.get('CDLFALLINGTHREEMETHODS',0) < 0]):
            bearish_score += SCORE_CONFIG['BEAR_STRONG_CANDLE']
            factors.append(f"BEAR_STRONG_CANDLE ({SCORE_CONFIG['BEAR_STRONG_CANDLE']})")

        # Penalty for high ATR regime
        if last['atr_ratio'] > 2.0:
            bearish_score += SCORE_CONFIG['PENALTY_HIGH_ATR_REGIME']
            factors.append(f"PENALTY_HIGH_ATR_REGIME ({SCORE_CONFIG['PENALTY_HIGH_ATR_REGIME']})")

        # Penalty if too close to EMA
        if abs(last['dist_from_ema']) < 0.5:
            bearish_score += SCORE_CONFIG['PENALTY_LOW_DIST_FROM_EMA']
            factors.append(f"PENALTY_LOW_DIST_FROM_EMA ({SCORE_CONFIG['PENALTY_LOW_DIST_FROM_EMA']})")

        if bearish_score >= SCORE_CONFIG['MIN_BEAR_SCORE_THRESHOLD']:
            signal = -1
            signal_reason += f'Bearish Signal (Score: {bearish_score}, Factors: {', '.join(factors)}); '

    if signal == 0:
        if global_trend == 'bullish':
            signal_reason += f'No bullish signal (Score: {bullish_score}, Threshold: {SCORE_CONFIG['MIN_BULL_SCORE_THRESHOLD']}); '
        elif global_trend == 'bearish':
            signal_reason += f'No bearish signal (Score: {bearish_score}, Threshold: {SCORE_CONFIG['MIN_BEAR_SCORE_THRESHOLD']}); '
        print(f"      ℹ️ No signal generated for {symbol}: {signal_reason}")
        return 0, None, None, None, signal_reason

    # --- Entry / SL / TP using BASE_TIMEFRAME ATR (consistent) ---
    # For entry price we use the most recent close or live bid/ask if available and desired
    entry_df = None
    if CONSIDER_SPREAD:
        bid, ask = fetch_binance_bid_ask(symbol) # Updated to fetch_binance_bid_ask
        if bid is None or ask is None or pd.isna(bid) or pd.isna(ask):
            signal_reason += 'Could not fetch bid/ask; '
            print(f"      ❌ Skipping signal for {symbol}: {signal_reason}")
            return 0, None, None, None, signal_reason
        current_bid = current_ask = bid # Changed to bid for consistency with buy/sell logic
    else:
        current_bid = current_ask = last['close']

    atr = last['ATR_14']
    if pd.isna(atr) or atr <= 0:
        signal_reason += 'ATR invalid; '
        print(f"      ❌ Skipping signal for {symbol}: {signal_reason}")
        return 0, None, None, None, signal_reason

    if signal == 1:
        entry_price = current_ask
        sl_calc = entry_price - ATR_SL_FACTOR * atr
        tp_calc = entry_price + ATR_TP_FACTOR * atr
    else:
        entry_price = current_bid
        sl_calc = entry_price + ATR_SL_FACTOR * atr
        tp_calc = entry_price - ATR_TP_FACTOR * atr

    # Final checks: SL must be on logical side (structure). Enforce a maximum percentage trap.
    # Compute percent distance
    sl_pct = abs(entry_price - sl_calc) / entry_price
    if sl_pct > STOP_LOSS_MAX_PERCENTAGE:
        # cap SL distance to allowed maximum
        if signal == 1:
            sl_calc = entry_price * (1 - STOP_LOSS_MAX_PERCENTAGE)
        else:
            sl_calc = entry_price * (1 + STOP_LOSS_MAX_PERCENTAGE)
        signal_reason += f'SL capped to {STOP_LOSS_MAX_PERCENTAGE*100:.2f}%; '

    # Ensure TP respects minimum TP% (avoid TP smaller than minimum)
    tp_pct = abs(tp_calc - entry_price) / entry_price
    if tp_pct < TAKE_PROFIT_MIN_PERCENTAGE:
        if signal == 1:
            tp_calc = entry_price * (1 + TAKE_PROFIT_MIN_PERCENTAGE)
        else:
            tp_calc = entry_price * (1 - TAKE_PROFIT_MIN_PERCENTAGE)
        signal_reason += f'TP raised to min {TAKE_PROFIT_MIN_PERCENTAGE*100:.2f}%; '

    # Finalize
    sl = sl_calc
    tp = tp_calc

    signal_reason += f'Entry: {entry_price:.8f}; SL: {sl:.8f}; TP: {tp:.8f}; '
    print(f"      ✅ Signal generated for {symbol}: {signal_reason}")
    return signal, entry_price, sl, tp, signal_reason

# ===============================

# Position sizing

# ===============================

def compute_position_size(account_balance, risk_per_trade, entry_price, sl_price, min_notional=10.0):
    # risk_per_trade is fraction of account
    if entry_price == sl_price:
        return 0.0
    risk_amount = account_balance * risk_per_trade
    # dollar distance per unit = abs(entry - sl)
    per_unit_loss = abs(entry_price - sl_price)
    if per_unit_loss <= 0:
        return 0.0
    base_qty_value = risk_amount / per_unit_loss
    # If trading quote currency (USDT pairs) the required position size in base asset is base_qty_value/entry_price
    # We return both value_in_quote and qty_in_base
    value_in_quote = min_notional if base_qty_value < min_notional else base_qty_value
    qty_in_base = value_in_quote / entry_price
    return value_in_quote, qty_in_base

# ===============================

# 11. CORRELATION ENGINE (improved alignment)

# ===============================

def compute_correlation(price_dict, window=100):
    # Build aligned returns on index intersection
    closes = {}
    for symbol, df in price_dict.items():
        if df is None or df.empty or 'close' not in df.columns:
            continue
        closes[symbol] = df['close']
    if not closes:
        return pd.DataFrame()
    price_df = pd.concat(closes, axis=1).dropna()
    if price_df.empty or price_df.shape[1] < 2:
        return pd.DataFrame(np.eye(len(closes)), index=list(closes.keys()), columns=list(closes.keys()))
    returns = price_df.pct_change().dropna()
    if returns.empty:
        return pd.DataFrame(np.eye(len(closes)), index=list(closes.keys()), columns=list(closes.keys()))
    return returns.tail(window).corr()

# ===============================

# PortfolioState (unchanged but added helper to list symbols)

# ===============================

class PortfolioState:
    def __init__(self):
        self.open_trades = []

    def total_risk(self):
        return sum(t.get("risk", 0) for t in self.open_trades)

    def symbol_exposure(self, symbol):
        return sum(1 for t in self.open_trades if t.get("symbol") == symbol)

# ===============================

# can_open_trade unchanged but safety checks improved

# ===============================

def can_open_trade(trade, portfolio, corr):
    if len(portfolio.open_trades) >= PORTFOLIO_CONFIG["max_open_trades"]:
        print(f"   - Candidate {trade['symbol']} rejected: Max open trades reached.")
        asyncio.run(send_telegram(f"❌ {trade['symbol']} rejected: Max open trades reached."))
        return False
    if portfolio.total_risk() + trade["risk"] > PORTFOLIO_CONFIG["max_total_risk"]:
        print(f"   - Candidate {trade['symbol']} rejected: Total risk too high.")
        asyncio.run(send_telegram(f"❌ {trade['symbol']} rejected: Total risk too high."))
        return False
    if portfolio.symbol_exposure(trade["symbol"]) >= PORTFOLIO_CONFIG["max_per_symbol"]:
        print(f"   - Candidate {trade['symbol']} rejected: Max exposure per symbol reached.")
        asyncio.run(send_telegram(f"❌ {trade['symbol']} rejected: Max exposure per symbol reached."))
        return False
    # correlation
    if corr is not None and not corr.empty and trade['symbol'] in corr.index:
        for t in portfolio.open_trades:
            if t.get('symbol') in corr.columns:
                if corr.loc[trade['symbol'], t['symbol']] > PORTFOLIO_CONFIG["correlation_threshold"]:
                    print(f"   - Candidate {trade['symbol']} rejected due to high correlation with {t['symbol']}.")
                    asyncio.run(send_telegram(f"❌ {trade['symbol']} rejected due to high correlation with {t['symbol']}.."))
                    return False
    return True

# ===============================

# TELEGRAM (keeps placeholder token safe)

# ===============================

async def send_telegram(msg):
    # do not send when placeholder tokens still present
    if TELEGRAM_TOKEN == "6247722895:AAEN6xEpnih_OdzSYX8D7Ni2q9mc7vfpBG" or TELEGRAM_CHAT_ID == "137856163":
        print("(Telegram skipped - placeholder token)", msg)
        return
    try:
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
        print("✅ Telegram message sent successfully.")
    except Exception as e:
        print(f"❌ Error sending Telegram message: {e}")

# ===============================

# LIVE PORTFOLIO RUNNER (kept main loop but safer and better logging)

# ===============================

def portfolio_live_runner(symbols):
    portfolio = PortfolioState()
    print("🔎 Live Portfolio Loop Running")

    # pre-clean symbol list (fix obvious typos and duplicates while preserving user's set)
    symbols = [s.strip() for s in symbols if isinstance(s, str) and s.strip()]

    while True:
        print(f"\n🔄 New Cycle {datetime.now(timezone.utc).isoformat()} UTC")
        current_utc_time_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC') # Added timestamp capture
        price_data = {}
        candidates = []

        # Check and update open trades
        new_open_trades = []
        closed_count = 0
        for trade in portfolio.open_trades:
            # retrieve current price depending on spread consideration
            if CONSIDER_SPREAD:
                bid, ask = fetch_binance_bid_ask(trade['symbol']) # Updated to fetch_binance_bid_ask
                if bid is None or ask is None:
                    print(f"⚠️ No bid/ask for {trade['symbol']}. Keeping open.")
                    new_open_trades.append(trade)
                    continue
                current_bid = current_ask = bid # Changed to bid for consistency with buy/sell logic
            else:
                ohlcv_df = fetch_binance_ohlcv(trade['symbol'], TIMEFRAME, limit=1) # Updated to fetch_binance_ohlcv
                if ohlcv_df.empty:
                    print(f"⚠️ No OHLCV for {trade['symbol']}. Keeping open.")
                    new_open_trades.append(trade)
                    continue
                current_bid = current_ask = ohlcv_df['close'].iloc[-1]

            closed = False
            if trade['direction'] == 'BUY':
                if current_bid <= trade['sl']:
                    asyncio.run(send_telegram(f"{current_utc_time_str} 🛑 {trade['symbol']} BUY closed at SL. Entry {trade['entry']:.6f}, SL {trade['sl']:.6f}, Price {current_bid:.6f}"))
                    closed = True
                elif current_bid >= trade['tp']:
                    asyncio.run(send_telegram(f"{current_utc_time_str} 💰 {trade['symbol']} BUY closed at TP. Entry {trade['entry']:.6f}, TP {trade['tp']:.6f}, Price {current_bid:.6f}"))
                    closed = True
            else:
                if current_ask >= trade['sl']:
                    asyncio.run(send_telegram(f"{current_utc_time_str} 🛑 {trade['symbol']} SELL closed at SL. Entry {trade['entry']:.6f}, SL {trade['sl']:.6f}, Price {current_ask:.6f}"))
                    closed = True
                elif current_ask <= trade['tp']:
                    asyncio.run(send_telegram(f"{current_utc_time_str} 💰 {trade['symbol']} SELL closed at TP. Entry {trade['entry']:.6f}, TP {trade['tp']:.6f}, Price {current_ask:.6f}"))
                    closed = True

            if closed:
                closed_count += 1
            else:
                new_open_trades.append(trade)

        portfolio.open_trades = new_open_trades
        print(f"ℹ️ {closed_count} trades closed. {len(portfolio.open_trades)} remain open.")

        # Process symbols
        for symbol in symbols:
            print(f"\n----- Processing {symbol} -----")
            all_tf_data_for_symbol = fetch_all_timeframe_data(symbol)
            if BASE_TIMEFRAME not in all_tf_data_for_symbol or all_tf_data_for_symbol.get(BASE_TIMEFRAME).empty:
                print(f"❌ Skipping {symbol}: missing base timeframe data")
                continue

            processed = {}
            for tf, df_tf in all_tf_data_for_symbol.items():
                if df_tf.empty:
                    processed[tf] = pd.DataFrame()
                    continue
                df_tf = add_candles(df_tf)
                df_tf = add_indicators(df_tf)
                processed[tf] = df_tf

            base_df = processed.get(BASE_TIMEFRAME)
            if base_df is None or base_df.empty:
                print(f"❌ {symbol}: base df empty after processing")
                continue

            print(f"   Ready to generate signals for {symbol}...")
            signal, entry, sl_calc, tp_calc, reason = generate_signals(symbol, processed)
            if signal == 0:
                # Already printed reason within generate_signals if no signal
                continue

            # --- NEW: Early Telegram notification for discovered candidate trade ---
            asyncio.run(send_telegram(f"{current_utc_time_str} 🔔 CANDIDATE: {symbol} {'BUY' if signal == 1 else 'SELL'} (Score: {reason.split('Score: ')[1].split(',')[0]}, Reason: {reason})"))
            # -------------------------------------------------------------------

            final_sl, final_tp = sl_calc, tp_calc
            # compute position size
            value_in_quote, qty_in_base = compute_position_size(ACCOUNT_BALANCE, PORTFOLIO_CONFIG['max_risk_per_trade'], entry, final_sl)
            if qty_in_base <= 0 or value_in_quote <= 0:
                print(f"❌ {symbol}: position size computed as zero or invalid. Skipping.")
                asyncio.run(send_telegram(f"{current_utc_time_str} ❌ {symbol}: position size invalid. Entry {entry}, SL {final_sl}"))
                continue

            candidate = {
                'symbol': symbol,
                'direction': 'BUY' if signal == 1 else 'SELL',
                'entry': entry,
                'sl': final_sl,
                'tp': final_tp,
                'risk': PORTFOLIO_CONFIG['max_risk_per_trade'],
                'value_in_quote': value_in_quote,
                'qty_in_base': qty_in_base
            }

            # store base_df for correlation
            price_data[symbol] = base_df
            candidates.append((candidate, reason))

        # Evaluate candidates by correlation and portfolio rules
        if candidates:
            print(f"Evaluating {len(candidates)} candidates")
            corr_matrix = compute_correlation(price_data) if len(price_data) > 1 else pd.DataFrame()
            approved = []
            for cand, reason in candidates:
                if can_open_trade(cand, portfolio, corr_matrix):
                    approved.append(cand)
                    asyncio.run(send_telegram(f"{current_utc_time_str} ✅ TRADE APPROVED: {cand['symbol']} {cand['direction']} Entry {cand['entry']:.6f} SL {cand['sl']:.6f} TP {cand['tp']:.6f} Size(quote) {cand['value_in_quote']:.2f}"))
            portfolio.open_trades.extend(approved)
            print(f"✅ {len(approved)} trades opened")
        else:
            print("ℹ️ No trade candidates this cycle")

        # Sleep until next cycle (based on BASE_TIMEFRAME)
        sleep_seconds = TF_SECONDS.get(BASE_TIMEFRAME, 60)
        print(f"⏱ Sleeping {sleep_seconds} seconds until next cycle")
        time.sleep(min(60, sleep_seconds))  # For testing keep short; change min to sleep_seconds for production

if __name__ == '__main__':
    # Get a list of all valid trading symbols from Binance exchange info
    binance_symbols = {s['symbol'] for s in exchange_info['symbols'] if s['status'] == 'TRADING'}

    # Original list of symbols
    all_symbols = [ "TRIAUSDT", "MOLTUSDT", "ZAMAUSDT", "GWEIUSDT", "BIRBUSDT", "FIGHTUSDT", "SENTUSDT", "SKRUSDT", "ELSAUSDT", "SPORTFUNUSDT", "LITUSDT", "FOGOUSDT", "FRAXUSDT", "RIVERUSDT", "YEEUSDT", "BREVUSDT", "WHITEWHALEUSDT", "ZKPUSDT", "TAKEUSDT", "THQUSDT", "BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "DOGEUSDT", "ZECUSDT", "BCHUSDT", "XAUtUSDT", "ALGOUSDT", "ARBUSDT", "ATOMUSDT", "AVAXUSDT", "PREUSDT", "ASTERUSDT", "FILUSDT", "ETCUSDT", "LINKUSDT", "ADAUSDT", "DOTUSDT", "UNIUSDT", "LTCUSDT", "XLMUSDT", "ICPUSDT", "TONUSDT", "SUIUSDT", "BEAMXUSDT", "FLOWUSDT", "AKTUSDT", "WLDUSDT", "JASMYUSDT", "ALCHUSDT", "PENGUUSDT", "ZEREBROUSDT", "XNAUSDT", "PEPE2USDT", "AIDOGEUSDT", "SLPUSDT", "USTCUSDT", "FETUSDT", "MEMEUSDT", "BONEUSDT", "ACTUSDT", "BANUSDT", "BITCOINUSDT", "NOTUSDT", "ZKUSDT", "DOGUSDT", "AGIUSDT", "API3USDT", "ARKMUSDT", "ATHUSDT", "BBUSDT", "BICOUSDT", "BLASTUSDT", "BLURUSDT", "CVXUSDT", "DUSKUSDT", "DYMUSDT", "GASUSDT", "GLMRUSDT", "GLMUSDT", "IDUSDT", "IOUSDT", "LPTUSDT", "MAGICUSDT", "MERLUSDT", "METISUSDT", "MNTUSDT", "PEOPLEUSDT", "PIXELUSDT", "RATSUSDT", "REZUSDT" ]

    # Filter the symbols to include only those valid on Binance
    SYMBOLS = [s for s in all_symbols if s in binance_symbols]
    print(f"Filtered SYMBOLS list with {len(SYMBOLS)} valid symbols.")
    print("🚀 Improved Portfolio Engine Started")
    try:
        portfolio_live_runner(SYMBOLS)
    except KeyboardInterrupt:
        print("🛑 Stopped by user")
    except Exception as e:
        print(f"🛑 Unexpected error: {e}")