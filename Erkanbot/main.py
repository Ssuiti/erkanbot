import os
import time
import datetime
import requests
import numpy as np
import pandas as pd
import yfinance as yf

# Ayarlar (Render Environment Variables veya buraya doğrudan yazılabilir)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "BURAYA_BOT_TOKEN_YAZIN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "BURAYA_CHAT_ID_YAZIN")

SYMBOL = "GC=F"
TIMEFRAME = "5m"
HTF_TIMEFRAME = "1h"

SL_BUF = 0.3
MIN_SL_A = 0.8
MAX_SL_A = 2.5
TP1_R = 1.0
TP2_R = 2.0
SESS_START = 7
SESS_END = 17
RG_LO = 0.6
RG_HI = 2.5

REV_MIN = 4
REV_ADX = 35
RSI_OS = 30
PUL_ADX = 20
PB_BARS = 6
PB_RSI = 45
SWEEP_MIN = 0.1
ASIA_MIN_R = 1.0
ASIA_MAX_R = 12.0

def send_telegram(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Telegram hatasi: {e}")

def calc_atr(df, length=14):
    high, low, close = df['High'], df['Low'], df['Close'].shift(1)
    tr = pd.concat([high - low, (high - close).abs(), (low - close).abs()], axis=1).max(axis=1)
    return tr.rolling(length).mean()

def calc_rsi(series, length=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    return 100 - (100 / (1 + (avg_gain / (avg_loss + 1e-9))))

def calc_adx(df, length=14):
    high, low, close = df['High'], df['Low'], df['Close']
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    up = high - high.shift(1)
    down = low.shift(1) - low
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr_s = pd.Series(tr).ewm(alpha=1/length, adjust=False).mean()
    p_dm_s = pd.Series(plus_dm, index=df.index).ewm(alpha=1/length, adjust=False).mean()
    m_dm_s = pd.Series(minus_dm, index=df.index).ewm(alpha=1/length, adjust=False).mean()
    dx = 100 * (abs((100 * p_dm_s / (tr_s + 1e-9)) - (100 * m_dm_s / (tr_s + 1e-9))) / 
                ((100 * p_dm_s / (tr_s + 1e-9)) + (100 * m_dm_s / (tr_s + 1e-9)) + 1e-9))
    return dx.ewm(alpha=1/length, adjust=False).mean()

def calc_bb(series, length=20, mult=2.0):
    mid = series.rolling(length).mean()
    std = series.rolling(length).std()
    return mid, mid + mult * std, mid - mult * std

def calc_stoch_rsi(rsi_series, length=14, k_len=3, d_len=3):
    min_r = rsi_series.rolling(length).min()
    max_r = rsi_series.rolling(length).max()
    raw = (rsi_series - min_r) / (max_r - min_r + 1e-9) * 100
    k = raw.rolling(k_len).mean()
    d = k.rolling(d_len).mean()
    return k, d

def calc_vwap(df):
    typ = (df['High'] + df['Low'] + df['Close']) / 3
    vol = df['Volume']
    cum_v = vol.groupby(df['Datetime'].dt.date).cumsum()
    cum_p = (typ * vol).groupby(df['Datetime'].dt.date).cumsum()
    return (cum_p / (cum_v + 1e-9)).fillna(typ)

def fetch_candles(symbol, interval, period="5d"):
    try:
        t = yf.Ticker(symbol)
        df = t.history(period=period, interval=interval)
        if df.empty: return None
        df = df.reset_index()
        col = 'Datetime' if 'Datetime' in df.columns else 'Date'
        df['Datetime'] = pd.to_datetime(df[col], utc=True)
        return df
    except Exception as e:
        print(f"Veri hatasi: {e}")
        return None

def check_signals(prev_bar_time):
    df = fetch_candles(SYMBOL, TIMEFRAME)
    if df is None or len(df) < 100:
        return prev_bar_time

    i = len(df) - 2
    c_bar = df.iloc[i]
    p_bar = df.iloc[i-1]
    curr_time = c_bar['Datetime']

    if prev_bar_time == curr_time:
        return prev_bar_time

    df_htf = fetch_candles(SYMBOL, HTF_TIMEFRAME)
    htf_bias = 0
    if df_htf is not None and len(df_htf) > 50:
        e20 = df_htf['Close'].ewm(span=20, adjust=False).mean()
        e50 = df_htf['Close'].ewm(span=50, adjust=False).mean()
        diff = e20.iloc[-2] - e50.iloc[-2]
        htf_bias = 1 if diff > 0 else (-1 if diff < 0 else 0)

    df['atr'] = calc_atr(df, 14)
    df['atr_avg'] = df['atr'].rolling(100).mean()
    df['rsi'] = calc_rsi(df['Close'], 14)
    df['bb_mid'], df['bb_up'], df['bb_lo'] = calc_bb(df['Close'], 20, 2.0)
    df['st_k'], df['st_d'] = calc_stoch_rsi(df['rsi'], 14, 3, 3)
    df['adx'] = calc_adx(df, 14)
    df['ema21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['ema55'] = df['Close'].ewm(span=55, adjust=False).mean()
    df['ema200'] = df['Close'].ewm(span=200, adjust=False).mean()
    df['vwap'] = calc_vwap(df)

    c_bar = df.iloc[i]
    p_bar = df.iloc[i-1]

    hr = c_bar['Datetime'].hour + c_bar['Datetime'].minute / 60.0
    if not ((hr >= SESS_START and hr < SESS_END) and (RG_LO <= (c_bar['atr'] / (c_bar['atr_avg'] + 1e-9)) <= RG_HI)):
        return curr_time

    asia = df[(df['Datetime'].dt.date == c_bar['Datetime'].date()) & (df['Datetime'].dt.hour < 7)]
    asia_hi = asia['High'].max() if len(asia) > 0 else np.nan
    asia_lo = asia['Low'].min() if len(asia) > 0 else np.nan

    rng = max(c_bar['High'] - c_bar['Low'], 0.01)
    bull_bar = c_bar['Close'] > c_bar['Open']
    bear_bar = c_bar['Close'] < c_bar['Open']
    bull_w = (min(c_bar['Open'], c_bar['Close']) - c_bar['Low']) / rng >= 0.4
    bear_w = (c_bar['High'] - max(c_bar['Open'], c_bar['Close'])) / rng >= 0.4

    # 1) Donus
    c1L = c_bar['Low'] <= c_bar['bb_lo'] or p_bar['Low'] <= p_bar['bb_lo']
    c1S = c_bar['High'] >= c_bar['bb_up'] or p_bar['High'] >= p_bar['bb_up']
    c2L = df['rsi'].iloc[i-2:i+1].min() <= RSI_OS
    c2S = df['rsi'].iloc[i-2:i+1].max() >= (100 - RSI_OS)
    c3L = df['st_k'].iloc[i-2:i+1].min() <= 20 and c_bar['st_k'] > c_bar['st_d']
    c3S = df['st_k'].iloc[i-2:i+1].max() >= 80 and c_bar['st_k'] < c_bar['st_d']
    prev_lo = df['Low'].iloc[i-22:i-2].min()
    prev_hi = df['High'].iloc[i-22:i-2].max()
    lo3 = df['Low'].iloc[i-2:i+1].min()
    hi3 = df['High'].iloc[i-2:i+1].max()
    c4L = lo3 < prev_lo and c_bar['Close'] > prev_lo
    c4S = hi3 > prev_hi and c_bar['Close'] < prev_hi
    c5L = bull_w or ((min(p_bar['Open'], p_bar['Close']) - p_bar['Low']) / max(p_bar['High'] - p_bar['Low'], 0.01) >= 0.4)
    c5S = bear_w or ((p_bar['High'] - max(p_bar['Open'], p_bar['Close'])) / max(p_bar['High'] - p_bar['Low'], 0.01) >= 0.4)

    revL = c_bar['adx'] <= REV_ADX and bull_bar and c_bar['Close'] > p_bar['Close'] and sum([c1L, c2L, c3L, c4L, c5L]) >= REV_MIN and (c1L or c4L)
    revS = c_bar['adx'] <= REV_ADX and bear_bar and c_bar['Close'] < p_bar['Close'] and sum([c1S, c2S, c3S, c4S, c5S]) >= REV_MIN and (c1S or c4S)

    # 2) Trend Cekilme
    tchL = df['Low'].iloc[i-PB_BARS:i].apply(lambda x: x <= c_bar['ema21'] or x <= c_bar['vwap']).any()
    tchS = df['High'].iloc[i-PB_BARS:i].apply(lambda x: x >= c_bar['ema21'] or x >= c_bar['vwap']).any()
    up_t = c_bar['ema21'] > c_bar['ema55'] and c_bar['Close'] > c_bar['ema200'] and htf_bias == 1
    dn_t = c_bar['ema21'] < c_bar['ema55'] and c_bar['Close'] < c_bar['ema200'] and htf_bias == -1
    pulL = c_bar['adx'] >= PUL_ADX and up_t and tchL and df['rsi'].iloc[i-PB_BARS:i].min() <= PB_RSI and bull_bar and c_bar['Close'] > c_bar['ema21'] and c_bar['Close'] > p_bar['High'] and c_bar['rsi'] > p_bar['rsi']
    pulS = c_bar['adx'] >= PUL_ADX and dn_t and tchS and df['rsi'].iloc[i-PB_BARS:i].max() >= (100 - PB_RSI) and bear_bar and c_bar['Close'] < c_bar['ema21'] and c_bar['Close'] < p_bar['Low'] and c_bar['rsi'] < p_bar['rsi']

    # 3) Asya Supurme
    swp_l = False
    swp_s = False
    if pd.notna(asia_hi) and pd.notna(asia_lo) and (ASIA_MIN_R <= ((asia_hi - asia_lo) / (c_bar['atr'] + 1e-9)) <= ASIA_MAX_R):
        swp_l = (lo3 < asia_lo - c_bar['atr'] * SWEEP_MIN) and (c_bar['Close'] > asia_lo) and bull_bar
        swp_s = (hi3 > asia_hi + c_bar['atr'] * SWEEP_MIN) and (c_bar['Close'] < asia_hi) and bear_bar

    direction, strat_name, structure = 0, "", 0.0
    if revL: direction, strat_name, structure = 1, "Donus", lo3
    elif revS: direction, strat_name, structure = -1, "Donus", hi3
    elif pulL: direction, strat_name, structure = 1, "Trend Cekilme", df['Low'].iloc[i-4:i+1].min()
    elif pulS: direction, strat_name, structure = -1, "Trend Cekilme", df['High'].iloc[i-4:i+1].max()
    elif swp_l: direction, strat_name, structure = 1, "Asya Supurme", lo3
    elif swp_s: direction, strat_name, structure = -1, "Asya Supurme", hi3

    if direction != 0:
        ent = c_bar['Close']
        atr_v = c_bar['atr']
        sl_s = structure - atr_v * SL_BUF if direction == 1 else structure + atr_v * SL_BUF
        dst = min(max(abs(ent - sl_s), atr_v * MIN_SL_A), atr_v * MAX_SL_A)
        sl = ent - direction * dst
        tp1 = ent + direction * dst * TP1_R
        tp2 = ent + direction * dst * TP2_R

        action = "🟢 Erkan bey alalim" if direction == 1 else "🔴 Erkan bey satalim"
        msg = (
            f"<b>{action}</b>\n\n"
            f"📌 <b>Strateji:</b> {strat_name}\n"
            f"📊 <b>Varlik:</b> XAUUSD (Altin)\n"
            f"💵 <b>Giris:</b> {ent:.2f}\n"
            f"🛑 <b>SL:</b> {sl:.2f} (-{dst:.2f})\n"
            f"🎯 <b>TP1:</b> {tp1:.2f} (+{dst * TP1_R:.2f})\n"
            f"🎯 <b>TP2:</b> {tp2:.2f} (+{dst * TP2_R:.2f})\n"
            f"⏰ <b>Zaman:</b> {curr_time.strftime('%H:%M UTC')}"
        )
        send_telegram(msg)
        print(f"[{datetime.datetime.now()}] Sinyal gonderildi: {strat_name}")

    return curr_time

if __name__ == "__main__":
    print("Render botu aktif!")
    send_telegram("🚀 <b>ErkanBot Render Uzerinde Baslatildi!</b>\n7/24 kesintisiz XAUUSD taraniyor...")
    last_bar = None
    while True:
        try:
            last_bar = check_signals(last_bar)
            time.sleep(20)
        except Exception as e:
            print(f"Dongu hatasi: {e}")
            time.sleep(20)
