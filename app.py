import numpy as np
import pandas as pd
import ta
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="EUR/USD ANN Signal Demo", layout="wide")

FEATURES = ['ret_1', 'ret_5', 'candle_dir', 'rsi', 'macd_hist',
            'atr_pct', 'dist_sma20', 'bb_pctb']
THR, SL_MULT, RR = 0.40, 1.5, 1.5      # project parameters (chosen on validation data)


class NpScaler:
    """Same as sklearn StandardScaler.transform: (X - mean) / scale."""
    def __init__(self, mean, scale):
        self.mean, self.scale = mean, scale

    def transform(self, X):
        return (X - self.mean) / self.scale


@st.cache_resource
def load_assets():
    s = np.load('scaler.npz')
    scaler = NpScaler(s['mean'], s['scale'])
    d = np.load('weights.npz')
    W = [d[f'arr_{i}'] for i in range(6)]

    def predict(X):
        h = np.maximum(X @ W[0] + W[1], 0)             # Dense 64, ReLU
        h = np.maximum(h @ W[2] + W[3], 0)             # Dense 32, ReLU
        z = h @ W[4] + W[5]                            # Dense 3
        e = np.exp(z - z.max(axis=1, keepdims=True))   # softmax
        return e / e.sum(axis=1, keepdims=True)
    return scaler, predict


@st.cache_data(ttl=300)
def fetch_data():
    raw = yf.download('EURUSD=X', period='60d', interval='1h',
                      progress=False, auto_adjust=False)
    if raw.empty:
        return None
    raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
    d = raw.rename(columns=str.lower)[['open', 'high', 'low', 'close']].dropna()
    d.index = pd.to_datetime(d.index).tz_convert('UTC').tz_localize(None)
    now = pd.Timestamp.now(tz='UTC').tz_localize(None)
    return d[d.index + pd.Timedelta(hours=1) <= now]       # closed candles only


def build_features(d):
    """Same feature definitions as training."""
    d = d.copy()
    c = d['close']
    d['ret_1'] = c.pct_change(1)
    d['ret_5'] = c.pct_change(5)
    d['candle_dir'] = np.sign(d['close'] - d['open'])
    d['rsi'] = ta.momentum.rsi(c, window=14)
    d['macd_hist'] = ta.trend.macd_diff(c) / c
    d['atr'] = ta.volatility.average_true_range(d['high'], d['low'], c, window=14)
    d['atr_pct'] = d['atr'] / c
    d['dist_sma20'] = c / ta.trend.sma_indicator(c, 20) - 1
    d['bb_pctb'] = ta.volatility.bollinger_pband(c, window=20)
    return d


# ---------------- Page ----------------
st.title("EUR/USD Hourly ANN Signal Demo")
st.caption("Educational machine-learning project. Not financial advice. "
           "Paper/demo use only; past performance does not predict future results.")

if st.button("🔄 Refresh Market Data Now"):
    st.cache_data.clear()
    st.rerun()

try:
    scaler, predict = load_assets()
except Exception as e:
    st.error(f"Could not load the model files: {e}")
    st.stop()

with st.spinner("Downloading recent EUR/USD hourly candles..."):
    try:
        live = fetch_data()
    except Exception as e:
        live = None
        st.error(f"Data download failed: {e}")

if live is None or len(live) < 100:
    st.error("No market data returned (rate limit, library error, or market closed). "
             "Try again in a few minutes.")
    st.stop()

f = build_features(live).dropna(subset=FEATURES)
probs_all = predict(scaler.transform(f[FEATURES].values))
pred_all, conf_all = probs_all.argmax(axis=1), probs_all.max(axis=1)
sig_all = np.where((pred_all != 1) & (conf_all >= THR),
                   np.where(pred_all == 2, 'BUY', 'SELL'), 'HOLD')

# Latest closed candle
probs, signal = probs_all[-1], sig_all[-1]
row = f.iloc[-1]
entry, risk = float(row['close']), SL_MULT * float(row['atr'])
sl = tp = None
if signal == 'BUY':
    sl, tp = entry - risk, entry + RR * risk
elif signal == 'SELL':
    sl, tp = entry + risk, entry - RR * risk

c1, c2, c3, c4 = st.columns(4)
c1.metric("Signal", signal)
c2.metric("Confidence", f"{probs.max():.1%}")
c3.metric("Last closed candle (UTC)", f.index[-1].strftime("%Y-%m-%d %H:%M"))
c4.metric("Reference price", f"{entry:.5f}")

if signal == 'HOLD':
    st.info(f"No trade: the model's top probability is below the {THR:.0%} confidence "
            f"threshold, or its top class is HOLD.")
else:
    a, b, c = st.columns(3)
    a.metric("Entry (last close, approx.)", f"{entry:.5f}")
    b.metric("Stop loss (1.5 x ATR)", f"{sl:.5f}")
    c.metric("Take profit (RR 1.5)", f"{tp:.5f}")

st.subheader("Model probabilities")
st.bar_chart(pd.Series({'SELL': probs[0], 'HOLD': probs[1], 'BUY': probs[2]}))

st.subheader("Recent candles")
recent = live.tail(120)
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(recent.index, recent['close'], color='black', lw=1, label='Close')
if signal != 'HOLD':
    ax.axhline(entry, color='blue', ls='--', lw=0.9, label='Entry')
    ax.axhline(sl, color='red', ls='--', lw=0.9, label='SL')
    ax.axhline(tp, color='green', ls='--', lw=0.9, label='TP')
ax.set_ylabel("Price")
ax.legend(loc='upper left')
fig.autofmt_xdate()
st.pyplot(fig)

st.subheader("Most recent BUY/SELL signals (last ~60 days)")
hist = pd.DataFrame({'candle_start (UTC)': f.index, 'signal': sig_all,
                     'confidence': conf_all.round(3), 'close': f['close'].values})
hist = hist[hist['signal'] != 'HOLD'].tail(10).iloc[::-1]
st.dataframe(hist, hide_index=True)

with st.expander("About this demo and its limits"):
    st.write(
        "- MLP classifier (8 inputs, 64-32 hidden units, 3 outputs) trained on 2005-2015 "
        "EUR/USD hourly data.\n"
        "- Labels: future 4-candle move compared with 0.8 x ATR. Signals need confidence of at least 40%.\n"
        "- SL = 1.5 x ATR, TP = 1.5 x risk. Entry shown is the last close; a real fill would be the next open.\n"
        "- Historical backtests were weak: classification precision was only slightly above chance, "
        "and trading results were worse than random on 2015-2020 data.\n"
        "- Live data comes from Yahoo Finance and differs from the training data source."
    )
