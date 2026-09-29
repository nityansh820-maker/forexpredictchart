import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from tensorflow.keras.models import load_model

st.set_page_config(page_title="EUR/USD Forex ANN Dashboard (15M)", layout="wide")

st.title("📈 EUR/USD Live Forex Trading Signal Assistant (15-Min)")
st.caption("Powered by Artificial Neural Network (Keras MLP)")

# Auto-refresh page every 60 seconds automatically
components.html(
    """
    <script>
        setTimeout(function() {
            window.parent.location.reload();
        }, 60000);
    </script>
    """,
    height=0
)

# Refresh Button
if st.button("🔄 Refresh Market Data Now"):
    st.rerun()

# 1. Load Pre-Trained Model & Scaler
@st.cache_resource
def load_assets():
    scaler = joblib.load('scaler.pkl')
    model  = load_model('forex_ann.keras')
    return scaler, model

scaler, model = load_assets()

# 2. Indicator Calculation Pipeline
def calculate_indicators(df):
    df = df.copy()

    close = df['Close'].squeeze()
    open_p = df['Open'].squeeze()
    high = df['High'].squeeze()
    low = df['Low'].squeeze()

    # RSI 14
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD Histogram
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df['macd_hist'] = (macd - signal) / close

    # ATR & ATR%
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    df['atr_pct'] = df['atr'] / close

    # Distance to SMA 20
    sma20 = close.rolling(20).mean()
    df['dist_sma20'] = (close - sma20) / sma20

    # Bollinger Bands %B
    std20 = close.rolling(20).std()
    upper = sma20 + (2 * std20)
    lower = sma20 - (2 * std20)
    df['bb_pctb'] = (close - lower) / (upper - lower)

    # Returns & Direction
    df['ret_1'] = close.pct_change(1)
    df['ret_5'] = close.pct_change(5)
    df['direction'] = np.where(close > open_p, 1, -1)

    return df

# 3. Fetch Live 15-Minute Market Data
st.info("Fetching latest live 15-minute EUR/USD candles...")
data = yf.download('EURUSD=X', period='5d', interval='15m', progress=False)

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

df = calculate_indicators(data).dropna()

# 4. Neural Network Inference on Latest Candle
features = ['rsi', 'macd_hist', 'atr_pct', 'dist_sma20', 'bb_pctb', 'ret_1', 'ret_5', 'direction']
X_live = df[features].iloc[-1:].values
X_scaled = scaler.transform(X_live)

probs = model.predict(X_scaled)
classes = ['SELL', 'HOLD', 'BUY']
pred_idx = int(np.argmax(probs))
pred_class = classes[pred_idx]
confidence = float(np.max(probs) * 100)

latest = df.iloc[-1]

close_val = latest['Close']
current_price = float(close_val.iloc) if isinstance(close_val, (pd.Series, np.ndarray)) else float(close_val)

atr_raw = latest['atr']
atr_val = float(atr_raw.iloc) if isinstance(atr_raw, (pd.Series, np.ndarray)) else float(atr_raw)

# Risk Management calculation
sl = None
tp = None
if pred_class == 'BUY':
    sl = current_price - (1.5 * atr_val)
    tp = current_price + (2.25 * atr_val)
elif pred_class == 'SELL':
    sl = current_price + (1.5 * atr_val)
    tp = current_price - (2.25 * atr_val)

# 5. Display Metric Cards
c1, c2, c3, c4 = st.columns(4)
c1.metric(label="Live Rate (EUR/USD)", value=f"${current_price:.5f}")
c2.metric(label="ANN Prediction (15M)", value=str(pred_class), delta=f"{confidence:.1f}% Prob.")
c3.metric(label="Stop Loss (SL)", value=f"${sl:.5f}" if sl is not None else "N/A")
c4.metric(label="Take Profit (TP)", value=f"${tp:.5f}" if tp is not None else "N/A")

# 6. Interactive Plotly Candlestick Chart (Last 40 15M candles)
chart_df = df.tail(40)
fig = go.Figure(data=[go.Candlestick(
    x=chart_df.index,
    open=chart_df['Open'].squeeze(),
    high=chart_df['High'].squeeze(),
    low=chart_df['Low'].squeeze(),
    close=chart_df['Close'].squeeze(),
    name="EUR/USD 15M"
)])
fig.update_layout(
    title="Recent 15-Minute Candlestick Price Movement",
    xaxis_rangeslider_visible=False,
    template="plotly_dark",
    height=450
)
st.plotly_chart(fig, use_container_width=True)
