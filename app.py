import os
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
