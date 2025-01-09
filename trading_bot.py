import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import plotly.graph_objects as go
from textblob import TextBlob
import numpy as np
from datetime import datetime
import time
from transformers import pipeline
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import plotly.express as px
import openai  # Ensure OpenAI API is installed and set up
import pytz
import matplotlib.pyplot as plt

def configure_page():
    """
    Configures the Streamlit page to use full screen, expand the sidebar,
    and set the zoom level to 90%.
    """
    # Configure the page settings
    st.set_page_config(
        page_title="Real-Time Stock Analysis",
        layout="wide",  # Ensures the app takes the full screen width
        initial_sidebar_state="expanded"  # Keeps the sidebar expanded
    )

    # Inject custom CSS for zoom
    st.markdown(
        """
        <style>
            body {
                zoom: 90%; /* Set zoom to 90% */
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Call the configure_page function at the start of the script
configure_page()


def fetch_data(symbol, interval="1m", period="3mo"):
    """Fetch live or historical data for a single stock."""
    try:
        # Download data
        df = yf.download(tickers=symbol, interval=interval, period=period, progress=False)
        if df.empty:
            raise ValueError(f"No data fetched for {symbol} with interval {interval} and period {period}.")

        # Reset index and standardize Datetime column
        df.reset_index(inplace=True)
        if 'Datetime' not in df.columns:
            df.rename(columns={'index': 'Datetime'}, inplace=True)
        df['Datetime'] = pd.to_datetime(df['Datetime'])

        # Flatten multi-level columns if present
        df.columns = ['_'.join(filter(None, col)) if isinstance(col, tuple) else col for col in df.columns]

        # Rename columns to ensure compatibility
        column_mapping = {
            f'High_{symbol}': 'High',
            f'Low_{symbol}': 'Low',
            f'Close_{symbol}': 'Close',
            f'Open_{symbol}': 'Open',
            f'Volume_{symbol}': 'Volume'
        }
        df.rename(columns=column_mapping, inplace=True)

        # Ensure required columns exist
        required_columns = ['High', 'Low', 'Close', 'Open']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise KeyError(f"Missing columns for {symbol}: {', '.join(missing_columns)}")

        # Fill missing data
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)

        return df
    except Exception as e:
        st.warning(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()
    
    
def fetch_premarket_data(symbol, interval="1m", period="1d"):
    """Fetch pre/post-market data by using extended data logic."""
    try:
        ticker = yf.Ticker(symbol)
        # Fetch the regular trading data
        regular_data = ticker.history(period=period, interval=interval, prepost=False)
        # Fetch full data including pre/post-market
        extended_data = ticker.history(period=period, interval=interval, prepost=True)
        
        # Subtract regular hours to get only pre/post-market data
        prepost_data = extended_data[~extended_data.index.isin(regular_data.index)]
        prepost_data.reset_index(inplace=True)
        prepost_data['Datetime'] = pd.to_datetime(prepost_data['Datetime'])
        return prepost_data
    except Exception as e:
        st.warning(f"Error fetching pre/post-market data for {symbol}: {e}")
        return pd.DataFrame()



def calculate_indicators(df):
    """Add technical indicators to the data."""
    required_columns = ['High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.warning(f"The DataFrame is missing required columns: {', '.join(missing_columns)}")
        return df

    try:
        # Ensure 'Close' is a 1D Series
        close_series = df['Close']
        if isinstance(close_series, pd.DataFrame):
            close_series = close_series.squeeze()  # Convert to Series if needed

        # Function to calculate EMA
        def calculate_ema(series, window):
            ema = [series.iloc[0]]  # Start with the first value of the series
            alpha = 2 / (window + 1)
            for price in series.iloc[1:]:
                ema.append((price - ema[-1]) * alpha + ema[-1])
            return pd.Series(ema, index=series.index)

        # Function to calculate VWAP
        def calculate_vwap(df, window):
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            cumulative_tp_vol = (typical_price * df['Volume']).rolling(window=window).sum()
            cumulative_vol = df['Volume'].rolling(window=window).sum()
            vwap = cumulative_tp_vol / cumulative_vol
            return vwap

        # Function to calculate RSI
        def calculate_rsi(series, window):
            delta = series.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            avg_gain = gain.rolling(window=window, min_periods=1).mean()
            avg_loss = loss.rolling(window=window, min_periods=1).mean()

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi

        # Function to calculate MACD and Signal Line
        def calculate_macd(series, fast_window, slow_window, signal_window):
            ema_fast = calculate_ema(series, fast_window)
            ema_slow = calculate_ema(series, slow_window)
            macd_line = ema_fast - ema_slow
            signal_line = calculate_ema(macd_line, signal_window)
            macd_diff = macd_line - signal_line
            return macd_line, signal_line, macd_diff

        # Calculate indicators
        df['9_EMA'] = calculate_ema(close_series, window=9)
        df['21_EMA'] = calculate_ema(close_series, window=21)
        df['VWAP'] = calculate_vwap(df, window=14)
        df['RSI'] = calculate_rsi(close_series, window=14)

        macd_line, signal_line, macd_diff = calculate_macd(close_series, fast_window=12, slow_window=26, signal_window=9)
        df['MACD_Line'] = macd_line
        df['Signal_Line'] = signal_line
        df['MACD'] = macd_diff

        # Add Bollinger Bands
        bb = ta.volatility.BollingerBands(close_series, window=20, window_dev=2)
        df['BB_Upper'] = bb.bollinger_hband()
        df['BB_Lower'] = bb.bollinger_lband()

        # Add ATR
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)

        return df
    except Exception as e:
        st.error(f"Error calculating indicators: {e}")
        return df




def detect_candlestick_patterns(row, prev_row1, prev_row2):
    """Detect multiple candlestick patterns for a given row and previous rows."""
    patterns_detected = []

    # Hammer
    if ((row['High'] - row['Low']) > 3 * (row['Open'] - row['Close'])) and \
       ((row['Close'] - row['Low']) / (0.001 + row['High'] - row['Low']) > 0.6) and \
       ((row['Open'] - row['Low']) / (0.001 + row['High'] - row['Low']) > 0.6):
        patterns_detected.append("Hammer")

    # Shooting Star
    if ((row['High'] - row['Low']) > 3 * (row['Open'] - row['Close'])) and \
       ((row['High'] - row['Close']) / (0.001 + row['High'] - row['Low']) > 0.6) and \
       ((row['High'] - row['Open']) / (0.001 + row['High'] - row['Low']) > 0.6) and \
       (row['Open'] > row['Close']):
        patterns_detected.append("Shooting Star")

    # Bullish Engulfing
    if prev_row1 is not None and \
       (prev_row1['Open'] > prev_row1['Close']) and \
       (row['Close'] > row['Open']) and \
       (row['Close'] > prev_row1['Open']) and \
       (row['Open'] < prev_row1['Close']):
        patterns_detected.append("Bullish Engulfing")

    # Bearish Engulfing
    if prev_row1 is not None and \
       (prev_row1['Close'] > prev_row1['Open']) and \
       (row['Open'] > row['Close']) and \
       (row['Open'] > prev_row1['Close']) and \
       (row['Close'] < prev_row1['Open']):
        patterns_detected.append("Bearish Engulfing")

    # Doji
    if abs(row['Open'] - row['Close']) <= (row['High'] - row['Low']) * 0.1:
        patterns_detected.append("Doji")

    # Morning Star
    if prev_row2 is not None and \
       prev_row1 is not None and \
       (prev_row2['Close'] < prev_row2['Open']) and \
       (prev_row1['Close'] > prev_row1['Open']) and \
       (row['Open'] < row['Close']) and \
       (row['Close'] > (prev_row2['Open'] + prev_row2['Close']) / 2):
        patterns_detected.append("Morning Star")

    # Evening Star
    if prev_row2 is not None and \
       prev_row1 is not None and \
       (prev_row2['Close'] > prev_row2['Open']) and \
       (prev_row1['Close'] < prev_row1['Open']) and \
       (row['Open'] > row['Close']) and \
       (row['Close'] < (prev_row2['Open'] + prev_row2['Close']) / 2):
        patterns_detected.append("Evening Star")

    return patterns_detected


def calculate_market_profile(data, bins=20):
    """
    Calculate market profile (volume profile) to identify support and resistance zones.
    """
    if data.empty or 'High' not in data.columns or 'Low' not in data.columns or 'Volume' not in data.columns:
        raise ValueError("Data must contain 'High', 'Low', and 'Volume' columns.")

    # Calculate typical price levels (midpoint between High and Low)
    typical_prices = (data['High'] + data['Low']) / 2

    # Group volumes into bins based on price levels
    hist, bin_edges = np.histogram(typical_prices, bins=bins, weights=data['Volume'])

    # Identify High Volume Nodes (HVNs) and Low Volume Nodes (LVNs)
    high_volume_threshold = np.percentile(hist, 75)  # Top 25% of volumes
    low_volume_threshold = np.percentile(hist, 25)  # Bottom 25% of volumes

    # Correctly map volume bins back to price levels
    high_volume_indices = np.where(hist >= high_volume_threshold)[0]
    low_volume_indices = np.where(hist <= low_volume_threshold)[0]

    high_volume_nodes = bin_edges[high_volume_indices] if len(high_volume_indices) > 0 else []
    low_volume_nodes = bin_edges[low_volume_indices] if len(low_volume_indices) > 0 else []

    # Use the highest HVN as resistance and lowest HVN as support
    resistance = max(high_volume_nodes) if len(high_volume_nodes) > 0 else None
    support = min(high_volume_nodes) if len(high_volume_nodes) > 0 else None

    return support, resistance

def integrate_market_profile(data):
    """
    Integrate Market Profile into support and resistance calculation.
    """
    hvns, lvns = calculate_market_profile(data, bins=20)

    # Use the highest HVN as resistance and lowest HVN as support
    resistance = max(hvns) if len(hvns) > 0 else None
    support = min(hvns) if len(hvns) > 0 else None

    return support, resistance
 


def detect_patterns(data):
    """
    Detect patterns in the data.
    """
    data['Double_Top'] = detect_double_top(data)
    data['Double_Bottom'] = detect_double_bottom(data)
    data['Gap_and_Go'] = detect_gap_and_go(data)
    data['Gap_Fill'] = detect_gap_fill(data)
    return data


# Pattern Detection Functions

def detect_double_top(data):
    return (
        (data['High'] == data['High'].shift(1))
        & (data['High'].shift(-1) == data['High'])
    )


def detect_double_bottom(data):
    return (
        (data['Low'] == data['Low'].shift(1))
        & (data['Low'].shift(-1) == data['Low'])
    )


def detect_ascending_triangle(data):
    return (
        (data['High'] == data['High'].rolling(window=5).max())
        & (data['Low'] > data['Low'].shift(1))
    )


def detect_descending_triangle(data):
    return (
        (data['Low'] == data['Low'].rolling(window=5).min())
        & (data['High'] < data['High'].shift(1))
    )



def detect_gap_and_go(data):
    return (data['Open'] > data['Close'].shift(1)) & (data['Close'] > data['Open'])


def detect_gap_fill(data):
    return (data['Open'] > data['Close'].shift(1)) & (data['Close'] < data['Open'])


def generate_trade_recommendations(data):
    """
    Generate trade recommendations based on patterns and indicators.
    """
    last_row = data.iloc[-1]
    recommendations = []


    # Recommendations for Double Top
    if last_row.get('Double_Top') and last_row['RSI'] > 70:
        recommendations.append(("SELL", last_row['Close'], "Double Top with RSI Overbought"))

    # Recommendations for Double Bottom
    if last_row.get('Double_Bottom') and last_row['RSI'] < 30:
        recommendations.append(("BUY", last_row['Close'], "Double Bottom with RSI Oversold"))


    # Recommendations for Gap and Go
    if last_row.get('Gap_and_Go') and last_row['Close'] > last_row['Open']:
        recommendations.append(("BUY", last_row['Close'], "Gap and Go Pattern"))

    # Recommendations for Gap Fill
    if last_row.get('Gap_Fill') and last_row['Close'] < last_row['Open']:
        recommendations.append(("SELL", last_row['Close'], "Gap Fill Pattern"))

    if recommendations:
        return recommendations[0]  # Return the first signal
    return None, None, "No trade signals detected"



def check_trade_signal(data):
    """Check the latest trade signal with rigorous conditions, detailed reasoning, and enhanced analysis."""
    if data.empty or len(data) < 1:
        return None, None, None, "No data available to evaluate signals."

    # Ensure the required indicators exist
    required_columns = ['9_EMA', '21_EMA', 'Close', 'VWAP', 'BB_Upper', 'BB_Lower', 'RSI', 'ATR', 'High', 'Low']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        return None, None, None, f"Missing required columns: {', '.join(missing_columns)}"

    # Select the last row
    row = data.iloc[-1]

    # Extract indicator values
    ema_9 = row['9_EMA']
    ema_21 = row['21_EMA']
    close = row['Close']
    vwap = row['VWAP']
    bb_upper = row['BB_Upper']
    bb_lower = row['BB_Lower']
    rsi = row['RSI']
    atr = row['ATR']

    # Validate all indicators
    if pd.isnull([ema_9, ema_21, close, vwap, bb_upper, bb_lower, rsi, atr]).any():
        return None, None, None, "One or more indicator values are missing or invalid."

    # Define thresholds
    atr_threshold_low = 0.005 * close  # 0.5% of Close
    atr_threshold_high = 0.02 * close  # 2% of Close

    # Define weights for indicators
    weights = {
        "EMA": 0.2,
        "VWAP": 0.15,
        "Bollinger Bands": 0.1,
        "RSI": 0.1,
        "ATR": 0.1,
        "SMA": 0.2,
        "Support/Resistance": 0.1,
        "Candlestick Patterns": 0.15,
    }

    # Evaluate indicators and assign scores
    score = 0
    total_weight = sum(weights.values())
    reasoning_parts = []


    # Calculate support and resistance levels using Market Profile
    try:
        support, resistance = calculate_market_profile(data)
    except Exception as e:
        return None, None, None, f"Error calculating support and resistance: {e}"

    # Handle None values safely in further checks
    if support is None or resistance is None:
        return None, None, None, "Support or resistance could not be determined."

     # Trend Confirmation (EMA)
    if ema_9 > ema_21:
        score += weights["EMA"]
        reasoning_parts.append(
            f"- **Trend Confirmation**: Short-term EMA (9_EMA = {ema_9:.2f}) is above long-term EMA (21_EMA = {ema_21:.2f}), indicating an uptrend (+{weights['EMA']*100:.0f}%)."
        )
    else:
        reasoning_parts.append(
            f"- **Trend Confirmation**: Short-term EMA (9_EMA = {ema_9:.2f}) is below long-term EMA (21_EMA = {ema_21:.2f}), indicating a downtrend (0%)."
        )

    # Price Action Alignment with VWAP
    if close > vwap:
        score += weights["VWAP"]
        reasoning_parts.append(
            f"- **Price Action Alignment with VWAP**: Close price ({close:.2f}) is above VWAP ({vwap:.2f}), signaling bullish sentiment (+{weights['VWAP']*100:.0f}%)."
        )
    else:
        reasoning_parts.append(
            f"- **Price Action Alignment with VWAP**: Close price ({close:.2f}) is below VWAP ({vwap:.2f}), signaling bearish sentiment (0%)."
        )

    # Bollinger Bands Positioning
    if close > (bb_upper + bb_lower) / 2:
        score += weights["Bollinger Bands"]
        reasoning_parts.append(
            f"- **Bollinger Band Positioning**: Close price ({close:.2f}) is in the upper half of Bollinger Bands (BB_Upper = {bb_upper:.2f}, BB_Lower = {bb_lower:.2f}) (+{weights['Bollinger Bands']*100:.0f}%)."
        )
    else:
        reasoning_parts.append(
            f"- **Bollinger Band Positioning**: Close price ({close:.2f}) is in the lower half of Bollinger Bands (BB_Upper = {bb_upper:.2f}, BB_Lower = {bb_lower:.2f}) (0%)."
        )

    # RSI Momentum
    if 50 <= rsi <= 70:
        score += weights["RSI"]
        reasoning_parts.append(
            f"- **RSI Momentum**: RSI ({rsi:.2f}) indicates bullish momentum (+{weights['RSI']*100:.0f}%)."
        )
    elif 30 <= rsi < 50:
        reasoning_parts.append(
            f"- **RSI Momentum**: RSI ({rsi:.2f}) indicates bearish momentum (0%)."
        )
    else:
        reasoning_parts.append(
            f"- **RSI Momentum**: RSI ({rsi:.2f}) indicates neutral momentum (0%)."
        )

    # ATR Volatility Check
    if atr_threshold_low <= atr <= atr_threshold_high:
        score += weights["ATR"]
        reasoning_parts.append(
            f"- **Volatility Check (ATR)**: ATR ({atr:.2f}) is within the acceptable range ({atr_threshold_low:.2f} - {atr_threshold_high:.2f}) (+{weights['ATR']*100:.0f}%)."
        )
    else:
        reasoning_parts.append(
            f"- **Volatility Check (ATR)**: ATR ({atr:.2f}) is outside the acceptable range ({atr_threshold_low:.2f} - {atr_threshold_high:.2f}) (0%)."
        )

    # Simple Moving Average (SMA)
    sma_50 = data['Close'].rolling(window=50).mean().iloc[-1]
    if close > sma_50:
        score += weights["SMA"]
        reasoning_parts.append(
            f"- **Simple Moving Average (SMA)**: Close price ({close:.2f}) is above the 50-period SMA ({sma_50:.2f}), indicating bullish momentum (+{weights['SMA']*100:.0f}%)."
        )
    else:
        reasoning_parts.append(
            f"- **Simple Moving Average (SMA)**: Close price ({close:.2f}) is below the 50-period SMA ({sma_50:.2f}), indicating bearish momentum (0%)."
        )

    # Support and Resistance Levels
    if close > resistance:
        score += weights["Support/Resistance"]
        reasoning_parts.append(
            f"- **Support/Resistance**: Close price ({close:.2f}) is breaking above resistance ({resistance:.2f}) (+{weights['Support/Resistance']*100:.0f}%)."
        )
    elif close < support:
        score -= weights["Support/Resistance"]
        reasoning_parts.append(
            f"- **Support/Resistance**: Close price ({close:.2f}) is breaking below support ({support:.2f}) (-{weights['Support/Resistance']*100:.0f}%)."
        )
    else:
        reasoning_parts.append(
            f"- **Support/Resistance**: Close price ({close:.2f}) is within support ({support:.2f}) and resistance ({resistance:.2f}) range (0%)."
        )
        
        # Candlestick Patterns
    patterns = {
        "Hammer": row.get('Hammer', False),
        "Shooting Star": row.get('Shooting Star', False),
        "Bullish Engulfing": row.get('Bullish Engulfing', False),
        "Bearish Engulfing": row.get('Bearish Engulfing', False),
        "Doji": row.get('Doji', False),
        "Morning Star": row.get('Morning Star', False),
        "Evening Star": row.get('Evening Star', False),
    }
    
    for pattern, detected in patterns.items():
        if detected:
            score += weights["Candlestick Patterns"]
            reasoning_parts.append(f"- **Candlestick Pattern Detected**: {pattern} indicates potential trend reversal (+{weights['Candlestick Patterns']*100:.0f}%).")



    # Calculate final weighted score
    weighted_score = score / total_weight  # Normalize score to a 0-1 range

    # Determine signal based on weighted score
    if weighted_score >= 0.7:  # Threshold for a Buy signal
        reasoning = "Buy signal generated because the following conditions were met:\n" + "\n".join(reasoning_parts)
        return "BUY", row['Datetime'], close, reasoning
    elif weighted_score <= 0.3:  # Threshold for a Sell signal
        reasoning = "Sell signal generated because the following conditions were met:\n" + "\n".join(reasoning_parts)
        return "SELL", row['Datetime'], close, reasoning
    else:
        reasoning = "No signal generated because the overall weighted score did not meet the thresholds:\n" + "\n".join(reasoning_parts)
        return None, None, None, reasoning




def plot_interactive_chart(
    ticker, 
    data, 
    signal=None, 
    signal_time=None, 
    signal_price=None, 
    show_fibonacci=False, 
    show_ema=False, 
    show_macd=False, 
    show_vwap=False, 
    show_ichimoku=False, 
    show_bollinger=False, 
    show_sma=False, 
    show_candlestick_patterns=False,
    show_support=False,  
    show_resistance=False  
):
    """
    Plot an interactive chart with Plotly, with optional indicators and buy/sell signals.
    """
    import plotly.graph_objects as go

    market_tz = pytz.timezone("US/Eastern")

    # Localize and convert timestamps
    if data['Datetime'].dt.tz is None:
        data['Datetime'] = data['Datetime'].dt.tz_localize(market_tz)
    else:
        data['Datetime'] = data['Datetime'].dt.tz_convert(market_tz)

    fig = go.Figure()

    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=data['Datetime'],
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Candlestick'
    ))

    # Add main price line
    fig.add_trace(go.Scatter(x=data['Datetime'], y=data['Close'], mode='lines', name='Price'))

    # Add buy/sell markers
    if signal and signal_time and signal_price:
        fig.add_trace(go.Scatter(
            x=[signal_time],
            y=[signal_price],
            mode='markers+text',
            name=f'{signal} Signal',
            text=[signal],
            textposition='top center',
            marker=dict(color='green' if signal == "BUY" else 'red', size=12)
        ))

    # Add EMA indicators
    if show_ema:
        if '9_EMA' in data.columns and '21_EMA' in data.columns:
            fig.add_trace(go.Scatter(
                x=data['Datetime'], y=data['9_EMA'],
                mode='lines', name='9 EMA', line=dict(dash='dot', color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=data['Datetime'], y=data['21_EMA'],
                mode='lines', name='21 EMA', line=dict(dash='dot', color='purple')
            ))

    # Add MACD
    if show_macd and 'MACD' in data.columns:
        fig.add_trace(go.Scatter(
            x=data['Datetime'], y=data['MACD'],
            mode='lines', name='MACD', line=dict(color='brown')
        ))

    # Add VWAP
    if show_vwap and 'VWAP' in data.columns:
        fig.add_trace(go.Scatter(
            x=data['Datetime'], y=data['VWAP'],
            mode='lines', name='VWAP', line=dict(color='gold')
        ))

    # Add Bollinger Bands
    if show_bollinger and 'BB_Upper' in data.columns and 'BB_Lower' in data.columns:
        fig.add_trace(go.Scatter(
            x=data['Datetime'], y=data['BB_Upper'], 
            mode='lines', name='BB Upper', line=dict(color='orange', dash='dot')
        ))
        fig.add_trace(go.Scatter(
            x=data['Datetime'], y=data['BB_Lower'], 
            mode='lines', name='BB Lower', line=dict(color='orange', dash='dot')
        ))

    # Add SMA
    if show_sma:
        sma_50 = data['Close'].rolling(window=50).mean()
        fig.add_trace(go.Scatter(
            x=data['Datetime'], y=sma_50, 
            mode='lines', name='50 SMA', line=dict(color='pink', dash='dash')
        ))

    # Add support and resistance lines
    if show_support or show_resistance:
        support, resistance = calculate_market_profile(data)
        if show_support and support is not None:
            fig.add_hline(
                y=support,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Support: {support:.2f}",
                annotation_position="bottom right"
            )
        if show_resistance and resistance is not None:
            fig.add_hline(
                y=resistance,
                line_dash="dash",
                line_color="green",
                annotation_text=f"Resistance: {resistance:.2f}",
                annotation_position="top right"
            )



    # Add Fibonacci retracements
    if show_fibonacci:
        fibonacci_levels = calculate_fibonacci_retracements(data)
        for level, value in fibonacci_levels.items():
            if isinstance(value, pd.Series):
                value = value.iloc[0]  # Convert Series to scalar if needed
            fig.add_hline(
                y=value,
                line_dash="dash",
                line_color="orange",
                opacity=0.8
            )
            fig.add_annotation(
                xref="paper",
                y=value,
                x=1.01,  # Place annotation outside the plot area
                text=f"{level}",
                showarrow=False,
                font=dict(size=10, color="orange")
            )

    # Add Ichimoku Clouds
    if show_ichimoku:
        if 'Leading Span A' in data.columns and 'Leading Span B' in data.columns:
            fig.add_trace(go.Scatter(
                x=data['Datetime'], y=data['Leading Span A'],
                mode='lines', name='Leading Span A', line=dict(color='green')
            ))
            fig.add_trace(go.Scatter(
                x=data['Datetime'], y=data['Leading Span B'],
                mode='lines', name='Leading Span B', line=dict(color='red')
            ))

    # Highlight Candlestick Patterns
    if show_candlestick_patterns:
        patterns = ['Hammer', 'Shooting Star', 'Bullish Engulfing', 'Bearish Engulfing', 'Doji', 'Morning Star', 'Evening Star']
        for pattern in patterns:
            if pattern in data.columns:
                pattern_indices = data.index[data[pattern] == True]
                for idx in pattern_indices:
                    fig.add_annotation(
                        x=data.loc[idx, 'Datetime'],
                        y=data.loc[idx, 'High'] if pattern != 'Shooting Star' else data.loc[idx, 'Low'],
                        text=pattern,
                        showarrow=True,
                        arrowhead=1,
                        arrowcolor='blue',
                        font=dict(size=10, color="blue")
                    )

    # Adjust the layout
    fig.update_layout(
        title=f"{ticker} Real-Time Stock Analysis",
        xaxis_title="Datetime",
        yaxis_title="Price",
        template="plotly_dark",
        height=600,
        xaxis_rangeslider_visible=False  # Hide range slider
    )

    # Automatically adjust Y-axis range based on data
    fig.update_yaxes(
        range=[
            data['Low'].min() * 0.99,  # Slight padding for visibility
            data['High'].max() * 1.01
        ]
    )

    return fig




def plot_dynamic_chart(ticker, interval, refresh_rate, show_fibonacci, show_ema, show_macd, show_vwap, show_ichimoku):
    """Fetch data and dynamically update the chart for the given ticker."""
    # Create placeholders for dynamic updates
    placeholder_signal = st.empty()
    placeholder_chart = st.empty()


    while True:
        # Fetch Data
        data = fetch_data(ticker, interval=interval, period="1d")

        if data.empty:
            st.warning(f"No data available for {ticker}.")
            break

        # Calculate Indicators
        data = calculate_indicators(data)

        # Check Trade Signals
        signal, signal_time, signal_price = check_trade_signal(data)

        # Display Trade Signals
        with placeholder_signal:
            if signal == "BUY":
                st.success(f"**BUY Signal detected for {ticker} at {signal_price:.2f} on {signal_time}**")
            elif signal == "SELL":
                st.error(f"**SELL Signal detected for {ticker} at {signal_price:.2f} on {signal_time}**")
            else:
                st.info("No signals detected.")

        # Plot Interactive Chart
        fig = plot_interactive_chart(
            ticker, data, signal, signal_time, signal_price,
            show_fibonacci, show_ema, show_macd, show_vwap, show_ichimoku
        )


        # Wait for refresh
        time.sleep(refresh_rate)



def enhanced_options_flow(ticker):
    """Fetch and analyze options flow for the given ticker."""
    try:
        options_data = yf.Ticker(ticker).option_chain()
        calls = options_data.calls
        puts = options_data.puts

        # Calculate put/call ratio
        total_calls = calls['openInterest'].sum()
        total_puts = puts['openInterest'].sum()
        put_call_ratio = total_puts / total_calls if total_calls > 0 else 0

        st.subheader(f"📈 Enhanced Options Flow for {ticker}")
        st.write(f"**Put/Call Ratio:** {put_call_ratio:.2f}")
        st.write("📉 **Calls**")
        st.dataframe(calls)
        st.write("📉 **Puts**")
        st.dataframe(puts)
    except Exception as e:
        st.warning(f"Error fetching enhanced options flow for {ticker}: {e}")
        
# Fibonacci Retracements
def calculate_fibonacci_retracements(df):
    """Calculate Fibonacci retracements for the given DataFrame."""
    high = df['High'].max()
    low = df['Low'].min()
    levels = {
        "0.0%": high,
        "23.6%": high - 0.236 * (high - low),
        "38.2%": high - 0.382 * (high - low),
        "50.0%": (high + low) / 2,
        "61.8%": high - 0.618 * (high - low),
        "100.0%": low,
    }
    return levels

# Ichimoku Clouds
def calculate_ichimoku_clouds(df):
    """Calculate Ichimoku Clouds for the given DataFrame."""
    high_series = df['High']
    low_series = df['Low']
    close_series = df['Close']

    # Ichimoku parameters
    nine_period_high = high_series.rolling(window=9).max()
    nine_period_low = low_series.rolling(window=9).min()
    conversion_line = (nine_period_high + nine_period_low) / 2

    twenty_six_period_high = high_series.rolling(window=26).max()
    twenty_six_period_low = low_series.rolling(window=26).min()
    base_line = (twenty_six_period_high + twenty_six_period_low) / 2

    leading_span_a = ((conversion_line + base_line) / 2).shift(26)
    leading_span_b = ((high_series.rolling(window=52).max() + low_series.rolling(window=52).min()) / 2).shift(26)

    df['Conversion Line'] = conversion_line
    df['Base Line'] = base_line
    df['Leading Span A'] = leading_span_a
    df['Leading Span B'] = leading_span_b

    return df



# Main Streamlit Application
def main():
    # Sidebar for user inputs
    tickers_input = st.sidebar.text_area(
        "Enter Tickers (comma-separated, e.g., AAPL, MSFT, TSLA)",
        value="AAPL, MSFT, TSLA",
        key="tickers_input_main"
    )
    st.title("Real-Time Stock Analysis")
    st.sidebar.title("Settings")
    interval = st.sidebar.selectbox("Interval", ["1m", "5m", "15m", "1h", "1d"], index=0)
    data_refresh_rate = st.sidebar.slider("Data Refresh Rate (seconds)", min_value=1, max_value=60, value=2)
    chart_refresh_rate = st.sidebar.slider("Chart Refresh Rate (seconds)", min_value=5, max_value=300, value=10)
    show_fibonacci = st.sidebar.checkbox("Show Fibonacci Retracements", value=True)
    show_ichimoku = st.sidebar.checkbox("Show Ichimoku Clouds", value=True)
    show_ema = st.sidebar.checkbox("Show EMA Indicators", value=True)
    show_vwap = st.sidebar.checkbox("Show VWAP", value=True)
    show_bollinger = st.sidebar.checkbox("Show Bollinger Bands", value=True)
    show_sma = st.sidebar.checkbox("Show Simple Moving Average (SMA)", value=True)
    show_support_resistance = st.sidebar.checkbox("Show Support/Resistance Levels", value=True)
    show_candlestick_patterns = st.sidebar.checkbox("Show Candlestick Patterns", value=True)

    # Initialize session state for refresh counters and last chart update timestamps
    if "refresh_counters" not in st.session_state:
        st.session_state.refresh_counters = {}
    if "last_chart_update" not in st.session_state:
        st.session_state.last_chart_update = {}

    # Validate and process tickers
    if tickers_input:
        selected_tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]
        tabs = st.tabs(selected_tickers)  # Create one tab per ticker

        for i, ticker in enumerate(selected_tickers):
            with tabs[i]:  # Each tab corresponds to one ticker
                st.subheader(f"{ticker} - Real-Time Chart")
                placeholder_chart = st.empty()  # Placeholder for the chart
                placeholder_signal = st.empty()  # Placeholder for signal alerts

                # Initialize session state for ticker
                if ticker not in st.session_state.refresh_counters:
                    st.session_state.refresh_counters[ticker] = 0
                if ticker not in st.session_state.last_chart_update:
                    st.session_state.last_chart_update[ticker] = time.time()

                # Main loop for updates
                while True:
                    # Fetch data
                    data = fetch_data(ticker, interval=interval, period="1d")
                    if data.empty:
                        st.warning(f"No data available for {ticker}.")
                        break

                    # Update Data and Indicators
                    try:
                        data = calculate_indicators(data)
                    except Exception as e:
                        st.error(f"Error calculating indicators for {ticker}: {e}")
                        break

                    # Real-Time Signal Alerts
                    try:
                        signal, signal_time, signal_price, reasoning = check_trade_signal(data)
                        if signal == "BUY":
                            placeholder_signal.success(
                                f"**BUY Signal detected for {ticker} at {signal_price:.2f} on {signal_time}**\n\n"
                                f"Reasoning:\n{reasoning}"
                            )
                        elif signal == "SELL":
                            placeholder_signal.error(
                                f"**SELL Signal detected for {ticker} at {signal_price:.2f} on {signal_time}**\n\n"
                                f"Reasoning:\n{reasoning}"
                            )
                        else:
                            placeholder_signal.info(f"No signals detected.\n\nReasoning:\n{reasoning}")
                    except Exception as e:
                        placeholder_signal.error(f"Error checking trade signals: {e}")

                    # Check if it's time to refresh the chart
                    current_time = time.time()
                    if current_time - st.session_state.last_chart_update[ticker] >= chart_refresh_rate:
                        # Increment the refresh counter
                        st.session_state.refresh_counters[ticker] += 1
                        refresh_count = st.session_state.refresh_counters[ticker]

                        # Generate a unique key using the ticker and refresh count
                        chart_key = f"{ticker}_chart_{refresh_count}"

                        # Plot Interactive Chart
                        support, resistance = calculate_market_profile(data)  # Ensure support and resistance are calculated before plotting

                        fig = plot_interactive_chart(
                            ticker=ticker,
                            data=data,
                            show_fibonacci=show_fibonacci,
                            show_ema=show_ema,
                            show_vwap=show_vwap,
                            show_ichimoku=show_ichimoku,
                            show_bollinger=show_bollinger,
                            show_sma=show_sma,
                            show_support=True,  # Enable support line
                            show_resistance=True,  # Enable resistance line
                            )

                        
                        placeholder_chart.plotly_chart(fig, use_container_width=True, key=chart_key)

                        # Update the last chart update timestamp
                        st.session_state.last_chart_update[ticker] = current_time

                    # Wait for data refresh
                    time.sleep(data_refresh_rate)


if __name__ == "__main__":
    main()








