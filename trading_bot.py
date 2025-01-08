import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import plotly.graph_objects as go
from textblob import TextBlob
import numpy as np
from datetime import datetime
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from transformers import pipeline
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import plotly.express as px
import openai  # Ensure OpenAI API is installed and set up
import pytz

# Configure the page to use the full screen
st.set_page_config(
    page_title="Real-Time Stock Analysis",
    layout="wide",  # Ensures the app takes the full screen width
    initial_sidebar_state="expanded"  # Keeps the sidebar expanded
)

def fetch_data(symbol, interval="1m", period="1d"):
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

        return df
    except Exception as e:
        st.error(f"Error calculating indicators: {e}")
        return df




def check_trade_signal(data):
    """Check the latest trade signal."""
    if data.empty or len(data) < 1:
        return None, None, None

    # Ensure the required indicators exist
    required_columns = ['9_EMA', '21_EMA', 'Close', 'VWAP']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        st.warning(f"Missing required columns for trade signal: {', '.join(missing_columns)}")
        return None, None, None

    # Select the last row
    row = data.iloc[-1]

    # Evaluate buy/sell conditions
    ema_9 = row['9_EMA']
    ema_21 = row['21_EMA']
    close = row['Close']
    vwap = row['VWAP']

    if pd.isnull([ema_9, ema_21, close, vwap]).any():
        return None, None, None

    if ema_9 > ema_21 and close > vwap:
        return "BUY", row['Datetime'], close
    elif ema_9 < ema_21 and close < vwap:
        return "SELL", row['Datetime'], close

    return None, None, None



def display_ticker_analysis():
    st.title("Ticker Analysis")

    # Sidebar for user input
    tickers_input = st.sidebar.text_area(
        "Enter Tickers (comma-separated, e.g., AAPL, MSFT, TSLA)",
        value="AAPL, MSFT, TSLA"
    )
    interval = st.sidebar.selectbox("Interval", ["1m", "5m", "15m", "1h", "1d"], index=1)
    period = st.sidebar.selectbox("Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y"], index=1)

    # Process tickers
    if tickers_input:
        tickers = [ticker.strip().upper() for ticker in tickers_input.split(",")]
        for ticker in tickers:
            st.subheader(f"Analysis for {ticker}")
            data = fetch_data(ticker, interval=interval, period=period)

            if not data.empty:
                # Calculate indicators
                data = calculate_indicators(data)
                st.write(f"Latest Data for {ticker}")
                st.dataframe(data.tail(10))  # Show last 10 rows

                # Check trade signal
                signal, signal_time, signal_price = check_trade_signal(data)
                if signal:
                    st.success(f"Trade Signal: {signal} at {signal_price:.2f} ({signal_time})")
                else:
                    st.info("No trade signal detected.")
            else:
                st.warning(f"No data available for {ticker}.")


def plot_interactive_chart(ticker, data, signal=None, signal_time=None, signal_price=None, show_fibonacci=False, show_ema=False, show_macd=False, show_vwap=False, show_ichimoku=False):
    """
    Plot an interactive chart with Plotly, with optional indicators and buy/sell signals.
    """
    import plotly.graph_objects as go
    import pytz

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

    # Add Fibonacci retracements
    if show_fibonacci:
        fibonacci_levels = calculate_fibonacci_retracements(data)
        for level, value in fibonacci_levels.items():
            # Ensure `value` is scalar
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

    # Customize layout
    fig.update_layout(
        title=f"{ticker} - Real-Time Chart",
        xaxis_title="Datetime",
        yaxis_title="Price",
        template="plotly_white",
        height=700,
        margin=dict(r=150)  # Add margin to make space for Fibonacci annotations
    )

    return fig



def plot_dynamic_chart(ticker, interval, refresh_rate, show_fibonacci, show_ema, show_macd, show_vwap, show_ichimoku):
    """Fetch data and dynamically update the chart for the given ticker."""
    # Create placeholders for dynamic updates
    placeholder_signal = st.empty()
    placeholder_chart = st.empty()

    # Static elements (sentiment, sector performance, etc.)
    sentiment = display_google_news_sentiment(ticker)
    st.metric(label="Sentiment", value=sentiment)

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

        # Update Chart in Placeholder
        with placeholder_chart:
            st.plotly_chart(fig, use_container_width=True)

        # Wait for refresh
        time.sleep(refresh_rate)

# Load FinBERT model and tokenizer
finbert_model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
finbert_tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
finbert_pipeline = pipeline("sentiment-analysis", model=finbert_model, tokenizer=finbert_tokenizer)

# Step 1: Scrape Google News
def fetch_google_news_articles(ticker):
    """Fetch Google News headlines and links for a specific ticker."""
    try:
        # Google News search URL
        url = f"https://www.google.com/search?q={ticker}+stock+news&tbm=nws"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract article titles and links
        articles = []
        for item in soup.find_all('div', class_='BNeawe vvjwJb AP7Wnd'):
            title = item.text
            parent = item.find_parent('a')
            link = parent['href'] if parent else None
            if link:
                articles.append({'title': title, 'link': link})
        return articles[:5]  # Limit to top 5 articles
    except Exception as e:
        print(f"Error fetching Google News: {e}")
        return []

# Step 2: Fetch Article Content
def fetch_article_content(link):
    """Fetch the content of an article given its link."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(link, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract content (this depends on the article's structure)
        paragraphs = soup.find_all('p')
        content = ' '.join([p.text for p in paragraphs])
        return content.strip()
    except Exception as e:
        print(f"Error fetching article content: {e}")
        return "No content available."

# Step 3: Analyze Sentiment with FinBERT
def analyze_articles_with_sentiment(articles):
    """Perform sentiment analysis on article titles and content."""
    results = []
    sentiment_scores = {"Positive": 0, "Negative": 0, "Neutral": 0}
    for article in articles:
        title = article['title']
        link = article['link']

        # Fetch article content
        content = fetch_article_content(link)

        # Combine title and content for sentiment analysis
        text_to_analyze = f"{title}. {content}"
        sentiment_result = finbert_pipeline([text_to_analyze])[0]

        # Update sentiment counts
        sentiment_scores[sentiment_result['label'].capitalize()] += 1

        # Append the results
        results.append({
            'title': title,
            'link': link,
            'content': content,
            'sentiment': sentiment_result['label'].capitalize(),
            'confidence': sentiment_result['score']
        })

    # Aggregate sentiment score
    total_articles = len(articles)
    aggregate_score = (
        (sentiment_scores["Positive"] - sentiment_scores["Negative"]) / total_articles
        if total_articles > 0 else 0
    )

    return results, aggregate_score

# Step 4: Display Results in Streamlit
def display_google_news_sentiment(ticker):
    """Display Google News sentiment analysis for the selected ticker."""
    articles = fetch_google_news_articles(ticker)
    if not articles:
        st.warning("No news articles found.")
        return

    # Perform sentiment analysis
    analyzed_articles, aggregate_score = analyze_articles_with_sentiment(articles)

    # Display aggregate sentiment
    st.subheader(f"Aggregate Sentiment for {ticker}")
    if aggregate_score > 0:
        st.success(f"Overall Sentiment: Positive (Score: {aggregate_score:.2f})")
    elif aggregate_score < 0:
        st.error(f"Overall Sentiment: Negative (Score: {aggregate_score:.2f})")
    else:
        st.info(f"Overall Sentiment: Neutral (Score: {aggregate_score:.2f})")

    # Display individual article results
    st.subheader(f"News Sentiment Details for {ticker}")
    for article in analyzed_articles:
        st.write(f"**Title:** {article['title']}")
        st.write(f"**Link:** [Read full article]({article['link']})")
        st.write(f"**Sentiment:** {article['sentiment']} (Confidence: {article['confidence']:.2f})")
        st.write("---")



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


        
# Set your OpenAI API key
openai.api_key = "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"

def ai_chatbot_tab():
    """AI chatbot for answering user queries."""
    st.header("AI Chatbot")

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Input box for user query
    user_input = st.text_input("Ask me anything about trading, alerts, or the app:", key="chat_input")

    # Generate response
    if user_input:
        # Call OpenAI API for the response
        response = generate_ai_response(user_input)

        # Add user query and response to chat history
        st.session_state.chat_history.append({"user": user_input, "bot": response})

    # Display chat history
    st.subheader("Chat History")
    for chat in st.session_state.chat_history:
        st.write(f"**You:** {chat['user']}")
        st.write(f"**AI:** {chat['bot']}")
        
def suggest_portfolio_allocation(risk_level):
    """Suggest portfolio allocation based on risk level."""
    allocations = {
        "low": {"Bonds": 70, "Stocks": 20, "Real Estate": 10},
        "medium": {"Bonds": 40, "Stocks": 50, "Real Estate": 10},
        "high": {"Bonds": 10, "Stocks": 80, "Crypto": 10},
    }
    return allocations.get(risk_level, "Risk level not recognized.")


def analyze_technical_indicators(ticker):
    """Analyze stock with technical indicators."""
    data = yf.download(ticker, period="1mo", interval="1d")
    data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
    data['MACD'] = ta.trend.macd_diff(data['Close'])
    return data.iloc[-1][['RSI', 'MACD']]

        
def generate_ai_response(user_input):
    """Generate a response from the AI model with a financial focus."""
    try:
        # Query OpenAI GPT model
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a highly skilled financial advisor and data analyst. "
                        "You provide insights on stock trading, options strategies, portfolio management, "
                        "and risk analysis. Use clear language, and tailor your advice to the user's question. "
                        "If asked about specific stocks, include real-time or historical data analysis where appropriate."
                    )
                },
                {"role": "user", "content": user_input}
            ],
            max_tokens=300,
            temperature=0.7,
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"An error occurred: {e}"

def detect_candlestick_patterns(df):
    """Manually calculate multiple candlestick patterns."""
    df['Hammer'] = (
        ((df['High'] - df['Low']) > 3 * (df['Open'] - df['Close'])) &
        ((df['Close'] - df['Low']) / (0.001 + df['High'] - df['Low']) > 0.6) &
        ((df['Open'] - df['Low']) / (0.001 + df['High'] - df['Low']) > 0.6)
    )
    
    df['Bullish Engulfing'] = (
        (df['Open'].shift(1) > df['Close'].shift(1)) &
        (df['Close'] > df['Open']) &
        (df['Close'] > df['Open'].shift(1)) &
        (df['Open'] < df['Close'].shift(1))
    )
    
    df['Bearish Engulfing'] = (
        (df['Close'].shift(1) > df['Open'].shift(1)) &
        (df['Open'] > df['Close']) &
        (df['Open'] > df['Close'].shift(1)) &
        (df['Close'] < df['Open'].shift(1))
    )
    
    df['Doji'] = (
        abs(df['Open'] - df['Close']) <= (df['High'] - df['Low']) * 0.1
    )
    
    df['Morning Star'] = (
        (df['Close'].shift(2) < df['Open'].shift(2)) &
        (df['Close'].shift(1) > df['Open'].shift(1)) &
        (df['Open'] < df['Close']) &
        (df['Close'] > (df['Open'].shift(2) + df['Close'].shift(2)) / 2)
    )
    
    df['Evening Star'] = (
        (df['Close'].shift(2) > df['Open'].shift(2)) &
        (df['Close'].shift(1) < df['Open'].shift(1)) &
        (df['Open'] > df['Close']) &
        (df['Close'] < (df['Open'].shift(2) + df['Close'].shift(2)) / 2)
    )
    
    df['Hanging Man'] = (
        ((df['High'] - df['Low']) > 3 * (df['Open'] - df['Close'])) &
        ((df['Close'] - df['Low']) / (0.001 + df['High'] - df['Low']) > 0.6) &
        ((df['Open'] - df['Low']) / (0.001 + df['High'] - df['Low']) > 0.6) &
        (df['Close'] < df['Open'])  # Bearish body
    )
    
    df['Shooting Star'] = (
        ((df['High'] - df['Low']) > 3 * (df['Open'] - df['Close'])) &
        ((df['High'] - df['Close']) / (0.001 + df['High'] - df['Low']) > 0.6) &
        ((df['High'] - df['Open']) / (0.001 + df['High'] - df['Low']) > 0.6) &
        (df['Open'] > df['Close'])  # Bearish body
    )

    return df

def generate_recommendations(df):
    """Generate Buy/Sell recommendations based on candlestick patterns."""
    recommendations = []

    for index, row in df.iterrows():
        if row['Bullish Engulfing'] or row['Hammer'] or row['Morning Star']:
            recommendations.append({'Date': row['Datetime'], 'Action': 'Buy', 'Pattern': 'Bullish'})
        elif row['Bearish Engulfing'] or row['Hanging Man'] or row['Shooting Star'] or row['Evening Star']:
            recommendations.append({'Date': row['Datetime'], 'Action': 'Sell', 'Pattern': 'Bearish'})

    recommendations_df = pd.DataFrame(recommendations)

    # Format the 'Date' column to a more readable format (e.g., Jan 6, 2025, 03:15 PM)
    if not recommendations_df.empty:
        recommendations_df['Date'] = recommendations_df['Date'].dt.strftime("%b %d, %Y %I:%M %p")

    return recommendations_df



def plot_candlestick_with_recommendations(df, recommendations):
    """Plot candlestick chart with Buy/Sell recommendations."""
    fig = go.Figure()

    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=df['Datetime'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Candlestick'
    ))

    # Add Buy markers
    buy_signals = recommendations[recommendations['Action'] == 'Buy']
    fig.add_trace(go.Scatter(
        x=buy_signals['Date'],
        y=df.loc[df['Datetime'].isin(buy_signals['Date']), 'Close'],
        mode='markers+text',
        name='Buy Signal',
        marker=dict(color='green', size=10),
        text=buy_signals['Pattern'],
        textposition='top center'
    ))

    # Add Sell markers
    sell_signals = recommendations[recommendations['Action'] == 'Sell']
    fig.add_trace(go.Scatter(
        x=sell_signals['Date'],
        y=df.loc[df['Datetime'].isin(sell_signals['Date']), 'Close'],
        mode='markers+text',
        name='Sell Signal',
        marker=dict(color='red', size=10),
        text=sell_signals['Pattern'],
        textposition='top center'
    ))

    fig.update_layout(
        title="Candlestick Patterns with Buy/Sell Recommendations",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white"
    )
    return fig

def pattern_recognition_tab():
    """Tab for candlestick pattern recognition with recommendations."""
    st.header("Candlestick Pattern Recognition with Buy/Sell Recommendations")

    # Input for ticker search
    ticker = st.text_input("Enter Ticker Symbol (e.g., AAPL):", value="AAPL")
    interval = st.selectbox("Interval", ["15m", "30m", "1h", "3h", "6h"], index=2)  # Updated intervals
    period = st.selectbox("Period", ["1d", "5d", "1mo"], index=1)  # Adjusted for intraday data

    if ticker:
        # Fetch data for the entered ticker
        df = fetch_data(ticker, interval=interval, period=period)
        if not df.empty:
            # Detect patterns
            df = detect_candlestick_patterns(df)

            # Generate recommendations
            recommendations = generate_recommendations(df)

            # Plot chart with recommendations
            fig = plot_candlestick_with_recommendations(df, recommendations)
            st.plotly_chart(fig)

            # Display recommendations in a table
            st.write("Buy/Sell Recommendations")
            if not recommendations.empty:
                st.dataframe(recommendations)
            else:
                st.info("No Buy/Sell patterns detected for the selected ticker.")
        else:
            st.error("No data available for the selected ticker and time period.")
    else:
        st.warning("Please enter a valid ticker symbol.")


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
    st.sidebar.title("Stock Analysis Tools")
    interval = st.sidebar.selectbox("Interval", ["1m", "5m", "15m", "1h", "1d"], index=0)
    refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", min_value=1, max_value=60, value=5)
    show_fibonacci = st.sidebar.checkbox("Show Fibonacci Retracements", value=True)
    show_ichimoku = st.sidebar.checkbox("Show Ichimoku Clouds", value=True)
    show_ema = st.sidebar.checkbox("Show EMA Indicators", value=True)
    show_vwap = st.sidebar.checkbox("Show VWAP", value=True)

    # Define analysis type tabs
    analysis_type = st.tabs(["Trading", "AI Chatbot"])

    # Tab for AI Chatbot
    with analysis_type[1]:
        ai_chatbot_tab()


    # Validate and process tickers
    if tickers_input:
        selected_tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]
        tabs = st.tabs(selected_tickers)  # Create one tab per ticker

        for i, ticker in enumerate(selected_tickers):
            with tabs[i]:  # Each tab corresponds to one ticker
                st.subheader(f"{ticker} - Real-Time Chart")
                placeholder_chart = st.empty()  # Placeholder for the chart
                placeholder_signal = st.empty()  # Placeholder for signal alerts

                while True:
                    # Fetch data
                    data = fetch_data(ticker, interval=interval, period="1d")
                    if data.empty:
                        st.warning(f"No data available for {ticker}.")
                        break

                    # Calculate Indicators
                    try:
                        data = calculate_indicators(data)
                        if show_ichimoku:
                            data = calculate_ichimoku_clouds(data)
                    except Exception as e:
                        st.error(f"Error calculating indicators for {ticker}: {e}")
                        break

                    # Plot Interactive Chart
                    fig = plot_interactive_chart(
                        ticker=ticker,
                        data=data,
                        show_fibonacci=show_fibonacci,
                        show_ema=show_ema,
                        show_vwap=show_vwap,
                        show_ichimoku=show_ichimoku
                    )
                    # Add a unique key for the chart
                    placeholder_chart.plotly_chart(fig, use_container_width=True, key=f"{ticker}_chart_{int(time.time())}")
                    

                    # Real-Time Signal Alerts
                    with placeholder_signal:
                        try:
                            signal, signal_time, signal_price = check_trade_signal(data)
                            if signal == "BUY":
                                st.success(f"**BUY Signal detected for {ticker} at {signal_price:.2f} on {signal_time}**")
                            elif signal == "SELL":
                                st.error(f"**SELL Signal detected for {ticker} at {signal_price:.2f} on {signal_time}**")
                            else:
                                st.info("No signals detected.")
                        except Exception as e:
                            st.error(f"Error checking trade signals: {e}")
                            break

                    # Wait for refresh
                    time.sleep(refresh_rate)


if __name__ == "__main__":
    main()











