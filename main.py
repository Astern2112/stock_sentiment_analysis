import logging
from datetime import datetime, timedelta
from typing import List, Dict
from urllib.request import urlopen, Request
import configparser
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
from transformers import pipeline
import yfinance as yf
import numpy as np
import plotly.graph_objects as go

# Constants
FINVIZ_URL = 'https://finviz.com/quote.ashx?t='
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration
def load_config():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config

config = load_config()
TICKERS = config.get('DEFAULT', 'tickers', fallback='AAPL,MSFT').split(',')

# Initialize sentiment pipeline
sentiment_pipeline = pipeline('sentiment-analysis', model='distilbert/distilbert-base-uncased-finetuned-sst-2-english')

def fetch_news_table(ticker: str) -> BeautifulSoup:
    url = f"{FINVIZ_URL}{ticker}"
    req = Request(url=url, headers={'user-agent': USER_AGENT})
    try:
        with urlopen(req) as response:
            html = BeautifulSoup(response, 'html.parser')
        return html.find(id='news-table')
    except Exception as e:
        logging.error(f"Error fetching news for {ticker}: {e}")
        return None

def parse_news_data(news_tables: Dict[str, BeautifulSoup]) -> List[Dict]:
    parsed_data = []
    for ticker, news_table in news_tables.items():
        if news_table is None:
            continue
        for row in news_table.find_all('tr'):
            title_element = row.a
            date_element = row.td
            if not (title_element and date_element):
                continue
            
            title = title_element.text
            date_data = date_element.text.strip().split()
            if len(date_data) < 2:
                continue
            
            date = date_data[0]
            time = date_data[-1]  # Last element is always the time
            parsed_data.append({'ticker': ticker, 'date': date, 'time': time, 'title': title})
    return parsed_data

def convert_date(date_str: str, time_str: str) -> datetime:
    today = datetime.now().date()
    if date_str.lower() == 'today':
        date_obj = today
    elif date_str.lower() == 'yesterday':
        date_obj = today - timedelta(days=1)
    else:
        date_obj = datetime.strptime(date_str, '%b-%d-%y').date()
    time_obj = datetime.strptime(time_str, '%I:%M%p').time()
    return datetime.combine(date_obj, time_obj)

def adjust_sentiment_score(row: pd.Series) -> float:
    try:
        score = float(row['sentiment_score'])
        if row['sentiment'] == 'NEGATIVE':
            return -score
        elif row['sentiment'] == 'NEUTRAL':
            return 0
        return score
    except ValueError:
        logging.warning(f"Non-numeric sentiment score encountered: {row['sentiment_score']}")
        return 0

def normalize_sentiment_scores(df: pd.DataFrame) -> pd.DataFrame:
    for ticker in df['ticker'].unique():
        ticker_mask = df['ticker'] == ticker
        scores = df.loc[ticker_mask, 'sentiment_score']
        min_score, max_score = scores.min(), scores.max()
        df.loc[ticker_mask, 'sentiment_score'] = (scores - min_score) / (max_score - min_score) * 2 - 1
    return df

def plot_sentiment_trends(mean_df):
    plt.figure(figsize=(12, 8))
    colors = ['green', 'blue', 'red']  # Add more colors if you have more tickers
    
    for ticker, color in zip(mean_df.columns, colors):
        plt.plot(mean_df.index, mean_df[ticker], label=ticker, marker='o', color=color)
        plt.plot(mean_df.index, mean_df[ticker].rolling(window=3).mean(), 
                 label=f'{ticker} 3-day MA', linestyle='--', color=color)
    
    plt.title('Sentiment Score Trends by Ticker')
    plt.xlabel('Date')
    plt.ylabel('Sentiment Score')
    plt.legend(title='Ticker', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_sentiment_heatmap(mean_df):
    plt.figure(figsize=(12, 8))
    sns.heatmap(mean_df.T, cmap='RdYlGn', center=0, annot=True, fmt='.2f')
    plt.title('Sentiment Heatmap by Ticker and Date')
    plt.xlabel('Date')
    plt.ylabel('Ticker')
    plt.tight_layout()
    plt.show()

def generate_trading_signals(mean_df, buy_threshold=0.2, sell_threshold=-0.2):
    signals = mean_df.copy()
    
    # Convert to numeric, replacing any non-numeric values with NaN
    signals = signals.apply(pd.to_numeric, errors='coerce')
    
    # Generate signals
    signals = signals.applymap(lambda x: 'BUY' if x > buy_threshold else ('SELL' if x < sell_threshold else 'HOLD'))
    
    return signals

def plot_trading_signals(signals):
    plt.figure(figsize=(12, 8))
    
    for i, ticker in enumerate(signals.columns):
        for j, signal in enumerate(signals[ticker]):
            if signal == 'BUY':
                plt.scatter(signals.index[j], i, color='green', marker='^', s=100)
            elif signal == 'SELL':
                plt.scatter(signals.index[j], i, color='red', marker='v', s=100)
            elif signal == 'HOLD':
                plt.scatter(signals.index[j], i, color='yellow', marker='o', s=100)
    
    plt.yticks(range(len(signals.columns)), signals.columns)
    plt.title('Trading Signals by Ticker')
    plt.xlabel('Date')
    plt.ylabel('Ticker')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def fetch_stock_data(tickers, start_date, end_date):
    stock_data = yf.download(tickers, start=start_date, end=end_date)
    return stock_data['Close']

def plot_sentiment_and_price(mean_df, price_df):
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    ax2 = ax1.twinx()
    
    for ticker in mean_df.columns:
        ax1.plot(mean_df.index, mean_df[ticker], label=f'{ticker} Sentiment')
        ax2.plot(price_df.index, price_df[ticker], label=f'{ticker} Price', linestyle='--')

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Sentiment Score')
    ax2.set_ylabel('Stock Price')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.title('Sentiment Scores and Stock Prices')
    plt.tight_layout()
    plt.show()

def fetch_historical_news(tickers, days_back=30):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    all_news = {}
    for ticker in tickers:
        ticker_news = []
        current_date = start_date
        while current_date <= end_date:
            url = f"{FINVIZ_URL}{ticker}&d={current_date.strftime('%m/%d/%Y')}"
            news_table = fetch_news_table(ticker, url)
            if news_table:
                ticker_news.extend(parse_news_data({ticker: news_table}))
            current_date += timedelta(days=1)
        all_news[ticker] = ticker_news
    return all_news

def preprocess_text(text):
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Join tokens back into string
    return ' '.join(tokens)

def create_interactive_plot(mean_df, price_df):
    fig = go.Figure()
    
    for ticker in mean_df.columns:
        fig.add_trace(go.Scatter(x=mean_df.index, y=mean_df[ticker],
                                 mode='lines+markers',
                                 name=f'{ticker} Sentiment'))
        fig.add_trace(go.Scatter(x=price_df.index, y=price_df[ticker],
                                 mode='lines',
                                 name=f'{ticker} Price',
                                 yaxis='y2'))
    
    fig.update_layout(
        title='Sentiment Scores and Stock Prices',
        xaxis_title='Date',
        yaxis_title='Sentiment Score',
        yaxis2=dict(title='Stock Price', overlaying='y', side='right'),
        legend_title='Legend',
        hovermode='x unified'
    )
    
    fig.show()

def main():
    # Fetch news data
    news_tables = {ticker: fetch_news_table(ticker) for ticker in TICKERS}
    
    # Parse news data
    parsed_data = parse_news_data(news_tables)
    
    # Create DataFrame
    df = pd.DataFrame(parsed_data)
    
    # Convert date and time to datetime
    df['datetime'] = df.apply(lambda row: convert_date(row['date'], row['time']), axis=1)
    df['date'] = df['datetime'].dt.date
    df = df.drop(columns=['time'])
    
    # Perform sentiment analysis
    df['sentiment'] = df['title'].apply(lambda x: sentiment_pipeline(x)[0]['label'])
    df['sentiment_score'] = df['title'].apply(lambda x: sentiment_pipeline(x)[0]['score'])
    
    # Adjust sentiment scores
    df['sentiment_score'] = df.apply(adjust_sentiment_score, axis=1)
    
    # Normalize sentiment scores
    df = normalize_sentiment_scores(df)
    
    # Group by ticker and date, calculate mean sentiment score
    mean_df = df.groupby(['ticker', 'date'])['sentiment_score'].mean().unstack().transpose()
    
    # Ensure all values are numeric
    mean_df = mean_df.apply(pd.to_numeric, errors='coerce')
    
    # Plot results
    plot_sentiment_trends(mean_df)
    plot_sentiment_heatmap(mean_df)
    
    # Generate and plot trading signals
    trading_signals = generate_trading_signals(mean_df, buy_threshold=0.2, sell_threshold=-0.2)
    plot_trading_signals(trading_signals)
    
    # Fetch and plot stock price data
    start_date = df['date'].min()
    end_date = df['date'].max()
    price_df = fetch_stock_data(TICKERS, start_date, end_date)
    plot_sentiment_and_price(mean_df, price_df)
    
    # Create interactive plot
    # create_interactive_plot(mean_df, price_df)

if __name__ == "__main__":
    main()
