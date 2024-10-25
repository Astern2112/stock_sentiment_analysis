from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from main import (
    fetch_news_table,
    parse_news_data,
    sentiment_pipeline,
    adjust_sentiment_score,
    normalize_sentiment_scores,
    convert_date,
    TICKERS
)

app = FastAPI()

class SentimentResponse(BaseModel):
    ticker: str
    average_sentiment: float
    latest_sentiment: float
    sentiment_trend: list

@app.get("/")
async def root():
    return {"message": "Welcome to the Stock Sentiment Analysis API"}

@app.get("/sentiment/{ticker}", response_model=SentimentResponse)
async def get_sentiment(ticker: str):
    if ticker not in TICKERS:
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} not found")

    news_table = fetch_news_table(ticker)
    if not news_table:
        raise HTTPException(status_code=503, detail=f"Unable to fetch news for {ticker}")

    parsed_data = parse_news_data({ticker: news_table})
    df = pd.DataFrame(parsed_data)
    
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No news data found for {ticker}")

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
    
    average_sentiment = df['sentiment_score'].mean()
    latest_sentiment = df.iloc[0]['sentiment_score'] if not df.empty else 0
    
    # Calculate sentiment trend (last 5 days)
    sentiment_trend = df.groupby('date')['sentiment_score'].mean().sort_index().tail(5).tolist()
    
    return SentimentResponse(
        ticker=ticker,
        average_sentiment=float(average_sentiment),
        latest_sentiment=float(latest_sentiment),
        sentiment_trend=sentiment_trend
    )

@app.get("/tickers")
async def get_tickers():
    return {"tickers": TICKERS}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)