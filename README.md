# Stock Sentiment Analysis and Trading Signals

## Overview

This project analyzes sentiment from news articles for selected stocks and generates trading signals based on the sentiment analysis. It uses Natural Language Processing (NLP) techniques to process news data, calculate sentiment scores, and visualize the results alongside stock prices.

## Features

- Fetches and processes news articles for specified stocks
- Performs sentiment analysis on news content
- Generates trading signals (BUY, SELL, HOLD) based on sentiment scores
- Visualizes sentiment scores and stock prices
- Provides an API for retrieving sentiment data

## Requirements

- Python 3.7+
- pip (Python package installer)

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/yourusername/stock-sentiment-analysis.git
   cd stock-sentiment-analysis
   ```

2. Create a virtual environment (optional but recommended):

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:

   ```
   pip install -r requirements.txt
   ```

4. Download NLTK data:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

## Configuration

Edit the `config.ini` file to specify the stock tickers you want to analyze:

## Usage

1. Run the main script:

   ```
   python main.py
   ```

   This will fetch news data, perform sentiment analysis, generate trading signals, and create visualizations.

2. To start the API server:
   ```
   uvicorn api:app --reload
   ```
   The API will be available at `http://localhost:8000`.

## API Endpoints

- GET `/sentiment/{ticker}`: Retrieve sentiment data for a specific ticker
  - Returns average sentiment, latest sentiment, and sentiment trend

## Project Structure

- `main.py`: Main script for data processing and analysis
- `api.py`: FastAPI server for sentiment data retrieval
- `config.ini`: Configuration file for stock tickers
- `requirements.txt`: List of Python dependencies

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
