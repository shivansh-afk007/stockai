import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from sentiment import sentiment_analyzer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_historical_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Get historical stock data."""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(start=start_date, end=end_date)
        return data
    except Exception as e:
        logger.error(f"Error fetching historical data: {str(e)}")
        return pd.DataFrame()

def prepare_training_data(symbols: list, start_date: str, end_date: str) -> tuple:
    """Prepare training data for sentiment model."""
    X_train = []
    y_train = []
    
    for symbol in symbols:
        try:
            # Get historical data
            data = get_historical_data(symbol, start_date, end_date)
            if data.empty:
                continue
            
            # Calculate daily returns
            data['Returns'] = data['Close'].pct_change()
            data = data.dropna()
            
            # Get sentiment scores for each day
            for date, row in data.iterrows():
                # Get news from the previous day
                prev_date = date - timedelta(days=1)
                company_name = symbol.replace(".NS", "")
                
                # Get sentiment scores
                articles = sentiment_analyzer.scrape_news(company_name)
                if not articles:
                    continue
                
                # Analyze each article
                finbert_scores = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
                gpt4_scores = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
                
                for article in articles:
                    # Get FinBERT sentiment
                    article_finbert = sentiment_analyzer.get_finbert_sentiment(article["title"])
                    
                    # Get GPT-4 sentiment (simplified for training)
                    article_gpt4 = {
                        "positive": article_finbert["positive"],  # Simplified for training
                        "neutral": article_finbert["neutral"],
                        "negative": article_finbert["negative"]
                    }
                    
                    # Accumulate scores
                    for sentiment in ["positive", "neutral", "negative"]:
                        finbert_scores[sentiment] += article_finbert[sentiment]
                        gpt4_scores[sentiment] += article_gpt4[sentiment]
                
                # Calculate average scores
                num_articles = len(articles)
                avg_finbert_scores = {k: v/num_articles for k, v in finbert_scores.items()}
                avg_gpt4_scores = {k: v/num_articles for k, v in gpt4_scores.items()}
                
                # Create feature vector
                features = [
                    avg_finbert_scores["positive"], avg_finbert_scores["neutral"], avg_finbert_scores["negative"],
                    avg_gpt4_scores["positive"], avg_gpt4_scores["neutral"], avg_gpt4_scores["negative"]
                ]
                
                # Create target (1 for positive return, 0 for negative)
                target = 1 if row['Returns'] > 0 else 0
                
                X_train.append(features)
                y_train.append(target)
                
        except Exception as e:
            logger.error(f"Error preparing data for {symbol}: {str(e)}")
            continue
    
    return np.array(X_train), np.array(y_train)

def train_sentiment_model():
    """Train the sentiment model with historical data."""
    # List of Indian stocks to train on
    symbols = [
        "RELIANCE.NS",
        "TCS.NS",
        "HDFCBANK.NS",
        "INFY.NS",
        "ICICIBANK.NS",
        "HINDUNILVR.NS",
        "SBIN.NS",
        "BHARTIARTL.NS",
        "ITC.NS",
        "KOTAKBANK.NS"
    ]
    
    # Training period (last 6 months)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
    
    logger.info("Preparing training data...")
    X_train, y_train = prepare_training_data(symbols, start_date, end_date)
    
    if len(X_train) > 0:
        logger.info(f"Training model with {len(X_train)} samples...")
        sentiment_analyzer.train_model(X_train, y_train)
        logger.info("Model training completed!")
    else:
        logger.error("No training data available!")

if __name__ == "__main__":
    train_sentiment_model() 