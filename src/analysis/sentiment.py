import os
import logging
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedSentimentAnalyzer:
    """Enhanced sentiment analyzer using FinBERT, GPT-4, and Logistic Regression."""
    
    def __init__(self):
        """Initialize the sentiment analyzer with required models."""
        # Load FinBERT
        self.finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        
        # Initialize GPT-4
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        # Initialize or load Logistic Regression model
        self.model_path = "models/sentiment_model.joblib"
        self.scaler_path = "models/sentiment_scaler.joblib"
        
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.lr_model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
        else:
            self.lr_model = LogisticRegression(random_state=42)
            self.scaler = StandardScaler()
        
        # Headers for web scraping
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    def get_finbert_sentiment(self, text: str) -> Dict[str, float]:
        """Get sentiment scores using FinBERT."""
        inputs = self.finbert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.finbert_model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
            
        return {
            "positive": float(probabilities[0][2]),
            "neutral": float(probabilities[0][1]),
            "negative": float(probabilities[0][0])
        }

    async def get_gpt4_sentiment(self, text: str) -> Dict[str, Any]:
        """Get sentiment analysis from GPT-4."""
        try:
            response = await openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a financial sentiment analyzer. Analyze the following text and provide sentiment scores and key points."},
                    {"role": "user", "content": text}
                ],
                temperature=0.3,
                max_tokens=150
            )
            
            # Extract sentiment from GPT-4's response
            analysis = response.choices[0].message.content
            
            # Parse GPT-4's response to extract sentiment scores
            # This is a simplified version - you might want to make it more sophisticated
            sentiment_scores = {
                "positive": 0.0,
                "neutral": 0.0,
                "negative": 0.0
            }
            
            if "positive" in analysis.lower():
                sentiment_scores["positive"] = 0.7
            elif "negative" in analysis.lower():
                sentiment_scores["negative"] = 0.7
            else:
                sentiment_scores["neutral"] = 0.7
                
            return {
                "scores": sentiment_scores,
                "analysis": analysis
            }
            
        except Exception as e:
            logger.error(f"Error getting GPT-4 sentiment: {str(e)}")
            return {
                "scores": {"positive": 0.33, "neutral": 0.34, "negative": 0.33},
                "analysis": "Error analyzing sentiment"
            }

    def combine_sentiments(self, finbert_scores: Dict[str, float], gpt4_scores: Dict[str, float]) -> np.ndarray:
        """Combine FinBERT and GPT-4 scores for Logistic Regression."""
        features = [
            finbert_scores["positive"], finbert_scores["neutral"], finbert_scores["negative"],
            gpt4_scores["positive"], gpt4_scores["neutral"], gpt4_scores["negative"]
        ]
        return np.array(features).reshape(1, -1)

    def predict_market_movement(self, combined_features: np.ndarray) -> Tuple[str, float]:
        """Predict market movement using Logistic Regression."""
        if not hasattr(self.lr_model, 'coef_'):
            # If model is not trained, return neutral prediction
            return "HOLD", 0.5
            
        scaled_features = self.scaler.transform(combined_features)
        prediction = self.lr_model.predict(scaled_features)
        probability = self.lr_model.predict_proba(scaled_features)
        
        if prediction[0] == 1:
            return "BUY", float(probability[0][1])
        else:
            return "SELL", float(probability[0][0])

    def scrape_news(self, company_name: str, days: int = 7) -> List[Dict[str, str]]:
        """Scrape financial news for a company."""
        try:
            # Format search query
            query = f"{company_name} stock news"
            query = query.replace(" ", "+")
            
            # Google News URL
            url = f"https://news.google.com/search?q={query}&hl=en-IN&gl=IN&ceid=IN%3Aen"
            
            # Fetch and parse the page
            response = requests.get(url, headers=self.headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find news articles
            articles = []
            for article in soup.select('article'):
                try:
                    title_elem = article.select_one('h3 > a')
                    if not title_elem:
                        continue
                        
                    title = title_elem.text
                    link = "https://news.google.com" + title_elem['href'][1:]
                    
                    # Get date if available
                    date_elem = article.select_one('time')
                    date = date_elem['datetime'] if date_elem else None
                    
                    articles.append({
                        "title": title,
                        "link": link,
                        "date": date
                    })
                    
                except Exception as e:
                    logger.error(f"Error parsing article: {str(e)}")
                    continue
                    
            return articles[:10]  # Return top 10 articles
            
        except Exception as e:
            logger.error(f"Error scraping news: {str(e)}")
            return []

    async def analyze_news_sentiment(self, company_name: str) -> Dict[str, Any]:
        """Analyze sentiment from news articles using multiple models."""
        try:
            # Scrape news articles
            articles = self.scrape_news(company_name)
            
            if not articles:
                return {
                    "overall_sentiment": "neutral",
                    "sentiment_scores": {"positive": 0.33, "neutral": 0.34, "negative": 0.33},
                    "market_prediction": "HOLD",
                    "confidence": 0.5,
                    "articles": []
                }
            
            # Analyze each article
            analyzed_articles = []
            total_finbert_scores = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
            total_gpt4_scores = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
            
            for article in articles:
                # Get FinBERT sentiment
                finbert_scores = self.get_finbert_sentiment(article["title"])
                
                # Get GPT-4 sentiment
                gpt4_result = await self.get_gpt4_sentiment(article["title"])
                gpt4_scores = gpt4_result["scores"]
                
                # Combine scores for overall article sentiment
                for sentiment in ["positive", "neutral", "negative"]:
                    total_finbert_scores[sentiment] += finbert_scores[sentiment]
                    total_gpt4_scores[sentiment] += gpt4_scores[sentiment]
                
                # Get market prediction for this article
                combined_features = self.combine_sentiments(finbert_scores, gpt4_scores)
                prediction, confidence = self.predict_market_movement(combined_features)
                
                # Add analyzed article
                article["sentiment"] = prediction
                article["confidence"] = confidence
                article["analysis"] = gpt4_result["analysis"]
                analyzed_articles.append(article)
            
            # Calculate average scores
            num_articles = len(articles)
            avg_finbert_scores = {k: v/num_articles for k, v in total_finbert_scores.items()}
            avg_gpt4_scores = {k: v/num_articles for k, v in total_gpt4_scores.items()}
            
            # Get overall market prediction
            combined_features = self.combine_sentiments(avg_finbert_scores, avg_gpt4_scores)
            overall_prediction, overall_confidence = self.predict_market_movement(combined_features)
            
            # Determine overall sentiment
            max_sentiment = max(avg_finbert_scores.items(), key=lambda x: x[1])[0]
            
            return {
                "overall_sentiment": max_sentiment,
                "sentiment_scores": avg_finbert_scores,
                "market_prediction": overall_prediction,
                "confidence": overall_confidence,
                "articles": analyzed_articles
            }
            
        except Exception as e:
            logger.error(f"Error analyzing news sentiment: {str(e)}")
            return {
                "overall_sentiment": "neutral",
                "sentiment_scores": {"positive": 0.33, "neutral": 0.34, "negative": 0.33},
                "market_prediction": "HOLD",
                "confidence": 0.5,
                "articles": []
            }

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the Logistic Regression model with historical data."""
        try:
            # Scale features
            X_scaled = self.scaler.fit_transform(X_train)
            
            # Train model
            self.lr_model.fit(X_scaled, y_train)
            
            # Save model and scaler
            os.makedirs("models", exist_ok=True)
            joblib.dump(self.lr_model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")

# Create singleton instance
sentiment_analyzer = EnhancedSentimentAnalyzer() 