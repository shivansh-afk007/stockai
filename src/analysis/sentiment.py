from transformers import AutoTokenizer, AutoModelForSequenceClassification
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Any
import re
import numpy as np
import torch
import time
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Class for analyzing sentiment of financial news and social media content."""
    
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """
        Initialize the SentimentAnalyzer.
        
        Args:
            model_name: Name of the pre-trained model to use
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.eval()  # Set model to evaluation mode
            
            # Headers for web scraping
            self.headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
        except Exception as e:
            logger.error(f"Error initializing SentimentAnalyzer: {str(e)}")
            raise
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary containing sentiment scores
        """
        try:
            # Tokenize text
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            # Get model outputs
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Convert predictions to probabilities
            positive_score = predictions[0][0].item()
            negative_score = predictions[0][1].item()
            neutral_score = predictions[0][2].item()
            
            # Determine sentiment label
            sentiment = "positive" if positive_score > max(negative_score, neutral_score) else \
                       "negative" if negative_score > max(positive_score, neutral_score) else "neutral"
            
            return {
                "sentiment": sentiment,
                "scores": {
                    "positive": positive_score,
                    "negative": negative_score,
                    "neutral": neutral_score
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing text: {str(e)}")
            raise
    
    def scrape_news(self, company_name: str, days: int = 7) -> List[Dict[str, str]]:
        """
        Scrape financial news for a company.
        
        Args:
            company_name: Name of the company
            days: Number of days of news to fetch
            
        Returns:
            List of dictionaries containing news articles
        """
        try:
            # Format search query
            query = f"{company_name} stock news"
            query = query.replace(" ", "+")
            
            # Google News URL
            url = f"https://news.google.com/search?q={query}&hl=en-IN&gl=IN&ceid=IN%3Aen"
            
            # Fetch and parse the page
            response = requests.get(url, headers=self.headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract news articles
            articles = []
            for article in soup.select('article'):
                try:
                    # Find title and link
                    title_elem = article.select_one('h3 a')
                    if title_elem and title_elem.text:
                        title = title_elem.text.strip()
                        link = "https://news.google.com" + title_elem['href'][1:]
                        
                        # Find date
                        time_elem = article.select_one('time')
                        date = time_elem['datetime'] if time_elem else datetime.now().strftime('%Y-%m-%d')
                        
                        articles.append({
                            "title": title,
                            "link": link,
                            "date": date
                        })
                except Exception as e:
                    logger.warning(f"Error parsing article: {str(e)}")
                    continue
            
            # If no articles found through Google News, use a fallback source
            if not articles:
                articles = [
                    {
                        "title": f"Latest market update for {company_name}",
                        "link": f"https://www.google.com/search?q={query}",
                        "date": datetime.now().strftime('%Y-%m-%d')
                    }
                ]
            
            return articles[:10]  # Return top 10 articles
            
        except Exception as e:
            logger.error(f"Error scraping news: {str(e)}")
            # Return a default article if scraping fails
            return [
                {
                    "title": f"Market analysis for {company_name}",
                    "link": f"https://www.google.com/search?q={query}",
                    "date": datetime.now().strftime('%Y-%m-%d')
                }
            ]
    
    def analyze_news_sentiment(self, company_name: str) -> Dict[str, Any]:
        """
        Analyze sentiment of recent news articles about a company.
        
        Args:
            company_name: Name of the company
            
        Returns:
            Dictionary containing aggregated sentiment analysis
        """
        try:
            # Scrape news articles
            articles = self.scrape_news(company_name)
            
            if not articles:
                return {
                    "overall_sentiment": "neutral",
                    "sentiment_scores": {"positive": 0, "negative": 0, "neutral": 0},
                    "articles": []
                }
            
            # Analyze sentiment for each article
            analyzed_articles = []
            sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
            
            for article in articles:
                sentiment_result = self.analyze_text(article["title"])
                sentiment_counts[sentiment_result["sentiment"]] += 1
                
                analyzed_articles.append({
                    "title": article["title"],
                    "sentiment": sentiment_result["sentiment"],
                    "scores": sentiment_result["scores"],
                    "date": article["date"],
                    "link": article["link"]
                })
            
            # Calculate overall sentiment
            total_articles = len(analyzed_articles)
            sentiment_scores = {
                k: v / total_articles for k, v in sentiment_counts.items()
            }
            
            overall_sentiment = max(sentiment_scores.items(), key=lambda x: x[1])[0]
            
            return {
                "overall_sentiment": overall_sentiment,
                "sentiment_scores": sentiment_scores,
                "articles": analyzed_articles
            }
            
        except Exception as e:
            logger.error(f"Error analyzing news sentiment: {str(e)}")
            raise

    def clean_text(self, text: str) -> str:
        """Clean and preprocess text."""
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = ' '.join(text.split())
        return text

    def get_news_articles(self, symbol: str, max_articles: int = 10) -> List[Dict[str, str]]:
        """
        Fetch news articles for a given stock symbol.
        
        Args:
            symbol (str): Stock symbol
            max_articles (int): Maximum number of articles to fetch
            
        Returns:
            list: List of dictionaries containing article information
        """
        try:
            # Check cache
            cache_key = f"{symbol}_news"
            if (cache_key in self.news_cache and 
                datetime.now() - self.cache_timestamp[cache_key] < self.cache_duration):
                return self.news_cache[cache_key]

            # Example URLs for financial news (you would need to implement proper news API integration)
            # This is a placeholder - in production, use proper financial news APIs
            news_sources = [
                f"https://economictimes.indiatimes.com/{symbol}/stocks/news",
                f"https://www.moneycontrol.com/stocks/{symbol}/news"
            ]
            
            articles = []
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            for url in news_sources:
                try:
                    response = requests.get(url, headers=headers, timeout=10)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        # Extract articles (implementation depends on website structure)
                        # This is a placeholder implementation
                        for article in soup.find_all('article')[:max_articles//2]:
                            title = article.find('h2')
                            if title:
                                articles.append({
                                    'title': self.clean_text(title.text),
                                    'date': datetime.now().strftime('%Y-%m-%d'),
                                    'source': url
                                })
                except Exception as e:
                    logger.warning(f"Error fetching news from {url}: {str(e)}")
                    continue

            # Cache the results
            self.news_cache[cache_key] = articles
            self.cache_timestamp[cache_key] = datetime.now()
            
            return articles[:max_articles]
            
        except Exception as e:
            logger.error(f"Error fetching news articles: {str(e)}")
            return []

    def analyze_sentiment(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment of given texts.
        
        Args:
            texts (list): List of texts to analyze
            
        Returns:
            list: List of sentiment analysis results
        """
        try:
            results = []
            for text in texts:
                if not text:
                    continue
                    
                sentiment = self.sentiment_analyzer(text)[0]
                results.append({
                    'text': text,
                    'sentiment': sentiment['label'],
                    'score': sentiment['score']
                })
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return []

    def get_stock_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Get overall sentiment analysis for a stock.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            dict: Dictionary containing sentiment analysis results
        """
        try:
            # Get news articles
            articles = self.get_news_articles(symbol)
            if not articles:
                return {'error': 'No articles found'}

            # Analyze sentiment of article titles
            titles = [article['title'] for article in articles]
            sentiments = self.analyze_sentiment(titles)

            # Calculate overall sentiment
            sentiment_scores = {
                'positive': 0,
                'negative': 0,
                'neutral': 0
            }

            for sentiment in sentiments:
                sentiment_scores[sentiment['sentiment'].lower()] += 1

            total = len(sentiments)
            sentiment_distribution = {
                k: v/total for k, v in sentiment_scores.items()
            }

            # Determine overall sentiment
            overall_sentiment = max(sentiment_distribution.items(), key=lambda x: x[1])[0]

            return {
                'overall_sentiment': overall_sentiment,
                'sentiment_distribution': sentiment_distribution,
                'detailed_analysis': sentiments,
                'articles': articles
            }
            
        except Exception as e:
            logger.error(f"Error getting stock sentiment: {str(e)}")
            return {'error': str(e)} 