import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.data.stock_data import StockDataFetcher
from src.analysis.technical import TechnicalAnalyzer
from src.analysis.sentiment import SentimentAnalyzer

class TestStockDataFetcher(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.fetcher = StockDataFetcher(cache_dir="test_cache")
        self.test_symbol = "RELIANCE.NS"
    
    def test_fetch_stock_data(self):
        """Test fetching stock data."""
        # Test with default parameters
        data = self.fetcher.fetch_stock_data(self.test_symbol)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertFalse(data.empty)
        
        # Check required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            self.assertIn(col, data.columns)
    
    def test_get_stock_info(self):
        """Test getting stock information."""
        info = self.fetcher.get_stock_info(self.test_symbol)
        self.assertIsInstance(info, dict)
        
        # Check required fields
        required_fields = ['longName', 'sector', 'industry']
        for field in required_fields:
            self.assertIn(field, info)

class TestTechnicalAnalyzer(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = TechnicalAnalyzer()
        
        # Create sample data
        dates = pd.date_range(start='2024-01-01', end='2024-02-01', freq='D')
        self.sample_data = pd.DataFrame({
            'Open': np.random.uniform(100, 110, len(dates)),
            'High': np.random.uniform(110, 120, len(dates)),
            'Low': np.random.uniform(90, 100, len(dates)),
            'Close': np.random.uniform(100, 110, len(dates)),
            'Volume': np.random.uniform(1000000, 2000000, len(dates))
        }, index=dates)
    
    def test_calculate_all_indicators(self):
        """Test calculating technical indicators."""
        result = self.analyzer.calculate_all_indicators(self.sample_data)
        self.assertIsInstance(result, pd.DataFrame)
        
        # Check if indicators are calculated
        expected_indicators = ['RSI', 'MACD', 'BB_High', 'BB_Low', 'SMA_20']
        for indicator in expected_indicators:
            self.assertIn(indicator, result.columns)
    
    def test_get_signals(self):
        """Test generating trading signals."""
        data = self.analyzer.calculate_all_indicators(self.sample_data)
        signals = self.analyzer.get_signals(data)
        
        self.assertIsInstance(signals, dict)
        self.assertIn('RSI', signals)
        self.assertIn('MACD', signals)
        self.assertIn('Overall', signals)
    
    def test_get_support_resistance(self):
        """Test calculating support and resistance levels."""
        levels = self.analyzer.get_support_resistance(self.sample_data)
        
        self.assertIsInstance(levels, dict)
        self.assertIn('support', levels)
        self.assertIn('resistance', levels)
        self.assertIn('pivot', levels)

class TestSentimentAnalyzer(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = SentimentAnalyzer()
        self.test_text = "Company reports strong quarterly earnings with positive outlook."
    
    def test_analyze_text(self):
        """Test sentiment analysis of text."""
        result = self.analyzer.analyze_text(self.test_text)
        
        self.assertIsInstance(result, dict)
        self.assertIn('sentiment', result)
        self.assertIn('scores', result)
        
        # Check sentiment scores
        scores = result['scores']
        self.assertIn('positive', scores)
        self.assertIn('negative', scores)
        self.assertIn('neutral', scores)
        
        # Check score probabilities sum to approximately 1
        total_score = sum(scores.values())
        self.assertAlmostEqual(total_score, 1.0, places=2)
    
    def test_scrape_news(self):
        """Test news scraping."""
        company_name = "Reliance Industries"
        articles = self.analyzer.scrape_news(company_name)
        
        self.assertIsInstance(articles, list)
        if articles:  # If any articles found
            article = articles[0]
            self.assertIn('title', article)
            self.assertIn('link', article)
            self.assertIn('date', article)
    
    def test_analyze_news_sentiment(self):
        """Test news sentiment analysis."""
        company_name = "Reliance Industries"
        result = self.analyzer.analyze_news_sentiment(company_name)
        
        self.assertIsInstance(result, dict)
        self.assertIn('overall_sentiment', result)
        self.assertIn('sentiment_scores', result)
        self.assertIn('articles', result)

if __name__ == '__main__':
    unittest.main() 