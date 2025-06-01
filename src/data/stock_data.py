import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockDataFetcher:
    def __init__(self, cache_dir: str = "cache"):
        """Initialize the StockDataFetcher with cache directory."""
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def _get_cache_path(self, symbol: str, start_date: str, end_date: str) -> str:
        """Generate cache file path for the given parameters."""
        return os.path.join(self.cache_dir, f"{symbol}_{start_date}_{end_date}.csv")
    
    def _is_cache_valid(self, cache_path: str, max_age_days: int = 1) -> bool:
        """Check if cached data is still valid."""
        if not os.path.exists(cache_path):
            return False
        
        file_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
        age = datetime.now() - file_time
        return age.days < max_age_days
    
    def fetch_stock_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch stock data for the given symbol and date range.
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE.NS' for NSE stocks)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with stock data
        """
        try:
            # Set default dates if not provided
            if not end_date:
                end_date = datetime.now().strftime("%Y-%m-%d")
            if not start_date:
                start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            
            cache_path = self._get_cache_path(symbol, start_date, end_date)
            
            # Try to load from cache if enabled
            if use_cache and self._is_cache_valid(cache_path):
                logger.info(f"Loading cached data for {symbol}")
                return pd.read_csv(cache_path, index_col=0, parse_dates=True)
            
            # Fetch new data
            logger.info(f"Fetching new data for {symbol}")
            stock = yf.Ticker(symbol)
            data = stock.history(start=start_date, end=end_date)
            
            # Cache the data
            if use_cache:
                data.to_csv(cache_path)
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            raise
    
    def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get general information about a stock.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary containing stock information
        """
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
            # Extract relevant information
            relevant_info = {
                "longName": info.get("longName"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "marketCap": info.get("marketCap"),
                "currency": info.get("currency"),
                "exchange": info.get("exchange"),
                "country": info.get("country")
            }
            
            return relevant_info
            
        except Exception as e:
            logger.error(f"Error fetching info for {symbol}: {str(e)}")
            raise 