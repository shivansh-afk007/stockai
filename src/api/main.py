from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import logging
import numpy as np
import pandas as pd

from ..data.stock_data import StockDataFetcher
from ..analysis.technical import TechnicalAnalyzer
from ..analysis.sentiment import SentimentAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="StockAI API",
    description="AI-powered stock analysis API for Indian markets",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize analyzers
stock_fetcher = StockDataFetcher()
technical_analyzer = TechnicalAnalyzer()
sentiment_analyzer = SentimentAnalyzer()

# Pydantic models for request/response
class StockAnalysisRequest(BaseModel):
    symbol: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    include_technical: bool = True
    include_sentiment: bool = True

class StockAnalysisResponse(BaseModel):
    symbol: str
    data: Dict[str, Any]
    technical_analysis: Optional[Dict[str, Any]]
    sentiment_analysis: Optional[Dict[str, Any]]
    timestamp: str

def convert_numpy_types(obj):
    """Convert numpy types to Python native types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='list')
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, pd.DatetimeIndex):
        return [d.isoformat() for d in obj]
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

@app.get("/")
async def root():
    """Root endpoint returning API information."""
    return {
        "name": "StockAI API",
        "version": "1.0.0",
        "status": "active"
    }

@app.post("/analyze", response_model=StockAnalysisResponse)
async def analyze_stock(request: StockAnalysisRequest):
    """
    Analyze a stock using technical and sentiment analysis.
    
    Args:
        request: StockAnalysisRequest object containing analysis parameters
        
    Returns:
        StockAnalysisResponse object containing analysis results
    """
    try:
        # Fetch stock data
        stock_data = stock_fetcher.fetch_stock_data(
            request.symbol,
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        if stock_data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for symbol {request.symbol}")
        
        # Get stock info
        stock_info = stock_fetcher.get_stock_info(request.symbol)
        
        # Convert dates to ISO format strings
        dates = [d.isoformat() for d in stock_data.index]
        
        # Prepare the response data
        response_data = {
            "symbol": request.symbol,
            "data": {
                "info": stock_info,
                "dates": dates,
                "open": stock_data['Open'].tolist(),
                "high": stock_data['High'].tolist(),
                "low": stock_data['Low'].tolist(),
                "close": stock_data['Close'].tolist(),
                "volume": stock_data['Volume'].tolist(),
                "latest_price": float(stock_data['Close'].iloc[-1]),
                "change_percent": float((stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[-2]) / 
                                 stock_data['Close'].iloc[-2] * 100),
            },
            "technical_analysis": None,
            "sentiment_analysis": None,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add technical analysis if requested
        if request.include_technical:
            technical_data = technical_analyzer.calculate_all_indicators(stock_data)
            signals = technical_analyzer.get_signals(technical_data)
            levels = technical_analyzer.get_support_resistance(stock_data)
            
            response_data["technical_analysis"] = {
                "signals": signals,
                "levels": levels,
                "latest_indicators": {
                    col: technical_data[col].iloc[-1]
                    for col in technical_data.columns
                    if col not in stock_data.columns
                }
            }
        
        # Add sentiment analysis if requested
        if request.include_sentiment:
            company_name = stock_info.get("longName", request.symbol)
            sentiment_data = sentiment_analyzer.analyze_news_sentiment(company_name)
            response_data["sentiment_analysis"] = sentiment_data
        
        # Convert numpy types to Python native types
        return convert_numpy_types(response_data)
        
    except Exception as e:
        logger.error(f"Error analyzing stock: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stock_info/{symbol}")
async def get_stock_info(symbol: str):
    """
    Get basic information about a stock.
    
    Args:
        symbol: Stock symbol
        
    Returns:
        Dictionary containing stock information
    """
    try:
        info = stock_fetcher.get_stock_info(symbol)
        return info
        
    except Exception as e:
        logger.error(f"Error fetching stock info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/technical/{symbol}")
async def get_technical_analysis(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """
    Get technical analysis for a stock.
    
    Args:
        symbol: Stock symbol
        start_date: Start date for analysis
        end_date: End date for analysis
        
    Returns:
        Dictionary containing technical analysis results
    """
    try:
        # Fetch stock data
        stock_data = stock_fetcher.fetch_stock_data(symbol, start_date, end_date)
        
        if stock_data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for symbol {symbol}")
        
        # Calculate indicators and signals
        technical_data = technical_analyzer.calculate_all_indicators(stock_data)
        signals = technical_analyzer.get_signals(technical_data)
        levels = technical_analyzer.get_support_resistance(stock_data)
        
        return {
            "signals": signals,
            "levels": levels,
            "latest_indicators": {
                col: technical_data[col].iloc[-1]
                for col in technical_data.columns
                if col not in stock_data.columns
            }
        }
        
    except Exception as e:
        logger.error(f"Error performing technical analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sentiment/{symbol}")
async def get_sentiment_analysis(symbol: str):
    """
    Get sentiment analysis for a stock.
    
    Args:
        symbol: Stock symbol
        
    Returns:
        Dictionary containing sentiment analysis results
    """
    try:
        # Get company name from stock info
        info = stock_fetcher.get_stock_info(symbol)
        company_name = info.get("longName", symbol)
        
        # Perform sentiment analysis
        sentiment_data = sentiment_analyzer.analyze_news_sentiment(company_name)
        return sentiment_data
        
    except Exception as e:
        logger.error(f"Error performing sentiment analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 