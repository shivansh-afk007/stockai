import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import ta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TechnicalAnalyzer:
    """Class for calculating technical indicators for stock analysis."""
    
    def __init__(self):
        """Initialize the TechnicalAnalyzer."""
        pass
    
    def calculate_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators for the given stock data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all technical indicators
        """
        try:
            df = data.copy()
            
            # RSI
            df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
            
            # MACD
            macd = ta.trend.MACD(df['Close'])
            df['MACD'] = macd.macd()
            df['MACD_Signal'] = macd.macd_signal()
            df['MACD_Hist'] = macd.macd_diff()
            
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df['Close'])
            df['BB_High'] = bollinger.bollinger_hband()
            df['BB_Low'] = bollinger.bollinger_lband()
            df['BB_Mid'] = bollinger.bollinger_mavg()
            
            # Moving Averages
            df['SMA_20'] = ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator()
            df['SMA_50'] = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator()
            df['EMA_20'] = ta.trend.EMAIndicator(df['Close'], window=20).ema_indicator()
            
            # Volume Indicators
            df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
            
            # Momentum Indicators
            df['Stoch_K'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close']).stoch()
            df['Stoch_D'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close']).stoch_signal()
            
            # Trend Indicators
            df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx()
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            raise
    
    def get_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate trading signals based on technical indicators.
        
        Args:
            data: DataFrame with technical indicators
            
        Returns:
            Dictionary with trading signals and their values
        """
        try:
            signals = {}
            
            # RSI Signals
            signals['RSI'] = {
                'value': data['RSI'].iloc[-1],
                'signal': 'Oversold' if data['RSI'].iloc[-1] < 30 else 'Overbought' if data['RSI'].iloc[-1] > 70 else 'Neutral'
            }
            
            # MACD Signals
            signals['MACD'] = {
                'value': data['MACD'].iloc[-1],
                'signal': 'Buy' if data['MACD'].iloc[-1] > data['MACD_Signal'].iloc[-1] else 'Sell'
            }
            
            # Bollinger Bands Signals
            close = data['Close'].iloc[-1]
            signals['BB'] = {
                'value': close,
                'signal': 'Oversold' if close < data['BB_Low'].iloc[-1] else 'Overbought' if close > data['BB_High'].iloc[-1] else 'Neutral'
            }
            
            # Moving Average Signals
            signals['MA_Cross'] = {
                'value': f"SMA20: {data['SMA_20'].iloc[-1]:.2f}, SMA50: {data['SMA_50'].iloc[-1]:.2f}",
                'signal': 'Buy' if data['SMA_20'].iloc[-1] > data['SMA_50'].iloc[-1] else 'Sell'
            }
            
            # Stochastic Signals
            signals['Stochastic'] = {
                'value': f"K: {data['Stoch_K'].iloc[-1]:.2f}, D: {data['Stoch_D'].iloc[-1]:.2f}",
                'signal': 'Oversold' if data['Stoch_K'].iloc[-1] < 20 else 'Overbought' if data['Stoch_K'].iloc[-1] > 80 else 'Neutral'
            }
            
            # Overall Signal
            buy_signals = sum(1 for signal in signals.values() if signal['signal'] in ['Buy', 'Oversold'])
            sell_signals = sum(1 for signal in signals.values() if signal['signal'] in ['Sell', 'Overbought'])
            
            signals['Overall'] = {
                'value': f"Buy Signals: {buy_signals}, Sell Signals: {sell_signals}",
                'signal': 'Strong Buy' if buy_signals >= 3 else 'Strong Sell' if sell_signals >= 3 else 'Neutral'
            }
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating trading signals: {str(e)}")
            raise
    
    def get_support_resistance(self, data: pd.DataFrame, window: int = 20) -> Dict[str, float]:
        """
        Calculate support and resistance levels using pivot points.
        
        Args:
            data: DataFrame with OHLCV data
            window: Window size for calculating levels
            
        Returns:
            Dictionary with support and resistance levels
        """
        try:
            df = data.tail(window)
            
            pivot = (df['High'].max() + df['Low'].min() + df['Close'].iloc[-1]) / 3
            r1 = 2 * pivot - df['Low'].min()
            s1 = 2 * pivot - df['High'].max()
            
            return {
                'resistance': r1,
                'pivot': pivot,
                'support': s1
            }
            
        except Exception as e:
            logger.error(f"Error calculating support/resistance levels: {str(e)}")
            raise 