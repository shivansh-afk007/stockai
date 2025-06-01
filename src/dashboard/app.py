import dash
from dash import html, dcc, Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd
import requests
import json
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Dash app
app = dash.Dash(
    __name__,
    title="StockAI Dashboard",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)

# API endpoint
API_BASE_URL = "http://localhost:8000"

# App layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("StockAI Dashboard", className="header-title"),
        html.P("AI-Powered Stock Analysis for Indian Markets", className="header-description")
    ], className="header"),
    
    # Main content
    html.Div([
        # Left panel - Input controls
        html.Div([
            html.Div([
                html.Label("Stock Symbol"),
                dcc.Input(
                    id="stock-input",
                    type="text",
                    placeholder="Enter stock symbol (e.g., RELIANCE.NS)",
                    className="input-field"
                )
            ], className="input-group"),
            
            html.Div([
                html.Label("Date Range"),
                dcc.DatePickerRange(
                    id="date-range",
                    start_date=(datetime.now() - timedelta(days=365)).date(),
                    end_date=datetime.now().date(),
                    className="date-picker"
                )
            ], className="input-group"),
            
            html.Div([
                html.Button("Analyze", id="analyze-button", className="button primary"),
                html.Div(id="error-message", className="error-message")
            ], className="button-group")
        ], className="left-panel"),
        
        # Right panel - Analysis results
        html.Div([
            # Stock info card
            html.Div([
                html.H3("Stock Information"),
                html.Div(id="stock-info", className="info-content")
            ], className="card"),
            
            # Price chart
            html.Div([
                html.H3("Price Chart"),
                dcc.Graph(id="price-chart")
            ], className="card"),
            
            # Technical analysis
            html.Div([
                html.H3("Technical Analysis"),
                html.Div([
                    html.Div(id="technical-signals", className="signals"),
                    html.Div(id="support-resistance", className="levels")
                ], className="technical-content")
            ], className="card"),
            
            # Sentiment analysis
            html.Div([
                html.H3("Sentiment Analysis"),
                html.Div([
                    html.Div(id="sentiment-summary", className="sentiment-summary"),
                    html.Div(id="news-list", className="news-list")
                ], className="sentiment-content")
            ], className="card")
        ], className="right-panel")
    ], className="main-content")
], className="container")

@app.callback(
    [Output("stock-info", "children"),
     Output("price-chart", "figure"),
     Output("technical-signals", "children"),
     Output("support-resistance", "children"),
     Output("sentiment-summary", "children"),
     Output("news-list", "children"),
     Output("error-message", "children")],
    [Input("analyze-button", "n_clicks")],
    [State("stock-input", "value"),
     State("date-range", "start_date"),
     State("date-range", "end_date")]
)
def update_analysis(n_clicks, symbol, start_date, end_date):
    """Update all components with new analysis results."""
    if not n_clicks or not symbol:
        return [dash.no_update] * 7
    
    try:
        # Call API for analysis
        response = requests.post(
            f"{API_BASE_URL}/analyze",
            json={
                "symbol": symbol,
                "start_date": start_date,
                "end_date": end_date,
                "include_technical": True,
                "include_sentiment": True
            }
        )
        
        if response.status_code != 200:
            error_detail = "Unknown error"
            try:
                error_detail = response.json().get("detail", "Unknown error")
            except:
                pass
            raise Exception(error_detail)
        
        data = response.json()
        
        # Stock info card
        stock_info = data["data"]["info"]
        info_card = html.Div([
            html.P(f"Company: {stock_info.get('longName', symbol)}", className="info-item"),
            html.P(f"Sector: {stock_info.get('sector', 'N/A')}", className="info-item"),
            html.P(f"Industry: {stock_info.get('industry', 'N/A')}", className="info-item"),
            html.P(f"Latest Price: ₹{data['data']['latest_price']:.2f}", className="info-item price"),
            html.P(
                f"Change: {data['data']['change_percent']:.2f}%",
                className=f"info-item change {'positive' if data['data']['change_percent'] > 0 else 'negative'}"
            )
        ])
        
        # Price chart
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=data["data"]["dates"],
            open=data["data"]["open"],
            high=data["data"]["high"],
            low=data["data"]["low"],
            close=data["data"]["close"],
            name="OHLC"
        ))
        fig.update_layout(
            title=f"{symbol} Price Chart",
            yaxis_title="Price",
            template="plotly_dark",
            hovermode="x unified"
        )
        
        # Technical signals
        tech_analysis = data.get("technical_analysis", {})
        signals = tech_analysis.get("signals", {})
        signal_elements = []
        for indicator, details in signals.items():
            signal_elements.append(html.Div([
                html.Strong(f"{indicator}: "),
                html.Span(f"{details['signal']} ({details['value']})")
            ], className=f"signal-item {details['signal'].lower()}"))
        
        # Support/Resistance levels
        levels = tech_analysis.get("levels", {})
        level_elements = [
            html.P(f"Resistance: ₹{levels.get('resistance', 0):.2f}", className="level resistance"),
            html.P(f"Pivot: ₹{levels.get('pivot', 0):.2f}", className="level pivot"),
            html.P(f"Support: ₹{levels.get('support', 0):.2f}", className="level support")
        ]
        
        # Sentiment analysis
        sentiment = data.get("sentiment_analysis", {})
        sentiment_summary = html.Div([
            html.P(f"Overall Sentiment: {sentiment.get('overall_sentiment', 'N/A').title()}", 
                  className=f"sentiment {sentiment.get('overall_sentiment', 'neutral').lower()}"),
            html.Div([
                html.P(f"Positive: {sentiment.get('sentiment_scores', {}).get('positive', 0):.1%}"),
                html.P(f"Neutral: {sentiment.get('sentiment_scores', {}).get('neutral', 0):.1%}"),
                html.P(f"Negative: {sentiment.get('sentiment_scores', {}).get('negative', 0):.1%}")
            ], className="sentiment-scores")
        ])
        
        # News list
        news_elements = []
        for article in sentiment.get("articles", []):
            news_elements.append(html.Div([
                html.A(
                    article["title"],
                    href=article["link"],
                    target="_blank",
                    className=f"news-title {article.get('sentiment', 'neutral').lower()}"
                ),
                html.P(f"Date: {article.get('date', 'N/A')}", className="news-date")
            ], className="news-item"))
        
        return (
            info_card,
            fig,
            html.Div(signal_elements),
            html.Div(level_elements),
            sentiment_summary,
            html.Div(news_elements),
            None
        )
        
    except Exception as e:
        logger.error(f"Error updating analysis: {str(e)}")
        return [dash.no_update] * 6 + [str(e)]

if __name__ == "__main__":
    app.run(debug=True) 