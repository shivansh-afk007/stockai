# StockAI - AI-Powered Stock Analysis for Indian Markets

StockAI is a comprehensive stock analysis platform that combines technical analysis, sentiment analysis, and machine learning to provide insights for Indian stock market investors.

## Features

- **Real-time Stock Data**: Fetch and cache real-time stock data from Yahoo Finance
- **Technical Analysis**: Calculate and visualize various technical indicators
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
  - Moving Averages (SMA, EMA)
  - Support and Resistance Levels
  - Volume Analysis
- **Sentiment Analysis**: Analyze market sentiment using news and social media
  - News sentiment analysis using FinBERT
  - Real-time news aggregation
  - Sentiment scoring and classification
- **Modern Web Interface**: Interactive dashboard built with Dash
  - Responsive design
  - Real-time updates
  - Interactive charts
  - User-friendly controls

## Project Structure

```
stockai/
├── requirements.txt        # Project dependencies
├── README.md              # Project documentation
├── src/                   # Source code
│   ├── data/             # Data fetching and processing
│   │   ├── __init__.py
│   │   └── stock_data.py
│   ├── analysis/         # Analysis modules
│   │   ├── __init__.py
│   │   ├── technical.py
│   │   └── sentiment.py
│   ├── api/              # FastAPI backend
│   │   ├── __init__.py
│   │   └── main.py
│   └── dashboard/        # Dash frontend
│       ├── __init__.py
│       ├── app.py
│       └── assets/
│           └── style.css
└── tests/                # Unit tests
    ├── __init__.py
    └── test_stock_analysis.py
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/stockai.git
cd stockai
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the FastAPI backend:
```bash
cd src/api
uvicorn main:app --reload
```

2. Start the Dash frontend:
```bash
cd src/dashboard
python app.py
```

3. Open your browser and navigate to:
- Dashboard: http://localhost:8050
- API Documentation: http://localhost:8000/docs

## API Endpoints

### Stock Analysis
- `POST /analyze`: Comprehensive stock analysis
  ```json
  {
    "symbol": "RELIANCE.NS",
    "start_date": "2024-01-01",
    "end_date": "2024-03-01",
    "include_technical": true,
    "include_sentiment": true
  }
  ```

### Technical Analysis
- `GET /technical/{symbol}`: Get technical analysis for a stock
  - Query parameters:
    - `start_date`: Analysis start date
    - `end_date`: Analysis end date

### Sentiment Analysis
- `GET /sentiment/{symbol}`: Get sentiment analysis for a stock

### Stock Information
- `GET /stock_info/{symbol}`: Get basic stock information

## Development

### Running Tests
```bash
python -m pytest tests/
```

### Code Style
The project follows PEP 8 style guidelines. Use `black` for code formatting:
```bash
black src/
```

## Dependencies

- **Data Processing**
  - yfinance: Yahoo Finance data
  - pandas: Data manipulation
  - numpy: Numerical computations

- **Analysis**
  - scikit-learn: Machine learning
  - ta: Technical analysis
  - transformers: FinBERT model
  - beautifulsoup4: Web scraping

- **Backend**
  - fastapi: API framework
  - uvicorn: ASGI server

- **Frontend**
  - dash: Web framework
  - plotly: Interactive charts

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [FinBERT](https://huggingface.co/ProsusAI/finbert) for financial sentiment analysis
- [ta](https://technical-analysis-library-in-python.readthedocs.io/) for technical analysis indicators
- [yfinance](https://github.com/ranaroussi/yfinance) for stock data

## Contact

For questions and feedback, please open an issue on GitHub. 