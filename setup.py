from setuptools import setup, find_packages

setup(
    name="stockai",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'yfinance>=0.2.36',
        'pandas>=2.2.0',
        'numpy>=1.26.3',
        'scikit-learn>=1.3.2',
        'fastapi>=0.109.0',
        'uvicorn>=0.27.0',
        'python-dotenv>=1.0.0',
        'requests>=2.31.0',
        'beautifulsoup4>=4.12.2',
        'transformers>=4.36.2',
        'torch>=2.1.2',
        'ta>=0.10.2',
        'kiteconnect>=4.2.0',
        'python-telegram-bot>=20.7',
        'plotly>=5.18.0',
        'dash>=2.14.2'
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-powered stock analysis for Indian markets",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/stockai",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
) 