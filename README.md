# CMPE 257 - Machine Learning - Individual Project
# Stock Market Predictor using Supervised Learning

### Aim
To examine a number of different techniques to predict future stock prices. We do this by applying supervised learning methods for stock price forecasting by interpreting the seemingly chaotic market data.

## Setup Instructions
	$ pip install -r requirements.txt
	$ python3 src/preprocessing.py
	$ python3 src/featureSelection.py
	$ python3 src/normalizeInterpolate.py
	$ python3 src/regression.py

### Methodology 
1. Pre-processing and Cleaning
2. Feature Selection
3. Data Normalization
3. Analysis of various supervised learning methods
4. Conclusions

### Datasets used
1. http://www.nasdaq.com/
2. https://in.finance.yahoo.com
3. https://www.google.com/finance

### References
- [Scikit-Learn](http://scikit-learn.org/stable/)
- [Theano](http://deeplearning.net/software/theano/)
- [Recurrent Neural Networks - LSTM Models](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [ARIMA Models](http://people.duke.edu/~rnau/411arim.htm)
- https://github.com/dv-lebedev/google-quote-downloader
- [Book Value](http://www.investopedia.com/terms/b/bookvalue.asp)
- http://www.investopedia.com/articles/basics/09/simplified-measuring-interpreting-volatility.asp
- [Volatility](http://www.stock-options-made-easy.com/volatility-index.html)
- https://github.com/dzitkowskik/StockPredictionRNN
- https://github.com/scorpionhiccup
- http://cs229.stanford.edu/proj2013/DaiZhang-MachineLearningInStockPriceTrendForecasting.pdf
- http://cs229.stanford.edu/proj2015/009_report.pdf