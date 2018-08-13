from __future__ import print_function
from modelHelper import addFeatures, applyTimeLag, performRegression
import sys
import os
import pickle
import traceback
from sklearn import preprocessing
import pandas as pd

def main(dir_path='./input/sample/', output_dir='./output/'):

    scores = {}
    files = os.listdir(dir_path)
    maxdelta = 30
    delta = range(8, maxdelta)
    print('Delta days accounted: ', max(delta))
    for file_name in files:
        try:
            symbol = file_name.split('.')[0]
            print(symbol)
            path = os.path.join(dir_path, file_name)
            out = pd.read_csv(path, index_col=2, parse_dates=[2])
            out.drop(out.columns[0], axis=1, inplace=True)
            datasets = [out]    
            cols = ['high', 'open', 'low', 'close', 'volume', 'adj_close']
            for dataset in datasets:
                columns = dataset.columns
                adjclose = columns[-2]
                returns = columns[-1]
                for dele in delta:
                    addFeatures(dataset, adjclose, returns, dele)
                dataset = dataset.iloc[max(delta):,:] 
            finance = pd.concat(datasets)
            high_value = 365
            high_value = min(high_value, finance.shape[0] - 1)
            lags = range(high_value, 30)
            print('Maximum time lag ', high_value)
            if 'symbol' in finance.columns:
                finance.drop('symbol', axis=1, inplace=True)
            print('Size of data frame: ', finance.shape)
            print('Number of None after merging: ', (finance.shape[0] * finance.shape[1]) - finance.count().sum())
            finance = finance.interpolate(method='time')
            print('Number of None after time interpolation: ', finance.shape[0]*finance.shape[1] - finance.count().sum())
            finance = finance.fillna(finance.mean())
            print('Number of None after mean interpolation: ', (finance.shape[0]*finance.shape[1] - finance.count().sum()))
            finance.columns = [str(col.replace('&', '_and_')) for col in finance.columns]
            finance.open = finance.open.shift(-1)
            print(high_value)
            finance = applyTimeLag(finance, [high_value], delta)
            print('Number of None after temporal shifting: ', (finance.shape[0] * finance.shape[1]) - finance.count().sum())
            print('Size of data frame after feature creation: ', finance.shape)
            mean_squared_errors, r2_scores = performRegression(finance, 0.95, symbol, output_dir)
            scores[symbol] = [mean_squared_errors, r2_scores]
        except Exception:
            pass
            traceback.print_exc()
    with open(os.path.join(output_dir, 'scores.pickle'), 'wb') as handle:
        pickle.dump(scores, handle)

if __name__ == '__main__':
    main()
