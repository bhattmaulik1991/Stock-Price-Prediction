import os
import sys
import pandas as pd
from sklearn import preprocessing

def main(pathOfDirectory='./input/sample/', oppathOfDirectory='./input/normalized/'):
    files = os.listdir(pathOfDirectory)
    for file_name in files:
        dframe = pd.read_csv(os.path.join(pathOfDirectory, file_name))
        cols = ['high', 'open', 'low', 'close', 'volume', 'adj_close']
        for col in cols:
            dframe[col] = dframe[col].interpolate('linear', order=2)
        for col in cols:
            preprocessing.normalize(dframe[col].values.reshape(-1,1), axis=1, norm='l2', copy=False)
        dframe.to_csv(os.path.join(oppathOfDirectory, file_name), encoding='utf-8')

if __name__ == '__main__':
    #main(sys.argv[1], sys.argv[2])
    main()