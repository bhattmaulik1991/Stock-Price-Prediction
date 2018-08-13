#! /usr/bin/python
import sys
import os
import csv
import pandas
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import StratifiedKFold
import numpy as np
import datetime

def main(pathOfDirectory='./input/sample/'):
	files = os.listdir(pathOfDirectory)
	X = np.array([])
	y = np.array([])
	ranking = []
	for nameOfFile in files:
		with open( os.path.join(pathOfDirectory, nameOfFile), 'r') as tfile:
			data = pandas.read_csv(os.path.join(pathOfDirectory, nameOfFile), header=0)
			reader = csv.reader(tfile)
			next(reader, None)
			col = np.array(list(data.adj_close))
			X = col[:20]
			y = [ col[-1], col[-2] ]
			for row in reader:
				if any(row[key] in (None, "") for key in range(len(row))):
					continue
				temp = row[2:7] + row[9:]
				for i in range(len(temp)):
					if "-" not in temp[i]:
						temp[i] = float(temp[i])
					else:
						temp[i] = 0
				if len(temp)!=20:
					temp = temp + [0] * (20-len(temp))
				if len(row[8])!=20:
					row[8] = row[8] + '0' * (20-len(row[8]))
				X=np.concatenate((X, np.array(temp)))
				y=np.concatenate((y, np.array([float(row[8]), 0.0])))
		estimator = LinearRegression()
		selector = RFECV(estimator, step=1, cv=StratifiedKFold(y, 2))
		X = []
		y = []
		ranking = [sum(x) for x in zip(ranking, [selector])]

if __name__ == '__main__':
	main()