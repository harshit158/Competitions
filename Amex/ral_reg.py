''' regression for predicting raly attendences '''
from __future__ import division
import pandas as pd
import matplotlib as plt
import csv
import numpy as np
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import math
import csv
from sklearn.tree import DecisionTreeRegressor

'''creating test and train'''
# import pandas as pd
# tr=pd.read_csv('tr_col2.csv')
# new=tr
# new=tr[tr.r_cen_reg.isnull()]
# new.to_csv('r_cen_empty.csv')
# new=tr[tr.r_cen_reg.notnull()]
# new.to_csv('r_cen_train.csv')


train_file=open('r_tok_train.csv','rb')
train=csv.reader(train_file)
next(train, None)
# next(train, None)


train_dict={}
target_dict={}
X=[]
y=[]
x_target=[]
i=1
for row in train:
	arr=[]	
	arr=[float(row[i]) for i in range(3,13)]  # donations and social shares
	arr.append(float(row[16])) # age
	arr.append(float(row[19])) # political affiliation
	arr.append(float(row[20])) # years
	arr.append(float(row[21])) # perc
	arr.append(float(row[22])) # unique
	arr.append(float(row[34])) # total rallies ORIGINAL
	arr.append(float(row[35])) # total donation per citizen including all parties
	arr.append(float(row[41])) # income


	X.append(arr)
	y.append(float(row[4]))

# print len(x_target), len(X[0]), len(y)
# print X[0]



# ''' reading target data '''

train_file=open('r_tok_empty.csv','rb')
train=csv.reader(train_file)
next(train, None)

x_target=[]
for row in train:
	arr=[]	
	arr=[float(row[i]) for i in range(3,13)]  # donations and social shares
	arr.append(float(row[16])) # age
	arr.append(float(row[19])) # political affiliation
	arr.append(float(row[20])) # years
	arr.append(float(row[21])) # perc
	arr.append(float(row[22])) # unique
	arr.append(float(row[34])) # total rallies ORIGINAL
	arr.append(float(row[35])) # total donation per citizen including all parties
	arr.append(float(row[41])) # income

	x_target.append(arr)

print 'total :' + str(len(x_target)+len(X))

# ''' Regressor '''

from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import cross_val_score	
regr = RandomForestRegressor(random_state=0, n_estimators=100)
regr.fit(X,y)
pred=regr.predict(x_target)
scores = regr.score(X,y)
print 'score :' +str(scores)
print pred


# ''' writing predictions to file '''
w=open('r_tok_predicted', 'wb')
for item in pred:
	w.write(str(int(round(item)))+'\n')

# raw_input('\nConcatenate')

# pr=pd.read_csv('r_cen_predicted.csv','rb')













	






