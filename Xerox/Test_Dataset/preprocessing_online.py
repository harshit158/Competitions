from __future__ import division
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier,AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import mixture
from sklearn import cross_validation
import pandas as pd
import numpy as np
import sys
##----------------------------------------------------------------------------------------------------------
# Get X_train ready first
# age = np.genfromtxt('id_age_train.csv',delimiter = ",")
# age = age[1:,1][:,None]  # Discarding header and ID Column
# X_train = np.load('VF.npy')
# X_train = np.concatenate((X_train,age), axis = 1)   ## AGE COLUMN IN TRAINING
# y_train = np.genfromtxt('id_label_train.csv',delimiter = ",")
# y_train = y_train[1:,1][:,None]
# np.save('X_train.npy',X_train)
# np.save('y_train.npy',y_train)

#----------------------------------------------------------------------------------------------------------
# Defina feature extractor
def extract(df,i):
	#1 Get slope
	clf = LinearRegression(n_jobs = -1)
	idx = range(4,8) + [9]

	
	df.loc[df.V3 <= 40,'V3'] = df.V3.median()
	df.loc[(df.V4 <= 8) & (df.V4 >= 40) ,'V4'] = df.V4.median()
	df.loc[df.V6 <= 90,'V6'] = df.V6.median()

	slopes = []
	for j in idx:
		## look out for unfilled data frame for slope calculation ##
		x = df.iloc[:,1]  # Time stamp
		y = df.iloc[:,j]  # Variable

		x = x.loc[y.notnull()]
		y = y.loc[y.notnull()]

		
		if x.shape[0] >= 2:

			x1 = x.iloc[0:len(x)//2]
			x2 = x.iloc[len(x)//2:len(x)]

			y1 = y.iloc[0:len(y)//2]
			y2 = y.iloc[len(y)//2:len(y)]
			
			clf.fit(x1[:,None],y1)
			coef1 = clf.coef_[0]
			clf.fit(x2[:,None],y2)
			coef2 = clf.coef_[0]
		elif x.shape[0] == 1:

			clf.fit(x[:,None],y)
			coef1 = clf.coef_[0]
			coef2 = coef1

		else:
			coef1 = 0
			coef2 = 0
		slopes.append(coef1)
		slopes.append(coef2)
	# print slopes
	df1 = df[['V3','V4','V5','V6','V12']]
	g0 = pd.isnull(df1).sum().values.tolist()
	df = df.fillna(df.mean())
	df1 = df[['V3','V4','V5','V6','V12']]

	g1 = df1.apply(lambda x: np.average(x , weights = np.array(x.index+1),axis = 0))
	g1 = np.array(g1).tolist()
	# print '1 ',g1

	g2 = df1.max().values.tolist() # max
	# print '2 ',g2

	g3 = df1.min().values.tolist() # min
	# print '3 ',g3

	g4 = (np.array(g2) - np.array(g3)).tolist() # diff
	# print '4 ',g4

	g5 = df1.tail(1).values.tolist() # last value
	g5 = g5[0]
	# print '5 ',g5

	g6 =  [sum(df.ICU)/len(df1)]
	# print '6 ', g6

	g7 = [age.AGE[i]]
	# print '7 ', g7
	final = np.concatenate((slopes,g0,g1,g2,g3,g4,g5,g6,g7),axis = 1)
	# print final.shape
	return final





#----------------------------------------------------------------------------------------------------------
#Get X_val ready
path1 = sys.argv[1]
vit=pd.read_csv(path1)
vit = vit.fillna(vit.mean())
vit['V12'] = vit.V1/vit.V2
path2 = sys.argv[2]
age = pd.read_csv(path2,index_col = 'ID')
l1,l2=age.index.min(),(age.index.max()+1)
x_submission=[]
stamps=[]
j=0
for i in range(l1,l2):
    lower=j
    print 'ID = ' + str(i)  + '           '+ str(j) 
    df=vit.iloc[lower:j+1]
    # df2=lab.iloc[lower:j+1]
    while(int(df.tail(1).ID)==i and j<vit.shape[0]):  

        icu=int(df.tail(1).ICU)
        if icu==1:

            x_sub = extract(df,i)

            x_submission.append(x_sub)

            stamps.append([int(df.tail(1).ID),int(df.tail(1).TIME)])


        j+=1
        # df2=df2.append(lab.iloc[j:j+1],ignore_index=True)
        df=df.append(vit.iloc[j:j+1], ignore_index=True) # appending each new row to old dataframe and passing to function

x_submission=np.array(x_submission)
print 'Saving x_submission and stamps'
print x_submission.shape
np.save('X_online_test.npy',x_submission)
np.save('stamps_online_test.npy',stamps)