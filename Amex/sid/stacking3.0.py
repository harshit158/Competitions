''' for prediction of final dataset '''


from __future__ import division
import csv
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier,AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier





def score_cal(pred,y_test,last):
	print pred.shape,y_test.shape,last.shape
	parties = {'ODYSSEY':0,'CENTAUR': 1, 'TOKUGAWA': 2, 'EBONY':3, 'COSMOS':4}
	sc=zip(pred, last ,y_test) # 0
	result = 0
	for a,b,c in sc:  # a = predicted , b = historic , c = actual
		if parties[c] == parties[a]  and parties[c] == b:
			result += 50
		elif parties[c] == parties[a]  and parties[c] != b:
			result += 100
		elif parties[c] != parties[a]  and parties[c] != b:
			result -= 50

	print 'SCORE: ', result, result*(14/8)
## --------------------------------------------------------------------------------------------------


X_training = np.load('X_training_ultimate2.0.npy')
X_submission = np.load('X_Final.npy')
y_training = np.load('y_training.npy')
# print sum(X_training)


# One hot encoding 
# one=OneHotEncoder(categorical_features=[0], sparse=False)
# X_training=one.fit_transform(X_training)
# X_submission=one.fit_transform(X_submission)



# Scaling 
# scale = StandardScaler()
# X_training_scaled=scale.fit_transform(X_training)
# X_submission_scaled=scale.transform(X_submission)


# print X_submission.shape


# clfs = [RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
#         RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
#         ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
#         ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
#         GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50),
#         LogisticRegression()
#         ]

# n_folds  =10
# skf = list(StratifiedKFold(y_training,n_folds))
# X_blended_training = np.zeros((X_training.shape[0],5*len(clfs)))
# #print X_blended_training.shape
# X_blended_submission = np.zeros((X_submission.shape[0],5*len(clfs)))
# for i,clf in enumerate(clfs):
# 	# if i==5:
# 	# 	X_training,X_submission=X_training_scaled,X_submission_scaled
# 	print "IN CLASSIFIER NUMBER: ",i+1
# 	X_blended_submission_clf = np.zeros((len(X_blended_submission),5))
# 	columns = np.array(range(5)) + i*5
# 	for j,(train,test) in enumerate(skf):
# 		print "FOLD: ",j+1
# 		clf.fit(X_training[train],y_training[train])
		
# 		X_blended_training[test[:, None],columns] = clf.predict_proba(X_training[test])
# 		X_blended_submission_clf = X_blended_submission_clf + clf.predict_proba(X_submission)
# 	X_blended_submission_clf = X_blended_submission_clf/len(skf)
# 	X_blended_submission[:,columns] = X_blended_submission_clf

# print X_blended_training.shape
# print X_blended_submission.shape
# print 'Saving X_blended_training,X_blended_submission'
# np.save('X_blended_training.npy',X_blended_training)
# np.save('X_blended_submission.npy',X_blended_submission)
# print 'Data saved as binary format'
# print 'Blending/Stacking Done'
# print "Cross Validating"




''' loading X_blended_training and X_blended_submission '''
X_blended_training= np.load('X_blended_training.npy')
X_blended_submission=np.load('X_blended_submission.npy')

#--------------------------------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------------------------------------------------

clf = [RandomForestClassifier(n_estimators=100 , n_jobs=-1 , max_depth = 10)     #score: 649600
	   #ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini', max_depth=10),
	   #LogisticRegression(),
	   #GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50)
	  ]


X_train,X_test,Y_train,Y_test =  train_test_split(X_blended_training, y_training, test_size=0.2, random_state = 0)
reverse_dic = {0:'ODYSSEY', 1:'CENTAUR', 2:'TOKUGAWA', 3:'EBONY', 4:'COSMOS'}

for m,clf in enumerate(clf):
	clf.fit(X_train,Y_train)
	accuracy = clf.score(X_test,Y_test)
	pred = clf.predict(X_test)
	last = X_test[:,0]
	print last.shape
	print "Cross validation accuracy", np.mean(pred == Y_test)
	print score_cal(pred,Y_test,last)
	print accuracy
	head = np.array(range(10))
	if accuracy > 0.85:
		clf = RandomForestClassifier(n_estimators=100 , n_jobs=-1 , max_depth = 10)
		clf.fit(X_blended_training,y_training)
		print '1'
		data1 = clf.predict(X_blended_submission)
		print '2'
		data2 = clf.predict_proba(X_blended_submission)	    ## check for the second best CLASSIFIER
		# try:
		# 	data = np.concatenate((data1,data2),axis = 1)
		# 	print data[head[:,None],:]
		# except Exception, e:
		# 	print e
		
		final = []
		for i in range(len(data1)):
			if data1[i] != 'ODYSSEY':
				row = ['C'+str(57093+i),data1[i]]
			else:  # hunt for second best
				m = 0
				for j in range(1,5):
					if data2[i][j] > m:
						m = data2[i][j]
						label = reverse_dic[j]
				
				row = ['C'+str(57093+i),label]

			final.append(row)

		with open('Dunjen_master_IITKharagpur_final.csv','wb') as f:
			a = csv.writer(f) 
			a.writerows(final)
		f.close()

