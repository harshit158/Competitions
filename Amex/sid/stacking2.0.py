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
from sklearn import mixture
from sklearn import cross_validation




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
# print X_training[0:2]
X_submission = np.load('X_submission_ultimate2.0.npy')
# print X_submission.shape
y_training = np.load('y_training.npy')
# print sum(X_training)

'''
# #  reading edu , region, occupation 
# encod_training=np.load('C:\Users\HS\Desktop\Amex\encode_training.npy')
# # print encod_training[0:2]
# encod_submission=np.load('C:\Users\HS\Desktop\Amex\encode_submission.npy')
# print encod_submission.shape


# #  concatenating 
# X_training=np.concatenate((X_training,encod_training), axis=1)
# X_submission=np.concatenate((X_submission, encod_submission), axis=1)


#One hot encoding 
# one=OneHotEncoder(categorical_features=[0], sparse=False)
# X_training=one.fit_transform(X_training)
# X_submission=one.fit_transform(X_submission)



# Scaling 
# scale = StandardScaler()
# X_training_scaled=scale.fit_transform(X_training)
# X_submission_scaled=scale.transform(X_submission)


print X_submission.shape


clfs = [RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
        RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
        ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
        ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
        GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50),
        # AdaBoostClassifier(algorithm='SAMME',base_estimator=DecisionTreeClassifier(max_depth=13),n_estimators=90,learning_rate=0.1),
        # LinearSVC(C=0.1, penalty="l1", dual=False),
        # mixture.GMM(n_components=5),
        # KNeighborsClassifier(n_neighbors=10),
        LogisticRegression()
        ]

n_folds  =10
skf = list(StratifiedKFold(y_training,n_folds))
X_blended_training = np.zeros((X_training.shape[0],5*len(clfs)))
#print X_blended_training.shape
X_blended_submission = np.zeros((X_submission.shape[0],5*len(clfs)))
for i,clf in enumerate(clfs):
	# if i==5:
	# 	X_training,X_submission=X_training_scaled,X_submission_scaled
	print "IN CLASSIFIER NUMBER: ",i+1
	X_blended_submission_clf = np.zeros((len(X_blended_submission),5))
	columns = np.array(range(5)) + i*5
	for j,(train,test) in enumerate(skf):
		print "FOLD: ",j+1
		clf.fit(X_training[train],y_training[train])
		
		X_blended_training[test[:, None],columns] = clf.predict_proba(X_training[test])
		X_blended_submission_clf = X_blended_submission_clf + clf.predict_proba(X_submission)
	X_blended_submission_clf = X_blended_submission_clf/len(skf)
	X_blended_submission[:,columns] = X_blended_submission_clf


print X_blended_training.shape
print X_blended_submission.shape
print 'Saving X_blended_training,X_blended_submission'
np.save('X_blended_training.npy',X_blended_training)
np.save('X_blended_submission.npy',X_blended_submission)
print 'Data saved as binary format'
print 'Blending/Stacking Done'
print "Cross Validating"


'''

''' loading X_blended_training and X_blended_submission '''
X_blended_training= np.load('X_blended_training.npy')
X_blended_submission=np.load('X_blended_submission.npy')

#--------------------------------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------------------------------------------------

print '\nRunning stackers \n'

clfs = [RandomForestClassifier(n_estimators=100 , n_jobs=-1 , max_depth = 10, oob_score=False),     #score: 649600
	   # ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini', max_depth=10),
	   # LogisticRegression(),
	   # GradientBoostingClassifier(learning_rate=0.01, subsample=0.5, max_depth=6, n_estimators=50)

	  ]


X_train,X_test,Y_train,Y_test =  train_test_split(X_blended_training, y_training, test_size=0.2, random_state = 0)

for m,clf in enumerate(clfs):
	clf.fit(X_train,Y_train)
	accuracy = clf.score(X_test,Y_test)
	print cross_validation.cross_val_score(clf, X_blended_training, y_training, cv=10, n_jobs=1).mean()
	pred = clf.predict(X_test)
	last = X_test[:,0]
	print last.shape
	print "Cross validation accuracy", np.mean(pred == Y_test)
	print score_cal(pred,Y_test,last)
	print accuracy

	if accuracy > 0.85:
		clf.fit(X_blended_training,y_training)
		data = clf.predict(X_blended_submission)
		final = []
		for i in range(len(data)):
			row = ['C'+str(42820+i),data[i]]
			final.append(row)
		with open('Dunjen_master_IITKharagpur_newfeature'+str(m)+'.csv','wb') as f:
			a = csv.writer(f) 
			a.writerows(final)
		f.close()
	
