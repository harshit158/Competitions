from __future__ import division
import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier,AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold

X_training = []
y_training = []
counter = 1
parties = {'ODYSSEY':[0],'CENTAUR': [1], 'TOKUGAWA': [2], 'EBONY':[3], 'COSMOS':[4]}
with open('tr_col.csv','rb') as f:
	reader = csv.reader(f)
	next(reader, None)
	for row in reader:
		y_training.append(row[1])
	 	row = parties[row[2]] + row[3:13] + row[15:23] + [row[34]] + row[36:41] + [row[41]]
		X_training.append(row)
		counter += 1
f.close()
print str(counter) + ' Lines of Training data Processed !!'
X_training = np.array(X_training)
y_training = np.array(y_training)
counter = 0
X_submission = []
with open('le_miss_filled.csv','rb') as f:
	reader = csv.reader(f)
	next(reader, None)
	for row in reader:
	 	age = np.mean([int(j) for j in row[15].strip('+').split('-')])
	 	row = parties[row[1]] +row[2:12] + row[14:15]+ [age] +row[16:22] + [row[23]] + row[25:30] + [row[30]]
		X_submission.append(row)
		counter += 1
f.close()
X_submission = np.array(X_submission)
print str(counter) + ' Lines of Final data preprocesed'

clfs = [RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini',max_depth=5),
        RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='entropy',max_depth=5),
        ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini',max_depth=5),
        ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='entropy',max_depth=5),
        GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50)]
n_folds = 10
skf = list(StratifiedKFold(y_training,n_folds))
X_blended_training = np.zeros((X_training.shape[0],5*len(clfs)))
print X_blended_training.shape
X_blended_submission = np.zeros((X_submission.shape[0],5*len(clfs)))
for i,clf in enumerate(clfs):
	print "IN CLASSIFIER NUMBER: ",i+1
	X_blended_submission_clf = np.zeros((len(X_blended_submission),5))
	columns = np.array(range(5)) + i*5
	for j,(train,test) in enumerate(skf):
		print "FOLD: ",j+1
		clf.fit(X_training[train],y_training[train])
		X_blended_training[test[:, None],columns] = clf.predict_proba(X_training[test])   # bug
		X_blended_submission_clf = X_blended_submission_clf + clf.predict_proba(X_submission)
	X_blended_submission_clf = X_blended_submission_clf/len(skf)
	X_blended_submission[:,columns] = X_blended_submission_clf

print 'Blending/Stacking Done'
clf = AdaBoostClassifier(n_estimators=100 , learning_rate=0.05, algorithm='SAMME' )
clf.fit(X_blended_training,y_training)
print 'Testing Training score: ',clf.score(X_blended_training,y_training)
data = clf.predict(X_blended_submission)
final = []
for i in range(len(data)):
	row = ['C'+str(42820+i),data[i]]
	final.append(row)
with open('Dunjen_master_IITKharagpur_25.csv','wb') as f:
	a = csv.writer(f) 
	a.writerows(final)
f.close()