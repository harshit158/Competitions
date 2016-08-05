from __future__ import division
import pandas as pd
import matplotlib as plt
import csv
import numpy as np
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import cross_validation, grid_search
import math
from sklearn.externals import joblib
from sklearn.preprocessing import OneHotEncoder



f=open('tr_col2.csv','rb')
reader=csv.reader(f)
next(reader, None)
# next(reader, None)

X=[]
y=[]
last=[]

party_dict={'CENTAUR':0,'COSMOS':1,'EBONY':2,'TOKUGAWA':3,'ODYSSEY':4}
don_party_dict={'CENTAUR':24,'COSMOS':28,'EBONY':25,'TOKUGAWA':26,'ODYSSEY':27}
occ_dict={'Others':0,'Teacher':1,'Director':2,'Consultant':3,'Middle Management':4,'House Husband':5,'Self Employed':6,'Doctor':7,'Social Worker':8,'Magician':9,'Pilot':11,'Janitor':12,'Senior Management':13,'Scientist':14,'Lawyer':15,'Chef':16,'Dentist':17,'Politician':18,'Service':19,'Military':20,'Factory Manager':21,'Student':22,'Amabassador':10}
edu_dict={'primary':0,'MBA':1,'professional':2,'doctorate':3}
region_raw=[3, 51, 29, 56, 27, 36,  8, 42, 17, 43, 45,  4, 55, 32, 18, 22, 37,
       39,  2, 47, 23, 13,  5, 21, 49, 34, 12, 28, 41, 48,  7, 54, 24, 46,
        9, 33, 25,  6, 16, 14, 19, 11, 15, 35, 52, 20, 10, 31,  1, 40, 26,
       38, 44, 50]


region_dict={}
i=0
for item in region_raw:
	region_dict[str(item)]=i
	i+=1

for row in reader:
	'''though many features are engineered , only few are taken ( see the final 'arr' being created ) ''' 

	encod_arry=[0,0,0,0,0] # one hot encoding for party
	encod_arry[party_dict[row[2]]]=1#float(row[don_party_dict[str(row[2])]])
	encode_without=[float(party_dict[row[2]]+1)]
	

	don_party_dict2={int(row[3]):'CENTAUR',int(row[7]):'COSMOS',int(row[4]):'EBONY',int(row[5]):'TOKUGAWA',int(row[6]):'ODYSSEY'}
	don_encode=[0,0,0,0,0]
	don_encode[party_dict[don_party_dict2[max(don_party_dict2.keys())]]]=1 # one hot encoding for max(donations)
	

	soc_party_dict2={float(row[8]):'CENTAUR',float(row[11]):'COSMOS',float(row[9]):'EBONY',float(row[10]):'TOKUGAWA',float(row[12]):'ODYSSEY'}
	soc_encode=[0,0,0,0,0]
	soc_encode[party_dict[soc_party_dict2[max(soc_party_dict2.keys())]]]=1 # one hot encoding for max(donations)

	final=map(lambda (x,y,z ):x+y+z, zip(encod_arry,don_encode, soc_encode))

	occ_arry=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] # one hot encoding for occupation
	occ_arry[occ_dict[row[13]]]=1
	occ_without=[float(occ_dict[row[13]]+1)] # WITHOUT one hot encoding

	edu_arry=[0,0,0,0]     # one hot encoding for education
	edu_arry[edu_dict[row[23]]]=1
	# edu_arry=[float(item)*float(row[16]) for item in edu_arry ]
	edu_without=[float(edu_dict[row[23]]+1)] # WITHOUT one hot encoding

	region_arry=[0 for i in range(54)]
	region_arry[region_dict[row[14]]]=1
	# region_arry=[float(item)*(float(row[20])) for item in region_arry]
	region_without=[float(region_dict[row[14]]+1)/(float(row[16])+float(row[20]))] # WITHOUT one hot encoding

	arr=[]
	arr=[float(row[i]) for i in range(3,13)]
	# arr=[float(row[i])/sum(arr) for i in range(3,8)] #normalizing donations by row

	# arr2=[float(row[i]) for i in range(8,13)]
	# # arr2=[float(row[i])/sum(arr2) for i in range(8,13)] # normalizing social shares by row
	# arr=arr+arr2

	arr.append(float(row[15])) # h_size
	arr.append(float(row[16])) # age
	arr.append(float(row[17])) # married
	arr.append(float(row[18])) # ownership
	arr.append(float(row[19])) # political affiliation
	arr.append(float(row[20])) # years
	arr.append(float(row[21])) # perc
	arr.append(float(row[22])) # unique
	# arr.append((float(row[24])*float(row[36]))) # cenXcenXral
	# arr.append((float(row[25])*float(row[37]))) # eboXeboXral
	# arr.append((float(row[26])*float(row[38]))) # tokXtokXral
	# arr.append((float(row[27])*float(row[39]))) # ody_odyXral
	# arr.append((float(row[28])*float(row[40]))) # cosXcosXral
	# arr.append(float(row[33]))#*float(row[21])*float(row[34])) #unique*pol *perc* total 
	arr.append(float(row[34])) # total rallies ORIGINAL
	# arr.append(tot) # tot rallies as per REGRESSION
	arr.append(float(row[35])) # total donation per citizen including all parties

	
	# arr.append(float(row[43])/float(row[34])) # raly centaur
	arr.append(float(row[36]))
	# arr.append(int(float(row[43])==0)) # 0 OR 1
	arr.append(float(row[37]))
	# arr.append(int(float(row[44])==0)) # 0 OR 1
	# arr.append(float(row[45])/float(row[34])) # raly tokugawa
	arr.append(float(row[38]))
	# arr.append(int(float(row[45])==0)) # 0 OR 1
  	# arr.append(float(row[46])/float(row[34])) # raly odyssey
  	arr.append(float(row[39]))
  	# arr.append(int(float(row[46])==0)) # 0 OR 1
	# arr.append(float(row[47])/float(row[34])) # raly cosmos
	arr.append(float(row[40]))
	# arr.append(int(float(row[47])==0)) # 0 OR 1
	arr.append(float(row[41])) # income
	# arr.append(float(row[42])) # total rallies - MODIFIED
	


	if '' in arr:
		continue

	'''final arr creation'''

	arr=[float(item) for item in arr]
	# arr= arr+ edu_without + encode_without + region_without + occ_without +soc_encode+don_encode+final+edu_arry+ region_arry  +occ_arry+encod_arry
	arr=edu_without + region_without + occ_without #+soc_encode+don_encode
	

	X.append(arr)
	# last.append(int(arr[-1]))
	y.append(row[1])

print len(X[0])

# arr_np=np.array(X)
# print arr_np.shape
# np.save('encode_training.npy',arr_np)
# # quit()

'''Feature selection techniques '''

#***************** Feature selection chi 2
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
# from sklearn.feature_selection import f_classif
# X= SelectKBest(chi2, k=25).fit_transform(X, y)
# print X[0]

#***************** Feature selection forest
# from sklearn.ensemble import ExtraTreesClassifier
# clf = ExtraTreesClassifier()
# X = clf.fit(X, y).transform(X)
# print (clf.feature_importances_)
# ar=clf.feature_importances_
# sorted_ar=sorted((e,i) for i,e in enumerate(ar))
# for item in sorted_ar:
# 	print item

#***************** Feature selection SVM
# from sklearn.svm import LinearSVC
# X = LinearSVC(C=0.01, penalty="l1", dual=False).fit_transform(X, y)


print 'No of features :'+str(len(X[0])) 

#****************** dividing into test train

def test_train(size):
	x_train,x_test,y_train,y_test=train_test_split(X,y,test_size= float(size))
	print len(x_train),len(y_train)
	return x_train,x_test,y_train,y_test
	


''' 

***********    working with 'Leaderboard' submission file *********************************************************************

'''

#************************getting data from Leaderboard

test_file=open('le_miss_filled.csv','rb')
test_reader=csv.reader(test_file)
next(test_reader, None)

party_dict={'CENTAUR':0,'COSMOS':1,'EBONY':2,'TOKUGAWA':3,'ODYSSEY':4}
don_party_dict={'CENTAUR':24,'COSMOS':28,'EBONY':25,'TOKUGAWA':26,'ODYSSEY':27}
occ_dict={'Others':0,'Teacher':1,'Director':2,'Consultant':3,'Middle Management':4,'House Husband':5,'Self Employed':6,'Doctor':7,'Social Worker':8,'Magician':9,'Amabassador':10,'Pilot':11,'Janitor':12,'Senior Management':13,'Scientist':14,'Lawyer':15,'Chef':16,'Dentist':17,'Politician':18,'Service':19,'Military':20,'Factory Manager':21,'Student':22}
edu_dict={'primary':0,'MBA':1,'professional':2,'doctorate':3}

x_lead=[]
for row in test_reader:

	arr=[]
	arr=[row[i] for i in range(2,12)]
	arr.append(int(row[14])) # h_size
	age_nums=row[15].strip('+').split('-') # taking mean of age groups
	age_nums=[int(item) for item in age_nums] # converting the splitted numbers from age groups into integers
	mean=sum((age_nums))/len(age_nums)
	arr.append(int(mean)) # age
	arr.append(int(mean)*float(row[18])) # age X political affiliation
	arr.append(float(row[16])) # married
	arr.append(float(row[17])) # ownership
	arr.append(float(row[18])) # political affiliation
	arr.append(float(row[19])) # years
	arr.append(float(row[20])) # perc
	arr.append(float(row[21])) # unique
	# arr.append(float(row[22])) # unique*pol 
	arr.append(float(row[2])*float(row[7])*float(row[25])) # cenXcenXral
	arr.append(float(row[3])*float(row[8])*float(row[26])) # eboXeboXral
	arr.append(float(row[4])*float(row[9])*float(row[27])) # tokXtokXral
	arr.append(float(row[5])*float(row[11])*float(row[28])) # ody_odyXral
	arr.append(float(row[6])*float(row[10])*float(row[29])) # cosXcosXral
	arr.append(float(row[23])) # total rallies attended
	arr.append(float(row[24])) # total donation per citizen including all parties
	arr.append(float(row[25])/float(row[23])) # raly centaur
	arr.append(float(row[25]))
	arr.append(int(float(row[25])==5.498)) # 0 OR 1
	arr.append(float(row[26])/float(row[23])) # raly ebony
	arr.append(float(row[26]))
	arr.append(int(float(row[26])==3.362)) # 0 OR 1
	arr.append(float(row[27])/float(row[23])) # raly tokugawa
	arr.append(float(row[27]))
	arr.append(int(float(row[27])==2.494)) # 0 OR 1
  	arr.append(float(row[28])/float(row[23])) # raly odyssey
  	arr.append(float(row[28]))
  	arr.append(int(float(row[28])==4.635)) # 0 OR 1
	arr.append(float(row[29])/float(row[23])) # raly cosmos
	arr.append(float(row[29]))
	arr.append(int(float(row[29])==6.369)) # 0 OR 1
	arr.append(float(row[30])) # income
	# arr.append(float(row[42])) # total rallies - MODIFIED


	''' categorical variables '''
	occ_without=[float(occ_dict[row[12]]+1)] # WITHOUT one hot encoding
	edu_without=[float(edu_dict[row[31]]+1)] # WITHOUT one hot encoding
	region_without=[float(region_dict[row[13]]+1)] # WITHOUT one hot encoding
	encode_without=[float(party_dict[row[1]]+1)] # WITHOUT one hot encoding


	party_dict={'CENTAUR':0,'COSMOS':1,'EBONY':2,'TOKUGAWA':3,'ODYSSEY':4}

	don_party_dict2={int(row[2]):'CENTAUR',int(row[6]):'COSMOS',int(row[3]):'EBONY',int(row[4]):'TOKUGAWA',int(row[5]):'ODYSSEY'}
	don_encode=[0,0,0,0,0]
	don_encode[party_dict[don_party_dict2[max(don_party_dict2.keys())]]]=1 # one hot encoding for max(donations)

	soc_party_dict2={float(row[7]):'CENTAUR',float(row[10]):'COSMOS',float(row[8]):'EBONY',float(row[9]):'TOKUGAWA',float(row[11]):'ODYSSEY'}
	soc_encode=[0,0,0,0,0]
	soc_encode[party_dict[soc_party_dict2[max(soc_party_dict2.keys())]]]=1 # one hot encoding for max(donations)

	encod_arry=[0,0,0,0,0] # one hot encoding for party
	encod_arry[party_dict[row[1]]]=1

	occ_arry=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] # one hot encoding for occupation
	occ_arry[occ_dict[row[12]]]=1

	edu_arry=[0,0,0,0]     # one hot encoding for education
	edu_arry[edu_dict[row[31]]]=1

	region_arry=[0 for i in range(54)]
	region_arry[region_dict[row[13]]]=1

	final=map(lambda (x,y,z ):x+y+z, zip(encod_arry,don_encode, soc_encode))






	if '' in arr:
		continue
	arr=[float(item) for item in arr]
	arr=arr + occ_without + edu_without   +region_without+ encode_without + final +encod_arry + soc_encode + don_encode + occ_arry + edu_arry + region_arry
	arr= edu_without + region_without  + occ_without
	x_lead.append(arr)

print len(x_lead[0])
arr_np=np.array(x_lead)
print arr_np.shape
np.save('encode_submission.npy',arr_np)
print arr_np.shape

quit()

# ''' the fucking Classifier '''

##**********************************
# #training classifier


'''calculating the final score '''

def score_cal(pred,y_test):
	sc=zip(pred, last ,y_test) # 0
	result = 0
	for a,b,c in sc:  # a = predicted , b = historic , c = actual
		if party_dict[c] == party_dict[a] and party_dict[c] == b - 1:
			result += 50
		elif party_dict[c] == party_dict[a] and party_dict[c] != b -1:
			result += 100
		elif party_dict[c] != party_dict[a] and party_dict[c] != b -1:
			result -= 50

	print 'SCORE: ', result, result*(14/8)

# #******NB
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()
# clf.fit(x_train, y_train)
# pred=clf.predict(x_test)

# print accuracy_score(y_test, pred)

#******SVM
# from sklearn import svm
# clf=svm.SVC()
# clf.fit(x_train, y_train)
# pred=clf.predict(x_test)
# print accuracy_score(y_test, pred)

#*****Linear SVC
from sklearn.svm import LinearSVC
clf1=LinearSVC(C=0.1, penalty="l1", dual=False)
def lin_svm(x_train,y_train,x_test,y_test):
	clf1.fit(x_train, y_train)
	pred=clf1.predict(x_test)
	print '\nSVM  : \n' +str(accuracy_score(y_test, pred))
	score_cal(pred,y_test)
	# prob= clf1.predict_proba(x_test)
	 
	return pred#, prob


#*****Gradient boost classifier
from sklearn.ensemble import GradientBoostingClassifier
clf2 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,max_depth=10, random_state=0, subsample=0.5, max_features=5)

def gradient(x_train,y_train,x_test,y_test):
	clf2.fit(x_train, y_train)
	pred=clf2.predict(x_test)
	print '\nGradient : \n' +str(accuracy_score(y_test, pred))
	score_cal(pred,y_test)
	# prob= clf1.predict_proba(x_test)
	 
	return pred#, prob

#***** Adaboost
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
clf3 = AdaBoostClassifier(algorithm='SAMME',base_estimator=DecisionTreeClassifier(max_depth=13),n_estimators=90,learning_rate=0.1)

def adaboost(x_train,y_train,x_test,y_test):
	clf3.fit(x_train, y_train)
	pred=clf3.predict(x_test)
	print '\nAdaboost : \n' +str(accuracy_score(y_test, pred))
	score_cal(pred,y_test)
	# prob= clf3.predict_proba(x_test)
	
	return pred#, prob

#****** Random 
from sklearn.ensemble import RandomForestClassifier
clf4 = RandomForestClassifier(n_estimators=300,criterion='entropy', n_jobs=-1, max_depth=30)#, random_state=50, oob_score=True, min_samples_leaf=50)

def forest(x_train,y_train,x_test,y_test):
	clf4.fit(x_train, y_train)
	pred=clf4.predict(x_test)
	print '\nRandom : \n' +str(accuracy_score(y_test, pred))
	score_cal(pred,y_test)
	# prob= clf4.predict_proba(x_test)
	 
	return pred



'''cross_validation '''

# param_grid= {
 		
# 		'max_features': ['auto', 'sqrt', 'log2']
			
# 			  }
# lrgs = grid_search.GridSearchCV(estimator=clf4, param_grid=param_grid, n_jobs=1, cv=10)
# scores = cross_validation.cross_val_score(lrgs, X, y, cv=10, n_jobs=1)
# # print 'best score :' + str(lrgs.best_score_)
# print 'best parameter :' +str(lrgs.max_depth)
# print 'best param combination : ' + str(lrgs.best_params_)
# print scores
# print len(scores)
# print scores.mean()
# lrgs=lrgs.fit(X,y)
# joblib.dump(lrgs, 'forest.pkl') 
# clf = joblib.load('forest.pkl')






''' prediction vs submission '''

x_train,x_test,y_train,y_test=test_train(0.2)
# print len(x_train), len(x_test), len(y_train), len(y_test)

# pred1=lin_svm(x_train,y_train,x_test,y_test)
# pred2=gradient(x_train,y_train,x_test,y_test)
# pred3=adaboost(x_train,y_train,x_test,y_test)
pred4=forest  (x_train,y_train,x_test,y_test)

# write=open('probs.csv','wb')
# writer=csv.writer(write)

# zipped=zip(prob1,prob2, prob3, prob4)
# print len(zipped)
# print len(zipped[0])
# prob_arr=[]
# i=0
# for item in zipped:
# 	feat=[]
# 	for inner in item:
# 		for p in inner:
# 			feat.append(float(p))
# 	prob_arr.append(feat)

# writer.writerows(prob_arr)
# print len(prob_arr)
# print len(prob_arr[0])
# print prob_arr[0]
# 		feat.append(float(inner))



'''writing result '''
# clf_dict={1:'svm',2:'grad', 3:'ada', 4:'forest'}
# i=1
# for item in [pred1, pred2, pred3, pred4]:
# 	w=open(clf_dict[i]+'_result','wb')
# 	for row in item:
# 		w.write(row+'\n')
# 	w.close()
# 	i+=1
# print len(pred4)

# for item in pred2:
# 	w=open('tr_grad_result','wb')
# 	for row in item:
# 		w.write(row+'\n')
# 	w.close()

'''
If Actual Vote  = Predicted Vote  and Actual Vote     =  Historical Vote, then score = 50
If Actual Vote  = Predicted Vote   and Actual Vote    ^=  Historical Vote, then score = 100
If Actual Vote  ^= Predicted Vote   and Actual Vote   =   Historical Vote, then score = 0
If Actual Vote  ^= Predicted Vote   and Actual Vote  ^=  Historical Vote, then score = -50
'''

''' k means '''
# from sklearn.cluster import KMeans
# from sklearn import metrics
# X=np.array(X)
# kmeans_model = KMeans(n_clusters=100, random_state=1, n_jobs=1).fit(X)
# labels = kmeans_model.labels_
# dist=kmeans_model.transform(X)
# print (dist[0][0])
# print len(labels)
# print kmeans_model.score(X)
# scr= (1/len(X)) * Sum(distance(x(i),c(centroid(i)))) for each i in [1,|x|]
# # metrics.silhouette_score(np.array(X), labels, metric='euclidean')