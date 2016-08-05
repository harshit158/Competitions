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
from sklearn import preprocessing


''' for neural net '''
''' ****************   Using Pybrain ****************** '''
from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer

alldata = ClassificationDataSet(12, 1, nb_classes=5)



f=open('tr_col2.csv','rb')
reader=csv.reader(f)
next(reader, None)
# next(reader, None)

X=[]
y=[]
last=[]

party_dict={'CENTAUR':0,'COSMOS':1,'EBONY':2,'TOKUGAWA':3,'ODYSSEY':4}
don_party_dict={'CENTAUR':24,'COSMOS':28,'EBONY':25,'TOKUGAWA':26,'ODYSSEY':27}
occ_dict={'Others':0,'Teacher':1,'Director':2,'Consultant':3,'Middle Management':4,'House Husband':5,'Self Employed':6,'Doctor':7,'Social Worker':8,'Magician':9,'Amabassador':10,'Pilot':11,'Janitor':12,'Senior Management':13,'Scientist':14,'Lawyer':15,'Chef':16,'Dentist':17,'Politician':18,'Service':19,'Military':20,'Factory Manager':21,'Student':22}
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
 
	arr=[]
	arr=[float(row[i]) for i in range(3,13)]
	arr.append(float(row[15])) # h_size
	arr.append(float(row[16])) # age

	if '' in arr:
		continue

	'''final arr creation'''

	# arr=[float(item) for item in arr]
	# arr= arr+ occ_without + edu_without +region_without+ encode_without  + final+encod_arry+soc_encode+don_encode +occ_arry+edu_arry+ region_arry# + ral_encode
	X.append(arr)
	y.append(party_dict[row[1]])

	# alldata.addSample(arr, party_dict[row[1]])

X=np.array(X)
X_scaled=preprocessing.normalize(X)
print X_scaled[0]
for i,item in enumerate(X_scaled):
	alldata.addSample(item,y[i])

print 'No of features ' + str(len(X[0]))

''' train-test data '''
# x_train,x_test,y_train,y_test=train_test_split(X,y,test_size= 0.2)
tstdata, trndata = alldata.splitWithProportion( 0.2 )

# trndata._convertToOneOfMany( )
# tstdata._convertToOneOfMany( )

# print tstdata['target']


fnn = buildNetwork( trndata.indim, 2, trndata.outdim, outclass=SoftmaxLayer, bias=True )
trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.0, learningrate=50,verbose=True, weightdecay=0.01)
# trainer.trainUntilConvergence( verbose = True, validationProportion = 0.2, maxEpochs = 1000, continueEpochs = 10 )
trainer.trainEpochs(5)
# p = fnn.activateOnDataset( tstdata )

# print p



# print len(alldata)




