from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
output_name=sys.argv[1]
x = pd.read_csv(output_name,names = ['ID','TIME','LABEL'])
x_predict = x.groupby(by = 'ID').max()
print sum(x_predict.LABEL == 1)
del x_predict['TIME']
x_predict =  x_predict.LABEL
x_predict.index = range(0,1198)
y = pd.read_csv('id_label_val.csv',names = ['ID','LABEL'])
TP = sum((y.LABEL == 1) & (x_predict == 1))
TN = sum((y.LABEL == 0) & (x_predict == 0))
FP = sum((y.LABEL == 0) & (x_predict == 1))
FN = sum((y.LABEL == 1) & (x_predict == 0))
# print TP,TN,FP,FN
print 'Sensitivity: ', (TP/(TP+FN))
print 'Specificity: ', (TN/(TN+FP))
print 'Accuracy: ', (TP+TN)/(TP+TN+FP+FN)


pred=pd.read_csv(output_name, names=['ID','TIME','LABEL'])
actual=pd.read_csv('id_label_val.csv',names=['ID','LABEL'],index_col='ID')

pred_arr=pred.groupby(by='ID').max()
# pred_arr2=pred_arr.copy()
pred_arr['LABEL_act']=actual['LABEL']
pred_arr=pred_arr[pred_arr.LABEL==1]
pred_arr=pred_arr[pred_arr.LABEL==pred_arr.LABEL_act]

index=pred_arr.index 
median1=[]
for i,item in enumerate(index):
    pred1=pred[pred.ID==item]
    median1.append(float((pred1.tail(1).TIME.values-pred1[pred1.LABEL==1].head(1).TIME.values)[0])/3600.0)
print len(median1)
print 'Median Prediction Time  :',np.median(median1)
# print len(median1)




# x_grp = x.groupby(by = 'ID')
# time = []
# counter = 0
# for name,group in x_grp:
# 	for i in group.iterrows():
# 		if i[1].LABEL == 1:
# 			# print name
# 			# print group
# 			counter += 1
# 			diff = group.tail(1).TIME - i[1].TIME
# 			# print name, diff.values[0]
# 			time.append(diff.values[0])
# 			break

# time = np.array(time,dtype = float)
# t = np.median(time)/(3600)
# print 'Median Prediction time: ', t
# if t >= 5:
# 	if t >= 5 and t <= 72:
# 		mpts = t/72
# 	elif t > 72:
# 		mpts = 1
# 	print 'SCORE: ',0.75*(TP/(TP+FN)) + 0.2*mpts + 0.05*((TN/(TN+FP)) - 0.99)*100
# else:
# 	print 'SCORE: 0'
# # z1 = x_grp.max()
# # z2 = (x_grp.LABEL == 1)
# # print z2
