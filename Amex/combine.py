from __future__ import division
import pandas as pd
import matplotlib as plt
import csv
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import math
from  collections import Counter


f=open('ensemble.csv','rb')
w=open('combined_result.csv','wb')
reader=csv.reader(f)
# next(reader, None)


i=2
for row in reader:
	# print row
	# break
	dic=Counter(row)
	inv = {v: k for k, v in dic.items()}

	if max(dic.values())==4:
		result=inv[4]
	elif max(dic.values())==3:
		result=inv[3]
	elif max(dic.values())==2:
		if 1 in dic.values():
			result=inv[2]
		else:
			result=row[0]
	elif max(dic.values())==1:
		result=row[0]
	i+=1
	
	# # 

	# print result[0]
	w.write(result+'\n')


# i=2
# for row in reader:
# 	# print row
# 	# break
# 	dic=Counter(row)
# 	inv = {v: k for k, v in dic.items()}
# 	# print dic.values()
# 	if max(dic.values())==3:
# 		result=inv[3]
# 	elif max(dic.values())==2:
# 		result=inv[2]
		
# 	elif max(dic.values())==1:
# 		result=row[2]
# 		# print i
# 	i+=1
# 	# # 

# 	# print result[0]
# 	w.write(result+'\n')










