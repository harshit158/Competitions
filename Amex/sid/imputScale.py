from __future__ import division
import csv
import numpy as np
from sklearn.preprocessing import Imputer
def check_condition(row):
	assert len(row) == 6
	rallies = row[:5]
	total = (row[-1])
	s = 0
	if total != '':
		# Three conditions available now, check each one by one 
		total = int(total)
		for r in rallies:
			if r != '':
				s += int(r)

		if s == total:
			for i in range(5):
				if rallies[i] == '':
					row[i] = 0
		elif s > total:
			# subtract the rest from centour
			assert row[0] != ''
			row[0] = int(row[0]) - (s - total)
		elif s < total:
			# calculate the difference and evenly distribute in b/w other rallies
			no = 0
			# print type(total)
			# print type(s)
			diff = total - s
			for i in range(5):
				if rallies[i] == '':
					no += 1
			for i in range(5):
				if rallies[i] ==  '':
					row[i] = diff/no 
	
	for k in range(len(row)):
		if row[k] == '':
			row[k] = np.nan
	return row
parties = {'ODYSSEY':[0],'CENTAUR': [1], 'TOKUGAWA': [2], 'EBONY':[3], 'COSMOS':[4]}
y_training = []
X_training = []
counter = 0
with open('Training_Dataset.csv','rb') as f:
	reader = csv.reader(f)
	next(reader, None)
	for row in reader:
		y_training.append(row[1])
		new = check_condition(row[24:28]+[row[29]] + [row[28]])
		assert len(new) == 6
		age = np.mean([int(j) for j in row[16].strip('+').split('-')])
		before = parties[row[2]] + row[3:13] + row[15:16] + [age] + row[17:23]
		for k in range(len(before)):
			if before[k] == '':
				before[k] = np.nan
			else:
	 			before[k] = ''.join(str(before[k]).strip('$').split(','))
	 	#print 'Before: ',before
		after = row[31]
		if after == '':
			after = np.nan
		else:
			after = ''.join(str(after).strip('$').split(','))
	 	#print 'After: ', after
	 	#print 'New: ',new
	 	row = before + new + [after]
	 	#print row

		X_training.append(row)
		counter += 1
f.close()
print str(counter) + ' Lines of Training data Processed !!'

imp = Imputer(missing_values = 'NaN', strategy = 'mean')   ## will be back
c = 0
for row in X_training:
	try:
		row = np.array(row,dtype = float)		
		c += 1
	except Exception, e:
		print c
		print row
		quit()
X_training = np.array(X_training,dtype = float)
print ' Training ',X_training.shape
X_training = imp.fit_transform(X_training)
np.save('X_training',X_training)
y_training = np.array(y_training)
#np.save('y_training.npy',y_training)
#------------------------------------
counter = 0
X_submission = []
with open('Leaderboard_Dataset.csv','rb') as f:
	reader = csv.reader(f)
	next(reader, None)
	for row in reader:
	 	age = np.mean([int(j) for j in row[15].strip('+').split('-')])
	 	new = check_condition(row[23:27] + [row[28]] + [row[27]])
	 	before = parties[row[1]] +row[2:12] + row[14:15]+ [age] + row[16:22]
	 	for k in range(len(before)):
	 		if before[k] == '':
	 			before[k] = np.nan
	 		else:
	 			before[k] = ''.join(str(before[k]).strip('$').split(','))
	 	after = row[30]
	 	if after == '':
	 		after = np.nan
	 	else:
	 		after = ''.join(str(after).strip('$').split(','))
	 	row = before + new + [after]
		X_submission.append(row)
		counter += 1
f.close()
print str(counter) + ' Lines of Leaderboard data processed'
X_submission = np.array(X_submission, dtype = float)
print X_submission.shape
X_submission = imp.fit_transform(X_submission)
np.save('X_submission',X_submission)



