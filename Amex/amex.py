import csv

f=open('modified.csv','rb')
reader=csv.reader(f)

max_party=[]
total=0
accuracy=0
c=0
for row in reader:
	if row[8]=='':
		continue
	dic={int(row[8]):'CENTAUR', int(row[9]):'EBONY', int(row[10]):'TOKUGAWA', int(row[11]):'COSMOS', int(row[12]):'ODYSSEY'}
	max_don=max(float(row[8]),float(row[9]),float(row[10]),float(row[11]),float(row[12]))
	# print dic[int(max_don)]

	if row[1]==dic[int(max_don)]:
		accuracy+=1
	max_party.append(dic[int(max_don)])
	c+=1
print str(accuracy) + ' /'+str(c)
	