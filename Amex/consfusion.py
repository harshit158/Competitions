import csv
import pandas as pd

en=pd.read_csv('ensemble_4clfs.csv')

f=open('ensemble_4clfs.csv','rb')
reader=csv_reader(f)

data=[][]
l=len(en.ran)
for i in range(4):
	for j in range(4):
		data[i][j]=sum(en.clf_row[i]==en.en.clf_col[j])	


write=open('confusion.csv','wb')
writer=csv.writer(f)


