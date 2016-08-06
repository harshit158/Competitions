
File 1: preprocessing.py
***************************
Extracts features from training and validation files 
This file uses training and validation files assumed to be in the same directory

OUTPUT : X_train.npy
                 	y_train.npy

File 2: preprocessing_online.py
**********************************
Extracts features from :
	Test files

OUTPUT : X_online_test.npy
	stamps_online_test.npy


File 3 : trainer.py
******************
Trains on files  X_train.npy and X_online_test generated in above steps and predicts the output

OUTPUT: output_logistic.csv
	output_extree_gini.csv
	output_extree_entropy.csv
	output_rand_gini.csv
	output_rand_entropy.csv
	outptu_gradboost.csv


File 4 : ensemble.py
**********************
This file generates the final predicted file

Input : Takes files generated in above step as input

OUTPUT : output.csv



******************
	
	






