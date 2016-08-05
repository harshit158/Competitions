from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score



neigh = KNeighborsClassifier(n_neighbors=14)


X_training=np.load('C:\Users\HS\Desktop\Amex\sid\X_training.npy')
y_training=np.load('C:\Users\HS\Desktop\Amex\sid\y_training.npy')
X_submission=np.load('C:\Users\HS\Desktop\Amex\sid\X_submission.npy')

encode_training=np.load('encode_training.npy') 
print encode_training.shape
encode_submission=np.load('encode_submission.npy')

# X_training=np.concatenate((X_training, encode_training), axis=1)
print X_training.shape

'''one hot'''
one=OneHotEncoder(categorical_features=[0], sparse=False)
X_training=one.fit_transform(X_training)
print X_training.shape
# print X_training[0]
# X_submission=one.fit_transform(X_submission)

# scale = StandardScaler()
# X_training=scale.fit_transform(X_training)
# print X_training.shape
# X_submission=scale.transform(X_submission)

x_train,x_test,y_train,y_test=train_test_split(X_training,y_training,test_size= 0.2)
neigh.fit(x_train, y_train)
pred=neigh.predict(x_test) 
print accuracy_score(y_test, pred)
