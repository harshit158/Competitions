import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet



X_training=np.load('C:\Users\HS\Desktop\Amex\sid\X_training.npy')
y_training=np.load('C:\Users\HS\Desktop\Amex\sid\y_training.npy')
X_submission=np.load('C:\Users\HS\Desktop\Amex\sid\X_submission.npy')

'''one hot'''
one=OneHotEncoder(categorical_features=[0], sparse=False)
X_training=one.fit_transform(X_training)
print X_training.shape

scale = StandardScaler()
X_training=scale.fit_transform(X_training)

net1 = NeuralNet(
        layers=[('input', layers.InputLayer),
                ('hidden', layers.DenseLayer),
                ('output', layers.DenseLayer),
                ],
        # layer parameters:
        input_shape=(None, 28*2),
        hidden_num_units=100,  # number of units in 'hidden' layer
        output_nonlinearity=lasagne.nonlinearities.softmax,
        output_num_units=5,  # 10 target values for the digits 0, 1, 2, ..., 9

        # optimization method:
        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.9,

        max_epochs=10,
        verbose=1,
        )

    # Train the network
    net1.fit(data['X_train'], data['y_train'])

    # Try the network on new data
    print("Feature vector (100-110): %s" % data['X_test'][0][100:110])
    print("Label: %s" % str(data['y_test'][0]))
    print("Predicted: %s" % str(net1.predict([data['X_test'][0]])))