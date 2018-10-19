import numpy as np
from sklearn.neural_network import MLPRegressor,MLPClassifier
import matplotlib.pyplot as plt

np.random.seed(3)
n = 1000
x_train = np.linspace(-4 , 4 , num= n )
y_train = x_train**2
X_train = np.reshape(x_train ,[n, 1])
Y_train = np.reshape(y_train ,[n ,])
fig = plt.figure()
plt.axis([-2,2,0,4])


clf = MLPRegressor(alpha=0.001, hidden_layer_sizes = (10,), max_iter = 50000,
                 activation = 'logistic', verbose = 'True', learning_rate = 'adaptive')
a = clf.fit(X_train, Y_train)


x_ = np.linspace(-2, 2, 100) # define axis

x_test = np.reshape(x_, [100, 1]) # [160, ] -> [160, 1]
y_pred = clf.predict(x_test) # predict network output given x_
y_test = x_test **2
plt.plot(x_test, y_test, 'b') # plot original function
plt.plot(x_test,y_pred, 'r') # plot network output
plt.suptitle("blue for real function\n red for prediction function")

plt.show()