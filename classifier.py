#importing libraries
import numpy as np
import pandas as pd
import matplotlib as plt

#importing dataset
dataset = pd.read_csv('SVMdataset.csv')
X = dataset.iloc[:,[2,3]].values
Y = dataset.iloc[:,4].values

#splitting the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, train_size = 0.25, random_state = 0)

#feature scaling for input variables
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(X_train)
x_test = sc.fit_transform(X_test)

#fit our SVM classifier to our training set
from sklearn.svm import SVC
svmclassifier = SVC(kernel = 'linear', random_state = 0)
svmclassifier.fit(x_train,Y_train)

#predicting tst set results
pred =svmclassifier.predict(x_test)

#creation of the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, pred)

# visualising training set results
from matplotlib.colors import ListedColormap
X_set, Y_set = x_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01), np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.pyplot.contourf(X1, X2, svmclassifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('#a3eeff', '#00d0ff')))
plt.pyplot.xlim(X1.min(), X1.max())
plt.pyplot.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.pyplot.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1], c = ListedColormap(('#6497a3', '#002730'))(i), label = j)
plt.pyplot.title('SVM - training set')
plt.pyplot.xlabel('Age')
plt.pyplot.ylabel('Estimated Salary')
plt.pyplot.legend()
plt.pyplot.show()

# visualising test set results
from matplotlib.colors import ListedColormap
X_set, Y_set = x_test, Y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01), np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.pyplot.contourf(X1, X2, svmclassifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('#a3eeff', '#00d0ff')))
plt.pyplot.xlim(X1.min(), X1.max())
plt.pyplot.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.pyplot.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1], c = ListedColormap(('#6497a3', '#002730'))(i), label = j)
plt.pyplot.title('SVM - testing set')
plt.pyplot.xlabel('Age')
plt.pyplot.ylabel('Estimated Salary')
plt.pyplot.legend()
plt.pyplot.show()