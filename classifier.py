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
svmclassifier = SVC(kernel = 'rbf', random_state = 0)
svmclassifier.fit(x_train,Y_train)

#predicting tst set results
pred =svmclassifier.predict(x_test)


#creation of the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, pred)

#defining function for plotting data
from matplotlib.colors import ListedColormap

def plotSets (X,Y, txt):
    X_set, Y_set = X, Y
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01), np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.pyplot.contourf(X1, X2, svmclassifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('#a3eeff', '#00d0ff')))
    plt.pyplot.xlim(X1.min(), X1.max())
    plt.pyplot.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(Y_set)):
        plt.pyplot.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1], c = ListedColormap(('#6497a3', '#002730'))(i), label = j)
    plt.pyplot.title('SVM - ' + txt)
    plt.pyplot.xlabel('Age')
    plt.pyplot.ylabel('Estimated Salary')
    plt.pyplot.legend()
    plt.pyplot.show()

# visualising training set results
plotSets(x_train, Y_train, "training set results" )

# visualising test set results
plotSets(x_test, Y_test, 'testing set results')

#getting the user prediction
X_userdata=[]
x_userdata=[]
x_pred=[]
names=[]
annotatelst=[]
while True:
    name = input("Enter name (enter 'done' when finished) : ")
    if name == "done":
        break
    age = int(input("Enter age: "))
    salary = int(input("Enter salary: "))
    names.append(name)
    X_userdata.append([age,salary])
    x_userdata = sc.fit_transform(X_userdata)
    print(x_userdata)
    x_pred = svmclassifier.predict(x_userdata)
    print(x_pred)
    #rebuilding the graph
    plotSets(x_userdata, x_pred, 'user input set results')

#writing back all user set and prediction result to local file
csv_entry=[]
print("The results are as follows:")
for i in range(len(names)):
    csv_entry.append([names[i],X_userdata[i][0], X_userdata[i][1], x_pred[i]])
print(csv_entry)
print("*where 0 - no purchased and 1 - purchased")
df = pd.DataFrame(csv_entry,columns = ['Name','Age','Salary','Purchased'])
df.to_csv('UserData.csv', mode='a', header=False)
    
print('Check UserData.csv for the stored changes for user predictions')
