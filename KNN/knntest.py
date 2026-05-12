import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

cmap = ListedColormap(['#FF0000','#00FF00','#0000FF'])

# iris dataset is used to classify iris flowers into 3 categories
iris = datasets.load_iris()

#Here X is the feature vector having 4 features : a) Sepal length, b) Sepal Width, c) Petal Length, d) Petal Width
X,y = iris.data , iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# print(X_train.shape)
# print(X_train[0])

# print(y_train.shape)
# print(y_train)

# plt.figure()
# X[:,0], X[:,1] are first and second features
# c = y means color variation according to values of y(target value)
# cmap as above
# edgecolors is black, k as in CMYK
# s is the size of points
# plt.scatter(X[:,0], X[:,1],c=y, cmap=cmap, edgecolors='k',s=20)
# plt.show()
from knn import KNN
clf = KNN(5)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

acc = np.sum(predictions==y_test) / len(y_test)
print(acc) 