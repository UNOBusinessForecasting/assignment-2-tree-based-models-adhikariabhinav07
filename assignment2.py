# Our import statements for this problem
import pandas as pd
import numpy as np
import patsy as pt

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data_train = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv")
y_t = data_train['meal']
x_t = data_train.drop(['meal','id','DateTime'], axis=1).dropna()
X, xt, Y, yt = train_test_split(x_t, y_t, test_size=0.33, random_state=42)
model = DecisionTreeClassifier(max_depth=5,min_samples_leaf=10)
modelFit = model.fit(X,Y)

print(f"\n\nIn-sample accuracy: {round(100 * accuracy_score(Y, model.predict(X)), 2)}%\n\n")
print(f"\n\nOut-of-sample accuracy: {round(100 * accuracy_score(yt, model.predict(xt)), 2)}%\n\n")


data_test = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv")
test = data_test.drop(["meal","id","DateTime"],axis = 1)
pred = model.predict(test)





