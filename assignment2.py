# Our import statements for this problem
import pandas as pd
import numpy as np
import patsy as pt

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data_train = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv")
data_test = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv")

model_t = DecisionTreeClassifier(max_depth=5,min_samples_leaf=10)
yt = data_test['meal']
xt = data_test.drop(['meal','id','DateTime'], axis=1).dropna()

x_t, xt, y_t, yt = train_test_split(x_t, y_t, test_size=0.33, random_state=42)
pred_t = model_t.fit(x_t,y_t)
print("\n\nout-sample accuracy: %s%%\n\n" 
 % str(round(100*accuracy_score(y_t, model.predict(x_t)), 2)))





model = DecisionTreeClassifier(max_depth=5,min_samples_leaf=10)

y = data_train['meal']
x = data_train.drop(['meal','id','DateTime'], axis=1).dropna()
pred = model.fit(x,y)

print("\n\nIn-sample accuracy: %s%%\n\n" 
 % str(round(100*accuracy_score(y, model.predict(x)), 2)))





#y, x = pt.dmatrices("meal ~ -1 + Total", data=data_train)
#xt, yt = pt.dmatrices("meal ~ -1 + Total", data=data_test)

