import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import pickle

#load data
data = pd.read_csv("WineQT.csv")

# check data
print(data["quality"])
print(min(data["quality"]))
print(max(data["quality"]))

# code data according to quality into good quality wines and bad quality wines
data = data.replace({"quality": {3: -1, 4: -1, 5: -1}})
data = data.replace({"quality": {6: 1, 7: 1, 8: 1}})

# divide into explainable and explained variables
Y = data["quality"]
X = data[data.columns.drop(['quality', 'Id'])]

#create sets to train the model
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.3, random_state=1)

# train the model
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, Y_train)

# see the model

print(clf.get_depth())
print(clf.get_n_leaves())
#ax = plt.gca()
#tree.plot_tree(clf, ax=ax)
#plt.show()
Y_train_pred = clf.predict(X_train)
print(Y_train_pred)
Y_val_pred = clf.predict(X_val)
print(Y_val_pred)
print(f'Train score {accuracy_score(Y_train_pred, Y_train)}')
print(f'Validation score {accuracy_score(Y_val_pred, Y_val)}')

# prune the tree
clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=4, min_samples_leaf=25, min_samples_split=25)
clf = clf.fit(X_train, Y_train)

#see the pruned amodel
Y_train_pred = clf.predict(X_train)
print(Y_train_pred)
Y_val_pred = clf.predict(X_val)
print(Y_val_pred)
print(f'Train score {accuracy_score(Y_train_pred, Y_train)}')
print(f'Validation score {accuracy_score(Y_val_pred, Y_val)}')
#ax = plt.gca()
#tree.plot_tree(clf, ax=ax)
#plt.show()

# save the trained model
saved_model = pickle.dumps(clf)
joblib.dump(clf, 'model.pkl')

