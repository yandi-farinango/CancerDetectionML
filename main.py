import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from knn import KNN
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from sklearn.model_selection import GridSearchCV
import seaborn as sns

cancer = datasets.load_breast_cancer()

df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns=np.append(cancer['feature_names'], ['target']))
sns.set_style('darkgrid')
#SNSPairplot first5 features
sns.pairplot(df_cancer, vars = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness'])
#SNSPairplot first5 features seperated by Positive/Negative
sns.pairplot(df_cancer, hue = 'target', vars = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness'])
#Heatmap for selecting features
plt.figure(figsize=(20,12))
sns.heatmap(df_cancer.corr(), annot = True)

X = df_cancer[['mean radius', 'mean perimeter', 'mean area', 'mean concave points', 'target']]
y = X['target']

#SNSPairplot Selected Features
sns.pairplot(X, hue = 'target', vars = ['mean radius', 'mean perimeter', 'mean area', 'mean concave points'])
plt.show()

X = X.drop(['target'], axis = 1)
X = X.to_numpy()
y = y.to_numpy()

#Normalizing Data
X_max = X.max()
X_min = X.min()
X_range = (X_max - X_min)
X_norm = (X - X_min)/(X_range)

#DataSplit
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=.15, random_state=20)
X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=.15, random_state=20)

# Default SVC Model
svc_model = SVC()
svc_model.fit(X_train, y_train)
y_predict = svc_model.predict(X_dev)

cm = np.array(confusion_matrix(y_dev, y_predict, labels = [1,0]))
confusion = pd.DataFrame(cm, index = ['Cancer', 'Healthy'],
                        columns = ['Predict Cancer', 'Predict Healthy'])
print('Defualt SVC Model')
print(confusion)
print(classification_report(y_dev, y_predict))


#Best SVC
param_grid = {'C': [.1, 1, 10, 100], 'gamma': [1, .1, .01, .001], 'kernel': ['rbf', 'linear']}
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 4)
grid.fit(X_train, y_train)
print(grid.best_params_)
#SVC With Best Param
grid_predict = grid.predict(X_dev)
cm_best = np.array(confusion_matrix(y_dev, grid_predict, labels = [1,0]))
confusion_best = pd.DataFrame(cm_best, index = ['Cancer', 'Healthy'],
                         columns = ['Predicted Cancer', 'Predicted Healthy'])
print('Best SVC Model')
print(confusion_best)
print(classification_report(y_dev, grid_predict))


# KNN K3 Prediction
clf = KNN(k=3)
clf.fit(X_train, y_train)
knn_predictions = clf.predict(X_dev)

cm_knn = np.array(confusion_matrix(y_dev, knn_predictions, labels = [1,0]))
confusion_knn = pd.DataFrame(cm_knn, index = ['Cancer', 'Healthy'],
                         columns = ['Predicted Cancer', 'Predicted Healthy'])
print('KNN-3 Prediction')
print(confusion_knn)
print(classification_report(y_dev, knn_predictions))


# KNN K1 Prediction
clf1 = KNN(k=1)
clf1.fit(X_train, y_train)
knn1_predictions = clf1.predict(X_dev)

cm_knn1 = np.array(confusion_matrix(y_dev, knn1_predictions, labels = [1,0]))
confusion_knn1 = pd.DataFrame(cm_knn1, index = ['Cancer', 'Healthy'],
                         columns = ['Predicted Cancer', 'Predicted Healthy'])
print('KNN-1 Prediction')
print(confusion_knn1)
print(classification_report(y_dev, knn1_predictions))


#Most Frequent Dummy Classifier
dummyclf = DummyClassifier(strategy="most_frequent")
dummyclf.fit(X_train, y_train)
dummy_prediction = dummyclf.predict(X_dev)

cm_dummy = np.array(confusion_matrix(y_dev, dummy_prediction, labels = [1,0]))
confusion_dummy = pd.DataFrame(cm_dummy, index = ['Cancer', 'Healthy'],
                         columns = ['Predicted Cancer', 'Predicted Healthy'])
print('Most Frequent Dummy Prediction')
print(confusion_dummy)
print(classification_report(y_dev, dummy_prediction))


#Stratified Dummy Classifier
dummyclf2 = DummyClassifier(strategy="stratified")
dummyclf2.fit(X_train, y_train)
dummy_prediction2 = dummyclf2.predict(X_dev)

cm_dummy2 = np.array(confusion_matrix(y_dev, dummy_prediction2, labels = [1,0]))
confusion_dummy2 = pd.DataFrame(cm_dummy2, index = ['Cancer', 'Healthy'],
                         columns = ['Predicted Cancer', 'Predicted Healthy'])
print('Stratified Dummy Prediction')
print(confusion_dummy2)
print(classification_report(y_dev, dummy_prediction2))


#Best SVC with Test
#SVC With Best Param
grid_predict_final = grid.predict(X_test)
cm_best_final = np.array(confusion_matrix(y_test, grid_predict_final, labels = [1,0]))
confusion_best_final = pd.DataFrame(cm_best_final, index = ['Cancer', 'Healthy'],
                         columns = ['Predicted Cancer', 'Predicted Healthy'])
print('Best SVC Model')
print(confusion_best_final)
print(classification_report(y_test, grid_predict_final))

# KNN Prediction
clf_final = KNN(k=3)
clf_final.fit(X_train, y_train)
knn_predictions_final = clf.predict(X_test)

cm_knn_final = np.array(confusion_matrix(y_test, knn_predictions_final, labels = [1,0]))
confusion_knn_final = pd.DataFrame(cm_knn_final, index = ['Cancer', 'Healthy'],
                         columns = ['Predicted Cancer', 'Predicted Healthy'])
print('KNN-3 Prediction')
print(confusion_knn_final)
print(classification_report(y_test, knn_predictions_final))