#Import Required Libraries
from scipy.io import loadmat
import pandas as pd, numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

#Load email body data from mat file
data=loadmat('./dbworld/MATLAB/dbworld_bodies.mat')
# Load Inputs and and Labels into separate Pandas Dataframes
vectors=pd.DataFrame(data['inputs'])
labels=pd.DataFrame(data['labels'])


df = np.count_nonzero(vectors,axis=0)
idf=np.log10(vectors.shape[0]/df)
vectors=vectors*idf


vectors_train, vectors_test, labels_train, labels_test = train_test_split(vectors, labels,stratify=labels,test_size=0.25)

#Get an instance of the Naive Bayes CLassifier
clf = BernoulliNB()
#Fit the classifier on our data
clf.fit(X=vectors_train,y=labels_train)
# Get predictions on training and test datasets separately
pred_test_nb = clf.predict(vectors_test)
pred_train_nb= clf.predict(vectors_train)

print('Naive Bayes Result\n')
print('Micro F1 Score (Bodies)-Training Data: ',f1_score(y_true=labels_train, y_pred = pred_train_nb, average='micro'))
print('Macro F1 Score (Bodies)-Training Data: ',f1_score(y_true=labels_train, y_pred = pred_train_nb, average='macro'))
print('Accuracy (Bodies)-Training Data: ',accuracy_score(y_true=labels_train, y_pred = pred_train_nb))

print('Micro F1 Score (Bodies)-Test Data: ',f1_score(y_true=labels_test, y_pred = pred_test_nb, average='micro'))
print('Macro F1 Score (Bodies)-Test Data: ',f1_score(y_true=labels_test, y_pred = pred_test_nb, average='macro'))
print('Accuracy (Bodies)-Test Data: ',accuracy_score(y_true=labels_test, y_pred = pred_test_nb))
print('#################################################################\n')


#Get an instance of the kNN CLassifier and set k=5
clf = KNeighborsClassifier(n_neighbors=5)
#Fit the classifier on our data
clf.fit(X=vectors_train,y=labels_train)
# Get predictions on training and test datasets separately
pred_test_knn = clf.predict(vectors_test)
pred_train_knn = clf.predict(vectors_train)

# Calculate F1 Score and Accuracy on Training and Test Datasets Separaltely
print('KNN Result\n')
print('Micro F1 Score (Bodies)-Training Data: ',f1_score(y_true=labels_train, y_pred = pred_train_knn, average='micro'))
print('Macro F1 Score (Bodies)-Training Data: ',f1_score(y_true=labels_train, y_pred = pred_train_knn, average='macro'))
print('Accuracy (Bodies)-Training Data: ',clf.score(X = vectors_train,y =labels_train))

print('Micro F1 Score (Bodies)-Test Data: ',f1_score(y_true=labels_test, y_pred = pred_test_knn, average='micro'))
print('Macro F1 Score (Bodies)-Test Data: ',f1_score(y_true=labels_test, y_pred = pred_test_knn, average='macro'))
print('Accuracy (Bodies)-Test Data: ',clf.score(X = vectors_test,y =labels_test))
print('#################################################################\n')



#Get an instance of the Rocchhio CLassifier
clf = NearestCentroid()
#Fit the classifier on our data
clf.fit(X=vectors_train,y=labels_train)
# Get predictions on training and test datasets separately
pred_test_roc = clf.predict(vectors_test)
pred_train_roc= clf.predict(vectors_train)


# Calculate F1 Score and Accuracy on Training and Test Datasets Separaltely
print('Rocchhio Result\n')
print('Micro F1 Score (Bodies)-Training Data: ',f1_score(y_true=labels_train, y_pred = pred_train_roc, average='micro'))
print('Macro F1 Score (Bodies)-Training Data: ',f1_score(y_true=labels_train, y_pred = pred_train_roc, average='macro'))
print('Accuracy (Bodies)-Training Data: ',accuracy_score(y_true=labels_train, y_pred = pred_train_roc))

print('Micro F1 Score (Bodies)-Test Data: ',f1_score(y_true=labels_test, y_pred = pred_test_roc, average='micro'))
print('Macro F1 Score (Bodies)-Test Data: ',f1_score(y_true=labels_test, y_pred = pred_test_roc, average='macro'))
print('Accuracy (Bodies)-Test Data: ',accuracy_score(y_true=labels_train, y_pred = pred_train_roc))
print('#################################################################\n')


