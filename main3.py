#THIS FILE IS USED TO SAVE THE LINEAR SVC CLASSIFIER FILE


import pickle
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import numpy as np
import  matplotlib.pyplot as plt



features = pickle.load(open('features'))
labels = pickle.load(open('labels'))

X_train, X_test, y_train, y_test = cross_validation.train_test_split(features, labels, test_size=0.2, random_state=42)
# print(X_train, y_train)
# print(X_train.dtype, y_train.dtype)
# print(X_train.shape, y_train.shape)

clf1 = LinearSVC(C=1.0, loss='l2', penalty='l2',multi_class='ovr')
clf2 = SVC( kernel = 'linear', probability=True)

clf1.fit(X_train, y_train)
clf2.fit(X_train, y_train)
#y_pred = clf.predict(X_test)


#Save the classifier model to disk
filename = 'SVC_Classifier_Probabilities.sav'
pickle.dump(clf2, open(filename, 'wb'))

filename = 'SVC_Classifier.sav'
pickle.dump(clf1, open(filename, 'wb'))


#Load the model freom the disk
loaded_model =  pickle.load(open(filename, 'rb'))
# result = loaded_model.score(X_test, y_test)
# print(result)
# y_pred = loaded_model.predict(X_test)







def plot_confusion_matrix(y_true,y_pred):
    cm_array = confusion_matrix(y_true,y_pred)
    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)
    plt.imshow(cm_array[:-1,:-1], interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix", fontsize=16)
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.set_label('Number of images', rotation=270, labelpad=30, fontsize=12)
    xtick_marks = np.arange(len(true_labels))
    ytick_marks = np.arange(len(pred_labels))
    plt.xticks(xtick_marks, true_labels, rotation=90)
    plt.yticks(ytick_marks,pred_labels)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 12
    fig_size[1] = 12
    plt.rcParams["figure.figsize"] = fig_size
    plt.show()



# print("Accuracy: {0:0.1f}%\n".format(accuracy_score(y_test,y_pred)*100))
# plot_confusion_matrix(y_test,y_pred)