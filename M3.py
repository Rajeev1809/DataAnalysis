#importing packages
from sklearn.preprocessing import normalize, MinMaxScaler, FunctionTransformer
from sklearn import preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_curve

pd.set_option('display.max_columns', None)

#giveing the column name manually
col_names = ['private', 'applications', 'accepted', 'enroll', 'top10', 'top25', 'fulltime', 'parttime', 'outstate', 'roomrent',
             'book',' personal','phd','terminal', 's_f_ratio','alumni_donate','expend','grad_rate']

#importing dataset
college = pd.read_csv(r'C:\Users\17783\Downloads\College.csv', header=0, names=col_names)
print(college.head())

#giving feature columns
feature_cols = ['applications', 'accepted', 'enroll','outstate', 'roomrent','book','expend']
X_raw = college[feature_cols]  # Features
X_values = X_raw.values

#Normalization
X = preprocessing.StandardScaler().fit_transform(X_values)
#X=X_values
#X = normalize(X_values)
#X= MinMaxScaler().fit_transform(X_values)
#X=FunctionTransformer().fit_transform(X_values)

y = college.private  # Target variable

# spliting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=16)

# conf matrix
def confusionMatrix(y,y_pred, Name):
    cnf_matrix = metrics.confusion_matrix(y, y_pred)
    print(cnf_matrix)

    plt.matshow(cnf_matrix, cmap='Blues', alpha=0.3)
    group_names = ['true Pos', 'false Pos', 'false Neg', 'true Neg']
    group_counts = ["{0:0.0f}".format(value) for value in
                    cnf_matrix.flatten()]

    labels = [f"{v1}\n{v2}" for v1, v2 in
              zip(group_names, group_counts)]
    labels = np.asarray(labels).reshape(2, 2)
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            plt.text(x=j, y=i, s=labels[i, j], va='center', ha='center', size='xx-large')
    plt.title(Name)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')


#Creating a model with train data

lr= LogisticRegression()
lr.fit(X_train,y_train)
y_pred_train=lr.predict(X_train)
print("The trained model score: ",lr.score(X_train, y_train))
confusionMatrix(y_train,y_pred_train, 'Confusion matrix for Train data')

#with test data
y_pred=lr.predict(X_test)
print("The test data model score: ",lr.score(X_test, y_test))
confusionMatrix(y_test,y_pred, 'Confusion matrix for test data')


# geting the model metrics
print('Precision: %.3f' % precision_score(y_test,y_pred, pos_label='Yes'))

print('Recall: %.3f' % recall_score(y_test, y_pred,pos_label='Yes'))

print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))

print('F1 score: %.3f' % f1_score(y_test, y_pred,pos_label='Yes'))


#Plot and interpret the ROC curve.
y_pred_proba = lr.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba, pos_label='Yes')
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.figure('ROC - AUC')
plt.title('ROC - AUC')
plt.plot(fpr, tpr, label="data 1, auc="+str(auc))
plt.legend(loc=5)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.show()

