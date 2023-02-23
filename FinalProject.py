# import required libraries
import numpy as np
import pandas as pd
from IPython.core.display_functions import display
from sklearn import linear_model, preprocessing
import itertools
import joblib
import sys
from sklearn.preprocessing import normalize


sys.modules['sklearn.externals.joblib'] = joblib

from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV, Lasso, LassoCV, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tnrange, tqdm_notebook

# Giving the column names
col_names = ['carat', 'cut', 'color', 'clarity', 'depth_per', 'table', 'price', 'length', 'width', 'depth']
pd.set_option('display.max_columns', None)

# Importing dataset
diamond = pd.read_csv(r"C:\Users\17783\Documents\ALY6015\diamonds.csv", header=0, names=col_names)

# Structure of dataset
print(diamond.info())
print(diamond['cut'].value_counts())
print(diamond['color'].value_counts())
print(diamond['clarity'].value_counts())

# Descriptive statistics of dataset
print(diamond.describe())

# Removal of duplicated values
diamond.drop_duplicates(keep='last', inplace=True)



# Imputation of the 0 values for length, width and depth using mean of the related variable
cat_cols = []
num_cols = []
for col in diamond.columns:
    if diamond[col].map(type).eq(str).any():
        cat_cols.append(col)
    else:
        num_cols.append(col)

num_details = diamond[num_cols]
cat_details = pd.DataFrame(diamond[cat_cols])
num_details = num_details.mask(num_details == 0).fillna(num_details.mean())
diamonds = pd.concat([num_details, cat_details], axis=1)

# Outliers removal with the help of IQR method
Q1 = diamonds.quantile(0.25)
Q3 = diamonds.quantile(0.75)
IQR = Q3 - Q1
print(IQR)
diamonds_updated = diamonds[~((diamonds < (Q1 - 1.5 * IQR)) | (diamonds > (Q3 + 1.5 * IQR))).any(axis=1)]

# Histogram to check the price frequency
diamonds_updated.price.hist(bins=45, rwidth=.8, figsize=(10, 4))

# Scatterplot to show the relationship between carat (weight) and the price of diamonds
plt.figure(figsize=(12, 10))
sns.scatterplot(x='price', y='carat', data=diamonds_updated)
title = plt.title('Diamond prices as per carat (weight of diamond)')

# Transformation of non-numeric values using Label Encoder
# diamonds_updated = pd.get_dummies(diamonds_updated, columns=['color'], drop_first=True)


# Heat map to show the relationship between the price and all other variables
plt.figure('heat map', figsize=(10, 8))
corr = diamonds_updated.corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap="Blues")
plt.show()
print(diamonds_updated.head(2))

# Draft report ---------------------------------------------------------------------------------------------------


threshold = 4000
diamonds_updated['above_threshold'] = np.where(diamonds_updated['price'] > threshold, 1, 0)

# Select predictors
X = diamonds_updated[['length', 'width', 'depth']]

# Select target
y = diamonds_updated['above_threshold']


def confusionMatrix(y, y_pred, Name):
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


def logistic_Regression(X, y):
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Fit logistic regression model
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)

    # Predict probabilities of being above or below the threshold
    y_train_pred = logreg.predict(X_train)

    # Calculate accuracy
    acc_train = accuracy_score(y_train, y_train_pred)
    print("Accuracy for train data: {:.2f}%".format(acc_train * 100))
    # Predict binary outcome
    y_test_pred = logreg.predict(X_test)

    # Calculate accuracy
    acc_test = accuracy_score(y_test, y_test_pred)

    print("Accuracy for test data: {:.2f}%".format(acc_test * 100))

    print('Precision: %.3f' % precision_score(y_test, y_test_pred))

    print('Recall: %.3f' % recall_score(y_test, y_test_pred))


    print('F1 score: %.3f' % f1_score(y_test, y_test_pred))

    confusionMatrix(y_train, y_train_pred, 'Confusion matrix for train data')
    confusionMatrix(y_test, y_test_pred, 'Confusion matrix for test data')

    y_pred_test = logreg.predict_proba(X_test)[::, 1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_test)
    auc = metrics.roc_auc_score(y_test, y_pred_test)
    plt.figure('ROC - AUC')
    plt.title('ROC - AUC')
    plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
    plt.legend(loc=5)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.show()


logistic_Regression(X, y)

# Feature selection in forward

# Build EXC classifier to use in feature selection

clf = ExtraTreesClassifier(n_estimators=60, n_jobs=-1, random_state=42)

# Build step forward feature selection
sfs1 = sfs(clf,
           k_features=5,
           forward=False,
           floating=False,
           verbose=2,
           scoring='accuracy',
           cv=5)

# Perform SFFS=
X1 = diamonds_updated.drop(columns=['price', 'above_threshold', 'cut', 'clarity', 'color'])
sfs1 = sfs1.fit(X1, y)

# Which features?
new_cols = list(sfs1.k_feature_names_)
print(new_cols)
X1 = X1[new_cols]
logistic_Regression(X1, y)


def Ridge_Lasso(X_train, X_test, y_train, y_test):
    ridge = Ridge()
    ridge.fit(X_train, y_train)
    y_pred_train = ridge.predict(X_train)
    y_pred_test = ridge.predict(X_test)

    # Root Mean Square Error (RMSE) for the training set
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    print(f"Ridge Regression RMSE for the training set: {rmse_train}")

    # Root Mean Square Error (RMSE) for the test set
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    print(f"Ridge Regression RMSE for the test set: {rmse_test}")

    alphas = np.logspace(-5, 5, 100)
    ridge_cv = RidgeCV(alphas=alphas, store_cv_values=True)
    ridge_cv.fit(X_train, y_train)
    cv_mse = np.mean(ridge_cv.cv_values_, axis=0)
    lambda_min = alphas[np.argmin(cv_mse)]
    mse_std = np.std(ridge_cv.cv_values_, axis=0)
    lambda_1se = alphas[np.argmin(cv_mse + mse_std)]
    print(lambda_min)
    print(lambda_1se)

    # Plot the results
    plt.plot(np.log(alphas), cv_mse, '-o')
    plt.axvline(np.log(lambda_min), color='red', linestyle='--')
    plt.axvline(np.log(lambda_1se), color='green', linestyle='--')
    plt.xlabel('Log(lambda)')

    plt.ylabel('Mean Squared Error')
    plt.title('Ridge Regression - MSE vs Lambda')
    plt.show()

    # Lasso Regression
    lasso = Lasso()
    lasso.fit(X_train, y_train)
    y_pred_train = lasso.predict(X_train)
    y_pred_test = lasso.predict(X_test)

    # Root Mean Square Error (RMSE) for the training set
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    print(f"Lasso Regression RMSE for the training set: {rmse_train}")

    # Root Mean Square Error (RMSE) for the test set
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    print(f"Lasso Regression RMSE for the test set: {rmse_test}")

    lasso_cv = LassoCV(alphas=alphas, cv=5, random_state=42)
    lasso_cv.fit(X_train, y_train)

    # mse = np.mean(lasso_cv.mse_path_, axis=1)
    # lambdas = lasso_cv.alphas_
    # min_mse_index = np.argmin(mse)
    lambda_min_lasso = lasso_cv.alpha_

    # Get standard deviation of MSE values
    mse = np.mean(lasso_cv.mse_path_, axis=1)

    # calculate the standard deviation of the mean squared errors
    mse_std = np.std(lasso_cv.mse_path_, axis=1)

    # find the value of alpha corresponding to one standard error
    lambda_1se_lasso = lasso_cv.alphas_[np.argmin(mse + mse_std)]

    print("Lambda at minimum MSE:", lambda_min_lasso)
    print("Lambda at one standard error:", lambda_1se_lasso)

    # Plot the relationship between lambda and mean squared errors
    mse = np.mean(lasso_cv.mse_path_, axis=1)
    plt.plot(lasso_cv.alphas_, mse)
    plt.axvline(np.log(lambda_min_lasso), color='red', linestyle='--')
    plt.axvline(np.log(lambda_1se_lasso), color='green', linestyle='--')
    plt.xscale('log')
    plt.xlabel('Log(lambda)')
    plt.ylabel('Mean Squared Error')
    plt.title('LassoCV: Relationship between lambda and mean squared errors')
    plt.show()

X1=preprocessing.StandardScaler().fit_transform(X1.values)
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y, test_size=0.3, random_state=16)
Ridge_Lasso(X1_train, X1_test, y1_train, y1_test)

# linear regression predictors and target variables declaration
diamonds_reg = diamonds_updated[['length', 'width', 'depth']]
x = diamonds_reg
y = diamonds_updated[['price']]

# training the linear regression model
model = LinearRegression()
model.fit(x, y)
r_sq = model.score(x, y)
print(f"coefficient of determination: {r_sq}")
print(f"intercept: {model.intercept_}")
print(f"slope: {model.coef_}")

# value prediction for trained linear model
y_predicted = model.predict(x)
# print(y_predicted)
rmse = mean_squared_error(y, y_predicted)
r2 = r2_score(y, y_predicted)
print('R2 score: ', r2)

# linear regression plot
for col in x.columns:
    plt.scatter(x[col], y, s=10)

# separate numeric and categorical columns
categorical_columns = []
numeric_columns = []
for col in diamonds_updated.columns:
    if diamonds_updated[col].map(type).eq(str).any():  # check if there are any strings in column
        categorical_columns.append(col)
    else:
        numeric_columns.append(col)
data_numeric = diamonds_updated[numeric_columns]
data_categorical = pd.DataFrame(diamonds_updated[categorical_columns])
x = data_numeric.drop(columns=['price', 'above_threshold'], axis=1)
y = data_numeric['price']

# forward feature selection using linear regression model
clf = LinearRegression()
sfs1 = sfs(clf,
           k_features=5,
           forward=True,
           verbose=2,
           scoring='neg_mean_squared_error')
sfs1 = sfs1.fit(x, y)
feat_names = list(sfs1.k_feature_names_)
print(feat_names)

# new data frame with the columns selected by feature selection technique
updated_data2 = diamonds_updated[feat_names]
x = updated_data2
x=preprocessing.StandardScaler().fit_transform(x.values)
y = diamonds_updated[['price']]

# linear regression with new columns
model = LinearRegression()
model.fit(x, y)
r_sq = model.score(x, y)
print(f"coefficient of determination: {r_sq}")
print(f"intercept: {model.intercept_}")
print(f"slope: {model.coef_}")

# new data fram to find the best model accuracy
diamonds_reg2 = diamonds_updated[['length', 'width', 'depth', 'price']]
X_values = diamonds_reg2.values[:, :-1]
test = normalize(X_values)
scaler = preprocessing.StandardScaler().fit(X_values)
X = scaler.transform(X_values)
y = diamonds_reg2[['price']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
data_numeric = data_numeric.drop(columns=['price', 'depth_per'], axis=1)


def fit_linear_reg(X, Y):
    # Fit linear regression model and return RSS and R squared values
    model_k = linear_model.LinearRegression(fit_intercept=True)
    model_k.fit(X, Y)
    RSS = mean_squared_error(Y, model_k.predict(X)) * len(Y)
    R_squared = model_k.score(X, Y)
    return RSS, R_squared

# Initialization variables
k = 11
RSS_list, R_squared_list, feature_list = [], [], []
numb_features = []

# Looping over k = 1 to k = 11 features in X
for k in tnrange(1, len(data_numeric.columns) + 1, desc='Loop...'):
    # Looping over all possible combinations: from 11 choose k
    for combo in itertools.combinations(data_numeric.columns, k):
        tmp_result = fit_linear_reg(data_numeric[list(combo)], y)  # Store temp result
        RSS_list.append(tmp_result[0])  # Append lists
        R_squared_list.append(tmp_result[1])
        feature_list.append(combo)
        numb_features.append(len(combo))

# Store in DataFrame
df = pd.DataFrame(
    {'numb_features': numb_features, 'RSS': RSS_list, 'R_squared': R_squared_list, 'features': feature_list})

df_min = df[df.groupby('numb_features')['RSS'].transform(min) == df['RSS']]
df_max = df[df.groupby('numb_features')['R_squared'].transform(max) == df['R_squared']]
display(df_min)
display(df_max)

df['min_RSS'] = df.groupby('numb_features')['RSS'].transform(min)
df['max_R_squared'] = df.groupby('numb_features')['R_squared'].transform(max)
print(df)
