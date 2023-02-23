#importing packages
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV, LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from mlxtend.feature_selection import SequentialFeatureSelector as sfs


pd.set_option('display.max_columns', None)

#giveing the column name manually
col_names = ['private', 'applications', 'accepted', 'enroll', 'top10', 'top25', 'fulltime', 'parttime', 'outstate', 'roomrent',
             'book',' personal','phd','terminal', 's_f_ratio','alumni_donate','expend','grad_rate']

#importing dataset
college = pd.read_csv(r'C:\Users\17783\Downloads\College.csv', header=0, names=col_names)
print(college.head())
print(college.info())


college= college[college['grad_rate']<100]
print(college.info())

college['private'] = college['private'].map({'Yes': 1, 'No': 0})
X_all= college.drop('grad_rate', axis='columns')
feature_cols = ['private','applications', 'accepted', 'enroll','outstate', 'roomrent','book','expend']
X_raw = college[feature_cols]  # Features
X_values = X_raw.values

#Normalization
X = preprocessing.StandardScaler().fit_transform(X_values)
#X=X_values
#X = normalize(X_values)
#X= MinMaxScaler().fit_transform(X_values)
#X=FunctionTransformer().fit_transform(X_values)

y = college.grad_rate  # Target variable

# spliting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=16)

def Ridge_Lasso(X_train, X_test, y_train, y_test):
    ridge=Ridge( )
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

    lasso_cv = LassoCV(alphas= alphas,cv=5, random_state=42)
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


Ridge_Lasso(X_train, X_test, y_train, y_test)
#Feature selection in forward

# Build EXC classifier to use in feature selection

clf = ExtraTreesClassifier(n_estimators=60, n_jobs=-1,random_state=42)

# Build step forward feature selection
sfs1 = sfs(clf,
           k_features=7,
           forward=True,
           floating=False,
           verbose=2,
           scoring='accuracy',
           cv=9)

# Perform SFFS
sfs1 = sfs1.fit(X_all, y)

# Which features?
new_cols = list(sfs1.k_feature_idx_)
print(new_cols)

X1= X_all.iloc[:, new_cols]
X1_values= X1.values
X1 = preprocessing.StandardScaler().fit_transform(X1_values)
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y, test_size=0.3, random_state=16)
lr = LinearRegression()
lr.fit(X1_train,y1_train)

y_pred_lr_train = lr.predict(X1_train)
y_pred_lr_test = lr.predict(X1_test)

# Root Mean Square Error (RMSE) for the training set
rmse_lr_train = np.sqrt(mean_squared_error(y1_train, y_pred_lr_train))
rmse_lr_test = np.sqrt(mean_squared_error(y1_test, y_pred_lr_test))
print(f"Linear Regression RMSE for the train set: {rmse_lr_train}")
print(f"Linear Regression RMSE for the test set: {rmse_lr_test}")

