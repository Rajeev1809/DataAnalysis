import itertools
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

data_loc = r'C:\Users\17783\Downloads\AmesHousing.csv'
df = pd.read_csv(data_loc, header=0)
print(df)

print(df.describe())

df.fillna(value=0, inplace=True)

newdf = df.select_dtypes(include=np.number)


df_new = newdf[['Year Built','Year Remod/Add','Total Bsmt SF','Overall Qual','Bedroom AbvGr','Gr Liv Area','TotRms AbvGrd','Fireplaces',
                'Full Bath', 'Half Bath', 'Garage Area','Pool Area', 'Yr Sold', 'SalePrice']]
plt.figure(figsize=(50, 12))
heatmap = sns.heatmap(df_new.corr(), vmin=-1, vmax=1, annot=True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12)
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.show()

plt.subplot(1,3,1)
plt.scatter(df_new['Overall Qual'], df_new['SalePrice'])
plt.xlabel("Rates the overall material and finish of the house")
plt.ylabel("Sale Price")
plt.title("Scatter plot for high correlation")


plt.subplot(1,3,2)
plt.scatter(df_new['Pool Area'], df_new['SalePrice'])
plt.xlabel("Pool Area")
plt.title("Scatter plot for low correlation")


plt.subplot(1,3,3)
plt.scatter(df_new['TotRms AbvGrd'], df_new['SalePrice'])
plt.xlabel("Total Rooms")
plt.title("Scatter plot for 0.5 correlation")
plt.show()

X= df_new[['Total Bsmt SF','Gr Liv Area','Garage Area']]

y= df_new['SalePrice']

reg = LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

reg.fit(X_train, y_train)
print("The intercept:",reg.intercept_)
print("The coefficient:",reg.coef_)
print("The Score:",reg.score(X_train, y_train))


y_pred = reg.predict((X_test))
print(y_pred)

for col in X.columns:
   plt.scatter(X[col], y, s=10)

plt.legend(['Total Bsmt SF','Gr Liv Area','Garage Area'])
plt.xlabel(' Area')
plt.ylabel(' Sale Price')
plt.title(' Regression Model')
plt.show()



df_col=df_new[['Total Bsmt SF','Gr Liv Area','Garage Area']]

vif_df = pd.DataFrame()
vif_df["feature"] = df_col.columns
# calculating VIF for each feature
vif_df["VIF"] = [variance_inflation_factor(df_col.values, i) for i in range(len(df_col.columns))]
print(vif_df)
#

house=newdf.join(df['Garage Type'])

print("Old Shape: ", house.shape)
# IQR


def deleteOutlier(House):
    temp=House
    columns = ['Total Bsmt SF', 'Gr Liv Area', 'Garage Area', 'SalePrice']
    for i in columns[0:2]:
        Q1 = np.percentile(temp[i], 25, method='midpoint')
        Q3 = np.percentile(temp[i], 75, method='midpoint')
        IQR = Q3 - Q1
        upper = np.where(temp[i] >= (Q3 + 1.5 * IQR))
        lower = np.where(temp[i] <= (Q1 - 1.5 * IQR))
        try:
            temp.drop(upper[0], inplace=True)
            temp.drop(lower[0], inplace=True)
        except KeyError:
            print('Element not found')


    return temp;

df_outlier= deleteOutlier(house)

LA1=sns.boxplot(df_outlier['Gr Liv Area'])
plt.show()

TB1=sns.boxplot(df_outlier['Total Bsmt SF'])
plt.show()


print("New Shape:",df_outlier.shape)

X1= df_outlier[['Total Bsmt SF','Gr Liv Area','Garage Area']]

y1= df_outlier['SalePrice']

reg1 = LinearRegression()

X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=0)

reg1.fit(X_train1, y_train1)
print("The intercept of updated model:",reg1.intercept_)
print("The coefficient of updated model:",reg1.coef_)
print("The Score of updated model:",reg1.score(X_train1, y_train1))
# #all subset

df1 = df_outlier[['Total Bsmt SF', 'Gr Liv Area', 'Garage Area', 'Garage Type', 'SalePrice']]
df1 = pd.get_dummies(df1, columns=['Garage Type'], drop_first=False)
print(df1.head(3))


def fit_linear_reg(X, Y):
    # Fit linear regression model and return RSS and R squared values
    model_k = linear_model.LinearRegression(fit_intercept=True)
    model_k.fit(X, Y)
    RSS = mean_squared_error(Y, model_k.predict(X)) * len(Y)
    R_squared = model_k.score(X, Y)
#    print(model_k.intercept_)
    print(model_k.coef_)
    print("Best model Intercept",model_k.intercept_)
    return RSS, R_squared


from tqdm import tnrange, tqdm_notebook

# Initialization variables
Y = df1.SalePrice
X = df1.drop(columns='SalePrice', axis=1)
k = 10
RSS_list, R_squared_list, feature_list = [], [], []
numb_features = []

# Looping over k = 1 to k = 11 features in X
for k in tnrange(1, len(X.columns) + 1, desc='Loop...'):

    # Looping over all possible combinations: from 11 choose k
    for combo in itertools.combinations(X.columns, k):
        tmp_result = fit_linear_reg(X[list(combo)], Y)  # Store temp result
        RSS_list.append(tmp_result[0])  # Append lists
        R_squared_list.append(tmp_result[1])
        feature_list.append(combo)
        numb_features.append(len(combo))

# Store in DataFrame
df = pd.DataFrame(
    {'numb_features': numb_features, 'RSS': RSS_list, 'R_squared': R_squared_list, 'features': feature_list})

HR2= max(df['R_squared'])
print("Best model score:",HR2)
df.to_csv(os.path.join(r'C:\Users\17783\Downloads', 'BS_data.csv'))



