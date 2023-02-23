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

pd.set_option('display.max_columns', None)

df= pd.read_csv(r"C:\Users\17783\Downloads\titanic.csv", header=0)
print(df.head(3))
print(df.dtypes)
print(df.describe())

plt.figure('hist of age 1')
plt.hist(df['Age'])

#
# df['Age'] = np.where(df['Age'].isnull(), 28, df["Age"])
# print(df['Age'].describe())
#
# plt.figure('hist of age')
# plt.hist(df['Age'])

df['Age'] = np.where(df['Age'].isnull(), 85, df["Age"])
print(df['Age'].describe())

plt.figure('hist of 3')
plt.hist(df['Age'])
plt.show()


df['Family']= df['SibSp']+df['Parch']
print(df.head(5))