import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)
data_loc = r'C:\Users\17783\Downloads\train.csv'
df = pd.read_csv(data_loc, header=0)
print(df.head(3))
print(df.describe())

plt.figure('Heat map')
heatmap= sns.heatmap(df.corr(),vmin=-1, vmax=1,annot=True, cmap='Blues')
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45)
heatmap.set_title('Correlation heatmap')


plt.figure("Histogram")
plt.hist(df.humidity,bins=10)



df['month']= pd.to_datetime(df['datetime']).dt.month

df['day']= pd.to_datetime(df['datetime']).dt.day

plt.figure('Boxplot')
plt.subplot(2,1,1)
sns.boxplot(x=df['month'], y= df['count'])
plt.xlabel('Month')
plt.subplot(2,1,2)
sns.boxplot(x=df['day'], y= df['count'])
plt.xlabel('Day')
plt.ylabel('Count')
plt.title('Boxplot')

df_new= df[['temp', 'humidity', 'season','casual']]

x= df_new.drop(columns='casual')
y=df_new['casual']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
reg=LinearRegression()
reg.fit(X_train, y_train)
print("The intercept:",reg.intercept_)
print("The coefficient:",reg.coef_)
print("The Score:",reg.score(X_train, y_train))
plt.show()


