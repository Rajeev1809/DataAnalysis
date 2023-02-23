import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# generate normally distributed sample data
crop_data= pd.read_csv(r'C:\Users\17783\Downloads\crop_data.csv', header=0)
data= crop_data['yield']

diamond = pd.read_csv(r"C:\Users\17783\Documents\ALY6015\diamonds.csv", header=0)
data1=diamond['price']

def hist(data):
    # plot histogram
    plt.hist(data, bins=20, color='blue', alpha=0.5)
    plt.xlabel('Data Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Normally Distributed Sample Data')
    plt.show()

    # plot Q-Q plot
    sns.set_style('whitegrid')
    sns.mpl.rcParams['figure.figsize'] = (10, 6)
    sns.mpl.rcParams['axes.titlesize'] = 'large'
    sns.mpl.rcParams['axes.labelsize'] = 'large'
    sns.mpl.rcParams['xtick.labelsize'] = 'large'
    sns.mpl.rcParams['ytick.labelsize'] = 'large'

    plt.figure()
    sns.barplot(data)
    plt.title('Q-Q Plot of Normally Distributed Sample Data')


hist(data)
hist(data1)
plt.show()