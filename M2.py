import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chisquare, f_oneway, chi2_contingency, chi2
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

#11-1.6
#alpha=0.10

obs=[12,8,24,6]
exp=[10,14,18,8]

score, p=chisquare(obs,f_exp=exp)
print(score, p)

dof = 4-1
alpha = 0.10
q = 1 - alpha

critic_value = chi2.ppf(q, dof)
print(critic_value)


#11.1.8
#
delay_obs= [125,10,25,40]
delay_exp=[141.6,16.4,18,24]
chi1=chisquare(delay_obs,f_exp=delay_exp)
print(chi1)
#
# #11.2.8

data_2014= [724,335,174,107]
data_2013=[370,292,152,140]
movie= [data_2014, data_2013]
mve_chi=chi2_contingency(movie)
print('movie', mve_chi)


#11.2.10
Army = [[10791,62491],[7816,42750],[932,9525],[11819,54344]]

stat, p, dof, expected_data = chi2_contingency(Army)
print(stat)


#12.1.8
condiments=[270,130,230,180,80,70,200]
cereals=[260,220,290,290,200,320,140]
desserts=[100,180,250,250,300,360,300,160]

anova_test = f_oneway(condiments,cereals,desserts)
print(anova_test)

#12.2.10
Cereal= [578,320,264,249,237]
Chocolate_Candy=[311,106,109,125,173]
Coffee =[261,185,302,689]
anova_test1 = f_oneway(Cereal,Chocolate_Candy,Coffee)
print(anova_test1)

#12.2.12
Eastern_third=[4946,5953,6202,7243,6113]
Middle_third =[6149,7451,6000,6479]
Western_third =[5282,8605,6528,6911]
anova_test2 = f_oneway(Eastern_third,Middle_third,Western_third)
print(anova_test2)

#12.3.10
data = {'Grow_light': ['1', '1', '1', '1', '1', '1', '2', '2', '2', '2', '2', '2'],
        'Plant_food': ['A', 'A', 'A', 'B', 'B', 'B', 'A', 'A', 'A', 'B', 'B', 'B'],
        'Growth': [9.2, 9.4, 8.9, 7.1, 7.2, 8.5, 8.5, 9.2, 8.9, 5.5, 5.8, 7.6]
       }
df = pd.DataFrame(data)

# Conduct the two-way ANOVA
model = ols('Growth ~ C(Grow_light) + C(Plant_food) + C(Grow_light):C(Plant_food)', data=df).fit()
results = anova_lm(model)

print(results)


#own
pd.set_option('display.max_columns', None)
data_loc = r'C:\Users\17783\Downloads\baseball.csv'
baseball = pd.read_csv(data_loc, header=0)

plt.figure('Histogram of Wins')
plt.hist(baseball['W'])
plt.xlabel('Wins')
plt.ylabel('frequency')
plt.title('Histrogram Of wins')
plt.show()

group = baseball['Year']//10*10  # or df['year'].round(-1)
grouped = baseball.groupby([group]).mean()['W']
print(grouped)

Test_data= []
for n in grouped.values:
    Test_data.append(n)

print(Test_data)

test_score, p= chisquare(Test_data)
print(test_score,p)

#own point 4
crop_data= pd.read_csv(r'C:\Users\17783\Downloads\crop_data.csv', header=0)
crop_data.rename(columns={'yield': 'yield_value'}, inplace=True)
crop_data["density"] = crop_data["density"].astype("category")
crop_data["fertilizer"] = crop_data["fertilizer"].astype("category")
crop_data["block"] = crop_data["block"].astype("category")

# Create the model
model = ols("yield_value ~ C(density) + C(fertilizer) + C(density):C(fertilizer)", data=crop_data).fit()

# Perform ANOVA
aov_table = sm.stats.anova_lm(model)

print(aov_table)

