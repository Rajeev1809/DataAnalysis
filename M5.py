import numpy as np
from numpy import std
import random
import scipy.stats as stats

#13.2.6
# State the hypotheses and identify the claim:
# The null hypothesis is that the median paid attendance is 3000, and the alternative
# hypothesis is that the median paid attendance is not 3000. The claim is that the median paid attendance is not 3000.

# Since we are testing for a two-tailed hypothesis with a significance level of 0.05, we need to find the critical values
# from a standard normal distribution that correspond to a p-value of 0.025 on each tail. Using the z-table or a z-score
# calculator, we find that the critical value is 1.96.
attendance = [6210, 3150, 2700, 3012, 4875, 3540, 6127, 2581, 2642, 2573,
              2792, 2800, 2500, 3700, 6030, 5437, 2758, 3490, 2851, 2720]
median_attendance = np.median(attendance)

std_attendance = std(attendance)
mean_attendance = 3000
n = len(attendance)
z = (median_attendance - mean_attendance) / (std_attendance / np.sqrt(n))
print(z)

#13.2.10

# State the hypotheses and identify the claim:
# The null hypothesis is that the median number of lottery tickets sold is 200, and the alternative hypothesis is that
# the median number of lottery tickets sold is less than 200. The claim is that the median number of lottery tickets sold is less than 200.
#
# Find the critical value(s):
# Since we are testing for a one-tailed hypothesis to the left with a significance level of 0.05, we need to find the
# critical value from a standard normal distribution that corresponds to a p-value of 0.05. Using a z-score calculator,
# we find that the critical value is -1.645.
p = 15/40

mean_p = 0.5

std_p = np.sqrt(mean_p * (1 - mean_p) / 40)
z = (p - mean_p) / std_p
print("lottery",z)

# 13.3.4

# State the hypotheses and identify the claim:
# The null hypothesis is that there is no difference in the sentence received by each gender, and the alternative
# hypothesis is that there is a difference in the sentence received by each gender. The claim is that there is a difference in the sentence received by each gender.
#
# Find the critical value:
# The critical value can be found using a table of critical values for the Wilcoxon rank sum test, which is
# based on the significance level (α = 0.05) and the sample size.
males = [8, 12, 6, 14, 22, 27, 3, 2, 2, 2, 4, 6, 19, 15, 13]
females = [7, 5, 2, 3, 21, 26, 3, 9, 4, 0, 17, 23, 12, 11, 16]
stat, p_value = stats.ranksums(males, females)
print("gender:", stat, p)


#13.3.8

nl = [89, 9, 8, 101, 90, 91, 9, 96, 108, 100, 9, 6, 8]
al = [108, 8, 9, 97, 100, 102, 9, 104, 95, 89, 8, 101, 6, 1]

stat, p_value = stats.ranksums(nl, al)

print("nlal:", stat, p)

# For the first hypothesis test with ws = 13, n = 15, α = 0.01, and two-tailed, the critical value can be found in the row with n = 15 and the column with α = 0.01. The critical value for this hypothesis test is 26, which means that if the calculated test statistic (ws) is greater than or equal to 26, the null hypothesis would be rejected.
#
# For the second hypothesis test with ws = 32, n = 28, α = 0.025, and one-tailed, the critical value can be found in the row with n = 28 and the column with α = 0.025. The critical value for this hypothesis test is 31, which means that if the calculated test statistic (ws) is greater than or equal to 31, the null hypothesis would be rejected.
#
# For the third hypothesis test with ws = 65, n = 20, α = 0.05, and one-tailed, the critical value can be found in the row with n = 20 and the column with α = 0.05. The critical value for this hypothesis test is 38, which means that if the calculated test statistic (ws) is greater than or equal to 38, the null hypothesis would be rejected.
#
# For the fourth hypothesis test with ws = 22, n = 14, α = 0.10, and two-tailed, the critical value can be found in the row with n = 14 and the column with α = 0.10. The critical value for this hypothesis test is 19, which means that if the calculated test statistic (ws) is greater than or equal to 19, the null hypothesis would be rejected.


# 13.5.2

western_hemisphere = [527, 406, 474, 381, 411]
europe = [520, 510, 513, 548, 496]
eastern_asia = [523, 547, 547, 391, 549]

# Perform the Kruskal-Wallis test

h, p_value = stats.kruskal(western_hemisphere, europe, eastern_asia)
print('aisa', h,p_value)

#13.6

subway = np.array([845, 494, 425, 313, 108, 41])
rail = np.array([39, 291, 142, 103, 33, 38])

# Find the Spearman rank correlation coefficient
correlation, p_value = stats.spearmanr(subway, rail)
print('correlation', correlation, p_value)

#14.3.16


def get_prizes(box):
    prizes = []
    while len(prizes) < 4:
        prize = random.choice(box)
        if prize not in prizes:
            prizes.append(prize)
    return len(prizes)

def experiment(n):
    boxes = [1, 2, 3, 4]
    ttl_boxes = 0
    for i in range(n):
        ttl_boxes += get_prizes(boxes)
    return ttl_boxes / n

repeats = 40
average = experiment(repeats)
print("The average number of boxes a person needs to buy to get all four prizes is: ", average)

#14.3.18


def lotto():
    tickets = 0
    letters = set()
    while len(letters) < 3:
        tickets += 1
        letter = random.choices(["b", "i", "g"], weights=[0.6, 0.3, 0.1])[0]
        letters.add(letter)
    return tickets

def avg_tickets(trial):
    total_tickets = 0
    for i in range(trial):
        total_tickets += lotto()
    return total_tickets / trial

trial = 30
print("Average number of tickets needed to win:", avg_tickets(trial))
