# Conduct ANOVA test

from scipy.stats import f_oneway

# Data for groups

group1 = [10,12,14,16,18]
group2 = [9,11,13,15,17]
group3 = [8,10,12,14,16]

# perform ANOVA
f_stat, p_value = f_oneway(group1, group2, group3)

print("F_Statistics: ", f_stat)
print("P_value: ", p_value)
