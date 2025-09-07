# Perform Chi_square Test

from scipy.stats import chi2_contingency

# Contigency table
data = [[50, 30, 20], [30, 40, 20]]

# Perform Chi square test
chi2, p_value, dof, expected = chi2_contingency(data)

print("Chi Square Statistics: ", chi2)
print("P-Value: ", p_value)
print("Expected Frequencies: \n", expected)
