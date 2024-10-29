import numpy as np
import matplotlib.pyplot as plt

# Generate the distribution
np.random.seed(0)  # For reproducibility
data = [min(int(np.random.exponential(scale=1)), 5) for _ in range(10000)]

# Normalize the data by converting frequencies to probabilities
plt.hist(data, bins=range(7), density=True, edgecolor='black', align='left')
plt.title('Normalized Distribution of the Number of Letters', fontsize=18)
plt.xlabel('Number of Letters', fontsize=18)
plt.ylabel('Probability', fontsize=18)
plt.xticks(range(6))
# plt.grid(True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.show()
