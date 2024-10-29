import matplotlib.pyplot as plt
import numpy as np
from scipy.special import comb


def probability(p):
    total_sum = 0
    for i in range(6):  # i from 0 to 5
        total_sum += comb(15, i) * (1 - p)**i * p**(15 - i)
    return total_sum

# Monte Carlo method to estimate p
def find_p(target=0.6, precision=1e-4):
    low, high = 0, 1
    while high - low > precision:
        mid = (low + high) / 2
        prob = probability(mid)
        if prob >= target:
            high = mid
        else:
            low = mid
    return mid

# Function to find p for varying M
def find_p_for_M(M_values, precision=1e-4):
    p_values = []
    for M in M_values:
        p_value = find_p(target=M, precision=precision)
        p_values.append(p_value)
    return p_values

# Define the range of M values
M_values = np.linspace(0.1, 0.9, 100)  # Vary M from 0.1 to 0.9

# Find corresponding p values
p_values = find_p_for_M(M_values)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(p_values, M_values, label="$W(p)$", color='blue')
plt.xlabel('$p$', fontsize=18)
plt.ylabel('$W$', fontsize=18)
plt.title('Plot of $W(p)$ varying with $p$', fontsize=18)
plt.grid(True)
plt.legend(fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()