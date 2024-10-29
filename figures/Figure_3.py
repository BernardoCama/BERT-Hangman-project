import matplotlib.pyplot as plt
from collections import Counter
import os
import sys

# Retrieve the path of the main.py file
cwd = os.path.split(os.path.abspath(__file__))[0]
sys.path.append(os.path.dirname(cwd))

# Load the datasets
with open(os.path.join(cwd, 'words_alpha_test_unique.txt'), 'r') as file1, open(os.path.join(cwd, 'words_250000_train.txt'), 'r') as file2:
    dataset1 = file1.read().splitlines()
    dataset2 = file2.read().splitlines()

# Distribution of letters in the words
letters_count_1 = [len(word) for word in dataset1]
letters_count_2 = [len(word) for word in dataset2]

# Calculate frequency distribution of letters in the words for both datasets
letters_freq_1 = Counter("".join(dataset1))
letters_freq_2 = Counter("".join(dataset2))

# Get common letters for both datasets to align the axes
all_letters = sorted(set(letters_freq_1.keys()).union(set(letters_freq_2.keys())))

freq_1 = [letters_freq_1[letter] / sum(letters_freq_1.values()) for letter in all_letters]
freq_2 = [letters_freq_2[letter] / sum(letters_freq_2.values()) for letter in all_letters]

# Aligning the axes for the number of letters plot by using the same range for both datasets
max_letters = max(max(letters_count_1), max(letters_count_2))

plt.figure(figsize=(14, 10))

# Word length distribution in a 2x2 subplot layout
plt.subplot(2, 2, 1)
plt.hist(letters_count_1, bins=range(1, max_letters+2), density=True, edgecolor='black', alpha=0.7)
plt.title('Distribution of Word Length in Validation Dataset', fontsize=18)
plt.xlabel('Number of Letters', fontsize=18)
plt.ylabel('Probability', fontsize=18)
plt.grid(True)
plt.xlim(1, max_letters + 1)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.subplot(2, 2, 2)
plt.hist(letters_count_2, bins=range(1, max_letters+2), density=True, edgecolor='black', alpha=0.7)
plt.title('Distribution of Word Length in Training Dataset', fontsize=18)
plt.xlabel('Number of Letters', fontsize=18)
plt.ylabel('Probability', fontsize=18)
plt.grid(True)
plt.xlim(1, max_letters + 1)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Letter frequency distribution
plt.subplot(2, 2, 3)
plt.bar(all_letters, freq_1, alpha=0.7)
plt.title('Distribution of Letters in Validation Dataset', fontsize=18)
plt.xlabel('Letters', fontsize=18)
plt.ylabel('Frequency', fontsize=18)
plt.grid(True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.subplot(2, 2, 4)
plt.bar(all_letters, freq_2, alpha=0.7)
plt.title('Distribution of Letters in Training Dataset', fontsize=18)
plt.xlabel('Letters', fontsize=18)
plt.ylabel('Frequency', fontsize=18)
plt.grid(True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.tight_layout()
plt.show()
