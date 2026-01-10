import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Paths
DATA_PATH = "../data/creditcard.csv"
FIGURE_PATH = "../results/figures"

os.makedirs(FIGURE_PATH, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)

# Basic info
print("Dataset Shape:", df.shape)
print("\nClass Distribution:\n", df['Class'].value_counts())

# Plot class distribution
plt.figure()
sns.countplot(x='Class', data=df)
plt.title("Fraud vs Non-Fraud Transactions")
plt.xlabel("Class (0 = Non-Fraud, 1 = Fraud)")
plt.ylabel("Count")
plt.savefig(f"{FIGURE_PATH}/class_distribution.png")
plt.show()
