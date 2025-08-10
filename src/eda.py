import pandas as pd
import matplotlib.pyplot as plt
# 2. Define the path to your CSV file
data_path = "data\liver_cirrhosis_edited.csv"

# 3. Load the dataset
df = pd.read_csv(data_path)
# 4. Look at the first 5 rows of the dataset
print("First 5 rows:")
print(df.head())
# 5. Print the shape of the dataset
print("\nShape of dataset:", df.shape)
# 6. Show basic info about each column
print("\nDataset info:")
print(df.info())
# 7. Check for missing values in each column
print("\nMissing values per column:")
print(df.isnull().sum())
# 8. Optional — view basic statistics for numeric columns
print("\nBasic statistics:")
print(df.describe())

#  Plot the distribution of the target variable 'Stage'
plt.figure(figsize=(6, 4))  # set plot size
df['Stage'].value_counts().sort_index().plot(
    kind='bar',
    color=['#4CAF50', '#FFC107', '#F44336']  # green, yellow, red for stages 1, 2, 3
)

plt.title('Distribution of Liver Cirrhosis Stages', fontsize=14)
plt.xlabel('Stage', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()