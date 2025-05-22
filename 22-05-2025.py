import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Show basic info
print("First 10 rows:")
print(df.head(10))
print("\nShape of dataset:", df.shape)
print("\nSummary statistics:")
print(df.describe())
print("\nInfo:")
print(df.info())

# Handle missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin'], inplace=True)

# Feature engineering
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# Age binning
df['AgeGroup'] = pd.cut(df['Age'], bins=5)

# One-hot encoding
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Drop irrelevant columns
df.drop(columns=['PassengerId', 'Ticket', 'Name'], inplace=True)

# Visualization - Age histogram
plt.hist(df['Age'], bins=20, color='skyblue')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Bar chart - Survival by gender
df.groupby('Sex_male')['Survived'].sum().plot(kind='bar', color=['orange', 'green'])
plt.title('Survival Count by Gender')
plt.xlabel('Sex (0 = Female, 1 = Male)')
plt.ylabel('Count')
plt.show()

# Boxplot - Fare by Pclass
df.boxplot(column='Fare', by='Pclass')
plt.title('Fare by Pclass')
plt.xlabel('Pclass')
plt.ylabel('Fare')
plt.suptitle('')
plt.show()

# NumPy stats
fare = df['Fare'].values
age = df['Age'].values

print("\nFare - Mean:", np.mean(fare), "Median:", np.median(fare), "Std:", np.std(fare))
print("Age - Mean:", np.mean(age), "Median:", np.median(age), "Std:", np.std(age))

# Min-max normalization
fare_norm = (fare - np.min(fare)) / (np.max(fare) - np.min(fare))
age_norm = (age - np.min(age)) / (np.max(age) - np.min(age))

# Correlation matrix
corr = df.corr(numeric_only=True)
plt.imshow(corr, cmap='coolwarm')
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title('Correlation Matrix')
plt.show()

# Final step: save cleaned data
df.to_csv("titanic_cleaned.csv", index=False)
print("\nCleaned dataset saved as 'titanic_cleaned.csv'")
