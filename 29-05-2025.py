import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("titanic.csv")

# Select features and target
features = ['Pclass', 'Sex', 'Age', 'Fare']
target = 'Survived'
df_model = df[features + [target]].copy()

# Handle missing values
df_model['Age'].fillna(df_model['Age'].median(), inplace=True)
df_model['Fare'].fillna(df_model['Fare'].median(), inplace=True)

# Encode categorical variable 'Sex'
le = LabelEncoder()
df_model['Sex'] = le.fit_transform(df_model['Sex'])  # male = 1, female = 0

# Split data
X = df_model[features]
y = df_model[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print results
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(report)

# 1. Survival count by Sex
plt.figure(figsize=(6, 4))
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title('Survival Count by Sex')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.legend(title='Survived')
plt.tight_layout()
plt.show()

# 2. Age distribution by Survival
plt.figure(figsize=(6, 4))
sns.histplot(data=df, x='Age', hue='Survived', multiple='stack', kde=True, bins=30)
plt.title('Age Distribution by Survival')
plt.xlabel('Age')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# 3. Fare distribution by Class and Survival
plt.figure(figsize=(6, 4))
sns.boxplot(x='Pclass', y='Fare', hue='Survived', data=df)
plt.title('Fare Distribution by Class and Survival')
plt.xlabel('Passenger Class')
plt.ylabel('Fare')
plt.tight_layout()
plt.show()
