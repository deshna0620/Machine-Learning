Titanic Data Preprocessing using Pandas, NumPy, and Matplotlib

This project performs data preprocessing on the Titanic dataset using Python libraries: Pandas, NumPy, and Matplotlib. The goal is to clean and prepare the data for further analysis or modeling.

Task Description

The task involves the following steps:

1. Load the Titanic dataset and display initial insights (first 10 rows, dataset shape, summary statistics, and data info).
2. Handle missing values by:
   - Filling missing 'Age' values with the median age.
   - Filling missing 'Embarked' values with the most frequent category.
   - Dropping the 'Cabin' column due to excessive missing values.
3. Feature engineering:
   - Create a new feature `FamilySize` by adding `SibSp` + `Parch` + 1.
   - Extract the passenger's title from the `Name` column.
4. Bin the `Age` column into 5 equal-width bins (`AgeGroup`).
5. Convert categorical variables `Sex` and `Embarked` into numeric columns using one-hot encoding.
6. Drop irrelevant columns such as `PassengerId`, `Ticket`, and `Name`.
7. Visualize the dataset:
   - Histogram of `Age`.
   - Bar chart of survival counts by gender.
   - Box plot of `Fare` distribution by passenger class.
8. Use NumPy to compute mean, median, and standard deviation for `Fare` and `Age` columns, and apply min-max normalization.
9. Compute and visualize the correlation matrix of numerical features.
10. Save the cleaned dataset as `titanic_cleaned.csv`.

How to Run

1. Ensure Python is installed (make sure to add Python to PATH during installation).
2. Install required libraries:
pip install pandas numpy matplotlib
3. Place the `Titanic-Dataset.csv` file in the project folder.
4. Run the Python script:
python 2025-05-22.py
5. The cleaned data will be saved as `titanic_cleaned.csv` in the same folder.

Files in Repository

- 2025-05-22.py` : Python script containing the preprocessing code.
- Titanic-Dataset.csv` : Original Titanic dataset.
