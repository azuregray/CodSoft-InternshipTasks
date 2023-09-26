# CODSOFT SEPTEMBER 2023 ML TASK 2 - Credit Card Fraud Detection

## Main Objective: *Fraud Credit Card Data Analysis and Detection*.

This repository contains Python code for detecting credit card fraud using a Random Forest classifier. The code preprocesses transaction data, handles class imbalance, applies dimensionality reduction, and evaluates the model's performance.

## Libraries and Techniques Used

- **Pandas** (`pandas`) for data manipulation and analysis.
- **NumPy** (`numpy`) for numerical operations.
- **Scikit-Learn** (`scikit-learn`) for machine learning tools, including RandomForestClassifier, StandardScaler, LabelEncoder, OneHotEncoder, and IncrementalPCA.
- **Imbalanced-Learn** (`imblearn`) for oversampling with SMOTE to address class imbalance.
- **TQDM** (`tqdm`) for displaying progress bars during data processing.

## Code Overview

1. **Import necessary libraries**: Import all the required libraries for the project, including pandas, numpy, scikit-learn components, imblearn, and tqdm for progress bars.
2. **Load training and testing datasets**: Load the training and testing data from CSV files (`fraudTrain.csv` and `fraudTest.csv`).
3. **Combine training and testing data**: Combine the datasets to ensure encoding consistency and feature extraction.
4. **Extract datetime features**: Extract relevant features from the "trans_date_trans_time" column, including day of the week and hour of the day.
5. **Drop irrelevant columns**: Drop columns that are irrelevant for fraud detection (customize based on your data).
6. **Separate features and target variable**: Separate the dataset into features (X_combined) and the target variable (y_combined).
7. **Encode categorical columns**: Use LabelEncoder to encode "merchant" and "category" columns and OneHotEncoder for other categorical variables.
8. **Standardize numeric features**: Standardize the numeric features.
9. **Combine encoded categorical and numeric features**: Combine one-hot encoded categorical features with standardized numeric features.
10. **Split data back into training and test datasets**: Split the combined data back into training and test datasets.
11. **Address class imbalance using SMOTE**: Use SMOTE to oversample the minority class in the training data.
12. **Apply Incremental PCA for dimensionality reduction**: Apply Incremental PCA to reduce the dimensionality of the data.
13. **Train the Random Forest model**: Define and train a Random Forest classifier with the resampled and dimensionality-reduced data.
14. **Predict using the trained model**: Predict fraud detection results using the trained Random Forest model.
15. **Evaluate model performance**: Calculate and display accuracy, confusion matrix, and classification report.


### Usage

1. Ensure you have the required libraries installed as mentioned in the requirements section.
2. Place your training and testing data in CSV files named `fraudTrain.csv` and `fraudTest.csv`.
3. Run the provided code to preprocess the data, train the Random Forest model, and evaluate fraud detection performance.
4. Review the output, including accuracy, confusion matrix, and classification report, to assess the model's performance.

### TL;DR 

> This code implements credit card fraud detection using a Random Forest classifier, involving data preprocessing, handling class imbalance with SMOTE, dimensionality reduction with Incremental PCA, and model evaluation. To use the code, ensure required libraries are installed, place your data in CSV files, run the code, and evaluate fraud detection performance with accuracy, confusion matrix, and classification report.

### Requirements

- Python 3.x
- Pandas (`pandas`)
- NumPy (`numpy`)
- Scikit-Learn (`scikit-learn`)
- TQDM (`tqdm`)
- **Imbalanced-Learn** (`imblearn`)

### Check out Video Demonstration of this Project on my YouTube Channel:

> [Take me there](https://youtu.be/-ZdOvb067as?si=0QoocA3fwCnNjS94)

### Author

Darshan S
> Contact me @ [LinkedIn](https://linkedin.com/in/arcticblue/) | [Instagram](https://instagram.com/thedarshgowda) | [Email](mailto:d7gowda@gmail.com)

