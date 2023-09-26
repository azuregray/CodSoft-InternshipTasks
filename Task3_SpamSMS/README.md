# CODSOFT SEPTEMBER 2023 ML TASK 3 - Spam SMS Detection

## Main Objective: *Spam SMS Data Learning, Identification, Classification and generate a Statistical Report*.

This folder contains Python code for detecting spam SMS messages using Natural Language Processing (NLP) techniques. The code preprocesses SMS data, applies TF-IDF vectorization, and trains a Multinomial Naive Bayes classifier to classify messages as either spam or legitimate. The goal of this project is to build an AI model capable of classifying SMS messages as spam or legitimate. The techniques used include TF-IDF vectorization and a Multinomial Naive Bayes classifier for accurate classification.

## Libraries and Techniques Used

- **Pandas** (`pandas`) for data manipulation and analysis.
- **NumPy** (`numpy`) for numerical operations.
- **Scikit-Learn** (`scikit-learn`) for machine learning tools, including MultinomialNB, TfidfVectorizer, and train_test_split.
- **TQDM** (`tqdm`) for displaying a progress bar during processing.

## Code Overview

1. **Import necessary libraries**: Import required libraries, including pandas, numpy, scikit-learn components, and tqdm for progress bars.
2. **Load the SMS Spam Collection dataset**: Load the SMS dataset from 'spam.csv', encoding it with 'latin-1'.
3. **Preprocess the data**: Remove duplicates, map labels to 'ham' (legitimate) and 'spam', and split the data into features (X) and labels (y).
4. **Split the data into training and testing sets**: Split the dataset into training and testing sets using train_test_split.
5. **Create a TF-IDF vectorizer**: Initialize a TF-IDF vectorizer to convert text data into numerical features.
6. **Fit the vectorizer to the training data**: Transform the SMS text data into TF-IDF features for training.
7. **Initialize a Naive Bayes classifier**: Create a Multinomial Naive Bayes classifier.
8. **Train the classifier**: Train the classifier using the TF-IDF transformed training data.
9. **Transform the test data**: Use the same vectorizer to transform the SMS text data into TF-IDF features for testing.
10. **Make predictions**: Predict whether SMS messages are spam or legitimate using the trained classifier.
11. **Calculate accuracy**: Calculate the accuracy of the model's predictions.
12. **Display classification report**: Generate a classification report that includes precision, recall, F1-score, and support for both 'Legitimate SMS' and 'Spam SMS'.

### Usage

1. Ensure you have the required libraries installed as mentioned in the requirements section.
2. Prepare your SMS dataset or use the provided 'spam.csv' dataset.
3. Run the provided code to preprocess the data, train the Multinomial Naive Bayes classifier, and evaluate the model's performance.
4. Review the accuracy and the classification report to assess the model's effectiveness in detecting spam SMS messages.

### TL;DR 

> This code implements a spam SMS detection model using TF-IDF vectorization and Multinomial Naive Bayes classification, with data preprocessing and evaluation, to classify SMS messages as spam or legitimate. To use the code, ensure the required libraries are installed, provide your SMS dataset or use the provided one, run the code to train the model, and evaluate its performance using accuracy and a classification report for spam and legitimate SMS classification.

### Requirements

- Python 3.x
- Pandas (`pandas`)
- NumPy (`numpy`)
- Scikit-Learn (`scikit-learn`)
- TQDM (`tqdm`)

### Check out Video Demonstration of this Project on my YouTube Channel:

> [Take me there](https://youtu.be/v35TquFBSVw?si=_6njO4PWjuPTH7hI)

### Author

Darshan S
> Contact me @ [LinkedIn](https://linkedin.com/in/arcticblue/) | [Instagram](https://instagram.com/thedarshgowda) | [Email](mailto:d7gowda@gmail.com)


