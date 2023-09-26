'''
Hello again everyone! This is Darshan S
Intern at CODSOFT, India.

I'm very thrilled to share my Third Task at CodSoft Internship September 2023 viz. SPAM SMS DETECTION MODEL.
More details on the structure, working and requirements are available in the README files of respective folder name corresponding to the Task Name.
'''

# AUTHOR: DARSHAN S
# TASK NAME: Spam SMS Detection
# 4th Task in the List of Tasks
# TASK CATEGORY: Machine Learning
# DATE OF SUBMISSION: 26 September 2023
# LinkedIn Profile: https://linkedin.com/in/arcticblue/
# GitHub Repository: https://github.com/azuregray/CodSoft-InternshipTasks/


# Lets start the code by importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# Load the SMS Spam Collection dataset while specifying appropriate encoding format.
data = pd.read_csv('spam.csv', encoding='latin-1')

# Preprocess the input data
data.drop_duplicates(inplace=True)
data['label'] = data['v1'].map({'ham': 'ham', 'spam': 'spam'})
X = data['v2']
y = data['label']

# Split the data into two sets: Training Set and Testing Set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit the vectorizer to the training data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Initialize a Naive Bayes classifier
classifier = MultinomialNB()

# Train the classifier
classifier.fit(X_train_tfidf, y_train)

# Transform the test data using the same vectorizer
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Make predictions
y_pred = classifier.predict(X_test_tfidf)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Display classification report with labels 'ham' and 'spam'
report = classification_report(y_test, y_pred, target_names=['Legitimate SMS', 'Spam SMS'])

# Create a progress bar
progress_bar = tqdm(total=100, position=0, leave=True)

# Simulate progress updates
for i in range(10, 101, 10):
    progress_bar.update(10)
    progress_bar.set_description(f'Progress: {i}%')

# Close the progress bar
progress_bar.close()

# Display the results on the interface where the code was initiated from.
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(report)
