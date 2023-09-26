# CODSOFT SEPTEMBER 2023 ML TASK 1 - Movie Genre Classification

## Main Objective: *Movie Genre Learning, Identification and Classification*.

This folder contains Python code for a movie genre prediction system using MultiOutput Naive Bayes. It predicts movie genres based on their plots.

## Libraries and Techniques Used

- **Pandas** (`pandas`) for data manipulation and analysis.
- **NumPy** (`numpy`) for numerical operations.
- **Scikit-Learn** (`scikit-learn`) for machine learning tools, including TfidfVectorizer, MultiOutputClassifier, MultinomialNB, and MultiLabelBinarizer.
- **TQDM** (`tqdm`) for displaying progress bars during data processing.

## Code Overview

1. **Import necessary libraries**: Import all the required libraries for the project, including pandas, numpy, scikit-learn components, and tqdm for progress bars.
2. **Define the list of genres**: Create a list of movie genres that will be used for classification.
3. **Define a fallback genre**: Specify a fallback genre to assign to movies with no predicted genre.
4. **Load training data**: Load the training data from `train_data.txt` using pandas, where each row contains a movie serial number, name, genre(s), and plot.
5. **Data preprocessing for training data**: Prepare the training data by converting movie plot text to lowercase and encoding genre labels using MultiLabelBinarizer.
6. **TF-IDF Vectorization**: Use TfidfVectorizer to convert the movie plot text into TF-IDF features.
7. **Train the MultiOutput Naive Bayes classifier**: Train a MultiOutputClassifier using a Multinomial Naive Bayes classifier with the training data.
8. **Load test data**: Load the test data from `test_data.txt` using pandas, which includes movie serial number, name, and plot.
9. **Data preprocessing for test data**: Preprocess the test data by converting movie plot text to lowercase.
10. **Vectorize the test data**: Transform the test data using the same TF-IDF vectorizer used for training.
11. **Predict genres on test data**: Predict movie genres on the test data using the trained model.
12. **Create a results DataFrame**: Create a DataFrame that contains movie names and predicted genres.
13. **Replace empty predicted genres**: Replace empty predicted genres with the fallback genre.
14. **Write results to a text file**: Save the prediction results to `model_evaluation.txt` with proper formatting and UTF-8 encoding.
15. **Calculate evaluation metrics**: Calculate evaluation metrics, including accuracy, precision, recall, and F1-score, using training labels as a proxy.
16. **Append metrics to the output file**: Append the evaluation metrics to the `model_evaluation.txt` file.


### Usage

1. Specify the configuration file (`config_file`) and frozen model file (`frozen_model`) for the pre-trained model.
2. Load class labels from a text file (`file_name`) to recognize objects.
3. Configure model input size, scaling, and color format.
4. Load an image or a video for object detection and recognition.
5. Display the recognized objects, their bounding boxes, and confidence scores.
6. For video, real-time object detection and recognition are performed.

### TL;DR 

> This code utilizes MultiOutput Naive Bayes to predict movie genres based on plot descriptions, involving data preprocessing, TF-IDF vectorization, and evaluation metrics calculation. To use the code, ensure required libraries are installed, provide training and test data, run the code, and review genre predictions and evaluation metrics in `model_evaluation.txt`.

### Requirements

- Python 3.x
- Pandas (`pandas`)
- NumPy (`numpy`)
- Scikit-Learn (`scikit-learn`)
- TQDM (`tqdm`)

### Check out Video Demonstration of this Project on my YouTube Channel:

> [Take me there](https://youtu.be/InSIB84O3Co?si=D3FlAk2JVehmfC1k)

### Author

Darshan S
> Contact me @ [LinkedIn](https://linkedin.com/in/arcticblue/) | [Instagram](https://instagram.com/thedarshgowda) | [Email](mailto:d7gowda@gmail.com)

