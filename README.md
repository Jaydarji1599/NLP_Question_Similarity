# Question Similarity Classification Project

This project aims to build a question similarity classification system using natural language processing and machine learning techniques. We use the Quora Question Pairs dataset to train and evaluate six different models to identify duplicate questions. This system can improve the user experience on question-and-answer platforms like Quora by reducing content redundancy and ensuring that users find the information they seek efficiently.

## Project Structure

The project consists of three Python files:

data.py: Responsible for reading the data, preprocessing, data analysis, data visualization, and data engineering. It outputs a new dataset as a .csv file containing the extracted features.
for3D.py: Generates a 3D representation of all the extracted features from the data.py file.
train.py: Trains all the models discussed in the project using the generated dataset with all the features and provides the results for each model.

## Usage

## Prerequisites
Python 3.11.1
Libraries: pandas, numpy, seaborn, matplotlib, scikit-learn, nltk, fuzzywuzzy, BeautifulSoup

## Steps
1. Clone this repository to your local machine.
2. Install the required libraries using pip install -r requirements.txt.
3. Run data.py to read the data, preprocess it, and perform data analysis and visualization tasks. This script   will output a .csv file containing the extracted features.
4. Run for3D.py to generate a 3D representation of all the extracted features.
5. Run train.py to train and evaluate the six different models using the generated dataset with all the features. 
6. The results for each model will be displayed on the console.

## Models

The following models are used in this project:

1. K-Nearest Neighbors (KNN) with k = 1, 5, and 10
2. Support Vector Machines (SVM) with linear kernel
3. Support Vector Machines (SVM) with Radial Basis Function (RBF) kernel
3. Random Forest

The project includes a comprehensive evaluation of the models using metrics like error, accuracy, precision, recall, and F1 score. These metrics help us compare the performance of different models on the testing dataset and select the best-performing model for our task.

## Contributing

Feel free to contribute to this project by submitting a pull request, reporting bugs, or suggesting new features. Your contributions are highly appreciated.