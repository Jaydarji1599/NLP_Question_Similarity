import numpy
import sys
from sklearn.ensemble import RandomForestClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import svm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import metrics
import statistics
import pandas as pd
import pickle
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import time
import warnings
warnings.filterwarnings('ignore')


def evaluate_score(output, test_label):
    accuracy = metrics.accuracy_score(test_label, output)
    error = 1 - accuracy
    precision, recall, f1_score, _ = precision_recall_fscore_support(test_label, output, average='binary')
    return error, accuracy, precision, recall, f1_score
    #error is equal to accuracy - 1 which is also same as 1-the fraction of misclassified cases
    #creating the mean of each error value and making an array
    #https://stats.stackexchange.com/questions/133458/is-accuracy-1-test-error-rate

def train(input_data, input_label, kernel_type_lin, kernel_type_rbf):
    # Create lists to store various metrics
    all_error = []
    errork1 = []
    errork5 = []
    errork10 = []
    error_svm_linear = []
    error_svm_rbf = []
    error_random_forest = []
    accuracyk1 = []
    accuracyk5 = []
    accuracyk10 = []
    accuracy_svm_linear = []
    accuracy_svm_rbf = []
    accuracy_random_forest = []

    # Use StratifiedShuffleSplit to split the data into train and test sets
    StratifiedK = StratifiedShuffleSplit(n_splits=5, train_size=0.8, test_size=0.2, random_state=0)
    StratifiedK.get_n_splits(input_data, input_label)

    for data_train, data_test in tqdm(StratifiedK.split(input_data, input_label), total=StratifiedK.get_n_splits()):
        # Split the data into training and testing sets
        training_set, testing_set = input_data[data_train], input_data[data_test]
        train_label, test_label = input_label[data_train], input_label[data_test]

        # Initialize classifiers
        knn1 = KNeighborsClassifier(n_neighbors=1)
        knn5 = KNeighborsClassifier(n_neighbors=5)
        knn10 = KNeighborsClassifier(n_neighbors=10)
        ran_clf = RandomForestClassifier(n_estimators=100)
        svm_lin_clf = svm.SVC(kernel=kernel_type_lin, gamma='scale')
        svm_rbf_clf = svm.SVC(kernel=kernel_type_rbf, gamma='scale')

        # Train the classifiers and save the models to disk
        start_time = time.time()
        svm_lin_clf.fit(training_set, train_label)
        svm_lin_model = 'svm_lin_model.sav'
        pickle.dump(svm_lin_clf, open(svm_lin_model, 'wb'))
        output_svm_lin = svm_lin_clf.predict(testing_set)

        svm_rbf_clf.fit(training_set, train_label)
        svm_rbf_model = 'svm_rbf_model.sav'
        pickle.dump(svm_rbf_clf, open(svm_rbf_model, 'wb'))
        output_svm_rbf = svm_rbf_clf.predict(testing_set)

        ran_clf.fit(training_set, train_label)
        ran_clf_model = 'ran_clf_model.sav'
        pickle.dump(ran_clf, open(ran_clf_model, 'wb'))
        ran_output = ran_clf.predict(testing_set)

        knn1.fit(training_set, train_label)
        knn1_model = 'knn1_model.sav'
        pickle.dump(knn1, open(knn1_model, 'wb'))
        outputk1 = knn1.predict(testing_set)

        knn5.fit(training_set, train_label)
        knn5_model = 'knn5_model.sav'
        pickle.dump(knn5, open(knn5_model, 'wb'))
        outputk5 = knn5.predict(testing_set)

        knn10.fit(training_set, train_label)
        knn10_model = 'knn10_model.sav'
        pickle.dump(knn10, open(knn10_model, 'wb'))
        outputk10 = knn10.predict(testing_set)

        end_time = time.time()
        knn_train_time = end_time - start_time
        print(f"Time before and during training for kNN model: {knn_train_time:.2f} seconds")

        # Evaluating classifiers
        error_svm_lin, accuracy_svm_lin = evaluate_score(output_svm_lin, test_label)
        error_svm_linear.append(error_svm_lin)
        accuracy_svm_linear.append(accuracy_svm_lin)

        error_svm_rbf1, accuracy_svm_rbf1 = evaluate_score(output_svm_rbf, test_label)
        error_svm_rbf.append(error_svm_rbf1)
        accuracy_svm_rbf.append(accuracy_svm_rbf1)

        error_random_forest1, accuracy_random_forest1 = evaluate_score(ran_output, test_label)
        error_random_forest.append(error_random_forest1)
        accuracy_random_forest.append(accuracy_random_forest1)

        err_k1, acc_k1 = evaluate_score(outputk1, test_label)
        errork1.append(err_k1)
        accuracyk1.append(acc_k1)

        err_k5, acc_k5 = evaluate_score(outputk5, test_label)
        errork5.append(err_k5)
        accuracyk5.append(acc_k5)

        err_k10, acc_k10 = evaluate_score(outputk10, test_label)
        errork10.append(err_k10)
        accuracyk10.append(acc_k10)

    # Computing the mean error and accuracy for each classifier
    all_error.append(statistics.mean(error_svm_linear))
    all_error.append(statistics.mean(accuracy_svm_linear))
    all_error.append(statistics.mean(error_svm_rbf))
    all_error.append(statistics.mean(accuracy_svm_rbf))
    all_error.append(statistics.mean(error_random_forest))
    all_error.append(statistics.mean(accuracy_random_forest))
    all_error.append(statistics.mean(errork1))
    all_error.append(statistics.mean(accuracyk1))
    all_error.append(statistics.mean(errork5))
    all_error.append(statistics.mean(accuracyk5))
    all_error.append(statistics.mean(errork10))
    all_error.append(statistics.mean(accuracyk10))
    print(all_error)

    # Return the mean errors, accuracies, and saved model files
    return all_error, knn1_model, knn5_model, knn10_model, svm_lin_model, svm_rbf_model, ran_clf_model


def predict(knn1_model, knn5_model, knn10_model, svm_lin_model, svm_rbf_model, ran_clf_model, testing_set, test_label):
    # Predicting test labels using trained models
    outputk1 = knn1_model.predict(testing_set)
    outputk5 = knn5_model.predict(testing_set)
    outputk10 = knn10_model.predict(testing_set)
    outputsvm_lin = svm_lin_model.predict(testing_set)
    outputsvm_rbf = svm_rbf_model.predict(testing_set)
    outputran_clf = ran_clf_model.predict(testing_set)

    # Evaluating the performance of each model
    errork1, accuracyk1, precisionk1, recallk1, f1_scorek1 = evaluate_score(outputk1, test_label)
    errork5, accuracyk5, precisionk5, recallk5, f1_scorek5 = evaluate_score(outputk5, test_label)
    errork10, accuracyk10, precisionk10, recallk10, f1_scorek10 = evaluate_score(outputk10, test_label)
    errorksvm_lin, accuracysvm_lin, precisionsvm_lin, recallsvm_lin, f1_scoresvm_lin = evaluate_score(outputsvm_lin, test_label)
    errorsvm_rbf, accuracysvm_rbf, precisionsvm_rbf, recallsvm_rbf, f1_scoresvm_rbf = evaluate_score(outputsvm_rbf, test_label)
    errorran_clf, accuracyran_clf, precisionran_clf, recallran_clf, f1_scoreran_clf = evaluate_score(outputran_clf, test_label)

    # Consolidating performance metrics in a dictionary
    data = {'Model': ['KNN-1', 'KNN-5', 'KNN-10', 'SVM-Linear', 'SVM-RBF', 'RandomForest'],
            'Error': [errork1, errork5, errork10, errorksvm_lin, errorsvm_rbf, errorran_clf],
            'Accuracy': [accuracyk1, accuracyk5, accuracyk10, accuracysvm_lin, accuracysvm_rbf, accuracyran_clf],
            'Precision': [precisionk1, precisionk5, precisionk10, precisionsvm_lin, precisionsvm_rbf, precisionran_clf],
            'Recall': [recallk1, recallk5, recallk10, recallsvm_lin, recallsvm_rbf, recallran_clf],
            'F1_Score': [f1_scorek1, f1_scorek5, f1_scorek10, f1_scoresvm_lin, f1_scoresvm_rbf, f1_scoreran_clf]}
    
    return data



if __name__ == '__main__':

    dataset = sys.argv[1]

    percentage = float(sys.argv[2])

    firstdata = pd.read_csv('dataset.csv')
    data = firstdata.sample(8000,random_state=2)

    data_set = pd.DataFrame(data)

    print(data_set.shape)
    final_df = data_set.drop(columns=['id','qid1','qid2','question1','question2'])

    input_label = numpy.array(final_df.iloc[:, 1])
    input_data = numpy.array(final_df.drop(columns=['is_duplicate', 'Unnamed: 0']))

    all_error, knn1_model, knn5_model, knn10_model, svm_lin_model, svm_rbf_model, ran_clf_model = train(input_data,input_label,'linear','rbf')

    datag = predict(knn1_model, knn5_model, knn10_model, svm_lin_model, svm_rbf_model, ran_clf_model, input_data, input_label)


