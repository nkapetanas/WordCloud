import string

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

DATASET_PATH_TRAIN = "C:/Users/Delta/PycharmProjects/WordCloud/dataset/train.csv"
DATASET_PATH_TEST = "C:/Users/Delta/PycharmProjects/WordCloud/dataset/test_without_labels.csv"

# Create the token pattern: TOKENS_ALPHANUMERIC
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'
stop = set(stopwords.words('english'))


def read_dataset(dataset):
    df = pd.read_csv(dataset, encoding='utf-8')
    return df


def remove_punctuation(text):
    no_punct = "".join([word for word in text if word not in string.punctuation])
    return no_punct


def remove_stopwords(text):
    textWithoutStopwords = [word for word in text if word not in stopwords.words('english')]
    return textWithoutStopwords


def clean_data(dataframe):
    dataframe['Content'] = dataframe['Content'].str.lower()
    dataframe['Content'] = dataframe['Content'].str.replace('[^\w\s]', '')
    dataframe['Content'] = dataframe['Content'].apply(
        lambda x: ' '.join([item for item in x.split() if item not in stop]))


def createCSV(prediction, csvName):
    np.savetxt(csvName,
               np.dstack((np.array(test_data["Id"].values), prediction))[0], "%d,%d",
               header="Id,Predicted")


def calculate_metrics(y_actual, y_predicted):
    accuracy = accuracy_score(y_actual, y_predicted)
    precision = precision_score(y_actual, y_predicted, average='micro')
    recall = recall_score(y_actual, y_predicted, average='micro')
    f1 = f1_score(y_actual, y_predicted, average='micro')

    return accuracy, precision, recall, f1
train_data = read_dataset(DATASET_PATH_TRAIN)

test_data = read_dataset(DATASET_PATH_TEST)
classifier = {
    'SGD': SGDClassifier(max_iter=5)
}
# cleaning of data
clean_data(train_data)

clean_data(test_data)
Encoder = LabelEncoder()

# list_of_train_data = np.array_split(train_data, 5)

train_data['Label_Encoded'] = Encoder.fit_transform(train_data['Label'])
# separating features for our model from the target variable
x_train_data = train_data['Content']

y_train_data = train_data['Label_Encoded']

test_data_ = test_data['Content']

cls_stats = {}

kfold = KFold(n_splits=5, random_state=42, shuffle=True)

fold = 0

vectorizer = HashingVectorizer(decode_error='ignore', n_features=2 ** 18, alternate_sign=False,
                               token_pattern=TOKENS_ALPHANUMERIC)
scores_svm_accuracy = []
scores_svm_precision = []
scores_svm_recall = []
scores_svm_f1 = []

scores_rf_accuracy = []
scores_rf_precision = []
scores_rf_recall = []
scores_rf_f1 = []

sgd_classifier = SGDClassifier(max_iter=1000, loss='hinge')

rand_forest_classifier = RandomForestClassifier(n_jobs=-1, max_depth=500)

for train_index, test_index in kfold.split(x_train_data):
    fold += 1
    print("Fold: %s" % fold)

    x_train_k, x_test_k = x_train_data.iloc[train_index], x_train_data.iloc[test_index]
    y_train_k, y_test_k = y_train_data.iloc[train_index], y_train_data.iloc[test_index]

    X_train = vectorizer.fit_transform(x_train_k)
    X_test = vectorizer.fit_transform(x_test_k)

    sgd_classifier.fit(X_train, y_train_k)
    predictedValues = sgd_classifier.predict(X_test)

    accuracy, precision, recall, f1 = calculate_metrics(y_test_k, predictedValues)
    scores_svm_accuracy.append(accuracy)
    scores_svm_precision.append(precision)
    scores_svm_recall.append(recall)
    scores_svm_f1.append(f1)

    rand_forest_classifier.fit(X_train, y_train_k)
    predictedValues_rand_forest = rand_forest_classifier.predict(X_test)

    accuracy, precision, recall, f1 = calculate_metrics(y_test_k, predictedValues_rand_forest)
    scores_rf_accuracy.append(accuracy)
    scores_rf_precision.append(precision)
    scores_rf_recall.append(recall)
    scores_rf_f1.append(f1)

print("SGDClassifier metrics")
print("Accuracy:" + str(np.mean(scores_svm_accuracy)))
print("Precision:" + str(np.mean(scores_svm_precision)))
print("Recall:" + str(np.mean(scores_svm_recall)))
print("F1:" + str(np.mean(scores_svm_f1)))

print("Random Forest metrics")
print("Accuracy:" + str(np.mean(scores_rf_accuracy)))
print("Precision:" + str(np.mean(scores_rf_precision)))
print("Recall:" + str(np.mean(scores_rf_recall)))
print("F1:" + str(np.mean(scores_rf_f1)))

test_data_ = vectorizer.fit_transform(test_data_)

predictedValues = sgd_classifier.predict(test_data_)
predictedValues_rand_forest = rand_forest_classifier.predict(test_data_)


createCSV(predictedValues, "testSet_categories.csv")
createCSV(predictedValues_rand_forest, "testSet_categories2.csv")
