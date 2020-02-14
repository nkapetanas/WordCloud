import string

import pandas as pd
import nltk
from sklearn.decomposition import TruncatedSVD

from sklearn.linear_model import SGDClassifier
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer, TfidfTransformer, TfidfVectorizer

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


def calculate_metrics(y_actual, y_predicted):
    accuracy = accuracy_score(y_actual, y_predicted)
    precision = precision_score(y_actual, y_predicted, average='micro')
    recall = recall_score(y_actual, y_predicted, average='micro')
    f1 = f1_score(y_actual, y_predicted, average='micro')

    return accuracy, precision, recall, f1

scores_svm_accuracy = []
scores_svm_precision = []
scores_svm_recall = []
scores_svm_f1 = []

scores_rf_accuracy = []
scores_rf_precision = []
scores_rf_recall = []
scores_rf_f1 = []

train_data = read_dataset(DATASET_PATH_TRAIN)
test_data = read_dataset(DATASET_PATH_TEST)

classifier = {
    'SGD': SGDClassifier(max_iter=5)
}
# cleaning of data
clean_data(train_data)
clean_data(test_data)

Encoder = LabelEncoder()
train_data['Label_Encoded'] = Encoder.fit_transform(train_data['Label'])

classes = np.unique(train_data['Label_Encoded'])

# list_of_train_data = np.array_split(train_data, 5)

# separating features for our model from the target variable
x_train_data = train_data['Content']
y_train_data = train_data['Label_Encoded']

cls_stats = {}

kfold = KFold(n_splits=5, random_state=42, shuffle=True)

sgd_classifier = SGDClassifier(max_iter=1000, loss='hinge')
rand_forest_classifier = RandomForestClassifier(n_jobs=-1, max_depth=500)

fold = 0

svd = TruncatedSVD(n_components=16)
tfidf_vectorizer = TfidfVectorizer(stop_words=stop)

for train_index, test_index in kfold.split(x_train_data):
    fold += 1
    print("Fold: %s" % fold)

    x_train_k, x_test_k = x_train_data.iloc[train_index], x_train_data.iloc[test_index]
    y_train_k, y_test_k = y_train_data.iloc[train_index], y_train_data.iloc[test_index]

    x_train_k_vectorized = tfidf_vectorizer.fit_transform(x_train_k)
    x_test_k_vectorized = tfidf_vectorizer.fit_transform(x_test_k)

    X_reduced = svd.fit_transform(x_train_k_vectorized)
    X_test = svd.fit_transform(x_test_k_vectorized)


    sgd_classifier.fit(X_reduced, y_train_k)
    predictedValues = sgd_classifier.predict(X_test)

    accuracy, precision, recall, f1 = calculate_metrics(y_test_k, predictedValues)

    scores_svm_accuracy.append(accuracy)
    scores_svm_precision.append(precision)
    scores_svm_recall.append(recall)
    scores_svm_f1.append(f1)

    rand_forest_classifier.fit(X_reduced, y_train_k)
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
