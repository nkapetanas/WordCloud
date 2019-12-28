import string

import pandas as pd
import nltk
# nltk.download()
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
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer, TfidfTransformer

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
    # dataframe['Content'] = dataframe['Content'].apply(lambda x: remove_punctuation(x))
    dataframe['Content'] = dataframe['Content'].str.replace('[^\w\s]', '')
    dataframe['Content'] = dataframe['Content'].apply(
        lambda x: ' '.join([item for item in x.split() if item not in stop]))
    # dataframe['Content'] = dataframe['Content'].apply(lambda x: remove_stopwords(x))


def calculate_metrics(y_actual, y_predicted):
    accuracy = accuracy_score(y_actual, y_predicted)
    precision = precision_score(y_actual, y_predicted, average='micro')
    recall = recall_score(y_actual, y_predicted, average='micro')
    f1 = f1_score(y_actual, y_predicted, average='micro')

    return accuracy, precision, recall, f1


train_data = read_dataset(DATASET_PATH_TRAIN)
test_data = read_dataset(DATASET_PATH_TEST)

# partial_fit_classifiers = {
#     'SGD': SGDClassifier(max_iter=5),
#     'Random Forest Classifier': RandomForestClassifier()
# }

partial_fit_classifiers = {
    'SGD': SGDClassifier(max_iter=5)
}
# cleaning of data
clean_data(train_data)
clean_data(test_data)

Encoder = LabelEncoder()
train_data['Label_Encoded'] = Encoder.fit_transform(train_data['Label'])

classes = np.unique(train_data['Label_Encoded'])

list_of_train_data = np.array_split(train_data, 5)

cls_stats = {}

for trainData in list_of_train_data:

    # separating features for our model from the target variable
    x_train_data = trainData['Content']
    y_train_data = trainData['Label_Encoded']

    x_train, x_test, y_train, y_test = train_test_split(x_train_data, y_train_data, test_size=0.2, shuffle=True)

    kfold = KFold(n_splits=5, random_state=42, shuffle=False)

    fold = 0
    for train_index, test_index in kfold.split(x_train):
        fold += 1

        print("Fold: %s" % fold)
        print(train_index)
        print(test_index)

        x_train, x_test = x_train.iloc[train_index], x_test.iloc[test_index]
        y_train, y_test = y_train.iloc[train_index], y_test.iloc[test_index]

        vectorizer = HashingVectorizer(decode_error='ignore', n_features=2 ** 18, alternate_sign=False,
                                       token_pattern=TOKENS_ALPHANUMERIC)

        X_train = vectorizer.fit_transform(x_train)
        X_test = vectorizer.fit_transform(x_test)
        for clf_name, clf in partial_fit_classifiers.items():
            # text_clf = Pipeline([('vect', CountVectorizer()),
            #                      ('tfidf', TfidfTransformer()),
            #                      ('clf', clf), ])
            # text_clf = Pipeline([('hashVect', HashingVectorizer(decode_error='ignore', n_features=2 ** 18, alternate_sign=False,
            #                                                     token_pattern=TOKENS_ALPHANUMERIC)),
            #                      ('clf', clf)])
            # text_clf.fit(x_train, y_train)

            clf.partial_fit(X_train, y_train, classes=classes)
            predicted = clf.predict(X_test)
            # predicted = text_clf.predict(x_test)
            accuracy, precision, recall, f1 = calculate_metrics(y_test, predicted)
            print(clf_name + ' accuracy = ' + str(accuracy * 100) + '%')
            print(clf_name + ' precision = ' + str(precision))
            print(clf_name + ' recall = ' + str(recall))
            print(clf_name + ' f1 = ' + str(f1))

# # create the transform
# vectorizer = HashingVectorizer(decode_error='ignore', n_features=2 ** 18, alternate_sign=False,
#                                token_pattern=TOKENS_ALPHANUMERIC)
#
# # encode document: encodes the sample document as a 2 ** 18-element sparse array
# vectorizer.fit_transform(x_train)

# text_clf_svm = Pipeline(
#     [('hashVect', HashingVectorizer(decode_error='ignore', n_features=2 ** 18, alternate_sign=False,
#                                     token_pattern=TOKENS_ALPHANUMERIC)),
#      ('svm', SVC(C=1.0, kernel='linear', degree=3, gamma='auto'))])
#
# _ = text_clf_svm.fit(x_train, y_train)
# predicted_svm = text_clf_svm.predict(x_test)
#
# accuracy, precision, recall, f1 = calculate_metrics(y_train, predicted_svm)

# def k_fold_cross(folds, degrees, X, y):
#     kfold = KFold(n_splits=folds, random_state=42, shuffle=False)
#
#     kf_dict = dict([("fold_%s" % i, []) for i in range(1, folds + 1)])
#     fold = 0
#
#     for train_index, test_index in kfold:
#         fold += 1
#
#         print("Fold: %s" % fold)
#
#         x_train, x_test = X.ix[train_index], X.ix[test_index]
#         y_train, y_test = y.ix[train_index], y.ix[test_index]
#
#         # create the transform
#         vectorizer = HashingVectorizer(decode_error='ignore', n_features=2 ** 18, alternate_sign=False,
#                                        token_pattern=TOKENS_ALPHANUMERIC)
#
#         # encode document: encodes the sample document as a 2 ** 18-element sparse array
#         vectorizer.fit_transform(train_data)
#
#         text_clf_svm = Pipeline(
#             [('hashVect', HashingVectorizer(decode_error='ignore', n_features=2 ** 18, alternate_sign=False,
#                                             token_pattern=TOKENS_ALPHANUMERIC)),
#              ('svm', SVC(C=1.0, kernel='linear', degree=3, gamma='auto'))])
#
#         _ = text_clf_svm.fit(x_train, y_train)
#         predicted_svm = text_clf_svm.predict(x_test)
#
#         kf_dict["fold_%s" % fold].append(test_mse)
#         # Convert these lists into numpy arrays to perform averaging
#         kf_dict["fold_%s" % fold] = np.array(kf_dict["fold_%s" % fold])
#
#     # Create the "average test MSE" series by averaging the
#     # test MSE for each degree of the linear regression model,
#     # across each of the k folds.
#     kf_dict["avg"] = np.zeros(degrees)
#     for i in range(1, folds + 1):
#         kf_dict["avg"] += kf_dict["fold_%s" % i]
#     kf_dict["avg"] /= float(folds)
#     return kf_dict
