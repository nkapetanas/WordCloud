import string

import pandas as pd
from sklearn.linear_model import SGDClassifier
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import HashingVectorizer

DATASET_PATH_TRAIN = "C:/Users/Delta/PycharmProjects/WordCloud/dataset/train.csv"
DATASET_PATH_TEST = "C:/Users/Delta/PycharmProjects/WordCloud/dataset/test_without_labels.csv"

# Create the token pattern: TOKENS_ALPHANUMERIC
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'


def read_dataset(dataset):
    df = pd.read_csv(dataset)
    return df


def remove_punctuation(text):
    no_punct = "".join([word for word in text if word not in string.punctuation])
    return no_punct


def remove_stopwords(text):
    textWithoutStopwords = [word for word in text if word not in stopwords.words('english')]
    return textWithoutStopwords


def clean_data(dataframe):
    dataframe['Content'] = dataframe['Content'].str.lower()
    dataframe['Content'] = dataframe['Content'].apply(lambda x: remove_punctuation(x))
    dataframe['Content'] = dataframe['Content'].apply(lambda x: remove_stopwords(x))
    all_content = dataframe['Content'].str.split(' ')

    all_text_cleaned = []
    for text in all_content:
        text = [x.strip(string.punctuation) for x in text]
        all_text_cleaned.append(text)
    text = [" ".join(text) for text in all_text_cleaned]
    return text


train_data = read_dataset(DATASET_PATH_TRAIN)
test_data = read_dataset(DATASET_PATH_TEST)

# cleaning of data
train_data_cleaned = clean_data(train_data)
test_data_cleaned = clean_data(test_data)

# create the transform
vectorizer = HashingVectorizer(decode_error='ignore', n_features=2 ** 18, alternate_sign=False,
                               token_pattern=TOKENS_ALPHANUMERIC)

# encode document: encodes the sample document as a 2 ** 18-element sparse array
vectorizer.fit_transform(train_data_cleaned)

text_clf_svm = Pipeline([('hashVect', HashingVectorizer(decode_error='ignore', n_features=2 ** 18, alternate_sign=False,
                                                        token_pattern=TOKENS_ALPHANUMERIC)),
                         ('svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42))])

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=i
)
_ = text_clf_svm.fit(x_train, y_train)
predicted_svm = text_clf_svm.predict(x_test)

accuracy_score(y_test, predicted_svm)
precision_score(y_test, predicted_svm)
recall_score(y_test, predicted_svm)
f1_score(y_test, predicted_svm)



cv = KFold(n_splits=5, random_state=42, shuffle=False)
