import string

import pandas as pd
from sklearn.linear_model import SGDClassifier
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import HashingVectorizer

DATASET_PATH_TRAIN = "C:/Users/Delta/PycharmProjects/WordCloud/dataset/train.csv"
DATASET_PATH_TEST = "C:/Users/Delta/PycharmProjects/WordCloud/dataset/test_without_labels.csv"

# Create the token pattern: TOKENS_ALPHANUMERIC
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'


def read_dataset(dataset):
    df = pd.read_csv(dataset)
    return df


def clean_data(dataframe):
    dataframe['Content'] = dataframe['Content'].str.lower()
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
vectorizer = HashingVectorizer(decode_error='ignore', n_features=2 ** 18, alternate_sign=False, token_pattern=TOKENS_ALPHANUMERIC)

# encode document: encodes the sample document as a 2 ** 18-element sparse array
vectorizer.fit_transform(train_data_cleaned)
