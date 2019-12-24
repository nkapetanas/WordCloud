import string

import numpy as np
import pandas as pd
from collections import Counter
from os import path
from PIL import Image
from string import punctuation
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt

DATASET_PATH_TRAIN = "C:/Users/Delta/PycharmProjects/WordCloud/dataset/train.csv"


def read_dataset(dataset):
    df = pd.read_csv(dataset)
    df.head()
    return df


def create_wordClouds(text, category):
    stopwords = set(STOPWORDS)

    # Create and generate a word cloud image:
    wc = WordCloud(stopwords=stopwords, background_color="white", max_font_size=50, max_words=200)

    counts_all = Counter()

    for line in text:
        counts_line = wc.process_text(line)
        counts_all.update(counts_line)

    wc.generate_from_frequencies(counts_all)
    worldcloud_toImage(wc, category)
    # wc.to_file('wc.png')


def plot_wordcloud(wordcloud):
    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


def get_Content_Text(df):
    words = "".join(df['Content'])
    return words


def remove_num(text):
    textWithoutNumbers = ''.join([i for i in text if not i.isdigit()])
    return textWithoutNumbers


def worldcloud_toImage(wordcloud, category):
    wordcloud.to_file(category + ".png")


df = read_dataset(DATASET_PATH_TRAIN)

df_Business = df.loc[df['Label'] == "Business"]
df_Entertainment = df.loc[df['Label'] == "Entertainment"]
df_Health = df.loc[df['Label'] == "Health"]
df_Technology = df.loc[df['Label'] == "Technology"]

df_Business['Content'] = df_Business['Content'].str.lower()
all_business = df_Business['Content'].str.split(' ')

all_business_cleaned = []

for text in all_business:
    text = [x.strip(string.punctuation) for x in text]
    all_business_cleaned.append(text)

text_business = [" ".join(text) for text in all_business_cleaned]

create_wordClouds(text_business, "Business")


# df_Business['Content'] = df_Business['Content'].apply(lambda x : x.strip().capitalize())

# df_Business['Content'] = df_Business['Content'].str.replace('\d+', '')
# df_Business['Content'] = df_Business['Content'].str.replace('_', '')
# df_Business['Content'] = df_Business['Content'].str.replace('?', '')
# df_Business['Content'] = df_Business['Content'].str.replace('•', '')
# df_Business['Content'] = df_Business['Content'].str.replace("@", '')
# df_Business['Content'] = df_Business['Content'].str.replace('▯', '')
# df_Business['Content'] = df_Business['Content'].str.replace("'", '')
# df_Business['Content'] = df_Business['Content'].str.replace(",", "")

# text = get_Content_Text(df_Business)


# removing other characters
def remove_u(text):
    text = text.replace('_', '')
    text = text.replace('?', '')
    text = text.replace('•', '')
    text = text.replace("@", '')
    text = text.replace('▯', '')
    text = text.replace("'", '')
    text = text.replace(",", "")
    return text

# worldCloud_toImage(word_cloud, "Business")
