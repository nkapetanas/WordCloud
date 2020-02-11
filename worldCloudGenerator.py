import string
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud, STOPWORDS

DATASET_PATH_TRAIN = "C:/Users/Delta/PycharmProjects/WordCloud/dataset/train.csv"


def read_dataset(dataset):
    df = pd.read_csv(dataset)
    return df


def create_wordClouds(text, category):
    stopwords = set(STOPWORDS)

    # Create and generate a word cloud image:
    wc = WordCloud(stopwords=stopwords, background_color="white", max_font_size=50, max_words=250)

    counts_all = Counter()

    for line in text:
        counts_line = wc.process_text(line)
        counts_all.update(counts_line)

    wc.generate_from_frequencies(counts_all)
    worldcloud_toImage(wc, category)


def plot_wordcloud(wordcloud):
    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


def get_Content_Text(df):
    words = "".join(df['Content'])
    return words


def worldcloud_toImage(wordcloud, category):
    wordcloud.to_file(category + ".png")


def clean_data(dataframe):
    dataframe['Content'] = dataframe['Content'].str.lower()
    all_content = dataframe['Content'].str.split(' ')
    all_text_cleaned = []
    for text in all_content:
        text = [x.strip(string.punctuation) for x in text]
        all_text_cleaned.append(text)
    text = [" ".join(text) for text in all_text_cleaned]
    return text


def removeSpecialChars(df):
    df['Content'] = df['Content'].str.replace('\d+', '')
    df['Content'] = df['Content'].str.replace('_', '')
    df['Content'] = df['Content'].str.replace('?', '')
    df['Content'] = df['Content'].str.replace('•', '')
    df['Content'] = df['Content'].str.replace("@", '')
    df['Content'] = df['Content'].str.replace('▯', '')
    df['Content'] = df['Content'].str.replace("'", '')
    df['Content'] = df['Content'].str.replace(",", "")
    df['Content'] = df['Content'].str.replace("$", "")
    return df

df = read_dataset(DATASET_PATH_TRAIN)
df_Business = df.loc[df['Label'] == "Business"]
df_Entertainment = df.loc[df['Label'] == "Entertainment"]
df_Health = df.loc[df['Label'] == "Health"]
df_Technology = df.loc[df['Label'] == "Technology"]

df_Business = removeSpecialChars(df_Business)
df_Entertainment = removeSpecialChars(df_Entertainment)
df_Health = removeSpecialChars(df_Health)
df_Technology = removeSpecialChars(df_Technology)

text_business = clean_data(df_Business)
text_entertainment = clean_data(df_Entertainment)
text_health = clean_data(df_Health)
text_technology = clean_data(df_Technology)

create_wordClouds(text_business, "Business")
create_wordClouds(text_entertainment, "Entertainment")
create_wordClouds(text_health, "Health")
create_wordClouds(text_technology, "Technology")