import numpy as np
import pandas as pd
from os import path
from PIL import Image
from jedi.refactoring import inline
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt

DATASET_PATH_TRAIN = "C:/Users/Delta/PycharmProjects/WordCloud/dataset/train.csv"


def read_dataset(dataset):
    df = pd.read_csv(dataset)
    df.head()
    return df


def create_wordClouds(text):
    stopwords = set(STOPWORDS)

    # Create and generate a word cloud image:
    # return WordCloud(max_font_size=25, max_words=250, background_color="white", stopwords=stopwords).generate(text)
    return WordCloud().generate(text)


def plot_wordcloud(wordcloud):
    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


def get_Content_Text(df):
    words = ",".join(df['Content'])
    return words



def worldcloud_toImage(wordcloud, category):
    wordcloud.to_file("img/" + category + ".png")


df = read_dataset(DATASET_PATH_TRAIN)

df_Business = df.loc[df['Label'] == "Business"]
df_Entertainment = df.loc[df['Label'] == "Entertainment"]
df_Health = df.loc[df['Label'] == "Health"]
df_Technology = df.loc[df['Label'] == "Technology"]

text = get_Content_Text(df_Business)


word_cloud = create_wordClouds(text)
plot_wordcloud(word_cloud)

# worldCloud_toImage(word_cloud, "Business")


