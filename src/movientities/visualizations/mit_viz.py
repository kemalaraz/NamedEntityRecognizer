import os
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter

class MitVisualizations:

    __files__ = ["train.txt", "test.txt"]

    @staticmethod
    def __open_file(filename:str):
        with open(filename, "r") as txt:
            word, label = [], []
            words, labels = [], []
            query, all_labels = [], []
            for line in txt.readlines():
                if line != "\n":
                    line = line.split(" ")
                    word.append(line[0])
                    label.append(line[-1][:-1])
                    query.append(line[0])
                    all_labels.append(line[-1][:-1])
                else:
                    words.append(word)
                    labels.append(label)
                    word, label = [], []
        return words, labels, query, all_labels

    @staticmethod
    def __count_characters(filepath:str):
        string = ""
        lengths = []
        with open(filepath) as f:
            for line in f.readlines():
                if line != "\n":
                    string += line.split(" ")[0]
                else:
                    lengths.append(len(string))
                    string = ""
        return lengths

    def __return_dataset(filepath:str):
        path = list(*os.walk(filepath))
        assert all(f in MitVisualizations.__files__ for f in path[2]),\
            f"All files in the folder must be in {MitVisualizations.__files__}"
        dataset = {}
        for p in path[2]:
            dataset["data" + "_" + p.split(".")[0]] = MitVisualizations.__open_file(os.path.join(path[0], p))
        return dataset

    @staticmethod
    def histogram(filepath:str, bins:int):
        name = filepath.split("/")[-2]
        data = MitVisualizations.__count_characters(filepath)
        plt.style.use('ggplot')
        plt.hist(data, bins=bins)
        plt.title(name + " character count")
        plt.xlabel('Number of characters')
        plt.ylabel('Count')
        plt.show()

    @staticmethod
    def pie(filepath:str):
        dataset = MitVisualizations.__return_dataset(filepath)
        sizes = [len(dataset['data_train'][0]), len(dataset['data_test'][0])]
        labels = [f"train = {len(dataset['data_train'][0])}", f"test = {len(dataset['data_test'][0])}"]
        plt.figure(figsize=(7,7))
        plt.pie(sizes, colors=sns.color_palette("hls", 2), labels=labels, autopct='%1.1f%%', startangle=90)
        plt.axis("equal")
        plt.savefig("Train-test pie chart")
        plt.show()

    @staticmethod
    def word_count(filepath:str):
        dataset = MitVisualizations.__return_dataset(filepath)
        # Number of words in each query
        words_each_query =  []
        for w_query in dataset["data_train"][0]:
            words_each_query.append(len(w_query))
        # words_each_query
        # Wordcloud
        wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate("+".join(dataset["data_train"][2]))
        plt.figure( figsize=(15,10))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.savefig("WordCloud")
        plt.show()

    @staticmethod
    def word_frequency(filepath:str):
        name = filepath.split("/")[-1]
        dataset = MitVisualizations.__return_dataset(filepath)
        # Number of words in queries
        words_each_query =  []
        for w_query in dataset["data_train"][0]:
            words_each_query.append(len(w_query))
        fig, ax = plt.subplots(figsize=(15,6))
        sns.distplot(words_each_query, bins=100, color="blue").set_title(f"{name} Word - Sentence Frequency")
        ax.set_xlabel('Number of words in a Sentence', fontsize = 12)
        ax.set_xlim([0, 50])
        plt.tick_params(axis='x', which='major', labelsize=8)
        ax.yaxis.tick_left()
        plt.savefig("Word - Sentence Frequency")

    @staticmethod
    def most_frequent(filepath:str, top:int=50):
        dataset = MitVisualizations.__return_dataset(filepath)
        word_counts = dict(Counter(dataset["data_train"][2]))
        most_word = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        sample_words = []
        sample_counts = []
        for i in range(top):
            sample_words.append(most_word[i][0])
            sample_counts.append(most_word[i][1])
        # Most occured 100 words' counts
        fig, ax = plt.subplots(figsize=(15,6))
        sns.barplot(x=sample_words, y=sample_counts).set_title(f"{top} Most Frequent Words")
        plt.xticks(rotation=90)
        ax.set_xlabel('Words', fontsize = 15)
        plt.tick_params(axis='x', which='major', labelsize=10)
        ax.yaxis.tick_left()
        plt.savefig(f"{top} Most Frequent Words")

    def label_counts(filepath:str):
        name = filepath.split("/")[-1]
        dataset = MitVisualizations.__return_dataset(filepath)
        label_counts = dict(Counter(dataset["data_train"][3]))
        most_label = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)

        # Number of labels
        sample_labs = []
        lab_counts = []
        for i in range(len(most_label)):
            sample_labs.append(most_label[i][0])
            lab_counts.append(most_label[i][1])

        fig, ax = plt.subplots(figsize=(15,6))
        sns.barplot(x=sample_labs, y=lab_counts).set_title(f"{name} Label Counts")
        plt.xticks(rotation=90)
        ax.set_xlabel('Labels', fontsize = 15)
        plt.tick_params(axis='x', which='major', labelsize=10)
        ax.yaxis.tick_left()

        xlocs, xlabs = plt.xticks()
        for i, v in enumerate(lab_counts):
            plt.text(xlocs[i] - 0.25, v + 0.01, str(v))

        plt.savefig("Label Counts")