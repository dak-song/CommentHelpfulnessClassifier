# Import Statements
import re
import nltk
import string
# from profanity import profanity
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.cluster import KMeans
from collections import Counter
from nltk.corpus.reader.bnc import BNCCorpusReader
import time
import math
import operator
import random
import pandas as pd
import numpy as np
from sklearn.externals import joblib  # pickle

# Stopwords and lemmatizer
nltk.download('wordnet')
nltk.download('stopwords')
stops = stopwords.words('english')
stops.append('i\'ve')
stops.append('i\'m')
stops.append('etc')
lmtzr = WordNetLemmatizer()


# Clustering Class
class Clustering(object):
    def __init__(self, algorithm, trainX, trainY, testX, testY):
        None

# RevRank Algorithm


class RevRank():
    def __init__(self, trainX, testX, testY, corpus_data, m=200):
        self.trainX, self.trainX_class, self.rawX = trainX
        self.testX, self.testY = testX, testY
        self.corpus_data = corpus_data
        self.trainX_freq = get_count(self.trainX_class)
        self.m = m

        print("Training")
        self.model = self.train()
        print("Featurizing")
        self.features_classes, self.features = self.featurize(self.trainX)
        print("Scoring")
        self.scores_classes, self.scores, self.scores_classes_avg = self.score(self.trainX)
        print("Labeling")
        self.labels_classes, self.labels = self.label(self.trainX)

    # Getter methods
    def get_model(self):
        return self.model

    def get_features(self):
        return self.features_classes, self.features

    def get_scores(self):
        return self.scores_classes, self.scores, self.scores_classes_avg

    def get_labels(self):
        return self.labels_classes, self.labels

    # Method to get the most helpful comments and corresponding indices in the training data for a particular class
    def get_most_helpful_per_class(self, major, k, reverse=False):
        major_scores = self.scores_classes[major]
        if reverse:
            major_scores.reverse()
        n = len(major_scores)
        if k > n:
            k = n

        helpful = []
        top_k_indices = []
        for i in range(k):
            index, score = major_scores[i]
            helpful.append(self.rawX[index])
            top_k_indices.append(index)

        return helpful, top_k_indices

    # Method to get the most helpful comments and corresponding indices in the training data
    def get_most_helpful_in_model(self, k, reverse=False):
        data_scores = self.scores
        data_scores_indices = [i[0] for i in sorted(enumerate(data_scores), key=operator.itemgetter(1), reverse=True)]
        if reverse:
            data_scores_indices.reverse()
        n = len(data_scores)
        if k > n:
            k = n

        helpful = []
        top_k_indices = data_scores_indices[:k]
        for index in top_k_indices:
            helpful.append(self.rawX[index])

        return helpful, top_k_indices

    # Method to get the most helpful comments and corresponding indices according to the data parameter.
    def get_most_helpful(self, data, k, c=20, l=25, reverse=False):
        data_scores_classes, data_scores, data_scores_classes_avg = self.score(data, c, l)
        data_scores_indices = [i[0] for i in sorted(enumerate(data_scores), key=operator.itemgetter(1), reverse=True)]
        if reverse:
            data_scores_indices.reverse()
        n = len(data_scores)
        if k > n:
            k = n

        helpful = []
        top_k_indices = data_scores_indices[:k]
        for index in top_k_indices:
            helpful.append(data[index])

        return helpful, top_k_indices

    # Helper methods
    def _getClass(self, comment):
        key = ''.join([i for i in comment[2] if not i.isdigit()])

        return key

    # Train method to train data and compute optimal feature vectors for each major
    def train(self, c=20):
        model = {}
        for major, freqs in self.trainX_freq.items():
            word_doms = []
            for word, wfreq in freqs.items():
                Bval = self.corpus_data[word] if word in self.corpus_data else 4
                Dval = wfreq * c * (1 / math.log(Bval))
                word_doms.append((word, Dval))
            word_doms = sorted(word_doms, key=operator.itemgetter(1), reverse=True)[:self.m]
            word_doms = [x[0] for x in word_doms]
            model[major] = word_doms

        return model

    # Featurize methods
    # Method to translate an example into a feature vector
    def featurize_example(self, comment):
        major = self._getClass(comment)
        words = comment[4:]
        xi = [0] * self.m
        major_optimal_features = self.model[major]

        for word in words:
            if word in major_optimal_features:
                word_index = major_optimal_features.index(word)
                xi[word_index] = 1

        return xi

    # Method to translate the data into feature vectors
    def featurize(self, data):
        features_classes = {}
        features = []
        for (i, comment) in enumerate(data):
            comment_major = self._getClass(comment)
            comment_feats = self.featurize_example(comment)
            features.append(comment_feats)
            if comment_major not in features_classes:
                features_classes[comment_major] = [(i, comment_feats)]
            else:
                features_classes[comment_major].append((i, comment_feats))

        return features_classes, features

    # Score methods
    # Method to score the example, get a score of how helpful the example is
    def score_example(self, comment, c=20, l=25):
        words = comment[4:]
        comment_feats = self.featurize_example(comment)
        r = len(words)
        d = sum([i * j for (i, j) in zip(comment_feats, [1] * self.m)])
        p = c if r <= l else 1
        score = (1.0 / p) * (d * 1.0 / r)

        return score

    # Method to score all the examples in training data
    def score(self, data, c=20, l=25):
        scores_classes = {}
        scores = []
        for (i, comment) in enumerate(data):
            comment_major = self._getClass(comment)
            comment_score = self.score_example(comment, c, l)
            scores.append(comment_score)
            if comment_major not in scores_classes:
                scores_classes[comment_major] = [(i, comment_score)]
            else:
                scores_classes[comment_major].append((i, comment_score))

        scores_classes_avg = {}
        for major, major_scores in scores_classes.items():
            s = sum([x[1] for x in major_scores])
            n = len(major_scores)
            scores_classes_avg[major] = (s * 1.0) / n

        for major, major_scores in scores_classes.items():
            major_scores = sorted(major_scores, key=operator.itemgetter(1), reverse=True)
            scores_classes[major] = major_scores

        return scores_classes, scores, scores_classes_avg

    # Label methods
    # Method to label the example as 1 for helpful, 0 for not helpful
    def label_example(self, comment, c=20, l=25):
        comment_major = self._getClass(comment)
        comment_major_avgscore = self.scores_classes_avg[comment_major]
        comment_score = self.score_example(comment, c, l)
        if comment_score >= comment_major_avgscore:
            return 1
        else:
            return 0

    # Method to label the examples in the data
    def label(self, data, c=20, l=25):
        labels_classes = {}
        labels = []
        for (i, comment) in enumerate(data):
            comment_major = self._getClass(comment)
            comment_label = self.label_example(comment, c, l)
            labels.append(comment_label)
            if comment_major not in labels_classes:
                labels_classes[comment_major] = [(i, comment_label)]
            else:
                labels_classes[comment_major].append((i, comment_label))

        return labels_classes, labels

    # Test methods
    # Method to measure accuracy of labeling
    def testAccuracy(self, c=20, l=25):
        test_labels_classes, test_labels = self.label(self.testX)
        correct = 0
        n = 0
        for i, label in enumerate(test_labels):
            real_label = self.testY[i]
            if real_label == label:
                correct += 1
            n += 1

        return (correct * 1.0) / n


# Method to parse the raw csv file of comments. Parse heuristic: Read file line by line, check if line starts with Name or not
def parse_data(path):
    og_data = []
    data = []
    classes = {}  # Dictionary of classes in the form of {(Last name, first name, classID, term): Comment}
    with open(path) as f:
        for line in f:
            nametag = line.find('\"')
            nametag2 = line.find('\"', 1, len(line))
            name = line[nametag + 1:nametag2]
            if nametag == 0 and nametag2 != -1 and name.isupper():
                # Split name into LN, FN+MI
                comment = name.split(",")
                # Split on commas, until comment
                rest = line[nametag2 + 2:].split(',', 2)
                comment = comment + rest[:2]
                ogwords = rest[2]
                words = re.sub(r'[^\w\s\']', ' ', ogwords.lower()).split()
                words = tokenize(words)
                if (len(words) > 0):
                    # Populate classes dictionary
                    # curr_key = (comment[0], comment[1], comment[2], comment[3])
                    curr_key = ''.join([i for i in comment[2] if not i.isdigit()])
                    if curr_key not in classes:
                        classes[curr_key] = words
                    else:
                        classes[curr_key] = classes[curr_key] + words
                    # Populate data
                    comment = comment + words
                    data.append(comment)
                    og_data.append(line)
            else:
                ogcomment = og_data[len(og_data) - 1]
                comment = data[len(data) - 1]
                # add words to this comment
                words = re.sub(r'[^\w\s\']', ' ', line.lower()).split()
                words = tokenize(words)

                if (len(words) > 0):
                    # Populate classes dictionary
                    # curr_key = (comment[0], comment[1], comment[2], comment[3])
                    curr_key = ''.join([i for i in comment[2] if not i.isdigit()])
                    if curr_key not in classes:
                        classes[curr_key] = words
                    else:
                        classes[curr_key] = classes[curr_key] + words
                    # Populate data
                    comment = comment + words
                    data[len(data) - 1] = comment
                    og_data[len(data) - 1] = ogcomment + line

    return data, classes, og_data

# Method to parse csv into pandas dataframe


def parse_dataframe(path, column_names):

    return pd.read_csv(path, sep=',', names=column_names)

# KMeans algorithm


class KM():
    # Input: training and test data (list of lists), test data labels (list)
    def __init__(self, trainX, testX, testY):
        #self.trainX, self.trainX_class, self.rawX = trainX
        #self.testX, self.testY = testX, testY
        self.trainX = trainX
        self.textX = testX
        self.testY = testY
        #self.trainX_freq = get_count(self.trainX_class)

    def fit(self, num_clusters):
        # tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
        #                                    min_df=0.2, stop_words='english',
        #                                    use_idf=True, ngram_range=(1, 3), lowercase=False)
        tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize, lowercase=False)
        tfidf_matrix = tfidf_vectorizer.fit(self.trainX)
        tfidf_matrix = tfidf_vectorizer.transform(self.trainX)
        print(tfidf_matrix.get_feature_names())
        km = KMeans(n_clusters=num_clusters)
        model = km.fit(tfidf_matrix)
        return model
        # clusters = km.labels_.tolist()
        # return clusters

    def predict(self, model, data):
        return model.predict(data)


# Method that returns dictionary of {(Last name, first name, classID, term) : Count of words in comment}


def get_count(classes):
    term_frequencies = {}
    for key, value in classes.items():
        counts = dict(Counter(value))  # vect.fit_transform(value)
        term_frequencies[key] = counts
    return term_frequencies

# Method to tokenize a list of words


def tokenize(words):
    # rid of \n and \t
    # words = [x.strip('\t') for x in words]
    words = [lmtzr.lemmatize(x) for x in words if x not in stops]
    return words

# Method to parse the BCN corpus data and return frequencies of each word


def parse_BCN_data(path):
    freq = {}
    with open(path) as f:
        for line in f:
            l = line.split()
            word = l[2]
            count = int(l[1]) / 100.0
            if word not in freq:
                freq[word] = count
            else:
                freq[word] = freq[word] + count

    return freq


def getRandomData(og_data, x):
    n = len(og_data)
    xs = random.sample(range(0, n), x)
    with open('annotate.txt', 'w') as f:
        for x in xs:
            d = og_data[x]
            f.write(d)
            f.write('\n')


if __name__ == '__main__':
    data, classes, og_data = parse_data("./comments.csv")
    parsed_comments = [[word for word in word_list[4:]] for word_list in data]  # list of lists of classes' parsed comments

    # freqs_bcn = parse_BCN_data("./lemma.al")
    # print("data: ", data)
    # print()
    # print("data with original comments: ", og_data)
    # print(freqs_bcn)
    # print(pd.read_csv('./comments.csv', sep=',').values[0][1])

    # RR = RevRank((data, classes, og_data), [], [], freqs_bcn)
    #df = parse_dataframe("./comments.csv", ['Name', 'Course', 'Term', 'Comment'])

    # Import test data
    mturk_labels = ['Instructor', 'Section', 'Term', 'Helpful1', 'Rank1', 'Helpful2', 'Rank2', 'Helpful3', 'Rank3']
    mturk = parse_dataframe("./MTurk_data.csv", mturk_labels)
    # List of rankings
    testY1 = mturk['Rank1'].tolist()
    testY2 = mturk['Rank2'].tolist()
    # List of binary helpfulness indicators
    helpfulY1 = mturk['Helpful1'].tolist()
    helpfulY2 = mturk['Helpful2'].tolist()

    # Split parsed_comments into training and test set
    KMtestX, KMtrainX = parsed_comments[:105], parsed_comments[105:]
    KMtestY = testY1
    KM = KM(KMtrainX, KMtestX, KMtestY)
    #print(KM.predict(comments=KM.get_all_comments(KM.get_values()), num_clusters=5))
    print(KM.predict(KM.fit(5), KMtestY))

    # Calculate inter-rater agreement (Between person 1 and 2)
    agreed_ranking = 0
    agreed_helpful = 0
    for i, score in enumerate(testY1):
        if score == testY2[i]:
            agreed_ranking += 1
        if helpfulY1[i] == helpfulY2[i]:
            agreed_helpful += 1

    agreement_ranking = agreed_ranking / len(testY1)
    print("Agreement on ranking: ", agreement_ranking)

    agreement_helpful = agreed_helpful / len(testY1)
    print("Agreement on helpful: ", agreement_helpful)

    # helpful, indices = RR.get_most_helpful_per_class("NURS", 3)
    # nhelpful, nindices = RR.get_most_helpful_per_class("NURS", 3, reverse=True)
    # print (helpful)
    # print()
    # print(nhelpful)
    #
    # helpful, indices = RR.get_most_helpful_in_model(10)
    # nhelpful, nindices = RR.get_most_helpful_in_model(10, reverse=True)
    # for h in helpful:
    #     print (h)
    #
    # print("\nNot Helpful\n")
    # for nh in nhelpful:
    #     print(nh)

    # for i, label in enumerate(RR.labels):
    #     print (og_data[i])
    #     print (label)
    #     print ()

    # getRandomData(og_data, 200)
    # print(len(data))
    # print(data[0])
    # s = 0
    # for d in data:
    #     s += len(d)
    # print (s * 1.0 / (len(data))) #32.55
    #
    # v = 0
    # for (key,val) in classes.items():
    #     v += len(val)
    # print (v * 1.0 / (len(classes))) #5556.14
    # freqs = get_count(classes)
    # print(len(freqs["NURS"])) #5493
    # print (freqs)

    # freqs_bcn = parse_BCN_data("./lemma.al")
    # print(freqs_bcn["great"]) #643.69
    # print(len(classes)) #107 documents, or majors
