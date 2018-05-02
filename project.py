# Import Statements
import re
import nltk
import string
# from profanity import profanity
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from collections import Counter
from nltk.corpus.reader.bnc import BNCCorpusReader
import time
import math
import operator
import random

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
        self.trainX, self.trainX_class = trainX
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
        for comment in data:
            comment_major = self._getClass(comment)
            comment_feats = self.featurize_example(comment)
            features.append(comment_feats)
            if comment_major not in features_classes:
                features_classes[comment_major] = [comment_feats]
            else:
                features_classes[comment_major].append(comment_feats)

        return features_classes, features

    # Score methods
    # Method to score the example, get a score of how helpful the example is
    def score_example(self, comment, c=20, l=25):
        words = comment[4:]
        comment_feats = self.featurize_example(comment)
        r = len(words)
        d = sum([i*j for (i, j) in zip(comment_feats, [1] * self.m)])
        p = c if r <= l else 1
        score = (1.0 / p) * (d * 1.0 / r)

        return score

    # Method to score all the examples in training data
    def score(self, data, c=20, l=25):
        scores_classes = {}
        scores = []
        for comment in data:
            comment_major = self._getClass(comment)
            comment_score = self.score_example(comment, c, l)
            scores.append(comment_score)
            if comment_major not in scores_classes:
                scores_classes[comment_major] = [comment_score]
            else:
                scores_classes[comment_major].append(comment_score)

        scores_classes_avg = {}
        for major, major_scores in scores_classes.items():
            s = sum(major_scores)
            n = len(major_scores)
            scores_classes_avg[major] = (s * 1.0) / n

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
        for comment in data:
            comment_major = self._getClass(comment)
            comment_label = self.label_example(comment, c, l)
            labels.append(comment_label)
            if comment_major not in labels_classes:
                labels_classes[comment_major] = [comment_label]
            else:
                labels_classes[comment_major].append(comment_label)

        return labels_classes, labels

    # Test methods
    # Method to measure accuracy of labeling
    def testAccuracy(self, c=20, l=25):
        test_labels_classes, test_labels = self.label(self.testX)
        correct = 0
        n = 0
        for i, label in test_labels:
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
    # make words lowercase
    # words = [x.lower() for x in words]
    # remove stopwords
    words = [lmtzr.lemmatize(x) for x in words if x not in stops]
    # rid of punctuation
    # words = [''.join(x for x in s if x not in string.punctuation) for s in words]
    # remove stopwords again
    # words = [x for x in words if x not in stops]
    # rid of '' again
    # words = [x for x in words if x]
    # rid of digits
    # words = [x for x in words if not any(c.isdigit() for c in x)]
    # words = [x for x in words if not (x.isdigit() or x[0] == '-' and x[1:].isdigit())]
    # words = [lmtzr.lemmatize(x) for x in words]
    # words = [x for x in words if len(x) > 2]
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

# Method to calculate the dominance of each word, and return a dictionary of
# {Major : {word : dominance value}} and a dictionary of {Major : top m dominance [word]}
def calc_dominance(freq, freq_corpus, c, m):
    dom = {}
    topdom = {}
    for major, counts in freq.items():
        dom[major] = {}
        for word, count in counts.items():
            Bval = freq_corpus[word] if word in freq_corpus else 4
            Dval = count * c * (1 / math.log(Bval))
            dom[major][word] = Dval

        major_doms = sorted(dom[major].items(), key=operator.itemgetter(1), reverse=True)
        major_doms = major_doms[:m]
        major_doms = [x[0] for x in major_doms]
        topdom[major] = major_doms

    return dom, topdom

def calc_feat_score(comment, topdom, m, c, l):
    key = ''.join([i for i in comment[2] if not i.isdigit()])
    words = comment[4:]
    comment_feats = [0] * m
    key_optimal_feats = topdom[key]
    # Compute features
    for word in words:
        if word in key_optimal_feats:
            word_i = key_optimal_feats.index(word)
            comment_feats[word_i] = 1

    # Compute score
    r = len(words)
    d = sum([i*j for (i, j) in zip(comment_feats, [1] * m)])
    p = c if r <= l else 1
    score = (1.0 / p) * (d * 1.0 / r)

    return comment, comment_feats, sum(comment_feats), score

def calc_feats_scores(data, topdom, m, c, l):
    features = []
    scores = []
    scores_classes = {}
    for comment in data:
        key = ''.join([i for i in comment[2] if not i.isdigit()])
        comment_detail = calc_feat_score(comment, topdom, m, c, l)
        features.append(comment_detail[1])
        scores.append(comment_detail[3])

        if key not in scores_classes:
            scores_classes[key] = [comment_detail]
        else:
            scores_classes[key].append(comment_detail)

    avg_classes = {}
    labels_classes = {}
    for major, detailed_comments in scores_classes.items():
        # Calculate average score for the major
        totalscore = sum([s for (i, j, k, s) in detailed_comments])
        avgscore = (totalscore * 1.0) / len(detailed_comments)
        avg_classes[major] = avgscore

        # Sort the detailed comments by score in the major
        sorted_d_c = sorted(detailed_comments, key=operator.itemgetter(3), reverse=True)
        scores_classes[major] = sorted_d_c

        # Create labels
        major_labels = []
        for comment_details in scores_classes[major]:
            if comment_details[3] >= avgscore:
                major_labels.append(1)
            else:
                major_labels.append(0)
        labels_classes[major] = major_labels

    # Create labels for data
    labels = []
    for i, comment in enumerate(data):
        key = ''.join([i for i in comment[2] if not i.isdigit()])
        avgscore = avg_classes[key]
        score = scores[i]
        if score >= avgscore:
            labels.append(1)
        else:
            labels.append(0)

    return features, scores, scores_classes, avg_classes, labels, labels_classes

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
    freqs_bcn = parse_BCN_data("./lemma.al")
    RR = RevRank((data, classes), [], [], freqs_bcn)
    for i, label in enumerate(RR.labels):
        print (og_data[i])
        print (label)
        print ()



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

    # dom, topdom = calc_dominance(freqs, freqs_bcn, 3, 200)
    # feats, scores, classes_details, avg_classes, labels, labels_classes = calc_feats_scores(data, topdom, 200, 20, 25)
    # print(len(feats))
    # print(feats[0])
    # print(len(scores))
    # print(scores[0])
    # print(classes_details["NURS"])
    # print(len(classes_details))

    # print(data[:10])
    # print(labels[:10])
    # print ()
    # print(classes_details["NURS"][:10])
    # print(labels_classes["NURS"][:10])
    # print ()
    # print (avg_classes)
