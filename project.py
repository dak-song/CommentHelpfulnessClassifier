# Import Statements
import re
import nltk
import string
from profanity import profanity
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from collections import Counter


# Clustering Class
class Clustering(object):
    def __init__(self, algorithm, trainX, trainY, testX, testY):
        None


# Method to parse the raw csv file of comments
# Parse heuristic: Read file line by line, check if line starts with Name or not
# 20858 Comments
def parse_data(path):
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
                words = rest[2]
                words = re.sub(r'[^\w\s\']', ' ', words.lower()).split()
                words = tokenize(words)
                if (len(words) > 0):
                    # Populate classes dictionary
                    curr_key = (comment[0], comment[1], comment[2], comment[3])
                    classes[curr_key] = words
                    # Populate data
                    comment = comment + words
                    data.append(comment)
            else:
                comment = data[len(data) - 1]
                # add words to this comment
                words = re.sub(r'[^\w\s\']', ' ', line.lower()).split()
                words = tokenize(words)

                if (len(words) > 0):
                    # Populate classes dictionary
                    curr_key = (comment[0], comment[1], comment[2], comment[3])
                    classes[curr_key] = words
                    # Populate data
                    comment = comment + words
                    data[len(data) - 1] = comment

    return data, classes


def get_count(classes):
    term_frequencies = {}
    for key, value in classes.items():
        counts = dict(Counter(value))  # vect.fit_transform(value)
        term_frequencies[key] = counts
    return term_frequencies



# Stopwords and lemmatizer
nltk.download('wordnet')
nltk.download('stopwords')
stops = stopwords.words('english')
stops.append('ive')
stops.append('im')
stops.append('etc')
lmtzr = WordNetLemmatizer()


def tokenize(words):
    # rid of \n and \t
    words = [x.strip('\t') for x in words]
    # make words lowercase
    words = [x.lower() for x in words]
    # remove stopwords
    words = [x for x in words if x not in stops]
    # rid of punctuation
    words = [''.join(x for x in s if x not in string.punctuation) for s in words]
    # remove stopwords again
    words = [x for x in words if x not in stops]
    # rid of '' again
    words = [x for x in words if x]
    # rid of digits
    words = [x for x in words if not any(c.isdigit() for c in x)]
    words = [x for x in words if not (x.isdigit() or x[0] == '-' and x[1:].isdigit())]
    words = [lmtzr.lemmatize(x) for x in words]
    words = [x for x in words if len(x) > 2]
    return words


if __name__ == '__main__':
    data, classes = parse_data("./comments.csv")
    print(len(data))

get_count(classes)
