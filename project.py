# Import Statements
import re


# Clustering Class
class Clustering(object):
    def __init__(self, algorithm, trainX, trainY, testX, testY):
        None


# Method to parse the raw csv file of comments
# Parse heuristic: Read file line by line, check if line starts with Name or not
# 20858 Comments
def parse_data(path):
    data = []
    with open(path) as f:
        for line in f:
            nametag = line.find('\"')
            nametag2 = line.find('\"', 1, len(line))
            name = line[nametag+1:nametag2]
            if nametag == 0 and nametag2 != -1 and name.isupper():
                # Split name into LN, FN+MI
                comment = name.split(",")
                # Split on commas, until comment
                rest = line[nametag2 + 2:].split(',', 2)
                comment = comment + rest[:2]
                words = rest[2]
                words = re.sub(r'[^\w\s\']', ' ', words.lower()).split()
                comment = comment + words
                data.append(comment)
            else:
                comment = data[len(data) - 1]
                # add words to this comment
                words = re.sub(r'[^\w\s\']', ' ', line.lower()).split()
                comment = comment + words
                data[len(data) - 1] = comment

    return data





if __name__ == '__main__':
    data = parse_data("./comments.csv")
    print(len(data))
