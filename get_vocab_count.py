import json
import re
from nltk.corpus import stopwords
from utils import *
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def clean_review(review):
    # 1. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", review)

    # 2. Convert to lower case, split into individual words
    words = letters_only.lower().split()

    # 3. Remove stop words
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if w not in stops]

    return " ".join(meaningful_words)


if __name__ == '__main__':
    # We load the reviews
    with open("./textme_review.json", 'r') as f:
        reviews = json.load(f)

    print "Cleaning and parsing the training set reviews..."
    clean_train_reviews = []
    for review in reviews:
        review_text = review["title"] + " " + review["body"]
        clean_train_reviews.append(clean_review(review_text))

    print "Creating the bag of words..."
    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.
    vectorizer = CountVectorizer(analyzer="word",
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 max_features=5000)

    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of
    # strings.
    train_data_features = vectorizer.fit_transform(clean_train_reviews)

    # Numpy arrays are easy to work with, so convert the result to an
    # array
    train_data_features = train_data_features.toarray()

    # Take a look at the words in the vocabulary
    vocab = vectorizer.get_feature_names()

    # Sum up the counts of each vocabulary word
    dist = np.sum(train_data_features, axis=0)

    # For each, print the vocabulary word and the number of times it
    # appears in the training set
    counter = Counter(dict(zip(vocab, dist)))
    for tag, count in counter.most_common(200):
        print tag, count
