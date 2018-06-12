from gensim.models import Word2Vec, KeyedVectors
import json
import re
from nltk.corpus import stopwords
from textblob import TextBlob
from utils import clean_text
from collections import Counter
import numpy as np


def load_google_w2v_model():
    model_path = 'w2v_models/google_news_w2v_model.bin.gz',
    w2v_model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    print(w2v_model.similar_by_word(word='text', topn=10))
    return w2v_model


def review_to_wordlist(review, remove_stopwords=False):
    # 1. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", review)

    # 2. Convert to lower case, split into individual words
    words = letters_only.lower().split()

    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if w not in stops]

    return words


def create_w2v_from_long_sentences(reviews):
    # We prepare the sentences
    sentences = []
    for index, review in enumerate(reviews):
        review_text = review["title"] + ". " + review["body"]
        sentences.append(review_to_wordlist(review_text))

    # We train the model
    w2v_model = Word2Vec(
        sentences,
        size=150,
        # window=20,
        min_count=2,
        workers=8,  # Number of threads to use to process
    )

    w2v_model.train(
        sentences,
        total_examples=len(sentences),
        epochs=10)

    return w2v_model


def create_w2v_from_unit_sentences(reviews):
    # We prepare the sentences
    sentences = []
    for index, review in enumerate(reviews):
        txblb_sentences = TextBlob(review["title"] + ". " + review["body"]).sentences
        txblb_sentences = [clean_text(review["title"] + ". " + review["body"])]
        for sentence in txblb_sentences:
            sentences.append(str(sentence))

    print("Working on {} sentences!".format(len(sentences)))

    w2v_model = Word2Vec(
        sentences=sentences,
        size=100,
        # window=10,
        min_count=2,
        workers=8,  # Number of threads to use to process
    )

    w2v_model.train(
        sentences,
        total_examples=len(sentences),
        epochs=10)

    w2v_model.save("textme_w2v_model")
    print("Word2Vec model successfully built and saved!")

    return w2v_model


def kaggle_w2v_method(reviews):
    # Download the punkt tokenizer for sentence splitting
    import nltk.data

    nltk.download()

    # Load the punkt tokenizer
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    # Define a function to split a review into parsed sentences
    def review_to_sentences(review, tokenizer, remove_stopwords=False):
        # Function to split a review into parsed sentences. Returns a
        # list of sentences, where each sentence is a list of words

        # 1. Use the NLTK tokenizer to split the paragraph into sentences
        raw_sentences = tokenizer.tokenize(review.strip())

        # 2. Loop over each sentence
        sentences = []
        for raw_sentence in raw_sentences:
            # If a sentence is empty, skip it
            if len(raw_sentence) > 0:
                # Otherwise, call review_to_wordlist to get a list of words
                sentences.append(review_to_wordlist(raw_sentence,
                                                    remove_stopwords))

        # Return the list of sentences (each sentence is a list of words,
        # so this returns a list of lists
        return sentences

    sentences = []  # Initialize an empty list of sentences

    print "Parsing sentences from training set"
    for review in reviews:
        review_text = review["title"] + ". " + review["body"]
        sentences += review_to_sentences(review_text, tokenizer)

    print "Working with {} sentences...".format(len(sentences))

    # Import the built-in logging module and configure it so that Word2Vec
    # creates nice output messages
    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)

    # Set values for various parameters
    num_features = 300  # Word vector dimensionality
    min_word_count = 10  # Minimum word count
    num_workers = 4  # Number of threads to run in parallel
    context = 10  # Context window size
    downsampling = 1e-3  # Downsample setting for frequent words

    # Initialize and train the model (this will take some time)
    print "Training model..."
    model = Word2Vec(
        sentences,
        workers=num_workers,
        size=num_features,
        min_count=min_word_count,
        window=context,
        sample=downsampling
    )

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    # It can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using Word2Vec.load()
    model_name = "{}features_{}minwords_{}context".format(num_features, min_word_count, context)
    model.save(model_name)

    return model


if __name__ == '__main__':
    model_name = "300features_10minwords_10context"

    try:
        w2v_model = Word2Vec.load(model_name)
    except Exception as e:
        print str(e)

        # We load the reviews
        with open("../reviews/textme_review.json", 'r') as f:
            reviews = json.load(f)

        print "Building a new model!"
        w2v_model = kaggle_w2v_method(reviews=reviews)

    print w2v_model.similar_by_word("call")

    print w2v_model.most_similar("number")
