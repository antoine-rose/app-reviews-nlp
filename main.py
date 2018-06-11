from gensim.models import Word2Vec, KeyedVectors
from gensim.utils import simple_preprocess
import json
from utils import *
from textblob import TextBlob
import nltk

# nltk.download('popular')


def build_w2v_model():
    with open("textme_review.json") as f:
        reviews = json.load(f)

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
        min_count=1,
        workers=8,  # Number of threads to use to process
    )

    w2v_model.train(
        sentences,
        total_examples=len(sentences),
        epochs=10)

    w2v_model.save("textme_w2v_model")
    print("Word2Vec model successfully built and saved!")

    return w2v_model


if __name__ == '__main__':
    try:
        model = Word2Vec.load("textme_w2v_model1")
    except Exception as e:
        print(str(e))
        print("Building a new model!")
        model = build_w2v_model()

    print(model.similar_by_word("call"))
