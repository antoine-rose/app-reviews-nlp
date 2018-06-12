from gensim.models import Word2Vec, KeyedVectors
from gensim.utils import simple_preprocess
import json
from utils import *
import nltk

# nltk.download('popular')


if __name__ == '__main__':
    try:
        model = Word2Vec.load("textme_w2v_model1")
    except Exception as e:
        print(str(e))
        print("Building a new model!")
        model = build_w2v_model()

    print(model.similar_by_word("call"))
