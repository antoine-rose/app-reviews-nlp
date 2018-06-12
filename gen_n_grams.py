from textblob import TextBlob
import json
from utils import *
from gensim.utils import simple_preprocess


def generate_n_grams(tokenized_sentence, ngram=2):
    my_list = []
    for index in range(len(tokenized_sentence) + 1 - ngram):
        my_list.append(tokenized_sentence[index:index + ngram])
    return my_list


if __name__ == '__main__':
    for n_gram in [2, 3, 4, 5]:

        n_grams_dict = {}

        with open("textme_review.json") as f:
            reviews = json.load(f)

        for review in reviews:
            if not review["language"] == "en":
                continue

            txblb_sentences = TextBlob(review["title"] + ". " + review["body"]).sentences

            for sentence in txblb_sentences:
                try:
                    n_grams_list = generate_n_grams(tokenized_sentence=simple_preprocess(str(sentence)), ngram=n_gram)
                    # n_grams = TextBlob(sentence=sentence).ngrams(ngram=n_gram)
                    for n_gram_item in n_grams_list:
                        n_gram_string = " ".join(n_gram_item)
                        if not n_grams_dict.get(n_gram_string, None):
                            n_grams_dict[n_gram_string] = 0
                        n_grams_dict[n_gram_string] += 1
                except:
                    pass

        with open("textme_{}_grams.json".format(n_gram), "w") as f:
            print("{} {}_grams saved!".format(len(n_grams_dict), n_gram))
            json.dump(n_grams_dict, f)
