"""
Add tags to reviews
1.  CC  Coordinating conjunction
2.  CD  Cardinal number
3.  DT  Determiner
4.  EX  Existential there
5.  FW  Foreign word
6.  IN  Preposition or subordinating conjunction
7.  JJ  Adjective
8.  JJR Adjective, comparative
9.  JJS Adjective, superlative
10. LS  List item marker
11. MD  Modal
12. NN  Noun, singular or mass
13. NNS Noun, plural
14. NNP Proper noun, singular
15. NNPS    Proper noun, plural
16. PDT Predeterminer
17. POS Possessive ending
18. PRP Personal pronoun
19. PRP$    Possessive pronoun
20. RB  Adverb
21. RBR Adverb, comparative
22. RBS Adverb, superlative
23. RP  Particle
24. SYM Symbol
25. TO  to
26. UH  Interjection
27. VB  Verb, base form
28. VBD Verb, past tense
29. VBG Verb, gerund or present participle
30. VBN Verb, past participle
31. VBP Verb, non-3rd person singular present
32. VBZ Verb, 3rd person singular present
33. WDT Wh-determiner
34. WP  Wh-pronoun
35. WP$ Possessive wh-pronoun
36. WRB Wh-adverb
"""

import json
from collections import Counter
from textblob import TextBlob
from utils import *

NOUNS_FEATURES = ['NN', 'NNS', ]
ADJECTIVES_FEATURES = ['JJ', 'JJR', 'JJS']
ADVERBS_FEATURES = ['RB', 'RBR', 'RBS']
VERBS_FEATURES = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
NUMBERS_FEATURES = ['CD']


def is_valid_keyphrase(tags):
    # Adjective + Noun : 'great app'
    cond1 = any(i in tags for i in NOUNS_FEATURES) and any(i in tags for i in ADJECTIVES_FEATURES)
    # Adjective + verb : 'easy to use'
    cond2 = any(i in tags for i in VERBS_FEATURES) and any(i in tags for i in ADJECTIVES_FEATURES)

    return cond1 or cond2


if __name__ == '__main__':
    counter = Counter()
    for n_gram in [2, 3, 4, 5]:
        # We load n_grams
        with open("textme_{}_grams.json".format(n_gram)) as f:
            n_grams_dict = json.load(f)

        valuable_n_grams_dict = {}
        counter_tmp = Counter(n_grams_dict).most_common(1000)
        for keyphrase, occurrence in counter_tmp:
            txblob_tags = TextBlob(keyphrase).tags
            pos_tags = [tag for word, tag in txblob_tags]
            if is_valid_keyphrase(pos_tags):
                valuable_n_grams_dict[keyphrase] = occurrence

        counter.update(valuable_n_grams_dict)

    most_commons = counter.most_common(300)
    print(most_commons)
