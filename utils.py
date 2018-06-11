import re


def clean_text(text):
    # We remove non letter characters
    return re.sub("[^a-zA-Z]", " ", text)