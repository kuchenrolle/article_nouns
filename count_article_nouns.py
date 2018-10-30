import spacy
import pandas as pd
from collections import Counter

BNC_SENTENCES = "/shared/ooominds1/corpora/BNC/bnc.clean.txt"
BNC_TAGGED = "/shared/ooominds1/corpora/BNC/BNC.tagged.txt"
RESULTS = "/shared/ooominds1/Shared/article_nouns/data/frequencies.tsv"


def most_frequent_nouns(corpus_path, top_n=None):
    """Counts nouns (tag="NN*") in corpus.
    Parameters
    ----------
    corpus_path : str or path
        path to corpus file
    top_n : int
        number of most frequent nouns to be returned

    Returns
    -------
    frequencies: Counter
        Counter with noun frequencies

    Notes
    -----
        If top_n is None (default) is an integer, only the top_n most
        frequent nouns are returned.
        Nouns with non-alphabetic characters are removed.
    """
    frequencies = Counter()
    with open(corpus_path, "r") as corpus:
        for line in corpus:
            try:
                token, tag = line.strip().lower().split("\t")
                if tag.startswith("nn"):
                    frequencies[token] += 1
            except ValueError:
                continue

    # remove words with non-alphabetic characters
    for noun in list(frequencies.keys()):
        if not noun.isalpha():
            del frequencies[noun]

    if top_n:
        frequencies = {n: f for n, f in frequencies.most_common(top_n)}
        return Counter(frequencies)
    else:
        return frequencies


def generate_sentences(corpus_path):
    """Generator that yields one line of a corpus at a time.
    """
    with open(corpus_path, "r") as corpus:
        for line in corpus:
            # skip document separators
            if line.startswith("---"):
                continue
            # skip empty lines
            if not line.strip():
                continue
            else:
                yield line


def process_sentence(sentence, target_nouns, nlp):
    """Generator that extracts all article-noun combinations from a sentence.
    Parameters
    ----------
    sentence : str
        A sentence
    target_nouns : collection
        A set of target nouns for which article-noun combinations are returned
    nlp : spacy text processing object
        A spacy NLP model that can identify, tag and parse noun chunks

    Returns
    -------
    noun_article: tuple
        Each article-noun combination is yielded separately.

    Notes
    -----
    Will return all determiners and general possessive markers.
    Non-generic possessives (Mary's, Teacher's, ...) are mapped to
    "other_possessive".
    Nouns not preceded a marker have "no_article" as the value of article.
    """
    sent = nlp(sentence)
    possessives = ("my", "your", "his", "her", "its", "their", "our")

    for noun_chunk in sent.noun_chunks:
        noun = noun_chunk.root.text.lower()
        if noun not in target_nouns:
            continue
        # iterate over each noun's dependencies until article is found
        for child in noun_chunk.root.children:

            # ignore non-alphabetic words
            if not child.text.isalpha():
                continue

            if child.dep_ == "det":
                article = child.text.lower()
                yield noun, article
                break
            elif child.dep_ == "poss":
                article = child.text.lower()
                if article not in possessives:
                    article = "other_possessive"
                yield noun, article
                break

        # if no article was found, set article to "no_article"
        else:
            article = "no_article"
            yield noun, article


def main():
    nlp = spacy.load("en")
    nouns = most_frequent_nouns(BNC_TAGGED, 1000)
    sentences = generate_sentences(BNC_SENTENCES)
    nouns_articles = {noun: Counter() for noun in nouns}

    for sentence in sentences:
        for noun, article in process_sentence(sentence, nouns, nlp):
            nouns_articles[noun][article] += 1
    nouns_articles = pd.DataFrame(nouns_articles).T.fillna(0).astype(int)

    # ignore hapax legomena (add counts to no_article)
    hapaxes = nouns_articles.columns[nouns_articles.sum(0) == 1]
    nouns_articles["no_article"] += nouns_articles[hapaxes].sum(1)
    nouns_articles.drop(columns=hapaxes, inplace=True)

    # save to file
    nouns_articles.to_csv(RESULTS, sep="\t")


if __name__ == "__main__":
    main()
