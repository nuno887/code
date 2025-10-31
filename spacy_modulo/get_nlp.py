import spacy
from .Entities import setup_entities


def get_nlp(disable_ner: bool = True, SerieIII: bool = True):
    exclude = ["ner"] if disable_ner else []
    nlp = spacy.load("pt_core_news_lg", exclude=exclude)
    setup_entities(nlp, SerieIII)
    return nlp
