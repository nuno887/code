
from spacy_modulo import get_nlp, setup_entities, OPTIONS

nlp = get_nlp(disable_ner=True, SerieIII=True)

__all__ = ["nlp", "get_nlp", "setup_entities", "OPTIONS"]
