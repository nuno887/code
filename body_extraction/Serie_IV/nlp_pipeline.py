# body_extractionIII/nlp_pipeline.py

from spacy_modulo import get_nlp, setup_entities, OPTIONS

# Use your module's loader. This mirrors the original behavior (NER disabled, Série III on).
nlp = get_nlp(disable_ner=True, SerieIII=True)

__all__ = ["nlp", "get_nlp", "setup_entities", "OPTIONS"]
