import os
import csv
import nltk
import spacy
import json

nlp = spacy.load("en_core_web_lg")

def get_candidates(sentence,ex_id = None):
    doc = nlp(sentence)
    candidates = []
    for cand in doc.ents:
        if cand.label_=="PERSON":
            candidates.append(cand.text)
    return candidates
